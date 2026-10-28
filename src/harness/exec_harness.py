from __future__ import annotations

import json
from typing import Any

from harness.http_client import HttpClient, UrllibHttpClient
from harness.types import AgentAction, CapturedCall, TaskCase, ToolObservation, TrialRecord


# TODO(decision): every write is forwarded; there is no clinician approval step.
class ExecHarness:
    def __init__(
        self,
        fhir_api_base: str,
        http: HttpClient | None = None,
        max_rounds: int = 8,
    ) -> None:
        base = fhir_api_base.rstrip("/") + "/"
        self.fhir_api_base = base
        self.http = http if http is not None else UrllibHttpClient()
        self.max_rounds = max_rounds

    def step(self, action: AgentAction) -> tuple[ToolObservation, CapturedCall]:
        if action.kind not in ("GET", "POST", "PUT"):
            raise ValueError(f"step() cannot execute {action.kind}")
        if not action.url:
            raise ValueError(f"{action.kind} action is missing a url")
        headers = {"Accept": "application/fhir+json"}
        if action.kind in ("POST", "PUT"):
            headers["Content-Type"] = "application/fhir+json"
        result = self.http.request(action.kind, action.url, body=action.body, headers=headers)
        location = _header(result.headers, "Location")
        observation = ToolObservation(
            status=result.status,
            body=result.body,
            headers=result.headers,
            location=location,
        )
        captured = CapturedCall(
            method=action.kind,
            url=action.url,
            request_body=action.body,
            status=result.status,
            response_body=result.body,
            response_headers=result.headers,
            location=location,
            elapsed_ms=result.elapsed_ms,
        )
        return observation, captured

    def run(self, task: TaskCase, agent: Any) -> TrialRecord:
        observation: ToolObservation | None = None
        calls: list[CapturedCall] = []
        for round_index in range(self.max_rounds):
            try:
                action = agent.next_action(observation)
            except Exception as error:
                return TrialRecord(
                    task_id=task.id,
                    passed=False,
                    failure_reason=f"agent error: {error}",
                    calls=calls,
                    finish_text=None,
                    rounds=round_index + 1,
                )
            if action.kind == "FINISH":
                passed, reason = evaluate_trial(task, calls, action.finish_payload)
                return TrialRecord(
                    task_id=task.id,
                    passed=passed,
                    failure_reason=reason,
                    calls=calls,
                    finish_text=action.finish_payload,
                    rounds=round_index + 1,
                )
            observation, captured = self.step(action)
            calls.append(captured)
        return TrialRecord(
            task_id=task.id,
            passed=False,
            failure_reason=f"stopped after {self.max_rounds} rounds without FINISH",
            calls=calls,
            finish_text=None,
            rounds=self.max_rounds,
        )


# TODO(decision): families other than task1 have no sol, so their answers are not graded.
def evaluate_trial(
    task: TaskCase,
    calls: list[CapturedCall],
    finish_text: str | None,
) -> tuple[bool, str | None]:
    """Pass on all-2xx calls plus FINISH, and sol when the task file has one."""

    for call in calls:
        if call.status < 200 or call.status >= 300:
            return False, f"{call.method} {call.url} returned status {call.status}"
    if finish_text is None:
        return False, "agent did not FINISH"
    if task.sol is not None:
        try:
            got = json.loads(finish_text)
        except json.JSONDecodeError:
            return False, "FINISH payload is not JSON"
        if got != task.sol:
            return False, f"FINISH {got!r} != sol {task.sol!r}"
    return True, None


def _header(headers: dict[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None
