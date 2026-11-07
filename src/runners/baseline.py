from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Protocol

from agent.fhir_rules import bundle_resources, parse_mrn
from analysis.agreement import agreement_counts, format_agreement
from fhir.snapshot_reset import restore_snapshot
from graders.refsol_grader import grade_with_refsol, load_refsol, refsol_skip_message
from graders.state_collector import collect_patient_state
from graders.state_diff_grader import gold_spec_for_task, grade_trial_state
from harness.exec_harness import ExecHarness
from harness.http_client import HttpClient, UrllibHttpClient
from harness.types import AgentAction, TaskCase, ToolObservation
from tasks_io import load_task_file, repo_root

DEFAULT_MODEL = "gpt-6-luna"
API_URL = "https://api.openai.com/v1/chat/completions"


class MissingApiKeyError(RuntimeError):
    pass


class ChatClient(Protocol):
    def complete(self, messages: list[dict], model: str) -> str: ...


def require_api_key() -> str:
    """Return OPENAI_API_KEY or raise MissingApiKeyError."""

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise MissingApiKeyError(
            "OPENAI_API_KEY is not set. Export it before running the baseline."
        )
    return key


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str | None = None,
        http: HttpClient | None = None,
        reasoning_effort: str | None = "none",
    ) -> None:
        if api_key is None:
            self.api_key = require_api_key()
        else:
            self.api_key = api_key.strip()
            if not self.api_key:
                raise MissingApiKeyError(
                    "OPENAI_API_KEY is not set. Export it before running the baseline."
                )
        self.http = http if http is not None else UrllibHttpClient()
        self.reasoning_effort = reasoning_effort

    def complete(self, messages: list[dict], model: str) -> str:
        """Send one chat completion and return the assistant text."""

        body: dict[str, Any] = {"model": model, "messages": messages}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_effort in (None, "none"):
            body["temperature"] = 0
        result = self.http.request(
            "POST",
            API_URL,
            body=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        if result.status != 200 or not isinstance(result.body, dict):
            raise RuntimeError(
                f"chat completion failed with status {result.status}: {result.body}"
            )
        try:
            content = result.body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"chat completion response has no message: {result.body}") from error
        if not isinstance(content, str):
            raise RuntimeError(f"chat completion content is not text: {content!r}")
        return content


def load_task_ids(path: Path) -> list[str]:
    """Read task ids, ignoring blank lines and comments."""

    ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def system_prompt(fhir_api_base: str) -> str:
    """Instructions for one FHIR action per reply."""

    base = fhir_api_base.rstrip("/") + "/"
    return (
        "You are a clinical agent using a FHIR R4 server. "
        f"The FHIR base URL is {base}. "
        "The current time is 2023-11-13T10:15:00+00:00. "
        "Reply with exactly one action and no markdown.\n"
        "GET <url>\n"
        "POST <url>\n"
        "<json body>\n"
        "PUT <url>\n"
        "<json body>\n"
        "FINISH\n"
        "<json answer>\n"
        "Use the base URL in every request. "
        "Record vitals as Observation resources, medication orders as MedicationRequest, "
        "and referrals or labs as ServiceRequest. "
        "Route is an object with a text field, and note is a list of objects with a text field. "
        "After the chart is correct, answer with FINISH and a JSON array."
    )


def user_prompt(task: TaskCase) -> str:
    """Task instruction plus the clinical context."""

    parts = [task.instruction.strip()]
    if task.context and task.context.strip():
        parts.append(task.context.strip())
    return "\n\n".join(parts)


def parse_model_action(text: str) -> AgentAction:
    """Parse a GET, POST, PUT, or FINISH reply."""

    cleaned = _strip_fences(text).strip()
    if not cleaned:
        raise ValueError("model returned an empty action")
    lines = cleaned.splitlines()
    head = lines[0].strip()
    kind, _, rest = head.partition(" ")
    kind = kind.upper()
    if kind == "FINISH":
        payload = "\n".join(lines[1:]).strip() or rest.strip() or "[]"
        return AgentAction(kind="FINISH", finish_payload=payload)
    if kind not in ("GET", "POST", "PUT"):
        raise ValueError(f"unrecognized action {head!r}")
    url = rest.strip()
    if not url:
        raise ValueError(f"{kind} is missing a url")
    body = None
    if kind in ("POST", "PUT"):
        raw = "\n".join(lines[1:]).strip()
        if not raw:
            raise ValueError(f"{kind} is missing a JSON body")
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError(f"{kind} body must be a JSON object")
    return AgentAction(kind=kind, url=url, body=body)


def render_observation(action: AgentAction, observation: ToolObservation) -> str:
    """Format a FHIR response, marking an accepted write for transcript graders."""

    body = observation.body
    text = body if isinstance(body, str) else json.dumps(body)
    if action.kind in ("POST", "PUT") and 200 <= observation.status < 300:
        location = observation.location or ""
        return (
            f"POST request accepted\nstatus: {observation.status}\nlocation: {location}\n{text}"
        )
    return f"status: {observation.status}\n{text}"


def history_for_refsol(messages: list[dict]) -> list[dict]:
    """Map chat messages to agent and user turns, omitting the system prompt."""

    history: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "assistant":
            history.append({"role": "agent", "content": message.get("content") or ""})
        else:
            history.append({"role": "user", "content": message.get("content") or ""})
    return history


def resource_ids(bundle: dict) -> list[str]:
    """Stable Type/id labels for resources in a bundle."""

    labels: list[str] = []
    for resource in bundle_resources(bundle):
        resource_id = resource.get("id")
        kind = str(resource.get("resourceType") or "Resource")
        labels.append(f"{kind}/{resource_id}" if resource_id else kind)
    return labels


class PromptAgent:
    def __init__(self, task: TaskCase, fhir_api_base: str, client: ChatClient, model: str) -> None:
        self.client = client
        self.model = model
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt(fhir_api_base)},
            {"role": "user", "content": user_prompt(task)},
        ]
        self.last_action: AgentAction | None = None
        self.messages_before_post: list[dict] | None = None
        self.pending_post_call: dict | None = None

    def next_action(self, observation: ToolObservation | None) -> AgentAction:
        if observation is not None and self.last_action is not None:
            self.messages.append(
                {"role": "user", "content": render_observation(self.last_action, observation)}
            )
        reply = self.client.complete(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": reply})
        action = parse_model_action(reply)
        if action.kind in ("POST", "PUT") and self.pending_post_call is None:
            self.messages_before_post = copy.deepcopy(self.messages[:-1])
            self.pending_post_call = {
                "kind": action.kind,
                "url": action.url,
                "body": action.body,
                "raw": reply,
            }
        self.last_action = action
        return action

    def prefix(self, task_id: str) -> dict:
        """Transcript up to the first write, or the full transcript when there is no write."""

        if self.pending_post_call is None:
            messages = copy.deepcopy(self.messages)
            pending = None
        else:
            messages = self.messages_before_post or []
            pending = self.pending_post_call
        return {
            "task_id": task_id,
            "model_id": self.model,
            "messages_before_post": messages,
            "pending_post_call": pending,
        }


def run_baseline(
    tasks: list[TaskCase],
    *,
    k: int,
    output_dir: Path,
    model: str,
    client: ChatClient,
    fhir_api_base: str,
    http: HttpClient,
    refsol_path: Path,
    no_reset: bool = False,
    reset: Callable[[], object] | None = None,
) -> list[dict]:
    """Run k trials per task, grade each chart, and write prefixes and results."""

    if k < 1:
        raise ValueError("k must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_dir = output_dir / "prefixes"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    module = load_refsol(refsol_path)
    skip_message = None if module is not None else refsol_skip_message(refsol_path)
    if skip_message:
        print(skip_message, file=sys.stderr)
    if no_reset:
        print("resets disabled: chart writes persist across trials")
    harness = ExecHarness(fhir_api_base, http=http)
    rows: list[dict] = []
    for task in tasks:
        patient_id = task.eval_mrn or parse_mrn(task.instruction)
        initial_ids: list[str] | None = None
        prefix_rel = f"prefixes/{task.id}.json"
        for trial in range(k):
            if not no_reset:
                if reset is None:
                    restore_snapshot()
                else:
                    reset()
            pre = collect_patient_state(http, fhir_api_base, patient_id)
            pre_ids = resource_ids(pre)
            if initial_ids is None:
                initial_ids = pre_ids
            agent = PromptAgent(task, fhir_api_base, client, model)
            record = harness.run(task, agent)
            post = collect_patient_state(http, fhir_api_base, patient_id)
            gold = gold_spec_for_task(task, pre)
            state = grade_trial_state(pre, post, gold, record.finish_text or "")
            refsol_passed: bool | None = None
            refsol_status = "skipped"
            refsol_message = skip_message
            if module is not None:
                try:
                    refsol_passed = grade_with_refsol(
                        module,
                        task,
                        history_for_refsol(agent.messages),
                        record.finish_text or "",
                        fhir_api_base,
                    )
                    refsol_status = "graded"
                    refsol_message = None
                except Exception as error:
                    refsol_passed = None
                    refsol_status = "error"
                    refsol_message = str(error)
            mismatches = list(state["mismatches"])
            state_passed = bool(state["passed"])
            if record.finish_text is None and record.failure_reason:
                state_passed = False
                if state["passed"]:
                    mismatches.append(f"trial did not finish: {record.failure_reason}")
            if trial == 0:
                (prefix_dir / f"{task.id}.json").write_text(
                    json.dumps(agent.prefix(task.id), indent=2) + "\n"
                )
            row = {
                    "task_id": task.id,
                    "trial": trial,
                    "model": model,
                    "reset": not no_reset,
                    "contaminated": sorted(pre_ids) != sorted(initial_ids),
                    "state_passed": state_passed,
                    "failure_reason": record.failure_reason,
                    "mismatches": mismatches,
                    "created": [_label(resource) for resource in state["created"]],
                    "updated": [
                        f"{item['resourceType']}/{item['id']}" for item in state["updated"]
                    ],
                    "deleted": [_label(resource) for resource in state["deleted"]],
                    "refsol_passed": refsol_passed,
                    "refsol_status": refsol_status,
                    "refsol_message": refsol_message,
                    "finish_text": record.finish_text,
                    "prefix_path": prefix_rel,
                    "rounds": record.rounds,
                    "pre_ids": pre_ids,
                    "post_ids": resource_ids(post),
            }
            rows.append(row)
            _write_outputs(output_dir, rows)
    return rows


def _write_outputs(output_dir: Path, rows: list[dict]) -> None:
    """Write results and the agreement table for every trial finished so far."""

    (output_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output_dir / "agreement.txt").write_text(format_agreement(agreement_counts(rows)) + "\n")


def _label(resource: dict) -> str:
    resource_id = resource.get("id")
    kind = resource.get("resourceType", "resource")
    if resource_id:
        return f"{kind}/{resource_id}"
    return str(kind)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a model baseline against FHIR and grade the chart diff."
    )
    parser.add_argument("--k", type=int, default=1, help="trials per task (use 1 or 5)")
    parser.add_argument("--output", type=Path, default=repo_root() / "runs" / "baseline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=Path, default=repo_root() / "tasks" / "baseline_100.txt")
    parser.add_argument(
        "--refsol",
        type=Path,
        default=repo_root() / "vendor" / "medagentbench" / "refsol.py",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="keep chart writes between trials so later trials see earlier orders",
    )
    parser.add_argument(
        "--fhir-base",
        default=os.environ.get("CAB_FHIR_BASE", "http://127.0.0.1:8080/fhir/"),
    )
    parser.add_argument("--reasoning-effort", default="none")
    args = parser.parse_args(argv)
    if args.k < 1:
        print("k must be at least 1", file=sys.stderr)
        return 2
    try:
        client = OpenAIChatClient(reasoning_effort=args.reasoning_effort or None)
    except MissingApiKeyError as error:
        print(str(error), file=sys.stderr)
        return 2
    catalog = load_task_file()
    selected: list[TaskCase] = []
    for task_id in load_task_ids(args.tasks):
        if task_id not in catalog:
            print(f"unknown task id {task_id}", file=sys.stderr)
            return 2
        selected.append(catalog[task_id])
    rows = run_baseline(
        selected,
        k=args.k,
        output_dir=args.output,
        model=args.model,
        client=client,
        fhir_api_base=args.fhir_base,
        http=UrllibHttpClient(),
        refsol_path=args.refsol,
        no_reset=args.no_reset,
    )
    print(format_agreement(agreement_counts(rows)))
    print(f"wrote {len(rows)} trial results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
