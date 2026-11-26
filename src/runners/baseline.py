from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Protocol

from agent.fhir_rules import bundle_resources, parse_mrn
from analysis.agreement import agreement_counts, format_agreement
from fhir.snapshot_reset import restore_snapshot
from graders.refsol_grader import grade_with_refsol, load_refsol, refsol_skip_message
from graders.state_collector import collect_patient_state, merge_resources, read_locations
from graders.state_diff_grader import gold_spec_for_task, grade_trial_state
from harness.exec_harness import ExecHarness
from harness.http_client import HttpClient, UrllibHttpClient
from harness.types import AgentAction, TaskCase, ToolObservation
from proxy.fault_proxy import FAULT_TYPES, MITIGATION_MODES, FaultProxy
from tasks_io import load_task_file, repo_root

logger = logging.getLogger(__name__)

PROVIDERS = ("openai", "groq")
DEFAULT_MODEL = "gpt-6-luna"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_URL = OPENAI_API_URL
GROQ_RATE_LIMIT_RETRIES = 3
GROQ_RATE_LIMIT_WAIT_CAP_S = 30.0
GROQ_RATE_LIMIT_DEFAULT_WAIT_S = 1.0
DEFAULT_MAX_COMPLETION_TOKENS = 4096
_ACTION_KINDS = ("GET", "POST", "PUT", "FINISH")
_DISTRIBUTION = "clinical-agent-bench-exec"
_USER_AGENT_VERSION_FALLBACK = "0.1.0"

_PROVIDER_CONFIG = {
    "openai": {
        "url": OPENAI_API_URL,
        "env": "OPENAI_API_KEY",
        "model": DEFAULT_MODEL,
        "reasoning_effort": True,
        "rate_limit_retries": 0,
    },
    "groq": {
        "url": GROQ_API_URL,
        "env": "GROQ_API_KEY",
        "model": GROQ_DEFAULT_MODEL,
        "reasoning_effort": False,
        "rate_limit_retries": GROQ_RATE_LIMIT_RETRIES,
    },
}


class MissingApiKeyError(RuntimeError):
    pass


class ChatClient(Protocol):
    def complete(self, messages: list[dict], model: str) -> str: ...


def require_api_key(env_var: str = "OPENAI_API_KEY") -> str:
    """Return the named API key or raise MissingApiKeyError."""

    return _api_key_or_raise(os.environ.get(env_var) or "", env_var)


def _api_key_or_raise(key: str, env_var: str) -> str:
    """Return a single-line key, or raise without including the key."""

    if "\r" in key or "\n" in key:
        raise MissingApiKeyError(
            f"{env_var} contains a carriage return or line feed. Export a single-line key."
        )
    stripped = key.strip()
    if not stripped:
        raise MissingApiKeyError(f"{env_var} is not set. Export it before running the baseline.")
    return stripped


def default_model(provider: str) -> str:
    """Model used when --model is omitted."""

    return _provider_config(provider)["model"]


def chat_user_agent() -> str:
    """User-Agent sent on every chat completion request."""

    try:
        package_version = version(_DISTRIBUTION)
    except PackageNotFoundError:
        package_version = _USER_AGENT_VERSION_FALLBACK
    return f"clinical-agent-bench/{package_version}"


class OpenAIChatClient:
    def __init__(
        self,
        api_key: str | None = None,
        http: HttpClient | None = None,
        reasoning_effort: str | None = "none",
        provider: str = "openai",
        sleep: Callable[[float], None] | None = None,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    ) -> None:
        config = _provider_config(provider)
        env_var = config["env"]
        if api_key is None:
            self.api_key = require_api_key(env_var)
        else:
            self.api_key = _api_key_or_raise(api_key, env_var)
        if max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be at least 1")
        self.provider = provider
        self.api_url = config["url"]
        self.send_reasoning_effort = bool(config["reasoning_effort"])
        self.rate_limit_retries = int(config["rate_limit_retries"])
        self.max_completion_tokens = max_completion_tokens if provider == "groq" else None
        self.http = http if http is not None else UrllibHttpClient()
        self.reasoning_effort = reasoning_effort
        self._sleep = time.sleep if sleep is None else sleep

    def complete(self, messages: list[dict], model: str) -> str:
        """Send one chat completion and return the assistant text."""

        body = _completion_body(
            messages,
            model,
            self.reasoning_effort,
            self.send_reasoning_effort,
            self.max_completion_tokens,
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": chat_user_agent(),
        }
        attempt = 0
        while True:
            result = self.http.request("POST", self.api_url, body=body, headers=headers)
            if attempt >= self.rate_limit_retries or not _retryable_limit(result):
                break
            attempt += 1
            wait = _retry_after_seconds(result.headers)
            logger.warning(
                "Groq rate limit (HTTP %s); retrying in %.2fs (%s/%s)",
                result.status,
                wait,
                attempt,
                self.rate_limit_retries,
            )
            self._sleep(wait)
        return _completion_text(result)


def _provider_config(provider: str) -> dict[str, Any]:
    """Endpoint, key variable, and request options for one provider."""

    try:
        return _PROVIDER_CONFIG[provider]
    except KeyError as error:
        raise ValueError(f"unknown provider {provider}") from error


def _completion_body(
    messages: list[dict],
    model: str,
    reasoning_effort: str | None,
    send_reasoning_effort: bool,
    max_completion_tokens: int | None,
) -> dict[str, Any]:
    """JSON body for one chat completion."""

    body: dict[str, Any] = {"model": model, "messages": messages}
    if send_reasoning_effort and reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if not send_reasoning_effort or reasoning_effort in (None, "none"):
        body["temperature"] = 0
    if max_completion_tokens is not None:
        body["max_completion_tokens"] = max_completion_tokens
    return body


def _retryable_limit(result: Any) -> bool:
    """True for a Groq 429, or a 413 whose request still fits the token limit."""

    if result.status == 429:
        return True
    return result.status == 413 and _temporary_token_overflow(result.body)


def _temporary_token_overflow(body: Any) -> bool:
    """True when a 413 asks for no more tokens than the stated limit."""

    error = _error_object(body)
    if error is None or error.get("code") != "rate_limit_exceeded":
        return False
    requested, limit = _token_counts(str(error.get("message") or ""))
    if requested is None or limit is None:
        return False
    return requested <= limit


def _token_counts(message: str) -> tuple[int | None, int | None]:
    """Requested and limit token counts named in a rate-limit message."""

    requested = re.search(r"Requested\s+(\d+)", message, re.IGNORECASE)
    limit = re.search(r"Limit\s+(\d+)", message, re.IGNORECASE)
    return (
        int(requested.group(1)) if requested else None,
        int(limit.group(1)) if limit else None,
    )


def _completion_text(result: Any) -> str:
    """Assistant text, a recovered action, or an error for a bad response."""

    if result.status == 0:
        raise RuntimeError(_transport_error(result.body))
    if result.status == 400 and _is_tool_use_failed(result.body):
        recovered = _recover_tool_action(result.body)
        if recovered is not None:
            logger.warning("recovered chat action from HTTP 400 tool_use_failed")
            return recovered
        raise RuntimeError("chat completion failed with HTTP 400: tool_use_failed")
    if result.status == 413:
        raise RuntimeError(_too_large_message(result.body))
    if result.status != 200 or not isinstance(result.body, dict):
        raise RuntimeError(f"chat completion failed with HTTP {result.status}: {result.body}")
    return _message_content(result.body)


def _transport_error(body: Any) -> str:
    """Wording for an HTTP client status of 0."""

    if isinstance(body, dict) and "error" in body:
        return f"transport error: {body['error']}"
    return f"transport error: {body}"


def _too_large_message(body: Any) -> str:
    """HTTP 413 wording, naming counts when the request cannot fit."""

    error = _error_object(body)
    message = str(error.get("message") or "") if error else ""
    requested, limit = _token_counts(message)
    if requested is not None and limit is not None and requested > limit:
        return (
            f"chat completion failed with HTTP 413: requested {requested} tokens "
            f"exceeds limit {limit}"
        )
    if error and error.get("code") == "rate_limit_exceeded":
        return f"chat completion failed with HTTP 413: rate_limit_exceeded: {message}"
    return f"chat completion failed with HTTP 413: {body}"


def _error_object(body: Any) -> dict | None:
    """Provider error object, when the body has one."""

    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if not isinstance(error, dict):
        return None
    return error


def _message_content(body: dict) -> str:
    """Return message content when the completion finished cleanly."""

    try:
        choice = body["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"chat completion response has no message: {body}") from error
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    if finish_reason not in (None, "stop"):
        raise RuntimeError(_early_stop_message(finish_reason, body))
    if content is None or (isinstance(content, str) and not content.strip()):
        shown = "missing" if finish_reason is None else finish_reason
        raise RuntimeError(f"chat completion content is empty (finish_reason={shown})")
    if not isinstance(content, str):
        raise RuntimeError(f"chat completion content is not text: {content!r}")
    return content


def _early_stop_message(finish_reason: Any, body: dict) -> str:
    """Error for a completion that did not finish with stop."""

    tokens = _reasoning_tokens(body)
    if tokens is None:
        return f"chat completion stopped early (finish_reason={finish_reason})"
    return (
        f"chat completion stopped early (finish_reason={finish_reason}, "
        f"reasoning_tokens={tokens})"
    )


def _reasoning_tokens(body: dict) -> int | None:
    """Reasoning token count from a completion usage object."""

    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    direct = usage.get("reasoning_tokens")
    if isinstance(direct, int):
        return direct
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
        return details["reasoning_tokens"]
    return None


def _is_tool_use_failed(body: Any) -> bool:
    """True when the provider rejected a tool call the client did not ask for."""

    error = _error_object(body)
    return error is not None and error.get("code") == "tool_use_failed"


def _recover_tool_action(body: Any) -> str | None:
    """Action text recovered from tool_use_failed, or None."""

    error = _error_object(body)
    if error is None:
        return None
    text = _action_text(error.get("failed_generation"))
    if text is None:
        return None
    try:
        parse_model_action(text)
    except (ValueError, json.JSONDecodeError):
        return None
    return text


def _action_text(raw: Any) -> str | None:
    """Turn failed_generation into one GET, POST, PUT, or FINISH action."""

    if isinstance(raw, str):
        text = raw.strip()
        if _leading_action(text):
            return text
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        return _action_text(parsed)
    if isinstance(raw, dict):
        if "arguments" in raw:
            return _action_from_arguments(raw.get("arguments"))
        return _action_from_arguments(raw)
    return None


def _leading_action(text: str) -> bool:
    """True when the first token is an action verb."""

    if not text:
        return False
    head = text.splitlines()[0].strip().split(" ", 1)[0].upper()
    return head in _ACTION_KINDS


def _action_from_arguments(arguments: Any) -> str | None:
    """Build an action from tool arguments."""

    if isinstance(arguments, str):
        text = arguments.strip()
        return text if _leading_action(text) else None
    if not isinstance(arguments, dict):
        return None
    action = str(arguments.get("action") or "").strip().upper()
    if action not in _ACTION_KINDS:
        return None
    if action == "FINISH":
        payload = _argument_payload(arguments)
        if payload is None:
            return "FINISH"
        rendered = payload if isinstance(payload, str) else json.dumps(payload)
        return f"FINISH\n{rendered}"
    url = arguments.get("url")
    if not isinstance(url, str) or not url.strip():
        return None
    line = f"{action} {url.strip()}"
    if action == "GET":
        return line
    payload = _argument_payload(arguments)
    if payload is None:
        return None
    rendered = payload if isinstance(payload, str) else json.dumps(payload)
    return f"{line}\n{rendered}"


def _argument_payload(arguments: dict) -> Any:
    """Body or payload field from tool arguments."""

    if "body" in arguments:
        return arguments["body"]
    if "payload" in arguments:
        return arguments["payload"]
    return None


def _retry_after_seconds(headers: dict[str, str] | None) -> float:
    """Seconds to wait after a rate limit, capped so one wait cannot stall a trial."""

    raw = ""
    for key, value in (headers or {}).items():
        if key.lower() == "retry-after":
            raw = str(value).strip()
            break
    wait = _parse_retry_after(raw)
    if not math.isfinite(wait) or wait < 0:
        return GROQ_RATE_LIMIT_DEFAULT_WAIT_S
    if wait > GROQ_RATE_LIMIT_WAIT_CAP_S:
        return GROQ_RATE_LIMIT_WAIT_CAP_S
    return wait


def _parse_retry_after(raw: str) -> float:
    """Retry-After as seconds, or the default when the header is missing."""

    if not raw:
        return GROQ_RATE_LIMIT_DEFAULT_WAIT_S
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError, OverflowError):
        return GROQ_RATE_LIMIT_DEFAULT_WAIT_S
    if parsed is None:
        return GROQ_RATE_LIMIT_DEFAULT_WAIT_S
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - datetime.now(timezone.utc)).total_seconds()


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
    """Parse one GET, POST, PUT, or FINISH reply, ignoring a trailing action."""

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
        body, _end = json.JSONDecoder().raw_decode(raw)
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
    fault: str = "none",
    seed: int = 0,
    fault_attempt: int = 1,
    mitigation: str = "off",
    fault_log: Path | None = None,
    provider: str = "openai",
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
    proxy = FaultProxy(
        http,
        fault=fault,
        seed=seed,
        on_attempt=fault_attempt,
        mitigation=mitigation,
        log_path=fault_log if fault_log is not None else output_dir / "faults.jsonl",
    )
    harness = ExecHarness(fhir_api_base, http=proxy)
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
            proxy.begin_trial(f"{task.id}:{trial}", task.id)
            pre = collect_patient_state(http, fhir_api_base, patient_id)
            pre_ids = resource_ids(pre)
            if initial_ids is None:
                initial_ids = pre_ids
            agent = PromptAgent(task, fhir_api_base, client, model)
            record = harness.run(task, agent)
            post = merge_resources(
                collect_patient_state(http, fhir_api_base, patient_id),
                read_locations(http, fhir_api_base, _locations_for_grade(proxy, record)),
            )
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
                    "provider": provider,
                    "model": model,
                    "fault": fault,
                    "seed": seed,
                    "mitigation": mitigation,
                    "fault_fired": proxy.fault_fired,
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


def _write_locations(record) -> list[str]:
    """Location URLs for writes the caller saw accepted."""

    locations: list[str] = []
    for call in record.calls:
        if call.method in ("POST", "PUT") and call.location and 200 <= call.status < 300:
            locations.append(call.location)
    return locations


def _locations_for_grade(proxy: FaultProxy, record) -> list[str]:
    """Caller-visible write Locations plus commits a timeout hid from the caller."""

    locations = _write_locations(record)
    seen = set(locations)
    for location in proxy.committed_locations():
        if location and location not in seen:
            locations.append(location)
            seen.add(location)
    return locations


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
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default="openai",
        help="chat completions provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (default: gpt-6-luna for openai, openai/gpt-oss-20b for groq)",
    )
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
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help="Groq max_completion_tokens (default: 4096)",
    )
    parser.add_argument(
        "--fault",
        default="none",
        choices=FAULT_TYPES,
        help="fault injected on one write attempt (default: none)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for deterministic faults")
    parser.add_argument(
        "--fault-attempt",
        type=int,
        default=1,
        help="1-based POST/PUT attempt that receives the fault",
    )
    parser.add_argument(
        "--mitigation",
        default="off",
        choices=MITIGATION_MODES,
        help="server precondition headers (default: off)",
    )
    parser.add_argument(
        "--fault-log",
        type=Path,
        default=None,
        help="JSONL path for proxy attempts (default: <output>/faults.jsonl)",
    )
    args = parser.parse_args(argv)
    if args.k < 1:
        print("k must be at least 1", file=sys.stderr)
        return 2
    if args.fault_attempt < 1:
        print("fault-attempt must be at least 1", file=sys.stderr)
        return 2
    if args.max_completion_tokens < 1:
        print("max-completion-tokens must be at least 1", file=sys.stderr)
        return 2
    model = args.model or default_model(args.provider)
    try:
        client = OpenAIChatClient(
            reasoning_effort=args.reasoning_effort or None,
            provider=args.provider,
            max_completion_tokens=args.max_completion_tokens,
        )
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
        model=model,
        provider=args.provider,
        client=client,
        fhir_api_base=args.fhir_base,
        http=UrllibHttpClient(),
        refsol_path=args.refsol,
        no_reset=args.no_reset,
        fault=args.fault,
        seed=args.seed,
        fault_attempt=args.fault_attempt,
        mitigation=args.mitigation,
        fault_log=args.fault_log,
    )
    print(format_agreement(agreement_counts(rows)))
    print(f"wrote {len(rows)} trial results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
