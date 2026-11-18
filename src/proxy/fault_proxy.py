from __future__ import annotations

import copy
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

from harness.http_client import HttpClient, HttpResult

FAULT_TYPES = (
    "none",
    "committed_timeout",
    "uncommitted_timeout",
    "transient_5xx",
    "conflict_409",
    "precondition_412",
    "unprocessable_422",
    "duplicate_retry",
)
MITIGATION_MODES = ("off", "preconditions")
IDEMPOTENCY_SYSTEM = "urn:cab:write"

_WRITE_METHODS = ("POST", "PUT")
_HISTORY = re.compile(r"/_history/\d+$")


class FaultProxy:
    def __init__(
        self,
        inner: HttpClient,
        *,
        fault: str = "none",
        seed: int = 0,
        on_attempt: int = 1,
        mitigation: str = "off",
        log_path: Path | None = None,
        trial_id: str = "0",
        task_id: str = "",
        run_id: str | None = None,
    ) -> None:
        """Wrap a FHIR client so one chosen write can fail in a fixed way."""

        if fault not in FAULT_TYPES:
            raise ValueError(f"unknown fault {fault}")
        if mitigation not in MITIGATION_MODES:
            raise ValueError(f"unknown mitigation {mitigation}")
        if on_attempt < 1:
            raise ValueError("on_attempt must be at least 1")
        self.inner = inner
        self.fault = fault
        self.seed = seed
        self.on_attempt = on_attempt
        self.mitigation = mitigation
        self.log_path = log_path
        self.trial_id = str(trial_id)
        self.task_id = str(task_id)
        self.run_id = uuid.uuid4().hex if run_id is None else str(run_id)
        self._writes = 0
        self._fault_fired = False
        self._type_seq: dict[str, int] = {}
        self._versions: dict[str, str] = {}
        self._committed: list[str] = []
        self._retry_key: tuple[str, str, str] | None = None
        self._retry_index: int | None = None
        self._last_post_key: tuple[str, str, str] | None = None
        self._last_post_index: int | None = None

    def begin_trial(self, trial_id: str, task_id: str) -> None:
        """Clear per-trial write counters, versions, and hidden commit locations."""

        self.trial_id = str(trial_id)
        self.task_id = str(task_id)
        self._writes = 0
        self._fault_fired = False
        self._type_seq = {}
        self._versions = {}
        self._committed = []
        self._retry_key = None
        self._retry_index = None
        self._last_post_key = None
        self._last_post_index = None

    @property
    def fault_fired(self) -> bool:
        """True once the armed fault was applied to a write in this trial."""

        return self._fault_fired

    def committed_locations(self) -> list[str]:
        """Locations HAPI accepted this trial, including writes hidden by a timeout."""

        return list(self._committed)

    def request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResult:
        """Forward one call, or inject the armed fault on the chosen write attempt."""

        started = time.perf_counter()
        method = method.upper()
        attempt: int | None = None
        if method in _WRITE_METHODS:
            self._writes += 1
            attempt = self._writes
        prepared_body, prepared_headers = self._prepare(method, url, body, headers)
        fault = "none"
        if attempt is not None and attempt == self.on_attempt and self.fault != "none":
            fault = self.fault
            self._fault_fired = True
        if fault == "duplicate_retry":
            first = self._send(method, url, prepared_body, prepared_headers, timeout)
            self._observe(method, url, first, remember_version=False)
            second = self._send(method, url, prepared_body, prepared_headers, timeout)
            self._observe(method, url, second)
            self._log(fault, attempt, 1, method, url, second.status, first)
            self._log(fault, attempt, 2, method, url, second.status, second)
            result = second
        elif fault == "committed_timeout":
            upstream = self._send(method, url, prepared_body, prepared_headers, timeout)
            self._observe(method, url, upstream, remember_version=False)
            result = _synthetic(504, _outcome("error", "timeout", "Gateway timeout."), started)
            self._log(fault, attempt, 1, method, url, 504, upstream)
        elif fault == "none":
            result = self._send(method, url, prepared_body, prepared_headers, timeout)
            self._observe(method, url, result)
            self._log(fault, attempt, 1, method, url, result.status, result)
        else:
            status, outcome = _injected_fault(fault, self.seed, _resource_type(url, prepared_body))
            result = _synthetic(status, outcome, started)
            self._log(fault, attempt, 0, method, url, status, None)
        self._note_caller_outcome(method, result.status)
        return result

    def _prepare(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> tuple[dict[str, Any] | None, dict[str, str]]:
        """Attach conditional-create or version headers when mitigation is on."""

        prepared_headers = dict(headers or {})
        if self.mitigation != "preconditions":
            return body, prepared_headers
        if method == "POST":
            payload = self._with_post_identity(url, body)
            value = str(payload["identifier"][-1]["value"])
            _set_header(prepared_headers, "If-None-Exist", _if_none_exist(value))
            return payload, prepared_headers
        if method == "PUT":
            match = self._versions.get(_resource_key(url))
            if match:
                _set_header(prepared_headers, "If-Match", match)
            return body, prepared_headers
        return body, prepared_headers

    def _with_post_identity(self, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        """Add this POST's idempotency identifier, reusing it only for a failed-write retry."""

        resource_type = _resource_type(url, body) or "Resource"
        canonical = _canonical_body(body)
        key = (resource_type, url.split("?", 1)[0].rstrip("/"), canonical)
        if self._retry_key == key and self._retry_index is not None:
            index = self._retry_index
        else:
            index = self._type_seq.get(resource_type, 0) + 1
            self._type_seq[resource_type] = index
        self._last_post_key = key
        self._last_post_index = index
        value = f"{self.run_id}:{self.trial_id}:{self.task_id}:{resource_type}:{index}"
        return _with_identifier(body if isinstance(body, dict) else {}, value)

    def _note_caller_outcome(self, method: str, status: int) -> None:
        """Arm identifier reuse only when the caller saw a failed POST and retries it next."""

        if method == "POST" and self._last_post_key is not None and self._last_post_index is not None:
            if 200 <= status < 300:
                self._retry_key = None
                self._retry_index = None
            else:
                self._retry_key = self._last_post_key
                self._retry_index = self._last_post_index
            return
        if method in _WRITE_METHODS:
            self._retry_key = None
            self._retry_index = None

    def _send(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
        timeout: float | None,
    ) -> HttpResult:
        """Forward one request without sharing mutable body or headers."""

        payload = copy.deepcopy(body) if isinstance(body, dict) else body
        return self.inner.request(method, url, body=payload, headers=dict(headers), timeout=timeout)

    def _observe(self, method: str, url: str, result: HttpResult, *, remember_version: bool = True) -> None:
        """Remember a Location HAPI accepted, and a version only when the caller saw it."""

        if 200 <= result.status < 300:
            location = _header(result.headers, "Location")
            if location:
                raw = _absolute(location, url).split("?", 1)[0]
                if raw not in self._committed:
                    self._committed.append(raw)
        if not remember_version:
            return
        token = _version_token(result)
        if token is None or result.status < 200 or result.status >= 300:
            return
        if method in _WRITE_METHODS:
            location = _header(result.headers, "Location")
            if location:
                self._versions[_resource_key(_absolute(location, url))] = token
            body = result.body if isinstance(result.body, dict) else None
            if body and body.get("id") and body.get("resourceType"):
                self._versions[_resource_key(_id_url(url, str(body["resourceType"]), str(body["id"])))] = token
            return
        if method != "GET" or "/_history/" in url.split("?", 1)[0]:
            return
        body = result.body if isinstance(result.body, dict) else None
        if not body or not body.get("id"):
            return
        if body.get("resourceType") in (None, "Bundle", "OperationOutcome", "Parameters"):
            return
        self._versions[_resource_key(url)] = token

    def _log(
        self,
        fault: str,
        attempt: int | None,
        forward: int,
        method: str,
        url: str,
        status: int,
        upstream: HttpResult | None,
    ) -> None:
        """Append one JSONL record for a caller attempt or an infra retry hop."""

        if self.log_path is None:
            return
        record = {
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "task_id": self.task_id,
            "fault": fault,
            "seed": self.seed,
            "mitigation": self.mitigation,
            "attempt": attempt,
            "forward": forward,
            "method": method,
            "url": url,
            "status": status,
            "upstream_status": None if upstream is None else upstream.status,
            "upstream_location": None if upstream is None else _header(upstream.headers, "Location"),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def _injected_fault(fault: str, seed: int, resource_type: str | None) -> tuple[int, dict[str, Any]]:
    """Status and OperationOutcome for a fault that does not touch the server."""

    if fault == "uncommitted_timeout":
        return 504, _outcome("error", "timeout", "Gateway timeout.")
    if fault == "transient_5xx":
        if seed % 2 == 0:
            return 503, _outcome("error", "transient", "The server is temporarily unavailable.")
        return 500, _outcome("error", "exception", "Internal server error.")
    if fault == "conflict_409":
        return 409, _outcome(
            "error",
            "conflict",
            "The resource has changed since it was last read.",
            _expr(resource_type, "meta.versionId"),
        )
    if fault == "precondition_412":
        return 412, _outcome(
            "error",
            "processing",
            "Precondition failed: If-Match did not match the current resource version.",
            _expr(resource_type, "meta.versionId"),
        )
    if fault == "unprocessable_422":
        return 422, _outcome(
            "error",
            "processing",
            "The resource could not be processed.",
            _expr(resource_type, "status"),
        )
    raise ValueError(f"unknown fault {fault}")


def _outcome(
    severity: str,
    code: str,
    diagnostics: str,
    expression: list[str] | None = None,
) -> dict[str, Any]:
    """Build a FHIR OperationOutcome with one issue."""

    issue: dict[str, Any] = {"severity": severity, "code": code, "diagnostics": diagnostics}
    if expression:
        issue["expression"] = expression
    return {"resourceType": "OperationOutcome", "issue": [issue]}


def _expr(resource_type: str | None, path: str) -> list[str] | None:
    """FHIRPath for one field when the resource type is known."""

    if not resource_type:
        return None
    return [f"{resource_type}.{path}"]


def _synthetic(status: int, body: dict[str, Any], started: float) -> HttpResult:
    """Build a caller-facing result that did not come from the upstream server."""

    return HttpResult(
        status=status,
        body=body,
        headers={"Content-Type": "application/fhir+json"},
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def _with_identifier(body: dict[str, Any], value: str) -> dict[str, Any]:
    """Return a copy of a resource with this trial's idempotency identifier."""

    payload = copy.deepcopy(body)
    kept = []
    for item in payload.get("identifier") or []:
        if isinstance(item, dict) and item.get("system") == IDEMPOTENCY_SYSTEM:
            continue
        kept.append(item)
    kept.append({"system": IDEMPOTENCY_SYSTEM, "value": value})
    payload["identifier"] = kept
    return payload


def _canonical_body(body: dict[str, Any] | None) -> str:
    """Stable JSON for a POST, ignoring the identifier this proxy adds."""

    payload = copy.deepcopy(body) if isinstance(body, dict) else {}
    identifiers = payload.get("identifier")
    if isinstance(identifiers, list):
        kept = [
            item
            for item in identifiers
            if not (isinstance(item, dict) and item.get("system") == IDEMPOTENCY_SYSTEM)
        ]
        if kept:
            payload["identifier"] = kept
        else:
            payload.pop("identifier", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _if_none_exist(value: str) -> str:
    """If-None-Exist search for one idempotency identifier."""

    token = quote(f"{IDEMPOTENCY_SYSTEM}|{value}", safe="|")
    return f"identifier={token}"


def _resource_type(url: str, body: dict[str, Any] | None) -> str | None:
    """Resource type from the body, otherwise from a type-like URL segment."""

    if isinstance(body, dict):
        kind = body.get("resourceType")
        if isinstance(kind, str) and kind and kind not in ("Bundle", "OperationOutcome", "Parameters"):
            return kind
    parts = [part for part in url.split("?", 1)[0].rstrip("/").split("/") if part]
    if parts and parts[-1][:1].isupper():
        return parts[-1]
    if len(parts) >= 2 and parts[-2][:1].isupper():
        return parts[-2]
    return None


def _resource_key(url: str) -> str:
    """Canonical resource URL without a query or _history suffix."""

    return _HISTORY.sub("", url.split("?", 1)[0]).rstrip("/")


def _id_url(request_url: str, resource_type: str, resource_id: str) -> str:
    """Collection or instance URL rewritten to the instance URL."""

    base = request_url.split("?", 1)[0].rstrip("/")
    suffix = "/" + resource_type
    if base.endswith(suffix):
        return base + "/" + resource_id
    return _resource_key(request_url)


def _absolute(url: str, base: str) -> str:
    """Resolve a Location header against the request URL."""

    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return urljoin(base, url)
    root = base.split("?", 1)[0]
    if not root.endswith("/"):
        root = root.rsplit("/", 1)[0] + "/"
    return urljoin(root, url)


def _version_token(result: HttpResult) -> str | None:
    """ETag from the response, or a weak ETag built from meta.versionId."""

    etag = _header(result.headers, "ETag")
    if etag:
        return etag
    body = result.body
    if not isinstance(body, dict):
        return None
    meta = body.get("meta")
    if isinstance(meta, dict) and meta.get("versionId") not in (None, ""):
        return f'W/"{meta["versionId"]}"'
    return None


def _header(headers: dict[str, str], name: str) -> str | None:
    """Return a header value, ignoring case."""

    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def _set_header(headers: dict[str, str], name: str, value: str) -> None:
    """Set a header, replacing any earlier spelling of the same name."""

    for key in list(headers):
        if key.lower() == name.lower():
            del headers[key]
    headers[name] = value
