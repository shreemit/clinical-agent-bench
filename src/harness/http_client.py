"""Minimal FHIR HTTP client. Stdlib only; no model-provider SDKs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class HttpResult:
    status: int
    body: Any
    headers: dict[str, str]
    elapsed_ms: float


class HttpClient(Protocol):
    def request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 120,
    ) -> HttpResult: ...


class UrllibHttpClient:
    """Real HTTP. HTTP error statuses are returned, not raised."""

    def request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 120,
    ) -> HttpResult:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            method=method,
            headers=dict(headers or {}),
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                status = response.status
                response_headers = {key: value for key, value in response.headers.items()}
        except urllib.error.HTTPError as error:
            raw = error.read()
            status = error.code
            response_headers = {key: value for key, value in (error.headers.items() if error.headers else [])}
        except urllib.error.URLError as error:
            elapsed_ms = (time.perf_counter() - started) * 1000
            return HttpResult(
                status=0,
                body={"error": str(error.reason)},
                headers={},
                elapsed_ms=elapsed_ms,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        text = raw.decode("utf-8", errors="replace")
        parsed: Any
        if not text:
            parsed = None
        else:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = text
        return HttpResult(status=status, body=parsed, headers=response_headers, elapsed_ms=elapsed_ms)
