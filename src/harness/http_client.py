from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

_TRANSPORT_ERRORS = (http.client.HTTPException, TimeoutError, OSError)


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
        timeout: float | None = None,
    ) -> HttpResult: ...


class UrllibHttpClient:
    """HTTP statuses and transport failures are returned, not raised."""

    def __init__(self, timeout: float = 120) -> None:
        self.timeout = timeout

    def request(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResult:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            method=method,
            headers=dict(headers or {}),
        )
        started = time.perf_counter()
        limit = self.timeout if timeout is None else timeout
        try:
            response = urllib.request.urlopen(request, timeout=limit)
        except urllib.error.HTTPError as error:
            try:
                raw = error.read()
            except _TRANSPORT_ERRORS as read_error:
                return _transport_failure(started, read_error)
            status = error.code
            response_headers = {key: value for key, value in (error.headers.items() if error.headers else [])}
        except urllib.error.URLError as error:
            return _transport_failure(started, error.reason)
        except _TRANSPORT_ERRORS as error:
            return _transport_failure(started, error)
        else:
            try:
                raw = response.read()
                status = response.status
                response_headers = {key: value for key, value in response.headers.items()}
            except _TRANSPORT_ERRORS as error:
                return _transport_failure(started, error)
            finally:
                try:
                    response.close()
                except _TRANSPORT_ERRORS:
                    pass
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


def _transport_failure(started: float, error: BaseException) -> HttpResult:
    return HttpResult(
        status=0,
        body={"error": str(error)},
        headers={},
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )
