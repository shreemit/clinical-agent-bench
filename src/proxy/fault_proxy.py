"""Fault-injecting FHIR proxy. Not part of Week 1.

TODO(decision): the five fault templates (400, 409, 412, 422, timeout-after-commit)
are specified in the lean plan and are left unimplemented until Week 3.
"""

from __future__ import annotations


def proxy_post(resource_type: str, body: bytes, headers: dict[str, str]) -> dict:
    raise NotImplementedError(
        f"fault_proxy is a Week 3 stub and did not forward {resource_type}."
    )
