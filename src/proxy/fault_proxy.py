from __future__ import annotations


# TODO(decision): 400, 409, 412, 422, and timeout-after-commit are not injected.
def proxy_post(resource_type: str, body: bytes, headers: dict[str, str]) -> dict:
    raise NotImplementedError(f"fault_proxy is not yet implemented for {resource_type}.")
