from __future__ import annotations


# TODO(decision): smoke writes are plain POST/PUT, with no If-None-Exist header.
def attach_idempotency(
    resource: dict,
    headers: dict,
    trial_id: str,
    task_id: str,
    mode: str,
) -> tuple[dict, dict]:
    raise NotImplementedError("attach_idempotency is not yet implemented.")
