"""If-None-Exist idempotency wrapper. Week 3 component; not on the Week 1 path.

TODO(decision): Modes A/B/C are specified in the lean plan and are not part of
the Week 1 mock-agent smoke test.
"""

from __future__ import annotations


def attach_idempotency(
    resource: dict,
    headers: dict,
    trial_id: str,
    task_id: str,
    mode: str,
) -> tuple[dict, dict]:
    raise NotImplementedError(
        "attach_idempotency is a Week 3 stub and is not on the Week 1 path."
    )
