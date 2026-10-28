"""Rung A2 clinician accept/reject gate.

Not called by the Week 1 harness.

TODO(decision): confirm A2 stays unwired until autonomy-rung experiments.
Week 1 forwards writes unconditionally.
"""

from __future__ import annotations

from typing import Any


class ClinicianGate:
    def approve(self, payload: dict[str, Any]) -> bool:
        raise NotImplementedError(
            "ClinicianGate is a later-week stub and is not on the Week 1 path."
        )
