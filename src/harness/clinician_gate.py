from __future__ import annotations

from typing import Any


class ClinicianGate:
    # TODO(decision): writes are forwarded with no clinician approval step.
    def approve(self, payload: dict[str, Any]) -> bool:
        raise NotImplementedError("ClinicianGate is not yet implemented.")
