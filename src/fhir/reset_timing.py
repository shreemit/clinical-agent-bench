from __future__ import annotations

import json
from dataclasses import dataclass

RESET_TARGET_MS = 5000


@dataclass(frozen=True)
class ResetReceipt:
    duration_ms: int
    stop_ms: int
    restore_copy_ms: int
    ready_ms: int
    under_5s: bool
    target_ms: int = RESET_TARGET_MS


def under_reset_target(duration_ms: float, target_ms: int = RESET_TARGET_MS) -> bool:
    return duration_ms < target_ms


def parse_reset_stdout(text: str) -> ResetReceipt:
    for line in text.splitlines():
        if line.startswith("RESET_JSON "):
            payload = json.loads(line[len("RESET_JSON ") :])
            return ResetReceipt(
                duration_ms=int(payload["duration_ms"]),
                stop_ms=int(payload["stop_ms"]),
                restore_copy_ms=int(payload["restore_copy_ms"]),
                ready_ms=int(payload["ready_ms"]),
                under_5s=bool(payload["under_5s"]),
                target_ms=int(payload.get("target_ms", RESET_TARGET_MS)),
            )
    raise ValueError("snapshot restore did not print RESET_JSON")
