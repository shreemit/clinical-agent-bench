from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fhir.reset_timing import ResetReceipt, parse_reset_stdout

DEFAULT_READY_TIMEOUT_S = 180


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def restore_snapshot(root: Path | None = None, ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S) -> ResetReceipt:
    """Restore the H2 snapshot and wait until HAPI answers metadata."""

    root = root or repo_root()
    script = root / "scripts" / "snapshot_restore.sh"
    env = os.environ.copy()
    env["CAB_READY_TIMEOUT_S"] = str(ready_timeout_s)
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "snapshot restore failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return parse_reset_stdout(completed.stdout)
