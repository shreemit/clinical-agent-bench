from __future__ import annotations

import subprocess
from pathlib import Path

from fhir.reset_timing import ResetReceipt, parse_reset_stdout


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def restore_snapshot(root: Path | None = None) -> ResetReceipt:
    root = root or repo_root()
    script = root / "scripts" / "snapshot_restore.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "snapshot restore failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return parse_reset_stdout(completed.stdout)
