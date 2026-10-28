from __future__ import annotations

import json
from pathlib import Path

from harness.types import TaskCase


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_task_file(path: Path | None = None) -> dict[str, TaskCase]:
    path = path or repo_root() / "vendor" / "medagentbench" / "test_data_v2.json"
    rows = json.loads(path.read_text())
    tasks: dict[str, TaskCase] = {}
    for row in rows:
        task = TaskCase(
            id=row["id"],
            instruction=row.get("instruction") or "",
            context=row.get("context") or "",
            sol=row["sol"] if "sol" in row else None,
            eval_mrn=row.get("eval_MRN"),
        )
        tasks[task.id] = task
    return tasks


def load_smoke_ids(path: Path | None = None) -> list[str]:
    path = path or repo_root() / "tasks" / "smoke_ids.txt"
    ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids
