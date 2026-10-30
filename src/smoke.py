from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from agent.mock_agent import MockAgent
from fhir.reset_timing import ResetReceipt
from fhir.snapshot_reset import restore_snapshot
from harness.exec_harness import ExecHarness
from harness.types import TrialRecord
from tasks_io import load_smoke_ids, load_task_file, repo_root

FHIR_API_BASE = "http://127.0.0.1:8080/fhir/"


@dataclass
class SmokeRow:
    task_id: str
    passed: bool
    failure_reason: str | None
    methods: str
    finish_text: str | None
    reset: ResetReceipt


def format_smoke_table(rows: list[SmokeRow]) -> str:
    header = f"{'task_id':<12} {'result':<6} {'calls':<16} {'finish':<24} {'reset_ms':>10} {'<5s':>5}"
    lines = [header, "-" * len(header)]
    for row in rows:
        finish = row.finish_text if row.finish_text is not None else ""
        if len(finish) > 22:
            finish = finish[:21] + "…"
        lines.append(
            f"{row.task_id:<12} {'PASS' if row.passed else 'FAIL':<6} {row.methods:<16} "
            f"{finish:<24} {row.reset.duration_ms:>10} {'yes' if row.reset.under_5s else 'no':>5}"
        )
    return "\n".join(lines)


def summarize_resets(rows: list[SmokeRow]) -> str:
    durations = [row.reset.duration_ms for row in rows]
    met = sum(1 for row in rows if row.reset.under_5s)
    if not durations:
        return "no resets"
    mean = sum(durations) / len(durations)
    copies = [row.reset.restore_copy_ms for row in rows]
    readies = [row.reset.ready_ms for row in rows]
    return (
        f"reset target <{rows[0].reset.target_ms} ms: {met}/{len(rows)} met; "
        f"duration min/mean/max {min(durations)}/{mean:.0f}/{max(durations)} ms; "
        f"copy mean {sum(copies)/len(copies):.0f} ms; "
        f"ready mean {sum(readies)/len(readies):.0f} ms"
    )


def run_smoke(fhir_api_base: str = FHIR_API_BASE, root: Path | None = None) -> tuple[list[SmokeRow], int]:
    root = root or repo_root()
    tasks = load_task_file()
    rows: list[SmokeRow] = []
    harness = ExecHarness(fhir_api_base)
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    smoke_log = runs / "smoke.jsonl"
    for task_id in load_smoke_ids():
        receipt = restore_snapshot(root)
        task = tasks[task_id]
        record = harness.run(task, MockAgent(task, fhir_api_base))
        row = SmokeRow(
            task_id=task_id,
            passed=record.passed,
            failure_reason=record.failure_reason,
            methods=",".join(call.method for call in record.calls) or "-",
            finish_text=record.finish_text,
            reset=receipt,
        )
        rows.append(row)
        with smoke_log.open("a") as handle:
            handle.write(json.dumps(_row_json(row, record)) + "\n")
    failures = sum(1 for row in rows if not row.passed)
    return rows, failures


def _row_json(row: SmokeRow, record: TrialRecord) -> dict:
    return {
        "task_id": row.task_id,
        "passed": row.passed,
        "failure_reason": row.failure_reason,
        "finish_text": row.finish_text,
        "reset": {
            "duration_ms": row.reset.duration_ms,
            "stop_ms": row.reset.stop_ms,
            "restore_copy_ms": row.reset.restore_copy_ms,
            "ready_ms": row.reset.ready_ms,
            "under_5s": row.reset.under_5s,
            "target_ms": row.reset.target_ms,
        },
        "calls": [_call_json(call) for call in record.calls],
    }


def _call_json(call) -> dict:
    body = call.response_body
    summary: dict = {}
    if isinstance(body, dict):
        summary = {
            "resourceType": body.get("resourceType"),
            "id": body.get("id"),
            "total": body.get("total"),
        }
        if body.get("resourceType") == "OperationOutcome":
            summary["issue"] = body.get("issue")
        # TODO(decision): GET bundles are summarized; POST and PUT bodies are stored in full.
        if call.method in ("POST", "PUT"):
            summary["body"] = body
    return {
        "method": call.method,
        "url": call.url,
        "status": call.status,
        "location": call.location,
        "elapsed_ms": round(call.elapsed_ms, 1),
        "request_body": call.request_body,
        "response": summary,
    }


# TODO(decision): smoke exits 0 when tasks pass even if reset exceeds 5 seconds.
def main() -> int:
    base = os.environ.get("CAB_FHIR_BASE", FHIR_API_BASE)
    rows, failures = run_smoke(fhir_api_base=base)
    print(format_smoke_table(rows))
    print(summarize_resets(rows))
    for row in rows:
        if row.failure_reason:
            print(f"{row.task_id}: {row.failure_reason}")
    print(f"tasks {len(rows) - failures}/{len(rows)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
