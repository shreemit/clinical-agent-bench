from __future__ import annotations

import argparse
import json
from pathlib import Path


def agreement_counts(rows: list[dict]) -> dict:
    """Count transcript-versus-state outcomes. Rows without a transcript grade are skipped."""

    table = {"pass_pass": 0, "pass_fail": 0, "fail_pass": 0, "fail_fail": 0}
    skipped = 0
    for row in rows:
        transcript = row.get("refsol_passed")
        state = row.get("state_passed")
        if transcript is None or state is None:
            skipped += 1
            continue
        transcript_label = "pass" if transcript else "fail"
        state_label = "pass" if state else "fail"
        table[f"{transcript_label}_{state_label}"] += 1
    total = sum(table.values())
    agreed = table["pass_pass"] + table["fail_fail"]
    return {**table, "skipped": skipped, "total": total, "agreed": agreed}


def format_agreement(summary: dict) -> str:
    """Render the 2x2 transcript-versus-state table."""

    if summary["total"] == 0:
        return (
            "Transcript grader skipped: no trial has both a transcript grade and a state grade.\n"
            f"skipped rows: {summary['skipped']}"
        )
    lines = [
        f"{'':<20} {'state pass':>12} {'state fail':>12}",
        f"{'transcript pass':<20} {summary['pass_pass']:>12} {summary['pass_fail']:>12}",
        f"{'transcript fail':<20} {summary['fail_pass']:>12} {summary['fail_fail']:>12}",
        f"agreed {summary['agreed']}/{summary['total']}",
    ]
    if summary["skipped"]:
        lines.append(f"skipped rows without a transcript grade: {summary['skipped']}")
    return "\n".join(lines)


def load_results(path: Path) -> list[dict]:
    """Load JSONL trial results."""

    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the transcript-versus-state agreement table."
    )
    parser.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    print(format_agreement(agreement_counts(load_results(args.results))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
