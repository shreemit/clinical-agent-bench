"""Targeted DELETE + $expunge reset from the lean plan. Not used in Week 1.

Week 1 restores a Docker volume snapshot instead (`scripts/snapshot_restore.sh`).
The plan's go/no-go asks whether DELETE + $expunge can restore the patient
bundle hash in under 5 seconds. That path is intentionally unwired.

TODO(decision): if the measured snapshot restore stays above 5 seconds, switch
the Week 1 reset to DELETE + $expunge, or keep the snapshot and report the miss.
"""

from __future__ import annotations


def reset_patient_after_trial(patient_id: str, created_locations: list[str]) -> dict:
    raise NotImplementedError(
        "DELETE + $expunge reset is a stub. Week 1 uses snapshot restore. "
        f"Refused to reset {patient_id} via {len(created_locations)} locations."
    )
