from __future__ import annotations


# TODO(decision): if snapshot restore stays above 5s, switch to DELETE + $expunge or keep reporting the miss.
def reset_patient_after_trial(patient_id: str, created_locations: list[str]) -> dict:
    raise NotImplementedError(
        "DELETE + $expunge is not yet implemented. "
        f"Refused to reset {patient_id} via {len(created_locations)} locations."
    )
