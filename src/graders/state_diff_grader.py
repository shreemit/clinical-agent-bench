from __future__ import annotations


# TODO(decision): pass/fail is HTTP success plus sol when present, not a chart diff.
def grade_trial_state(pre_bundle: dict, post_bundle: dict, gold: dict, final_text: str) -> dict:
    raise NotImplementedError("state_diff_grader is not yet implemented.")
