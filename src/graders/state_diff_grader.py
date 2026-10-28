"""Judge-free state-diff grader. Week 2 component.

TODO(decision): Week 1 smoke pass/fail is HTTP success plus `sol` when the
task file has one. It does not diff the patient chart.
"""

from __future__ import annotations


def grade_trial_state(pre_bundle: dict, post_bundle: dict, gold: dict, final_text: str) -> dict:
    raise NotImplementedError("state_diff_grader is a Week 2 stub.")
