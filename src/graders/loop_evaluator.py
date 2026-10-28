"""Repair-loop stage evaluator. Week 3 component."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LoopGrade:
    fault_family: str
    noticed: bool
    diagnostic_uptake: bool
    clean_intermediate_commits: bool
    claim_matches_record: bool


def grade_loop(*args: object, **kwargs: object) -> LoopGrade:
    raise NotImplementedError("loop_evaluator is a Week 3 stub.")
