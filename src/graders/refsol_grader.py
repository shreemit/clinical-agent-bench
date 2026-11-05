from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from harness.types import TaskCase


def refsol_skip_message(path: Path) -> str:
    """Message used when the authors' reference grader file is absent."""

    return f"refsol skipped: {path} is not present"


def load_refsol(path: Path):
    """Load a dropped-in refsol module, or None when the file is absent."""

    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("cab_refsol", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"refsol failed to load: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ImportError(f"refsol failed to load: {path}: {error}") from error
    return module


def grade_with_refsol(
    module,
    task: TaskCase,
    history: list[dict],
    result: str,
    fhir_api_base: str,
) -> bool:
    """Call taskN(case_data, results, fhir_api_base) on a dropped-in refsol module."""

    function = getattr(module, task.family, None)
    if function is None and hasattr(module, "eval"):
        function = module.eval
    if function is None:
        raise AttributeError(f"refsol has no {task.family} or eval function")
    case_data = {
        "id": task.id,
        "instruction": task.instruction,
        "context": task.context,
        "sol": task.sol,
        "eval_MRN": task.eval_mrn,
    }
    results = SimpleNamespace(
        history=[
            SimpleNamespace(role=item.get("role"), content=item.get("content") or "")
            for item in history
        ],
        result=result,
    )
    return function(case_data, results, fhir_api_base) is True
