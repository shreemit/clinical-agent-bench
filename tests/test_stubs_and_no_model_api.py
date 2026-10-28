from pathlib import Path

import pytest

from analysis.autonomy_certificate import autonomy_certificate
from fhir.reset_controller import reset_patient_after_trial
from graders.loop_evaluator import grade_loop
from graders.state_diff_grader import grade_trial_state
from harness.clinician_gate import ClinicianGate
from harness.idempotency import attach_idempotency
from proxy.fault_proxy import proxy_post
from runners.prefix_runner import run_from_prefix


def test_unimplemented_modules_raise():
    with pytest.raises(NotImplementedError):
        ClinicianGate().approve({})
    with pytest.raises(NotImplementedError):
        attach_idempotency({}, {}, "t", "task", "none")
    with pytest.raises(NotImplementedError):
        proxy_post("Observation", b"{}", {})
    with pytest.raises(NotImplementedError):
        grade_trial_state({}, {}, {}, "")
    with pytest.raises(NotImplementedError):
        grade_loop()
    with pytest.raises(NotImplementedError):
        autonomy_certificate()
    with pytest.raises(NotImplementedError):
        run_from_prefix()
    with pytest.raises(NotImplementedError):
        reset_patient_after_trial("S1", [])


def test_source_does_not_import_a_model_sdk():
    root = Path(__file__).resolve().parents[1] / "src"
    banned = ("openai", "anthropic", "groq", "together", "litellm")
    offenders = []
    for path in root.rglob("*.py"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            lowered = stripped.lower()
            for token in banned:
                if token in lowered:
                    offenders.append(f"{path}: {stripped}")
    assert offenders == []
