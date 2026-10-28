from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

from agent.fhir_rules import (
    FROZEN_NOW,
    bundle_resources,
    calculate_age,
    first_quoted,
    latest_quantity,
    magnesium_dose,
    parse_mrn,
    parse_name_and_dob,
    potassium_dose_meq,
    require_bp_code,
    require_loinc,
    require_ndc,
    require_snomed,
    resource_url_from_location,
)
from agent.payloads import (
    blood_pressure_observation,
    medication_request,
    potassium_followup,
    service_request,
)
from harness.types import AgentAction, TaskCase, ToolObservation


# TODO(decision): only task3 issues PUT; the benchmark tasks themselves only POST.
# TODO(decision): funcs_v1.json is vendored but each family uses a hand-written script.
class MockAgent:
    def __init__(self, task: TaskCase, fhir_api_base: str) -> None:
        self.task = task
        self.base = fhir_api_base.rstrip("/") + "/"
        self.phase = 0
        self.memory: dict[str, Any] = {}

    def next_action(self, observation: ToolObservation | None) -> AgentAction:
        if observation is not None and not _ok(observation.status) and self.phase > 0:
            return _finish({"error": observation.status})
        handler = {
            "task1": self._task1,
            "task2": self._task2,
            "task3": self._task3,
            "task4": self._task4,
            "task5": self._task5,
            "task6": self._task6,
            "task7": self._task7,
            "task8": self._task8,
            "task9": self._task9,
            "task10": self._task10,
        }.get(self.task.family)
        if handler is None:
            raise ValueError(f"no script for {self.task.family}")
        return handler(observation)

    def _task1(self, observation: ToolObservation | None) -> AgentAction:
        if self.phase == 0:
            given, family, dob = parse_name_and_dob(self.task.instruction)
            self.phase = 1
            return _get(self.base, "Patient", {"family": family, "given": given, "birthdate": dob})
        patients = bundle_resources(observation.body if observation else None)
        if not patients:
            return _finish(["Patient not found"])
        patient = patients[0]
        mrn = patient.get("id")
        return _finish([mrn])

    def _task2(self, observation: ToolObservation | None) -> AgentAction:
        mrn = self.task.eval_mrn or parse_mrn(self.task.instruction)
        if self.phase == 0:
            self.phase = 1
            return _get(self.base, "Patient", {"identifier": mrn})
        patients = bundle_resources(observation.body if observation else None)
        dob = datetime.fromisoformat(patients[0]["birthDate"])
        return _finish([calculate_age(dob)])

    def _task3(self, observation: ToolObservation | None) -> AgentAction:
        mrn = self.task.eval_mrn or parse_mrn(self.task.instruction)
        value = first_quoted(self.task.instruction)
        code_text = require_bp_code(self.task.context)
        if self.phase == 0:
            self.phase = 1
            return _post(self.base + "Observation", blood_pressure_observation(mrn, value, code_text))
        if self.phase == 1:
            location = observation.location if observation else None
            if not location:
                return _finish({"error": "POST did not return Location"})
            url, resource_id = resource_url_from_location(location, self.base)
            self.phase = 2
            return _put(url, blood_pressure_observation(mrn, value, code_text, resource_id))
        return _finish([])

    def _task4(self, observation: ToolObservation | None) -> AgentAction:
        mrn = self.task.eval_mrn or parse_mrn(self.task.instruction)
        if self.phase == 0:
            self.phase = 1
            return _get(self.base, "Observation", {"patient": mrn, "code": "MG", "_count": "5000"})
        _, value, _ = latest_quantity(bundle_resources(observation.body if observation else None), within_hours=24)
        return _finish([value if value is not None else -1])

    def _task5(self, observation: ToolObservation | None) -> AgentAction:
        mrn = self.task.eval_mrn or parse_mrn(self.task.instruction)
        if self.phase == 0:
            self.phase = 1
            return _get(self.base, "Observation", {"patient": mrn, "code": "MG", "_count": "5000"})
        _, value, _ = latest_quantity(bundle_resources(observation.body if observation else None), within_hours=24)
        if self.phase == 1:
            self.memory["value"] = value
            dose = None if value is None else magnesium_dose(value)
            if dose is None:
                return _finish([value if value is not None else -1])
            grams, hours = dose
            self.phase = 2
            return _post(
                self.base + "MedicationRequest",
                medication_request(
                    mrn,
                    require_ndc(self.task.context),
                    "IV",
                    grams,
                    "g",
                    rate_value=hours,
                    rate_unit="h",
                ),
            )
        return _finish([self.memory["value"] if self.memory["value"] is not None else -1])

    def _task6(self, observation: ToolObservation | None) -> AgentAction:
        mrn = self.task.eval_mrn or parse_mrn(self.task.instruction)
        if self.phase == 0:
            self.phase = 1
            return _get(self.base, "Observation", {"patient": mrn, "code": "GLU", "_count": "5000"})
        resources = bundle_resources(observation.body if observation else None)
        total = 0.0
        count = 0
        cutoff = FROZEN_NOW - timedelta(hours=24)
        for resource in resources:
            raw = resource.get("effectiveDateTime")
            if not raw or "valueQuantity" not in resource:
                continue
            if datetime.fromisoformat(raw) < cutoff:
                continue
            total += resource["valueQuantity"]["value"]
            count += 1
        return _finish([total / count if count else -1])

    def _task7(self, observation: ToolObservation | None) -> AgentAction:
        mrn = self.task.eval_mrn or parse_mrn(self.task.instruction)
        if self.phase == 0:
            self.phase = 1
            return _get(self.base, "Observation", {"patient": mrn, "code": "GLU", "_count": "5000"})
        _, value, _ = latest_quantity(bundle_resources(observation.body if observation else None))
        return _finish([value if value is not None else -1])

    def _task8(self, observation: ToolObservation | None) -> AgentAction:
        del observation
        mrn = self.task.eval_mrn or parse_mrn(self.task.instruction)
        if self.phase == 0:
            self.phase = 1
            return _post(
                self.base + "ServiceRequest",
                service_request(
                    mrn,
                    "http://snomed.info/sct",
                    require_snomed(self.task.context),
                    note=first_quoted(self.task.instruction),
                ),
            )
        return _finish([])

    def _task9(self, observation: ToolObservation | None) -> AgentAction:
        mrn = self.task.eval_mrn or parse_mrn(self.task.instruction)
        if self.phase == 0:
            self.phase = 1
            return _get(self.base, "Observation", {"patient": mrn, "code": "K", "_count": "5000"})
        if self.phase == 1:
            _, value, _ = latest_quantity(bundle_resources(observation.body if observation else None))
            self.memory["value"] = value
            dose = None if value is None else potassium_dose_meq(value)
            if dose is None:
                return _finish([value if value is not None else -1])
            self.phase = 2
            return _post(
                self.base + "MedicationRequest",
                medication_request(mrn, require_ndc(self.task.context), "oral", dose, "mEq"),
            )
        if self.phase == 2:
            self.phase = 3
            return _post(
                self.base + "ServiceRequest",
                potassium_followup(mrn, require_loinc(self.task.context)),
            )
        return _finish([self.memory["value"] if self.memory["value"] is not None else -1])

    def _task10(self, observation: ToolObservation | None) -> AgentAction:
        mrn = self.task.eval_mrn or parse_mrn(self.task.instruction)
        if self.phase == 0:
            self.phase = 1
            return _get(self.base, "Observation", {"patient": mrn, "code": "A1C", "_count": "5000"})
        if self.phase == 1:
            when, value, raw = latest_quantity(bundle_resources(observation.body if observation else None))
            self.memory["answer"] = [-1] if value is None else [value, raw]
            from agent.fhir_rules import ONE_YEAR_BEFORE

            needs_order = value is None or (when is not None and when < ONE_YEAR_BEFORE)
            if not needs_order:
                return _finish(self.memory["answer"])
            self.phase = 2
            return _post(
                self.base + "ServiceRequest",
                service_request(mrn, "http://loinc.org", require_loinc(self.task.context)),
            )
        return _finish(self.memory["answer"])


def _ok(status: int) -> bool:
    return 200 <= status < 300


def _finish(payload: Any) -> AgentAction:
    return AgentAction(kind="FINISH", finish_payload=json.dumps(payload))


def _get(base: str, resource: str, params: dict[str, str]) -> AgentAction:
    query = dict(params)
    query["_format"] = "json"
    return AgentAction(kind="GET", url=f"{base}{resource}?{urlencode(query)}")


def _post(url: str, body: dict[str, Any]) -> AgentAction:
    return AgentAction(kind="POST", url=url, body=body)


def _put(url: str, body: dict[str, Any]) -> AgentAction:
    return AgentAction(kind="PUT", url=url, body=body)

