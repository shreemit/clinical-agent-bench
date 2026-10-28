from __future__ import annotations

# TODO(decision): route is a CodeableConcept and note is an array, not the reference checker's string route and object note.

from typing import Any

from agent.fhir_rules import AUTHORED_ON, NEXT_MORNING


def blood_pressure_observation(mrn: str, value: str, code_text: str, resource_id: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "resourceType": "Observation",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs",
                    }
                ]
            }
        ],
        "code": {"text": code_text},
        "effectiveDateTime": AUTHORED_ON,
        "valueString": value,
        "subject": {"reference": f"Patient/{mrn}"},
    }
    if resource_id is not None:
        body["id"] = resource_id
    return body


def medication_request(
    mrn: str,
    ndc: str,
    route_text: str,
    dose_value: float,
    dose_unit: str,
    rate_value: float | None = None,
    rate_unit: str | None = None,
) -> dict[str, Any]:
    dose_and_rate: dict[str, Any] = {"doseQuantity": {"value": dose_value, "unit": dose_unit}}
    if rate_value is not None:
        dose_and_rate["rateQuantity"] = {"value": rate_value, "unit": rate_unit}
    return {
        "resourceType": "MedicationRequest",
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {
            "coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": ndc}]
        },
        "subject": {"reference": f"Patient/{mrn}"},
        "authoredOn": AUTHORED_ON,
        "dosageInstruction": [
            {
                "route": {"text": route_text},
                "doseAndRate": [dose_and_rate],
            }
        ],
    }


def service_request(
    mrn: str,
    system: str,
    code: str,
    note: str | None = None,
    occurrence: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "resourceType": "ServiceRequest",
        "status": "active",
        "intent": "order",
        "priority": "stat",
        "code": {"coding": [{"system": system, "code": code}]},
        "subject": {"reference": f"Patient/{mrn}"},
        "authoredOn": AUTHORED_ON,
    }
    if note is not None:
        body["note"] = [{"text": note}]
    if occurrence is not None:
        body["occurrenceDateTime"] = occurrence
    return body


def potassium_followup(mrn: str, loinc: str) -> dict[str, Any]:
    return service_request(mrn, "http://loinc.org", loinc, occurrence=NEXT_MORNING)
