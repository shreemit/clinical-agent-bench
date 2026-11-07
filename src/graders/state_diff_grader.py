from __future__ import annotations

import copy
import json
from typing import Any

from agent.fhir_rules import (
    ONE_YEAR_BEFORE,
    bundle_resources,
    first_quoted,
    latest_quantity,
    magnesium_dose,
    parse_mrn,
    potassium_dose_meq,
    require_bp_code,
    require_loinc,
    require_ndc,
    require_snomed,
)
from agent.payloads import (
    blood_pressure_observation,
    medication_request,
    potassium_followup,
    service_request,
)
from harness.types import TaskCase

_IGNORED = {"meta", "text", "implicitRules", "language"}
_NUMBER_TOLERANCE = 0.1
_READ_ONLY = {"task1", "task2", "task4", "task6", "task7"}


def diff_bundles(pre_bundle: dict, post_bundle: dict) -> dict:
    """Return resources created, updated, or deleted between two FHIR bundles."""

    pre_by, pre_anon = _index(pre_bundle)
    post_by, post_anon = _index(post_bundle)
    created: list[dict] = []
    updated: list[dict] = []
    deleted: list[dict] = []
    for key, resource in post_by.items():
        if key not in pre_by:
            created.append(_canonical(resource))
            continue
        before = _canonical(pre_by[key])
        after = _canonical(resource)
        if before != after:
            updated.append(
                {"resourceType": key[0], "id": key[1], "before": before, "after": after}
            )
    for key, resource in pre_by.items():
        if key not in post_by:
            deleted.append(_canonical(resource))
    remaining = list(post_anon)
    for resource in pre_anon:
        canon = _canonical(resource)
        match_at = next(
            (index for index, item in enumerate(remaining) if _canonical(item) == canon),
            None,
        )
        if match_at is None:
            deleted.append(canon)
        else:
            remaining.pop(match_at)
    created.extend(_canonical(item) for item in remaining)
    created.sort(key=_sort_key)
    deleted.sort(key=_sort_key)
    updated.sort(key=lambda item: (item["resourceType"], item["id"]))
    return {"created": created, "updated": updated, "deleted": deleted}


def grade_diff(diff: dict, gold: dict) -> dict:
    """Pass when the diff matches the expected fields. Numbers match within 0.1."""

    mismatches: list[str] = []
    created = list(diff.get("created") or [])
    used = [False] * len(created)
    for expected in gold.get("created") or []:
        found = False
        for index, actual in enumerate(created):
            if used[index]:
                continue
            if _match(expected, actual):
                used[index] = True
                found = True
                break
        if not found:
            kind = expected.get("resourceType", "resource") if isinstance(expected, dict) else "resource"
            mismatches.append(f"missing created resource: {kind}")
    for index, actual in enumerate(created):
        if not used[index]:
            mismatches.append(f"unexpected created resource: {_label(actual)}")

    updates = list(diff.get("updated") or [])
    used_updates = [False] * len(updates)
    for expected in gold.get("updated") or []:
        found = False
        for index, actual in enumerate(updates):
            if used_updates[index]:
                continue
            if expected.get("resourceType") and expected["resourceType"] != actual["resourceType"]:
                continue
            if expected.get("id") is not None and str(expected["id"]) != str(actual["id"]):
                continue
            if _match(expected.get("resource") or {}, actual["after"]):
                used_updates[index] = True
                found = True
                break
        if not found:
            mismatches.append(f"missing updated resource: {_expect_label(expected)}")
    for index, actual in enumerate(updates):
        if not used_updates[index]:
            mismatches.append(
                f"unexpected updated resource: {actual['resourceType']}/{actual['id']}"
            )

    deletes = list(diff.get("deleted") or [])
    used_deletes = [False] * len(deletes)
    for expected in gold.get("deleted") or []:
        found = False
        for index, actual in enumerate(deletes):
            if used_deletes[index]:
                continue
            if _match(expected, actual):
                used_deletes[index] = True
                found = True
                break
        if not found:
            mismatches.append(f"missing deleted resource: {_expect_label(expected)}")
    for index, actual in enumerate(deletes):
        if not used_deletes[index]:
            mismatches.append(f"unexpected deleted resource: {_label(actual)}")

    return {
        "passed": not mismatches,
        "mismatches": mismatches,
        "created": created,
        "updated": updates,
        "deleted": deletes,
    }


def grade_trial_state(
    pre_bundle: dict,
    post_bundle: dict,
    gold: dict,
    final_text: str = "",
) -> dict:
    """Grade a trial from the chart diff. The finish text is not used."""

    del final_text
    return grade_diff(diff_bundles(pre_bundle, post_bundle), gold or {})


def gold_spec_for_task(task: TaskCase, pre_bundle: dict) -> dict:
    """Expected chart diff for a correct solution given the pre-trial chart."""

    family = task.family
    if family in _READ_ONLY:
        return _empty_gold()
    mrn = task.eval_mrn or parse_mrn(task.instruction)
    if family == "task3":
        return _created(
            blood_pressure_observation(
                mrn, first_quoted(task.instruction), require_bp_code(task.context)
            )
        )
    if family == "task5":
        _when, value = _latest_value(pre_bundle, "MG", within_hours=24)
        dose = None if value is None else magnesium_dose(value)
        if dose is None:
            return _empty_gold()
        grams, hours = dose
        return _created(
            medication_request(
                mrn,
                require_ndc(task.context),
                "IV",
                grams,
                "g",
                rate_value=hours,
                rate_unit="h",
            )
        )
    if family == "task8":
        return _created(
            service_request(
                mrn,
                "http://snomed.info/sct",
                require_snomed(task.context),
                note=first_quoted(task.instruction),
            )
        )
    if family == "task9":
        _when, value = _latest_value(pre_bundle, "K", within_hours=None)
        dose = None if value is None else potassium_dose_meq(value)
        if dose is None:
            return _empty_gold()
        return _created(
            medication_request(mrn, require_ndc(task.context), "oral", _whole(dose), "mEq"),
            potassium_followup(mrn, require_loinc(task.context)),
        )
    if family == "task10":
        when, value = _latest_value(pre_bundle, "A1C", within_hours=None)
        if value is None or when is None or when >= ONE_YEAR_BEFORE:
            return _empty_gold()
        return _created(
            service_request(mrn, "http://loinc.org", require_loinc(task.context))
        )
    raise ValueError(f"no gold spec for {family}")


def _empty_gold() -> dict:
    return {"created": [], "updated": [], "deleted": []}


def _created(*resources: dict) -> dict:
    return {"created": list(resources), "updated": [], "deleted": []}


def _latest_value(
    bundle: dict, code: str, *, within_hours: float | None
) -> tuple[Any, float | None]:
    labs = [
        resource
        for resource in bundle_resources(bundle)
        if resource.get("resourceType") == "Observation" and _code_matches(resource, code)
    ]
    when, value, _raw = latest_quantity(labs, within_hours=within_hours)
    return when, value


def _code_matches(resource: dict, code: str) -> bool:
    code_obj = resource.get("code") or {}
    if code_obj.get("text") == code:
        return True
    for coding in code_obj.get("coding") or []:
        if isinstance(coding, dict) and coding.get("code") == code:
            return True
    return False


def _whole(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return float(value)


def _index(bundle: dict) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    by_key: dict[tuple[str, str], dict] = {}
    anon: list[dict] = []
    for resource in bundle_resources(bundle or {}):
        key = _key(resource)
        if key is None:
            anon.append(resource)
        else:
            by_key[key] = resource
    return by_key, anon


def _key(resource: dict) -> tuple[str, str] | None:
    resource_id = resource.get("id")
    if resource_id is None or resource_id == "":
        return None
    return (str(resource.get("resourceType") or ""), str(resource_id))


def _canonical(resource: dict) -> dict:
    body = copy.deepcopy(resource)
    for key in _IGNORED:
        body.pop(key, None)
    return _norm(body)


def _norm(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _norm(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_norm(item) for item in value]
    return value


def _sort_key(resource: dict) -> tuple:
    return (
        str(resource.get("resourceType")),
        str(resource.get("id", "")),
        json.dumps(resource, sort_keys=True, default=str),
    )


def _match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= _NUMBER_TOLERANCE
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(key in actual and _match(value, actual[key]) for key, value in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_match(left, right) for left, right in zip(expected, actual))
    return expected == actual


def _label(resource: dict) -> str:
    resource_id = resource.get("id")
    kind = resource.get("resourceType", "resource")
    if resource_id:
        return f"{kind}/{resource_id}"
    return str(kind)


def _expect_label(expected: dict) -> str:
    kind = expected.get("resourceType", "resource")
    resource_id = expected.get("id")
    if resource_id:
        return f"{kind}/{resource_id}"
    return str(kind)
