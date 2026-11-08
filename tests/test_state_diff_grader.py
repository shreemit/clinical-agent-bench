import copy
import json
from pathlib import Path

from agent.fhir_rules import ONE_YEAR_BEFORE, IncompleteSearchError
from graders.state_collector import collect_patient_state, merge_resources, read_locations
from graders.state_diff_grader import (
    diff_bundles,
    gold_spec_for_task,
    grade_diff,
    grade_trial_state,
)
from harness.http_client import HttpResult
from harness.types import TaskCase
from tasks_io import load_task_file

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "state_diffs"


def _bundle(resources):
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": resource} for resource in resources],
    }


def _obs(resource_id, value, when="2023-11-13T08:00:00+00:00", code="MG"):
    return {
        "resourceType": "Observation",
        "id": resource_id,
        "status": "final",
        "code": {"text": code},
        "effectiveDateTime": when,
        "valueQuantity": {"value": value, "unit": "mg/dL"},
        "subject": {"reference": "Patient/S1"},
    }


def _load_fixtures():
    paths = sorted(FIXTURES.glob("*.json"))
    return [json.loads(path.read_text()) for path in paths]


def _case(item):
    task = item["task"]
    return TaskCase(
        id=task["id"],
        instruction=task["instruction"],
        context=task.get("context") or "",
        sol=task.get("sol"),
        eval_mrn=task.get("eval_MRN"),
    )


def test_diff_ignores_metadata_and_reports_clinical_changes():
    before = _obs("bp1", "120/80", code="BP")
    before["valueString"] = "120/80"
    before.pop("valueQuantity")
    after = copy.deepcopy(before)
    after["meta"] = {"versionId": "3", "lastUpdated": "2023-11-13T10:16:00+00:00"}
    after["text"] = {"status": "generated", "div": "<div>ignored</div>"}
    after["language"] = "en"
    assert diff_bundles(_bundle([before]), _bundle([after])) == {
        "created": [],
        "updated": [],
        "deleted": [],
    }
    changed = copy.deepcopy(after)
    changed["valueString"] = "118/77 mmHg"
    diff = diff_bundles(_bundle([before]), _bundle([changed]))
    assert len(diff["updated"]) == 1
    assert diff["updated"][0]["id"] == "bp1"
    assert diff["created"] == []
    assert diff["deleted"] == []
    gold = {
        "created": [],
        "deleted": [],
        "updated": [
            {
                "resourceType": "Observation",
                "id": "bp1",
                "resource": {"valueString": "118/77 mmHg"},
            }
        ],
    }
    assert grade_diff(diff, gold)["passed"]
    wrong = copy.deepcopy(gold)
    wrong["updated"][0]["resource"]["valueString"] = "90/60"
    assert not grade_diff(diff, wrong)["passed"]


def test_diff_detects_create_and_delete_without_mutating_inputs():
    pre = _bundle([_obs("keep", 1.0), _obs("gone", 2.0)])
    created = _obs("new", 3.0)
    created["meta"] = {"versionId": "1"}
    post = _bundle([_obs("keep", 1.0), created])
    snapshot = json.dumps(pre)
    diff = diff_bundles(pre, post)
    assert json.dumps(pre) == snapshot
    assert [item["id"] for item in diff["created"]] == ["new"]
    assert [item["id"] for item in diff["deleted"]] == ["gone"]
    assert diff["updated"] == []
    assert "meta" not in diff["created"][0]


def test_unexpected_create_and_duplicate_fail():
    pre = _bundle([])
    order = {
        "resourceType": "MedicationRequest",
        "id": "m1",
        "status": "active",
        "subject": {"reference": "Patient/S1"},
        "dosageInstruction": [
            {"doseAndRate": [{"doseQuantity": {"value": 50, "unit": "mEq"}}]}
        ],
    }
    extra = copy.deepcopy(order)
    extra["id"] = "m2"
    diff = diff_bundles(pre, _bundle([order, extra]))
    gold = {
        "created": [
            {
                "resourceType": "MedicationRequest",
                "dosageInstruction": [
                    {"doseAndRate": [{"doseQuantity": {"value": 50, "unit": "mEq"}}]}
                ],
            }
        ],
        "updated": [],
        "deleted": [],
    }
    graded = grade_diff(diff, gold)
    assert not graded["passed"]
    assert any(item.startswith("unexpected created") for item in graded["mismatches"])
    close = grade_diff(
        diff_bundles(pre, _bundle([order])),
        {
            "created": [
                {
                    "resourceType": "MedicationRequest",
                    "dosageInstruction": [
                        {"doseAndRate": [{"doseQuantity": {"value": 50.05, "unit": "mEq"}}]}
                    ],
                }
            ],
            "updated": [],
            "deleted": [],
        },
    )
    assert close["passed"]
    extra_dose = copy.deepcopy(order)
    extra_dose["dosageInstruction"].append({"text": "also this"})
    extra_graded = grade_diff(
        diff_bundles(pre, _bundle([extra_dose])),
        gold,
    )
    assert not extra_graded["passed"]
    far = grade_diff(
        diff_bundles(pre, _bundle([order])),
        {
            "created": [
                {
                    "resourceType": "MedicationRequest",
                    "dosageInstruction": [
                        {"doseAndRate": [{"doseQuantity": {"value": 60, "unit": "mEq"}}]}
                    ],
                }
            ],
            "updated": [],
            "deleted": [],
        },
    )
    assert not far["passed"]


def test_finish_text_does_not_change_the_grade():
    pre = _bundle([])
    post = _bundle([])
    gold = {"created": [], "updated": [], "deleted": []}
    first = grade_trial_state(pre, post, gold, "ordered")
    second = grade_trial_state(pre, post, gold, "not ordered")
    assert first["passed"] and second["passed"]
    assert first["mismatches"] == second["mismatches"]


def test_protocol_holds_orders_for_stale_or_missing_labs():
    tasks = load_task_file()
    old = "2023-11-10T08:00:00+00:00"
    recent = "2023-11-13T08:00:00+00:00"
    task5 = tasks["task5_1"]
    stale = _bundle([_obs("old", 0.4, old, "MG")])
    assert gold_spec_for_task(task5, stale)["created"] == []
    missing = _bundle([{"resourceType": "Patient", "id": task5.eval_mrn}])
    assert gold_spec_for_task(task5, missing)["created"] == []
    task9 = tasks["task9_1"]
    assert gold_spec_for_task(task9, _bundle([]))["created"] == []
    task10 = tasks["task10_1"]
    assert gold_spec_for_task(task10, _bundle([]))["created"] == []
    exact = _bundle([_obs("a1c", 6.1, ONE_YEAR_BEFORE.isoformat(), "A1C")])
    assert gold_spec_for_task(task10, exact)["created"] == []
    old = _bundle([_obs("a1c", 8.2, "2020-01-01T00:00:00+00:00", "A1C")])
    ordered = gold_spec_for_task(task10, old)["created"]
    assert [item["resourceType"] for item in ordered] == ["ServiceRequest"]


def test_fourteen_gold_fixtures_pass():
    items = _load_fixtures()
    assert len(items) == 14
    assert len({item["name"] for item in items}) == 14
    families = {item["task"]["id"].split("_", 1)[0] for item in items}
    assert families == {f"task{index}" for index in range(1, 11)}
    for item in items:
        task = _case(item)
        assert gold_spec_for_task(task, item["pre_bundle"]) == item["gold"]
        graded = grade_trial_state(
            item["pre_bundle"], item["post_bundle"], item["gold"], "transcript says done"
        )
        assert graded["passed"], (item["name"], graded["mismatches"])
        assert any("meta" in entry["resource"] for entry in item["post_bundle"]["entry"])
        again = grade_trial_state(
            item["pre_bundle"], item["post_bundle"], item["gold"], "different transcript"
        )
        assert again["passed"]


def test_fixture_golds_lock_clinical_fields():
    by_name = {item["name"]: item for item in _load_fixtures()}
    mild = by_name["task5_order_mild"]["gold"]["created"][0]
    dose = mild["dosageInstruction"][0]["doseAndRate"][0]
    assert dose["doseQuantity"] == {"value": 1, "unit": "g"}
    assert dose["rateQuantity"] == {"value": 1, "unit": "h"}
    assert mild["medicationCodeableConcept"]["coding"][0]["code"] == "0338-1715-40"
    assert mild["dosageInstruction"][0]["route"] == {"text": "IV"}
    severe = by_name["task5_order_severe"]["gold"]["created"][0]
    severe_dose = severe["dosageInstruction"][0]["doseAndRate"][0]
    assert severe_dose["doseQuantity"]["value"] == 4
    assert severe_dose["rateQuantity"] == {"value": 4, "unit": "h"}
    assert by_name["task5_hold"]["gold"]["created"] == []
    potassium = by_name["task9_order"]["gold"]["created"]
    assert [item["resourceType"] for item in potassium] == ["MedicationRequest", "ServiceRequest"]
    assert potassium[0]["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"] == {
        "value": 50,
        "unit": "mEq",
    }
    assert potassium[0]["dosageInstruction"][0]["route"] == {"text": "oral"}
    assert potassium[1]["code"]["coding"][0]["code"] == "2823-3"
    assert potassium[1]["occurrenceDateTime"] == "2023-11-14T08:00:00+00:00"
    assert by_name["task9_hold"]["gold"]["created"] == []
    pressure = by_name["task3_record_bp"]["gold"]["created"][0]
    assert pressure["valueString"] == "118/77 mmHg"
    assert pressure["code"] == {"text": "BP"}
    assert pressure["resourceType"] == "Observation"
    referral = by_name["task8_referral"]["gold"]["created"][0]
    assert referral["code"]["coding"][0]["code"] == "306181000000106"
    assert "ACL tear" in referral["note"][0]["text"]
    a1c = by_name["task10_order"]["gold"]["created"][0]
    assert a1c["code"]["coding"][0]["code"] == "4548-4"
    assert by_name["task10_hold"]["gold"]["created"] == []
    for name in (
        "task1_lookup",
        "task2_age",
        "task4_magnesium",
        "task6_glucose_average",
        "task7_glucose_latest",
    ):
        assert by_name[name]["gold"] == {"created": [], "updated": [], "deleted": []}


def test_mutating_a_gold_chart_fails():
    item = next(row for row in _load_fixtures() if row["name"] == "task5_order_mild")
    post = copy.deepcopy(item["post_bundle"])
    for entry in post["entry"]:
        resource = entry["resource"]
        if resource["resourceType"] != "MedicationRequest":
            continue
        resource["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]["value"] = 99
    graded = grade_trial_state(item["pre_bundle"], post, item["gold"], "")
    assert not graded["passed"]


class _PagingHttp:
    def __init__(self):
        self.urls = []
        self.headers = []

    def request(self, method, url, body=None, headers=None, timeout=None):
        self.urls.append(url)
        self.headers.append(headers)
        if url.endswith("/Patient/S1"):
            return HttpResult(200, {"resourceType": "Patient", "id": "S1"}, {}, 1)
        if "_getpages=2" in url:
            return HttpResult(
                200,
                {
                    "resourceType": "Bundle",
                    "type": "searchset",
                    "entry": [
                        {"resource": {"resourceType": "Observation", "id": "b", "status": "final"}}
                    ],
                },
                {},
                1,
            )
        if "/Observation?" in url:
            return HttpResult(
                200,
                {
                    "resourceType": "Bundle",
                    "type": "searchset",
                    "link": [
                        {
                            "relation": "next",
                            "url": "http://fhir.test/fhir/Observation?_getpages=2",
                        }
                    ],
                    "entry": [
                        {"resource": {"resourceType": "Observation", "id": "a", "status": "final"}}
                    ],
                },
                {},
                1,
            )
        return HttpResult(
            200, {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}, {}, 1
        )


def test_collector_pages_patient_resources():
    http = _PagingHttp()
    bundle = collect_patient_state(http, "http://fhir.test/fhir", "S1")
    ids = [entry["resource"]["id"] for entry in bundle["entry"]]
    assert ids == ["S1", "a", "b"]
    assert any(
        "/Observation?patient=S1&_count=200&_sort=-_lastUpdated" in url for url in http.urls
    )
    assert all((headers or {}).get("Cache-Control") == "no-cache" for headers in http.headers)
    assert any("_getpages=2" in url for url in http.urls)
    assert http.urls[0].endswith("/Patient/S1")


def test_collector_rejects_a_short_search():
    class Short:
        def request(self, method, url, body=None, headers=None, timeout=None):
            if "/Patient/" in url and "?" not in url:
                return HttpResult(200, {"resourceType": "Patient", "id": "S1"}, {}, 1)
            if "/Observation?" in url:
                return HttpResult(
                    200,
                    {
                        "resourceType": "Bundle",
                        "type": "searchset",
                        "total": 4,
                        "entry": [
                            {"resource": {"resourceType": "Observation", "id": "only", "status": "final"}}
                        ],
                    },
                    {},
                    1,
                )
            return HttpResult(
                200, {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}, {}, 1
            )

    try:
        collect_patient_state(Short(), "http://fhir.test/fhir/", "S1")
    except IncompleteSearchError as error:
        assert "total 4" in str(error)
    else:
        raise AssertionError("expected IncompleteSearchError")


def test_collector_bypasses_a_stale_search_cache():
    class StaleCache:
        def __init__(self):
            self.searches = []

        def request(self, method, url, body=None, headers=None, timeout=None):
            if "/Patient/" in url and "?" not in url:
                return HttpResult(200, {"resourceType": "Patient", "id": "S1"}, {}, 1)
            if "/Observation?" in url:
                fresh = "no-cache" in ((headers or {}).get("Cache-Control") or "")
                self.searches.append(fresh)
                entries = [{"resource": {"resourceType": "Observation", "id": "old", "status": "final"}}]
                if fresh:
                    entries.append(
                        {
                            "resource": {
                                "resourceType": "Observation",
                                "id": "new",
                                "status": "final",
                                "valueString": "118/77 mmHg",
                            }
                        }
                    )
                return HttpResult(
                    200,
                    {"resourceType": "Bundle", "type": "searchset", "total": len(entries), "entry": entries},
                    {},
                    1,
                )
            return HttpResult(
                200, {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}, {}, 1
            )

    http = StaleCache()
    bundle = collect_patient_state(http, "http://fhir.test/fhir/", "S1")
    ids = [entry["resource"]["id"] for entry in bundle["entry"]]
    assert "new" in ids
    assert http.searches and all(http.searches)


def test_read_location_merges_over_a_stale_snapshot():
    class Direct:
        def request(self, method, url, body=None, headers=None, timeout=None):
            self.url = url
            return HttpResult(
                200,
                {"resourceType": "Observation", "id": "786002", "valueString": "118/77 mmHg"},
                {},
                1,
            )

    http = Direct()
    found = read_locations(
        http,
        "http://fhir.test/fhir/",
        ["http://fhir.test/fhir/Observation/786002/_history/1"],
    )
    assert http.url == "http://fhir.test/fhir/Observation/786002"
    stale = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": {"resourceType": "Observation", "id": "786002", "valueString": "old"}}],
    }
    merged = merge_resources(stale, found)
    values = [entry["resource"]["valueString"] for entry in merged["entry"]]
    assert values == ["118/77 mmHg"]


def test_collector_raises_on_search_error():
    class Bad:
        def request(self, method, url, body=None, headers=None, timeout=None):
            if "/Patient/" in url:
                return HttpResult(404, {"resourceType": "OperationOutcome"}, {}, 1)
            return HttpResult(500, {"resourceType": "OperationOutcome"}, {}, 1)

    try:
        collect_patient_state(Bad(), "http://fhir.test/fhir/", "S9")
    except RuntimeError as error:
        assert "500" in str(error)
    else:
        raise AssertionError("expected RuntimeError")
