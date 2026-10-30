import json
from urllib.parse import parse_qs, urlparse

from agent.mock_agent import MockAgent
from harness.exec_harness import ExecHarness, evaluate_trial
from harness.http_client import HttpResult
from harness.types import CapturedCall, TaskCase
from tasks_io import load_smoke_ids, load_task_file


class ScriptedHttp:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def request(self, method, url, body=None, headers=None, timeout=120):
        self.calls.append({"method": method, "url": url, "body": body, "headers": headers})
        status, response_body, response_headers = self.handler(method, url, body)
        return HttpResult(status=status, body=response_body, headers=response_headers, elapsed_ms=1.0)


def _patient(mrn, birth="1950-01-01", given="Pat", family="Example"):
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 1,
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": mrn,
                    "birthDate": birth,
                    "name": [{"family": family, "given": [given]}],
                }
            }
        ],
    }


def _obs(value, when, code="MG"):
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 1,
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": code},
                    "effectiveDateTime": when,
                    "valueQuantity": {"value": value, "unit": "mg/dL"},
                }
            }
        ],
    }


def _handler(method, url, body):
    if method == "GET" and "/Patient?" in url and "family=" in url:
        return 200, _patient("S6534835", "1932-12-29", "Peter", "Stafford"), {}
    if method == "GET" and "/Patient?" in url:
        return 200, _patient("S2874099", "1950-06-01"), {}
    if method == "GET" and "code=MG" in url:
        return 200, _obs(1.2, "2023-11-13T08:00:00+00:00", "MG"), {}
    if method == "GET" and "code=GLU" in url:
        return 200, {
            "resourceType": "Bundle",
            "type": "searchset",
            "entry": [
                {"resource": {"effectiveDateTime": "2023-11-13T08:00:00+00:00", "valueQuantity": {"value": 100}}},
                {"resource": {"effectiveDateTime": "2023-11-13T09:00:00+00:00", "valueQuantity": {"value": 140}}},
                {"resource": {"effectiveDateTime": "2023-11-10T09:00:00+00:00", "valueQuantity": {"value": 999}}},
            ],
        }, {}
    if method == "GET" and "code=K" in url:
        return 200, _obs(3.0, "2023-11-12T08:00:00+00:00", "K"), {}
    if method == "GET" and "code=A1C" in url:
        return 200, _obs(7.1, "2020-01-01T00:00:00+00:00", "A1C"), {}
    if method in ("POST", "PUT"):
        resource_id = "99"
        payload = dict(body or {})
        payload["id"] = body.get("id", resource_id) if body else resource_id
        headers = {}
        if method == "POST":
            headers["Location"] = url.rstrip("/") + f"/{resource_id}/_history/1"
        return 201 if method == "POST" else 200, payload, headers
    raise AssertionError(f"unexpected {method} {url}")


def test_smoke_ids_cover_ten_families():
    ids = load_smoke_ids()
    assert ids == [f"task{i}_1" for i in range(1, 11)]
    tasks = load_task_file()
    assert len(tasks) == 300
    assert tasks["task1_1"].sol == ["S6534835"]
    assert tasks["task3_1"].sol is None


def test_all_smoke_families_pass_against_scripted_http():
    tasks = load_task_file()
    http = ScriptedHttp(_handler)
    harness = ExecHarness("http://fhir.test/fhir/", http=http)
    for task_id in load_smoke_ids():
        task = tasks[task_id]
        record = harness.run(task, MockAgent(task, harness.fhir_api_base))
        assert record.passed, (task_id, record.failure_reason, record.finish_text)
        assert all(200 <= call.status < 300 for call in record.calls)


def test_task3_posts_then_puts_same_observation():
    task = load_task_file()["task3_1"]
    http = ScriptedHttp(_handler)
    record = ExecHarness("http://fhir.test/fhir/", http=http).run(task, MockAgent(task, "http://fhir.test/fhir/"))
    assert [call.method for call in record.calls] == ["POST", "PUT"]
    assert record.calls[0].request_body["valueString"] == "118/77 mmHg"
    assert record.calls[0].request_body["subject"] == {"reference": "Patient/S2380121"}
    assert record.calls[1].url.endswith("/Observation/99")
    assert record.calls[1].request_body["id"] == "99"
    assert record.calls[1].request_body["valueString"] == "118/77 mmHg"


def test_task5_orders_moderate_magnesium_dose():
    task = load_task_file()["task5_1"]
    http = ScriptedHttp(_handler)
    record = ExecHarness("http://fhir.test/fhir/", http=http).run(task, MockAgent(task, "http://fhir.test/fhir/"))
    posts = [call for call in record.calls if call.method == "POST"]
    assert len(posts) == 1
    dose = posts[0].request_body["dosageInstruction"][0]["doseAndRate"][0]
    assert dose["doseQuantity"] == {"value": 2, "unit": "g"}
    assert dose["rateQuantity"] == {"value": 2, "unit": "h"}
    assert posts[0].request_body["medicationCodeableConcept"]["coding"][0]["code"] == "0338-1715-40"
    assert json.loads(record.finish_text) == [1.2]


def test_task5_does_not_order_when_magnesium_is_normal():
    task = load_task_file()["task5_1"]

    def handler(method, url, body):
        if method == "GET":
            return 200, _obs(2.3, "2023-11-13T08:00:00+00:00"), {}
        raise AssertionError("no write expected")

    record = ExecHarness("http://fhir.test/fhir/", http=ScriptedHttp(handler)).run(
        task, MockAgent(task, "http://fhir.test/fhir/")
    )
    assert record.passed
    assert [call.method for call in record.calls] == ["GET"]
    assert json.loads(record.finish_text) == [2.3]


def test_task9_posts_potassium_and_morning_lab():
    task = load_task_file()["task9_1"]
    http = ScriptedHttp(_handler)
    record = ExecHarness("http://fhir.test/fhir/", http=http).run(task, MockAgent(task, "http://fhir.test/fhir/"))
    assert [call.method for call in record.calls] == ["GET", "POST", "POST"]
    med, lab = record.calls[1].request_body, record.calls[2].request_body
    assert med["resourceType"] == "MedicationRequest"
    assert med["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]["value"] == 50
    assert lab["resourceType"] == "ServiceRequest"
    assert lab["code"]["coding"][0]["code"] == "2823-3"
    assert lab["occurrenceDateTime"] == "2023-11-14T08:00:00+00:00"


def test_task1_sol_mismatch_fails():
    task = TaskCase("task1_x", "name A B and DOB of 2000-01-01", "", ["EXPECTED"], "S1")
    call = CapturedCall("GET", "http://x", None, 200, {}, {}, None, 1)
    passed, reason = evaluate_trial(task, [call], json.dumps(["OTHER"]))
    assert not passed
    assert "FINISH" in reason


def test_http_error_does_not_raise_and_fails_the_trial():
    task = load_task_file()["task8_1"]

    def handler(method, url, body):
        return 422, {"resourceType": "OperationOutcome", "issue": [{"diagnostics": "no"}]}, {}

    record = ExecHarness("http://fhir.test/fhir/", http=ScriptedHttp(handler)).run(
        task, MockAgent(task, "http://fhir.test/fhir/")
    )
    assert not record.passed
    assert "422" in record.failure_reason
    assert record.calls[0].status == 422


def test_task6_averages_only_the_last_24_hours():
    task = load_task_file()["task6_1"]
    record = ExecHarness("http://fhir.test/fhir/", http=ScriptedHttp(_handler)).run(
        task, MockAgent(task, "http://fhir.test/fhir/")
    )
    assert json.loads(record.finish_text) == [120.0]


def test_task10_does_not_order_when_the_chart_has_no_result():
    task = load_task_file()["task10_1"]

    def handler(method, url, body):
        if method == "GET":
            return 200, {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}, {}
        raise AssertionError("no write expected")

    record = ExecHarness("http://fhir.test/fhir/", http=ScriptedHttp(handler)).run(
        task, MockAgent(task, "http://fhir.test/fhir/")
    )
    assert record.passed, record.failure_reason
    assert [call.method for call in record.calls] == ["GET"]
    assert json.loads(record.finish_text) == [-1]


def test_task10_orders_when_the_result_is_older_than_a_year():
    task = load_task_file()["task10_1"]
    when = "2020-01-01T00:00:00+00:00"

    def handler(method, url, body):
        if method == "GET":
            return 200, _obs(7.1, when, "A1C"), {}
        if method == "POST":
            return 201, body, {"Location": url.rstrip("/") + "/99/_history/1"}
        raise AssertionError(method)

    record = ExecHarness("http://fhir.test/fhir/", http=ScriptedHttp(handler)).run(
        task, MockAgent(task, "http://fhir.test/fhir/")
    )
    assert record.passed, record.failure_reason
    assert [call.method for call in record.calls] == ["GET", "POST"]
    assert record.calls[1].request_body["code"]["coding"][0]["code"] == "4548-4"
    assert json.loads(record.finish_text) == [7.1, when]


def test_task10_does_not_order_a_result_dated_exactly_one_year_ago():
    task = load_task_file()["task10_1"]

    def handler(method, url, body):
        if method == "GET":
            return 200, _obs(6.4, "2022-11-13T10:15:00+00:00", "A1C"), {}
        raise AssertionError("no write expected")

    record = ExecHarness("http://fhir.test/fhir/", http=ScriptedHttp(handler)).run(
        task, MockAgent(task, "http://fhir.test/fhir/")
    )
    assert record.passed, record.failure_reason
    assert [call.method for call in record.calls] == ["GET"]
    assert json.loads(record.finish_text) == [6.4, "2022-11-13T10:15:00+00:00"]


def test_latest_lab_queries_ask_for_the_newest_row():
    tasks = load_task_file()
    http = ScriptedHttp(_handler)
    harness = ExecHarness("http://fhir.test/fhir/", http=http)
    record = harness.run(tasks["task4_1"], MockAgent(tasks["task4_1"], harness.fhir_api_base))
    assert record.passed, record.failure_reason
    query = parse_qs(urlparse(record.calls[0].url).query)
    assert query["_sort"] == ["-date"]
    assert query["_count"] == ["1"]
    assert query["date"] == ["ge2023-11-12T10:15:00+00:00"]

    record = harness.run(tasks["task7_1"], MockAgent(tasks["task7_1"], harness.fhir_api_base))
    query = parse_qs(urlparse(record.calls[0].url).query)
    assert query["_sort"] == ["-date"]
    assert query["_count"] == ["1"]
    assert "date" not in query


def test_task6_follows_the_next_page():
    task = load_task_file()["task6_1"]
    first = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 2,
        "link": [{"relation": "next", "url": "http://fhir.test/fhir/Observation?_getpages=2"}],
        "entry": [
            {"resource": {"effectiveDateTime": "2023-11-13T08:00:00+00:00", "valueQuantity": {"value": 80}}}
        ],
    }
    second = {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": 2,
        "entry": [
            {"resource": {"effectiveDateTime": "2023-11-13T09:00:00+00:00", "valueQuantity": {"value": 120}}}
        ],
    }

    def handler(method, url, body):
        if "_getpages=2" in url:
            return 200, second, {}
        return 200, first, {}

    record = ExecHarness("http://fhir.test/fhir/", http=ScriptedHttp(handler)).run(
        task, MockAgent(task, "http://fhir.test/fhir/")
    )
    assert record.passed, record.failure_reason
    assert [call.method for call in record.calls] == ["GET", "GET"]
    assert json.loads(record.finish_text) == [100.0]


def test_task6_fails_when_the_page_is_short_of_total():
    task = load_task_file()["task6_1"]

    def handler(method, url, body):
        return 200, {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": 4,
            "entry": [
                {"resource": {"effectiveDateTime": "2023-11-13T08:00:00+00:00", "valueQuantity": {"value": 80}}}
            ],
        }, {}

    record = ExecHarness("http://fhir.test/fhir/", http=ScriptedHttp(handler)).run(
        task, MockAgent(task, "http://fhir.test/fhir/")
    )
    assert not record.passed
    assert record.finish_text is None
    assert "total 4" in record.failure_reason


def test_task8_note_comes_from_the_instruction():
    task = load_task_file()["task8_1"]
    record = ExecHarness("http://fhir.test/fhir/", http=ScriptedHttp(_handler)).run(
        task, MockAgent(task, "http://fhir.test/fhir/")
    )
    note = record.calls[0].request_body["note"][0]["text"]
    assert "ACL tear" in note
    assert record.calls[0].request_body["code"]["coding"][0]["code"] == "306181000000106"
