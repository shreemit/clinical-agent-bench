import json
from urllib.parse import unquote

import pytest

from agent.fhir_rules import first_quoted, require_snomed
from agent.payloads import service_request
from harness.http_client import HttpResult
from proxy.fault_proxy import FAULT_TYPES, IDEMPOTENCY_SYSTEM, FaultProxy
from runners.baseline import DEFAULT_MODEL, run_baseline
from tasks_io import load_task_file
from test_baseline import Chart, ScriptedChat, _post_then_finish

URL = "http://fhir.test/fhir/Observation"
POST_BODY = {"resourceType": "Observation", "status": "final"}

_CALLS = {
    "none": 1,
    "committed_timeout": 1,
    "uncommitted_timeout": 0,
    "transient_5xx": 0,
    "conflict_409": 0,
    "precondition_412": 0,
    "unprocessable_422": 0,
    "duplicate_retry": 2,
}
_STATUS = {
    "none": 201,
    "committed_timeout": 504,
    "uncommitted_timeout": 504,
    "transient_5xx": 503,
    "conflict_409": 409,
    "precondition_412": 412,
    "unprocessable_422": 422,
    "duplicate_retry": 201,
}


class Recorder:
    def __init__(self, responses=None):
        self.calls = []
        self._responses = list(responses or [])
        self.next_id = 1

    def request(self, method, url, body=None, headers=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "headers": dict(headers or {}),
                "timeout": timeout,
            }
        )
        if self._responses:
            return self._responses.pop(0)
        if method == "GET":
            return HttpResult(200, {"resourceType": "Patient", "id": "p"}, {}, 1)
        resource_id = str(self.next_id)
        self.next_id += 1
        resource = dict(body or {})
        resource["id"] = resource_id
        resource["meta"] = {"versionId": "1"}
        location = url.split("?", 1)[0].rstrip("/") + f"/{resource_id}/_history/1"
        status = 201 if method == "POST" else 200
        return HttpResult(status, resource, {"Location": location, "ETag": 'W/"1"'}, 4)


def _proxy(recorder, tmp_path, **kwargs):
    defaults = {"fault": "none", "seed": 0, "on_attempt": 1, "mitigation": "off", "log_path": tmp_path / "faults.jsonl"}
    defaults.update(kwargs)
    return FaultProxy(recorder, **defaults)


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("fault", FAULT_TYPES)
def test_each_fault_is_deterministic(tmp_path, fault):
    first = Recorder()
    second = Recorder()
    kwargs = {"fault": fault, "seed": 0, "trial_id": "t", "task_id": "task8_1"}
    left = _proxy(first, tmp_path / "a", **kwargs).request("POST", URL, body=dict(POST_BODY))
    right = _proxy(second, tmp_path / "b", **kwargs).request("POST", URL, body=dict(POST_BODY))
    assert left.status == right.status == _STATUS[fault]
    assert left.body == right.body
    assert len(first.calls) == len(second.calls) == _CALLS[fault]
    logged = _lines(tmp_path / "a" / "faults.jsonl")
    assert logged
    for row in logged:
        for key in ("fault", "attempt", "method", "url", "status", "upstream_status", "upstream_location"):
            assert key in row
        assert row["fault"] == fault
        assert row["attempt"] == 1
        assert row["method"] == "POST"
        assert row["url"] == URL
        assert row["status"] == _STATUS[fault]
    if fault in ("uncommitted_timeout", "transient_5xx", "conflict_409", "precondition_412", "unprocessable_422"):
        assert logged[0]["forward"] == 0
        assert logged[0]["upstream_status"] is None
        assert logged[0]["upstream_location"] is None
        assert left.headers.get("Location") is None
        assert left.body["resourceType"] == "OperationOutcome"


def test_transient_5xx_follows_seed_parity(tmp_path):
    seen = {}
    for seed in (0, 1, 2, 3, 8):
        result = _proxy(Recorder(), tmp_path / str(seed), fault="transient_5xx", seed=seed).request(
            "POST", URL, body=dict(POST_BODY)
        )
        seen[seed] = result.status
        issue = result.body["issue"][0]
        assert issue["severity"] == "error"
        assert issue["diagnostics"]
    assert seen[0] == seen[2] == seen[8] == 503
    assert seen[1] == seen[3] == 500


def test_outcome_faults_name_a_field(tmp_path):
    expected = {
        "conflict_409": ("conflict", "Observation.meta.versionId"),
        "precondition_412": ("processing", "Observation.meta.versionId"),
        "unprocessable_422": ("processing", "Observation.status"),
    }
    for fault, (code, expression) in expected.items():
        result = _proxy(Recorder(), tmp_path / fault, fault=fault).request("POST", URL, body=dict(POST_BODY))
        issue = result.body["issue"][0]
        assert issue["severity"] == "error"
        assert issue["code"] == code
        assert issue["diagnostics"]
        assert issue["expression"] == [expression]


def test_reads_and_later_writes_pass_through(tmp_path):
    recorder = Recorder()
    proxy = _proxy(recorder, tmp_path, fault="conflict_409", on_attempt=1)
    assert proxy.request("GET", "http://fhir.test/fhir/Patient/1").status == 200
    assert proxy.request("GET", "http://fhir.test/fhir/Observation?patient=1").status == 200
    denied = proxy.request("POST", URL, body=dict(POST_BODY))
    assert denied.status == 409
    allowed = proxy.request("POST", URL, body={"resourceType": "Observation", "status": "amended"})
    assert allowed.status == 201
    assert [call["method"] for call in recorder.calls] == ["GET", "GET", "POST"]
    logged = _lines(tmp_path / "faults.jsonl")
    assert [row["attempt"] for row in logged] == [None, None, 1, 2]
    assert [row["fault"] for row in logged] == ["none", "none", "conflict_409", "none"]


def test_fault_on_the_second_write_leaves_the_others_alone(tmp_path):
    recorder = Recorder()
    proxy = _proxy(recorder, tmp_path, fault="uncommitted_timeout", on_attempt=2)
    assert proxy.request("POST", URL, body=dict(POST_BODY)).status == 201
    assert proxy.request("PUT", URL + "/1", body={"resourceType": "Observation", "id": "1"}).status == 504
    assert proxy.request("POST", URL, body={"resourceType": "Observation", "status": "amended"}).status == 201
    assert [call["method"] for call in recorder.calls] == ["POST", "POST"]


def test_committed_timeout_hides_a_real_location(tmp_path):
    recorder = Recorder()
    proxy = _proxy(recorder, tmp_path, fault="committed_timeout", trial_id="3", task_id="task3_1")
    result = proxy.request("POST", URL, body=dict(POST_BODY), headers={"Content-Type": "application/fhir+json"})
    assert result.status == 504
    assert result.body["issue"][0]["code"] == "timeout"
    assert "Location" not in result.headers
    assert len(recorder.calls) == 1
    row = _lines(tmp_path / "faults.jsonl")[0]
    assert row["status"] == 504
    assert row["upstream_status"] == 201
    assert row["upstream_location"].endswith("/Observation/1/_history/1")
    assert proxy.committed_locations() == [row["upstream_location"]]
    assert row["trial_id"] == "3"
    assert row["task_id"] == "task3_1"


def test_duplicate_retry_returns_the_second_response(tmp_path):
    first = HttpResult(201, {"id": "1", "resourceType": "Observation"}, {"Location": "http://fhir.test/fhir/Observation/1/_history/1"}, 2)
    second = HttpResult(200, {"id": "1", "resourceType": "Observation"}, {"Location": "http://fhir.test/fhir/Observation/1/_history/1"}, 3)
    recorder = Recorder([first, second])
    proxy = _proxy(recorder, tmp_path, fault="duplicate_retry")
    got = proxy.request("POST", URL, body=dict(POST_BODY))
    assert got is second
    assert len(recorder.calls) == 2
    assert recorder.calls[0]["body"] == recorder.calls[1]["body"]
    rows = _lines(tmp_path / "faults.jsonl")
    assert [row["forward"] for row in rows] == [1, 2]
    assert [row["upstream_status"] for row in rows] == [201, 200]
    assert [row["status"] for row in rows] == [200, 200]
    assert rows[0]["upstream_location"].endswith("/_history/1")


def test_clean_proxy_returns_the_upstream_result():
    upstream = HttpResult(201, {"ok": True}, {"Location": "http://fhir.test/fhir/Observation/1"}, 8)

    class One:
        def request(self, method, url, body=None, headers=None, timeout=None):
            return upstream

    assert FaultProxy(One(), fault="none", mitigation="off").request("POST", URL, body={}) is upstream


def test_begin_trial_rearms_the_fault(tmp_path):
    recorder = Recorder()
    proxy = _proxy(recorder, tmp_path, fault="conflict_409")
    assert proxy.request("POST", URL, body=dict(POST_BODY)).status == 409
    proxy.begin_trial("1", "task9_1")
    assert proxy.request("POST", URL, body=dict(POST_BODY)).status == 409
    assert recorder.calls == []


def test_unknown_fault_and_attempt_are_rejected():
    with pytest.raises(ValueError):
        FaultProxy(Recorder(), fault="dropped")
    with pytest.raises(ValueError):
        FaultProxy(Recorder(), on_attempt=0)
    with pytest.raises(ValueError):
        FaultProxy(Recorder(), mitigation="client")


def test_preconditions_give_each_successful_post_its_own_identifier(tmp_path):
    recorder = Recorder()
    proxy = _proxy(
        recorder,
        tmp_path,
        mitigation="preconditions",
        trial_id="7",
        task_id="task8_1",
        run_id="run",
    )
    body = dict(POST_BODY)
    proxy.request("POST", URL, body=body, headers={"Content-Type": "application/fhir+json"})
    proxy.request("POST", URL, body=body, headers={"Content-Type": "application/fhir+json"})
    assert "identifier" not in body
    first, second = recorder.calls[0], recorder.calls[1]
    assert first["body"]["identifier"] != second["body"]["identifier"]
    assert first["body"]["identifier"][-1] == {
        "system": IDEMPOTENCY_SYSTEM,
        "value": "run:7:task8_1:Observation:1",
    }
    assert second["body"]["identifier"][-1]["value"] == "run:7:task8_1:Observation:2"
    decoded = unquote(first["headers"]["If-None-Exist"].split("=", 1)[1])
    assert decoded == IDEMPOTENCY_SYSTEM + "|run:7:task8_1:Observation:1"
    assert first["headers"]["Content-Type"] == "application/fhir+json"
    proxy.request("POST", URL, body={"resourceType": "Observation", "status": "amended"})
    assert recorder.calls[2]["body"]["identifier"][-1]["value"] == "run:7:task8_1:Observation:3"
    proxy.request(
        "POST",
        "http://fhir.test/fhir/MedicationRequest",
        body={"resourceType": "MedicationRequest", "status": "active"},
    )
    assert recorder.calls[3]["body"]["identifier"][-1]["value"] == "run:7:task8_1:MedicationRequest:1"


def test_failed_post_retry_reuses_one_identifier(tmp_path):
    denied = HttpResult(409, {"resourceType": "OperationOutcome"}, {}, 1)
    recorder = Recorder([denied])
    proxy = _proxy(recorder, tmp_path, mitigation="preconditions", trial_id="7", task_id="task8_1", run_id="run")
    body = dict(POST_BODY)
    assert proxy.request("POST", URL, body=body).status == 409
    proxy.request("GET", URL + "?patient=1")
    assert proxy.request("POST", URL, body=body).status == 201
    assert recorder.calls[0]["body"]["identifier"] == recorder.calls[2]["body"]["identifier"]
    assert recorder.calls[0]["headers"]["If-None-Exist"] == recorder.calls[2]["headers"]["If-None-Exist"]
    assert proxy.request("POST", URL, body=body).status == 201
    assert recorder.calls[3]["body"]["identifier"][-1]["value"] == "run:7:task8_1:Observation:2"


def test_a_later_write_cancels_identifier_reuse(tmp_path):
    recorder = Recorder()
    proxy = _proxy(
        recorder,
        tmp_path,
        fault="conflict_409",
        mitigation="preconditions",
        trial_id="7",
        task_id="task8_1",
        run_id="run",
    )
    body = dict(POST_BODY)
    assert proxy.request("POST", URL, body=body).status == 409
    assert proxy.request("PUT", URL + "/1", body={"resourceType": "Observation", "id": "1"}).status == 200
    proxy.request("POST", URL, body=body)
    assert recorder.calls[-1]["body"]["identifier"][-1]["value"] == "run:7:task8_1:Observation:2"


def test_run_id_keeps_identifiers_apart_across_runs(tmp_path):
    kwargs = {"mitigation": "preconditions", "trial_id": "task8_1:0", "task_id": "task8_1"}
    left_rec, right_rec = Recorder(), Recorder()
    left = _proxy(left_rec, tmp_path / "a", **kwargs)
    right = _proxy(right_rec, tmp_path / "b", **kwargs)
    left.request("POST", URL, body=dict(POST_BODY))
    right.request("POST", URL, body=dict(POST_BODY))
    left_value = left_rec.calls[0]["body"]["identifier"][-1]["value"]
    right_value = right_rec.calls[0]["body"]["identifier"][-1]["value"]
    assert left_value != right_value
    assert left_value.endswith(":task8_1:0:task8_1:Observation:1")
    assert right_value.endswith(":task8_1:0:task8_1:Observation:1")
    kept = left.run_id
    left.begin_trial("other", "task9_1")
    assert left.run_id == kept


def test_fault_fired_is_per_trial(tmp_path):
    proxy = _proxy(Recorder(), tmp_path, fault="conflict_409", on_attempt=2)
    assert proxy.fault_fired is False
    assert proxy.request("POST", URL, body=dict(POST_BODY)).status == 201
    assert proxy.fault_fired is False
    proxy.request("GET", "http://fhir.test/fhir/Patient/1")
    assert proxy.request("POST", URL, body=dict(POST_BODY)).status == 409
    assert proxy.fault_fired is True
    proxy.begin_trial("1", "task8_1")
    assert proxy.fault_fired is False
    assert proxy.request("GET", "http://fhir.test/fhir/Patient/1").status == 200
    assert proxy.fault_fired is False


def test_hidden_put_does_not_advance_if_match(tmp_path):
    created = HttpResult(
        201,
        {"resourceType": "Observation", "id": "5", "meta": {"versionId": "1"}},
        {"Location": "Observation/5/_history/1", "ETag": 'W/"1"'},
        1,
    )
    updated = HttpResult(
        200,
        {"resourceType": "Observation", "id": "5", "meta": {"versionId": "2"}},
        {"Location": "Observation/5/_history/2", "ETag": 'W/"2"'},
        1,
    )
    recorder = Recorder([created, updated])
    proxy = _proxy(recorder, tmp_path, fault="committed_timeout", on_attempt=2, mitigation="preconditions")
    proxy.request("POST", URL, body=dict(POST_BODY))
    hidden = proxy.request("PUT", "http://fhir.test/fhir/Observation/5", body={"resourceType": "Observation", "id": "5"})
    assert hidden.status == 504
    proxy.request("PUT", "http://fhir.test/fhir/Observation/5", body={"resourceType": "Observation", "id": "5"})
    assert recorder.calls[1]["headers"]["If-Match"] == 'W/"1"'
    assert recorder.calls[2]["headers"]["If-Match"] == 'W/"1"'
    assert any(location.endswith("/Observation/5/_history/2") for location in proxy.committed_locations())


def test_duplicate_retry_still_forwards_twice_when_preconditions_are_on(tmp_path):
    recorder = Recorder()
    proxy = _proxy(recorder, tmp_path, fault="duplicate_retry", mitigation="preconditions", trial_id="9", task_id="task5_1")
    proxy.request("POST", URL, body=dict(POST_BODY))
    assert len(recorder.calls) == 2
    assert recorder.calls[0]["headers"]["If-None-Exist"] == recorder.calls[1]["headers"]["If-None-Exist"]
    assert recorder.calls[0]["body"]["identifier"] == recorder.calls[1]["body"]["identifier"]


def test_if_match_comes_from_the_last_read(tmp_path):
    seen = HttpResult(
        200,
        {"resourceType": "Observation", "id": "9", "meta": {"versionId": "8"}},
        {"ETag": 'W/"4"'},
        1,
    )
    recorder = Recorder([seen])
    proxy = _proxy(recorder, tmp_path, mitigation="preconditions")
    proxy.request("GET", "http://fhir.test/fhir/Observation/9")
    proxy.request("PUT", "http://fhir.test/fhir/Observation/9", body={"resourceType": "Observation", "id": "9"})
    assert recorder.calls[1]["headers"]["If-Match"] == 'W/"4"'


def test_if_match_uses_version_id_when_the_etag_is_absent(tmp_path):
    seen = HttpResult(
        200,
        {"resourceType": "Observation", "id": "9", "meta": {"versionId": "4"}},
        {},
        1,
    )
    recorder = Recorder([seen])
    proxy = _proxy(recorder, tmp_path, mitigation="preconditions")
    proxy.request("GET", "http://fhir.test/fhir/Observation/9")
    proxy.request("PUT", "http://fhir.test/fhir/Observation/9", body={"resourceType": "Observation", "id": "9"})
    assert recorder.calls[1]["headers"]["If-Match"] == 'W/"4"'


def test_put_after_create_sends_if_match_without_a_get(tmp_path):
    created = HttpResult(
        201,
        {"resourceType": "Observation", "id": "5", "meta": {"versionId": "1"}},
        {"Location": "Observation/5/_history/1", "ETag": 'W/"1"'},
        1,
    )
    recorder = Recorder([created])
    proxy = _proxy(recorder, tmp_path, mitigation="preconditions")
    proxy.request("POST", URL, body=dict(POST_BODY))
    proxy.request("PUT", "http://fhir.test/fhir/Observation/5", body={"resourceType": "Observation", "id": "5"})
    assert recorder.calls[1]["headers"]["If-Match"] == 'W/"1"'
    assert "http://fhir.test/fhir/Observation/5/_history/1" in proxy.committed_locations()


def test_put_without_a_read_does_not_invent_if_match(tmp_path):
    recorder = Recorder()
    proxy = _proxy(recorder, tmp_path, mitigation="preconditions")
    proxy.request("PUT", URL + "/1", body={"resourceType": "Observation", "id": "1"})
    assert "If-Match" not in recorder.calls[0]["headers"]
    plain = Recorder()
    _proxy(plain, tmp_path / "off", mitigation="off").request("POST", URL, body=dict(POST_BODY))
    assert "If-None-Exist" not in plain.calls[0]["headers"]
    assert "identifier" not in (plain.calls[0]["body"] or {})


def test_state_grade_sees_a_committed_timeout(tmp_path):
    task = load_task_file()["task8_1"]
    body = service_request(
        task.eval_mrn,
        "http://snomed.info/sct",
        require_snomed(task.context),
        note=first_quoted(task.instruction),
    )
    chart = Chart(task.eval_mrn)
    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "grade",
        model=DEFAULT_MODEL,
        client=ScriptedChat(_post_then_finish(body, "http://fhir.test/fhir/ServiceRequest")),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
        fault="committed_timeout",
        seed=0,
        mitigation="off",
    )
    assert rows[0]["state_passed"] is True
    assert any(item.startswith("ServiceRequest/") for item in rows[0]["created"])
    assert "504" in rows[0]["failure_reason"]
    assert rows[0]["fault"] == "committed_timeout"
    assert rows[0]["seed"] == 0
    assert rows[0]["mitigation"] == "off"
    assert rows[0]["fault_fired"] is True
    posts = [row for row in _lines(tmp_path / "grade" / "faults.jsonl") if row["method"] == "POST"]
    assert posts[0]["status"] == 504
    assert posts[0]["upstream_status"] == 201


def test_stale_search_grades_a_hidden_commit_from_its_location(tmp_path):
    task = load_task_file()["task8_1"]
    posted = service_request(
        task.eval_mrn,
        "http://snomed.info/sct",
        require_snomed(task.context),
        note=first_quoted(task.instruction),
    )

    class Stale:
        def __init__(self):
            self.resource = None

        def request(self, method, url, body=None, headers=None, timeout=None):
            if method == "GET" and f"/Patient/{task.eval_mrn}" in url and "?" not in url:
                return HttpResult(200, {"resourceType": "Patient", "id": task.eval_mrn}, {}, 1)
            if method == "POST":
                resource = json.loads(json.dumps(body))
                resource["id"] = "99"
                resource["meta"] = {"versionId": "1"}
                self.resource = resource
                return HttpResult(201, resource, {"Location": url.rstrip("/") + "/99/_history/1"}, 1)
            if method == "GET" and "?" not in url and url.rstrip("/").endswith("/99"):
                return HttpResult(200, self.resource, {}, 1)
            if method == "GET":
                return HttpResult(200, {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}, {}, 1)
            raise AssertionError(f"{method} {url}")

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "stale",
        model=DEFAULT_MODEL,
        client=ScriptedChat(_post_then_finish(posted, "http://fhir.test/fhir/ServiceRequest")),
        fhir_api_base="http://fhir.test/fhir/",
        http=Stale(),
        refsol_path=tmp_path / "absent.py",
        reset=lambda: None,
        fault="committed_timeout",
    )
    assert rows[0]["state_passed"] is True
    assert "ServiceRequest/99" in rows[0]["created"]


def test_preconditions_do_not_break_the_state_grade(tmp_path):
    task = load_task_file()["task8_1"]
    body = service_request(
        task.eval_mrn,
        "http://snomed.info/sct",
        require_snomed(task.context),
        note=first_quoted(task.instruction),
    )
    chart = Chart(task.eval_mrn)
    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "pre",
        model=DEFAULT_MODEL,
        client=ScriptedChat(_post_then_finish(body, "http://fhir.test/fhir/ServiceRequest")),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
        fault="none",
        mitigation="preconditions",
    )
    assert rows[0]["state_passed"] is True
    stored = [item for item in chart.resources if item.get("resourceType") == "ServiceRequest"]
    assert stored[0]["identifier"][0]["system"] == IDEMPOTENCY_SYSTEM


def test_a_trial_with_no_write_records_that_the_fault_did_not_fire(tmp_path):
    task = load_task_file()["task8_1"]
    chart = Chart(task.eval_mrn)
    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "nowrite",
        model=DEFAULT_MODEL,
        client=ScriptedChat(lambda _turns, _messages, _model: "FINISH\n[]"),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
        fault="conflict_409",
        seed=3,
        mitigation="preconditions",
    )
    assert rows[0]["fault"] == "conflict_409"
    assert rows[0]["seed"] == 3
    assert rows[0]["mitigation"] == "preconditions"
    assert rows[0]["fault_fired"] is False
    assert rows[0]["state_passed"] is False


def test_preconditions_do_not_hide_an_agent_duplicate(tmp_path):
    task = load_task_file()["task8_1"]
    body = service_request(
        task.eval_mrn,
        "http://snomed.info/sct",
        require_snomed(task.context),
        note=first_quoted(task.instruction),
    )
    posted = "POST http://fhir.test/fhir/ServiceRequest\n" + json.dumps(body)

    class ConditionalChart(Chart):
        def request(self, method, url, body=None, headers=None, timeout=None):
            if method == "POST":
                existing = _existing_for_if_none_exist(self.resources, headers)
                if existing is not None:
                    location = url.rstrip("/") + f"/{existing['id']}/_history/1"
                    return HttpResult(200, existing, {"Location": location, "ETag": 'W/"1"'}, 1)
            return super().request(method, url, body=body, headers=headers, timeout=timeout)

    def reply(turns, _messages, _model):
        if turns < 2:
            return posted
        return "FINISH\n[]"

    chart = ConditionalChart(task.eval_mrn)
    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "dup",
        model=DEFAULT_MODEL,
        client=ScriptedChat(reply),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
        fault="none",
        mitigation="preconditions",
    )
    stored = [item for item in chart.resources if item.get("resourceType") == "ServiceRequest"]
    assert len(stored) == 2
    assert rows[0]["state_passed"] is False
    assert rows[0]["fault_fired"] is False


def _existing_for_if_none_exist(resources, headers):
    token = None
    for key, value in (headers or {}).items():
        if key.lower() == "if-none-exist":
            token = value
    if not token or "=" not in token:
        return None
    system, _, value = unquote(token.split("=", 1)[1]).partition("|")
    for resource in resources:
        for identifier in resource.get("identifier") or []:
            if identifier.get("system") == system and identifier.get("value") == value:
                return resource
    return None
