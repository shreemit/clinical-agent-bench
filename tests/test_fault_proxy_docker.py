import json
import os
import uuid
import urllib.error
import urllib.request

import pytest

from harness.http_client import UrllibHttpClient
from proxy.fault_proxy import FaultProxy

pytestmark = pytest.mark.docker

PATIENT = "Patient/S2380121"
HEADERS = {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}
PROBE = "urn:cab:probe"


def _base() -> str:
    base = os.environ.get("CAB_FHIR_BASE", "http://127.0.0.1:8080/fhir/")
    url = base.rstrip("/") + "/metadata"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            if response.status != 200:
                pytest.skip(f"FHIR metadata returned {response.status}")
    except (urllib.error.URLError, TimeoutError) as error:
        pytest.skip(f"HAPI FHIR is not reachable at {url}: {error}")
    return base if base.endswith("/") else base + "/"


def _body(resource_type: str, unique: str) -> dict:
    probe = {"system": PROBE, "value": unique}
    subject = {"reference": PATIENT}
    if resource_type == "Observation":
        return {
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": "cab-fault"},
            "valueString": unique,
            "subject": subject,
            "identifier": [probe],
        }
    if resource_type == "MedicationRequest":
        return {
            "resourceType": "MedicationRequest",
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {"text": unique},
            "subject": subject,
            "identifier": [probe],
        }
    return {
        "resourceType": "ServiceRequest",
        "status": "active",
        "intent": "order",
        "code": {"text": unique},
        "subject": subject,
        "identifier": [probe],
    }


def _count(http: UrllibHttpClient, base: str, resource_type: str, unique: str) -> int:
    token = f"{PROBE}|{unique}".replace("|", "%7C")
    url = f"{base}{resource_type}?identifier={token}&_count=20"
    result = http.request(
        "GET",
        url,
        headers={"Accept": "application/fhir+json", "Cache-Control": "no-cache"},
    )
    assert result.status == 200, result.body
    return len((result.body or {}).get("entry") or [])


def test_committed_timeout_leaves_one_resource_and_a_504(tmp_path):
    base = _base()
    direct = UrllibHttpClient(timeout=120)
    unique = uuid.uuid4().hex
    proxy = FaultProxy(
        direct,
        fault="committed_timeout",
        seed=0,
        mitigation="off",
        log_path=tmp_path / "faults.jsonl",
        trial_id=unique,
        task_id="task3_1",
    )
    result = proxy.request("POST", base + "Observation", body=_body("Observation", unique), headers=HEADERS)
    assert result.status == 504, result.body
    assert result.body["resourceType"] == "OperationOutcome"
    assert _count(direct, base, "Observation", unique) == 1
    location = proxy.committed_locations()
    assert len(location) == 1
    fetched = direct.request("GET", location[0], headers={"Accept": "application/fhir+json"})
    assert fetched.status == 200, fetched.body
    assert fetched.body["valueString"] == unique


@pytest.mark.parametrize(
    ("fault", "status", "seed"),
    [
        ("uncommitted_timeout", 504, 0),
        ("conflict_409", 409, 0),
        ("precondition_412", 412, 0),
        ("unprocessable_422", 422, 0),
        ("transient_5xx", 503, 0),
        ("transient_5xx", 500, 1),
    ],
)
def test_uncommitted_faults_leave_no_resource(tmp_path, fault, status, seed):
    base = _base()
    direct = UrllibHttpClient(timeout=120)
    unique = uuid.uuid4().hex
    proxy = FaultProxy(
        direct,
        fault=fault,
        seed=seed,
        mitigation="off",
        log_path=tmp_path / "faults.jsonl",
        trial_id=unique,
        task_id="task3_1",
    )
    result = proxy.request("POST", base + "Observation", body=_body("Observation", unique), headers=HEADERS)
    assert result.status == status, result.body
    assert result.body["issue"][0]["severity"] == "error"
    assert _count(direct, base, "Observation", unique) == 0
    row = json.loads((tmp_path / "faults.jsonl").read_text().splitlines()[0])
    assert row["upstream_status"] is None
    assert row["upstream_location"] is None


@pytest.mark.parametrize("resource_type", ["Observation", "MedicationRequest", "ServiceRequest"])
def test_retry_after_committed_timeout_stays_one_resource(tmp_path, resource_type):
    base = _base()
    direct = UrllibHttpClient(timeout=120)
    unique = uuid.uuid4().hex
    proxy = FaultProxy(
        direct,
        fault="committed_timeout",
        seed=0,
        mitigation="preconditions",
        on_attempt=1,
        log_path=tmp_path / "faults.jsonl",
        trial_id=unique,
        task_id=resource_type,
    )
    url = base + resource_type
    body = _body(resource_type, unique)
    first = proxy.request("POST", url, body=body, headers=HEADERS)
    assert first.status == 504, first.body
    second = proxy.request("POST", url, body=body, headers=HEADERS)
    assert second.status == 200, second.body
    assert _count(direct, base, resource_type, unique) == 1
    assert second.body["id"]


@pytest.mark.parametrize("resource_type", ["Observation", "MedicationRequest", "ServiceRequest"])
def test_duplicate_retry_with_preconditions_stays_one_resource(tmp_path, resource_type):
    base = _base()
    direct = UrllibHttpClient(timeout=120)
    unique = uuid.uuid4().hex
    proxy = FaultProxy(
        direct,
        fault="duplicate_retry",
        seed=0,
        mitigation="preconditions",
        on_attempt=1,
        log_path=tmp_path / "faults.jsonl",
        trial_id=unique,
        task_id=resource_type,
    )
    result = proxy.request("POST", base + resource_type, body=_body(resource_type, unique), headers=HEADERS)
    assert result.status == 200, result.body
    assert _count(direct, base, resource_type, unique) == 1
    rows = [json.loads(line) for line in (tmp_path / "faults.jsonl").read_text().splitlines()]
    assert [row["forward"] for row in rows] == [1, 2]
    assert rows[0]["upstream_status"] == 201
    assert rows[1]["upstream_status"] == 200


def test_stale_if_match_put_is_rejected_and_does_not_overwrite(tmp_path):
    base = _base()
    direct = UrllibHttpClient(timeout=120)
    unique = uuid.uuid4().hex
    proxy = FaultProxy(
        direct,
        fault="none",
        mitigation="preconditions",
        log_path=tmp_path / "faults.jsonl",
        trial_id=unique,
        task_id="task3_1",
    )
    created = proxy.request("POST", base + "Observation", body=_body("Observation", unique), headers=HEADERS)
    assert created.status == 201, created.body
    instance = base + "Observation/" + created.body["id"]
    read = proxy.request("GET", instance, headers={"Accept": "application/fhir+json"})
    assert read.status == 200, read.body
    current = dict(read.body)
    current.pop("meta", None)
    current["valueString"] = "newer-" + unique
    updated = direct.request("PUT", instance, body=current, headers=HEADERS)
    assert updated.status == 200, updated.body
    stale = dict(read.body)
    stale.pop("meta", None)
    stale["valueString"] = "stale-" + unique
    rejected = proxy.request("PUT", instance, body=stale, headers=HEADERS)
    assert rejected.status == 409, rejected.body
    diagnostics = rejected.body["issue"][0]["diagnostics"]
    assert "not the current version" in diagnostics
    fetched = direct.request(
        "GET",
        instance,
        headers={"Accept": "application/fhir+json", "Cache-Control": "no-cache"},
    )
    assert fetched.status == 200, fetched.body
    assert fetched.body["valueString"] == "newer-" + unique
