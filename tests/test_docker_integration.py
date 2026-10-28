"""Live HAPI checks. Skipped unless the compose stack answers /fhir/metadata."""

import os
import urllib.error
import urllib.request

import pytest

from agent.mock_agent import MockAgent
from fhir.snapshot_reset import restore_snapshot
from harness.exec_harness import ExecHarness
from harness.types import AgentAction
from tasks_io import load_task_file

pytestmark = pytest.mark.docker


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


def test_task1_read_and_task3_post_put_hit_hapi():
    base = _base()
    tasks = load_task_file()
    harness = ExecHarness(base)
    read = harness.run(tasks["task1_1"], MockAgent(tasks["task1_1"], base))
    assert read.passed, read.failure_reason
    assert read.calls[0].method == "GET"
    assert read.finish_text == '["S6534835"]'

    write = harness.run(tasks["task3_1"], MockAgent(tasks["task3_1"], base))
    assert write.passed, write.failure_reason
    assert [call.method for call in write.calls] == ["POST", "PUT"]
    assert write.calls[0].status == 201
    assert write.calls[1].status == 200
    assert write.calls[0].response_body["resourceType"] == "Observation"


def test_snapshot_restore_drops_a_write_and_records_timing():
    """One real volume restore. Slow: it restarts HAPI.

    The 5 second research gate is recorded on the receipt. This test does not
    fail when the gate is missed.

    TODO(decision): fail this test when under_5s is false once a faster reset exists.
    """

    base = _base()
    harness = ExecHarness(base)
    _, created = harness.step(
        AgentAction(
            kind="POST",
            url=base + "Observation",
            body={
                "resourceType": "Observation",
                "status": "final",
                "code": {"text": "cab-exec-marker"},
                "valueString": "week1-reset-marker",
                "subject": {"reference": "Patient/S2380121"},
            },
        )
    )
    assert created.status == 201, created.response_body
    marker_id = created.response_body["id"]

    receipt = restore_snapshot()
    assert receipt.duration_ms > 0
    assert receipt.restore_copy_ms >= 0
    assert receipt.ready_ms > 0
    assert receipt.target_ms == 5000

    _, fetched = harness.step(AgentAction(kind="GET", url=f"{base}Observation/{marker_id}?_format=json"))
    assert fetched.status == 404, fetched.status
