import json
from pathlib import Path

import pytest

from agent.fhir_rules import first_quoted, require_snomed
from agent.payloads import blood_pressure_observation, service_request
from analysis.agreement import agreement_counts, format_agreement, load_results, main as agreement_main
from harness.http_client import HttpResult
from harness.types import TaskCase
from runners.baseline import (
    DEFAULT_MODEL,
    MissingApiKeyError,
    OpenAIChatClient,
    load_task_ids,
    parse_model_action,
    run_baseline,
)
from tasks_io import load_task_file, repo_root


class Chart:
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.initial = [{"resourceType": "Patient", "id": patient_id}]
        self.resources = [dict(item) for item in self.initial]
        self.next_id = 1
        self.reset_count = 0

    def reset(self):
        self.resources = [dict(item) for item in self.initial]
        self.next_id = 1
        self.reset_count += 1

    def request(self, method, url, body=None, headers=None, timeout=None):
        if method == "GET" and f"/Patient/{self.patient_id}" in url and "?" not in url:
            found = [
                item
                for item in self.resources
                if item.get("resourceType") == "Patient" and item.get("id") == self.patient_id
            ]
            if not found:
                return HttpResult(404, {"resourceType": "OperationOutcome"}, {}, 1)
            return HttpResult(200, found[0], {}, 1)
        if method == "GET":
            kind = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
            matches = [item for item in self.resources if item.get("resourceType") == kind]
            return HttpResult(
                200,
                {
                    "resourceType": "Bundle",
                    "type": "searchset",
                    "total": len(matches),
                    "entry": [{"resource": item} for item in matches],
                },
                {},
                1,
            )
        if method == "POST":
            resource = json.loads(json.dumps(body))
            resource["id"] = f"new-{self.next_id}"
            self.next_id += 1
            resource["meta"] = {"versionId": "1", "lastUpdated": "2023-11-13T10:16:00+00:00"}
            self.resources.append(resource)
            location = url.rstrip("/") + f"/{resource['id']}/_history/1"
            return HttpResult(201, resource, {"Location": location}, 1)
        raise AssertionError(f"unexpected {method} {url}")


class ScriptedChat:
    def __init__(self, reply_for):
        self.reply_for = reply_for
        self.calls = 0

    def complete(self, messages, model):
        self.calls += 1
        assistant_turns = sum(1 for message in messages if message["role"] == "assistant")
        return self.reply_for(assistant_turns, messages, model)


def _post_then_finish(body, url):
    posted = "POST " + url + "\n" + json.dumps(body)

    def reply_for(assistant_turns, messages, model):
        del messages, model
        if assistant_turns == 0:
            return posted
        return "FINISH\n[]"

    return reply_for


def _refsol(path: Path, source: str) -> Path:
    path.write_text(source)
    return path


def test_agreement_script_prints_the_table(tmp_path, capsys):
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"state_passed": True, "refsol_passed": True}),
                json.dumps({"state_passed": False, "refsol_passed": True}),
                json.dumps({"state_passed": True, "refsol_passed": False}),
                json.dumps({"state_passed": False, "refsol_passed": False}),
                json.dumps({"state_passed": True, "refsol_passed": None}),
            ]
        )
        + "\n"
    )
    assert agreement_main([str(path)]) == 0
    text = capsys.readouterr().out
    assert "transcript pass" in text
    assert "transcript fail" in text
    assert "agreed 2/4" in text
    assert "skipped rows without a transcript grade: 1" in text


def test_baseline_task_list_has_100_ids():
    ids = load_task_ids(repo_root() / "tasks" / "baseline_100.txt")
    assert len(ids) == 100
    assert len(set(ids)) == 100
    tasks = load_task_file()
    for family in range(1, 11):
        found = [task_id for task_id in ids if task_id.startswith(f"task{family}_")]
        assert found == [f"task{family}_{index}" for index in range(1, 11)]
        assert all(task_id in tasks for task_id in found)


def test_parse_model_action_strips_fences():
    action = parse_model_action("```\nFINISH\n[1]\n```")
    assert action.kind == "FINISH"
    assert action.finish_payload == "[1]"
    posted = parse_model_action('POST http://fhir.test/fhir/Observation\n{"resourceType": "Observation"}')
    assert posted.kind == "POST"
    assert posted.body["resourceType"] == "Observation"


def test_missing_api_key_fails_before_any_request(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class Boom:
        def request(self, *args, **kwargs):
            raise AssertionError("HTTP was called without a key")

    with pytest.raises(MissingApiKeyError, match="OPENAI_API_KEY"):
        OpenAIChatClient(http=Boom())
    with pytest.raises(MissingApiKeyError, match="OPENAI_API_KEY"):
        OpenAIChatClient(api_key="  ", http=Boom())
    from runners.baseline import main

    code = main(["--k", "1", "--output", str(tmp_path / "out")])
    assert code == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_chat_client_posts_the_configured_model():
    seen = {}

    class FakeHttp:
        def request(self, method, url, body=None, headers=None, timeout=None):
            seen["method"] = method
            seen["url"] = url
            seen["body"] = body
            seen["headers"] = headers
            return HttpResult(
                200,
                {"choices": [{"message": {"content": "FINISH\n[]"}}]},
                {},
                1,
            )

    client = OpenAIChatClient(api_key="test-key", http=FakeHttp())
    text = client.complete([{"role": "user", "content": "hi"}], model="gpt-6-luna")
    assert text == "FINISH\n[]"
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/chat/completions")
    assert seen["body"]["model"] == DEFAULT_MODEL
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["reasoning_effort"] == "none"
    assert seen["headers"]["Authorization"] == "Bearer test-key"


def test_k1_and_k5_reset_grade_and_save_the_prefix(tmp_path):
    tasks = load_task_file()
    task = tasks["task8_1"]
    body = service_request(
        task.eval_mrn,
        "http://snomed.info/sct",
        require_snomed(task.context),
        note=first_quoted(task.instruction),
    )
    chart = Chart(task.eval_mrn)
    refsol = _refsol(
        tmp_path / "refsol.py",
        "def task8(case_data, results, fhir_api_base):\n"
        "    for index, item in enumerate(results.history):\n"
        "        if item.role == 'agent' and item.content.startswith('POST'):\n"
        "            nxt = results.history[index + 1].content if index + 1 < len(results.history) else ''\n"
        "            return 'POST request accepted' in nxt\n"
        "    return False\n",
    )
    rows = run_baseline(
        [task],
        k=5,
        output_dir=tmp_path / "k5",
        model=DEFAULT_MODEL,
        client=ScriptedChat(_post_then_finish(body, "http://fhir.test/fhir/ServiceRequest")),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=refsol,
        reset=chart.reset,
    )
    assert len(rows) == 5
    assert chart.reset_count == 5
    assert all(row["state_passed"] and row["refsol_passed"] for row in rows)
    assert all(row["refsol_status"] == "graded" for row in rows)
    assert all(not row["contaminated"] for row in rows)
    assert rows[0]["reset"] is True
    prefix = json.loads((tmp_path / "k5" / "prefixes" / "task8_1.json").read_text())
    assert prefix["model_id"] == DEFAULT_MODEL
    assert prefix["pending_post_call"]["kind"] == "POST"
    assert prefix["pending_post_call"]["url"].endswith("/ServiceRequest")
    assert all(message["role"] != "assistant" for message in prefix["messages_before_post"])
    saved = load_results(tmp_path / "k5" / "results.jsonl")
    assert len(saved) == 5
    summary = agreement_counts(saved)
    assert summary["pass_pass"] == 5
    assert "transcript pass" in (tmp_path / "k5" / "agreement.txt").read_text()

    chart.reset_count = 0
    one = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "k1",
        model="gpt-6-luna",
        client=ScriptedChat(_post_then_finish(body, "http://fhir.test/fhir/ServiceRequest")),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=refsol,
        reset=chart.reset,
    )
    assert len(one) == 1 and one[0]["state_passed"]
    assert chart.reset_count == 1


def test_read_task_prefix_has_no_pending_write_and_refsol_sees_the_answer(tmp_path):
    task = load_task_file()["task1_1"]
    chart = Chart(task.eval_mrn)
    refsol = _refsol(
        tmp_path / "refsol.py",
        "import json\n"
        "def task1(case_data, results, fhir_api_base):\n"
        "    for item in results.history:\n"
        "        if item.role == 'agent' and 'POST' in item.content:\n"
        "            return False\n"
        "    try:\n"
        "        return json.loads(results.result) == case_data['sol']\n"
        "    except Exception:\n"
        "        return False\n",
    )

    def reply_for(assistant_turns, messages, model):
        del assistant_turns, messages, model
        return 'FINISH\n["S6534835"]'

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "read",
        model=DEFAULT_MODEL,
        client=ScriptedChat(reply_for),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=refsol,
        reset=chart.reset,
    )
    assert rows[0]["state_passed"] is True
    assert rows[0]["refsol_passed"] is True
    prefix = json.loads((tmp_path / "read" / "prefixes" / "task1_1.json").read_text())
    assert prefix["pending_post_call"] is None
    assert any("S6534835" in message["content"] for message in prefix["messages_before_post"])
    assert all(
        "The FHIR base URL" not in message["content"]
        for message in prefix["messages_before_post"]
        if message["role"] == "assistant"
    )


def test_transcript_pass_with_unchanged_chart_is_a_disagreement(tmp_path):
    task = load_task_file()["task3_1"]
    chart = Chart(task.eval_mrn)
    refsol = _refsol(
        tmp_path / "refsol.py",
        "def task3(case_data, results, fhir_api_base):\n    return True\n",
    )

    def reply_for(assistant_turns, messages, model):
        del assistant_turns, messages, model
        return "FINISH\n[]"

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "phantom",
        model=DEFAULT_MODEL,
        client=ScriptedChat(reply_for),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=refsol,
        reset=chart.reset,
    )
    assert rows[0]["refsol_passed"] is True
    assert rows[0]["state_passed"] is False
    assert any("missing created resource" in item for item in rows[0]["mismatches"])
    text = format_agreement(agreement_counts(rows))
    assert "1" in text.splitlines()[1]


def test_missing_refsol_is_skipped(tmp_path, capsys):
    task = load_task_file()["task2_1"]
    chart = Chart(task.eval_mrn)

    def reply_for(assistant_turns, messages, model):
        del assistant_turns, messages, model
        return "FINISH\n[73]"

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "skip",
        model=DEFAULT_MODEL,
        client=ScriptedChat(reply_for),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "missing_refsol.py",
        reset=chart.reset,
    )
    assert rows[0]["state_passed"] is True
    assert rows[0]["refsol_passed"] is None
    assert rows[0]["refsol_status"] == "skipped"
    message = capsys.readouterr().err
    assert "refsol skipped:" in message
    assert "missing_refsol.py is not present" in message
    report = (tmp_path / "skip" / "agreement.txt").read_text()
    assert "Transcript grader skipped" in report


def test_no_reset_keeps_writes_on_the_next_trial(tmp_path, capsys):
    task = load_task_file()["task3_1"]
    body = blood_pressure_observation(
        task.eval_mrn, "118/77 mmHg", "BP"
    )
    chart = Chart(task.eval_mrn)
    rows = run_baseline(
        [task],
        k=2,
        output_dir=tmp_path / "keep",
        model=DEFAULT_MODEL,
        client=ScriptedChat(_post_then_finish(body, "http://fhir.test/fhir/Observation")),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        no_reset=True,
        reset=chart.reset,
    )
    assert chart.reset_count == 0
    assert rows[0]["reset"] is False
    assert rows[0]["contaminated"] is False
    assert rows[1]["contaminated"] is True
    assert any(label.startswith("Observation/") for label in rows[0]["post_ids"])
    assert any(label in rows[1]["pre_ids"] for label in rows[0]["post_ids"] if label.startswith("Observation/"))
    assert "resets disabled" in capsys.readouterr().out


def test_wrong_dose_fails_the_state_grade(tmp_path):
    task = load_task_file()["task3_1"]
    body = blood_pressure_observation(task.eval_mrn, "90/60", "BP")
    chart = Chart(task.eval_mrn)
    rows = run_baseline(
        [TaskCase(task.id, task.instruction, task.context, task.sol, task.eval_mrn)],
        k=1,
        output_dir=tmp_path / "wrong",
        model=DEFAULT_MODEL,
        client=ScriptedChat(_post_then_finish(body, "http://fhir.test/fhir/Observation")),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
    )
    assert rows[0]["state_passed"] is False


def test_unfinished_trial_does_not_state_pass_a_read_task(tmp_path):
    task = load_task_file()["task1_1"]
    chart = Chart(task.eval_mrn)

    class Broken:
        def complete(self, messages, model):
            del messages, model
            raise RuntimeError("model unavailable")

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "dead",
        model=DEFAULT_MODEL,
        client=Broken(),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
    )
    assert rows[0]["finish_text"] is None
    assert rows[0]["failure_reason"]
    assert rows[0]["state_passed"] is False
    assert any("trial did not finish" in item for item in rows[0]["mismatches"])


def test_refsol_error_is_omitted_from_the_agreement_table(tmp_path):
    task = load_task_file()["task8_1"]
    body = service_request(
        task.eval_mrn,
        "http://snomed.info/sct",
        require_snomed(task.context),
        note=first_quoted(task.instruction),
    )
    chart = Chart(task.eval_mrn)
    refsol = _refsol(
        tmp_path / "refsol.py",
        "def task8(case_data, results, fhir_api_base):\n"
        "    raise TypeError('note is a list, not an object')\n",
    )
    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "refsol-error",
        model=DEFAULT_MODEL,
        client=ScriptedChat(_post_then_finish(body, "http://fhir.test/fhir/ServiceRequest")),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=refsol,
        reset=chart.reset,
    )
    assert rows[0]["state_passed"] is True
    assert rows[0]["refsol_passed"] is None
    assert rows[0]["refsol_status"] == "error"
    assert "note is a list" in rows[0]["refsol_message"]
    summary = agreement_counts(rows)
    assert summary["total"] == 0
    assert summary["skipped"] == 1


def test_reset_failure_keeps_trials_already_written(tmp_path):
    task = load_task_file()["task2_1"]
    chart = Chart(task.eval_mrn)
    calls = {"n": 0}

    def reset():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("snapshot restore failed")
        chart.reset()

    def reply_for(assistant_turns, messages, model):
        del assistant_turns, messages, model
        return "FINISH\n[1]"

    with pytest.raises(RuntimeError, match="snapshot restore failed"):
        run_baseline(
            [task],
            k=2,
            output_dir=tmp_path / "partial",
            model=DEFAULT_MODEL,
            client=ScriptedChat(reply_for),
            fhir_api_base="http://fhir.test/fhir/",
            http=chart,
            refsol_path=tmp_path / "absent.py",
            reset=reset,
        )
    saved = load_results(tmp_path / "partial" / "results.jsonl")
    assert len(saved) == 1
    assert saved[0]["trial"] == 0
    assert saved[0]["task_id"] == "task2_1"
