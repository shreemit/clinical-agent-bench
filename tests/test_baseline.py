import json
import logging
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from importlib.metadata import version
from pathlib import Path

import pytest

from agent.fhir_rules import first_quoted, require_bp_code, require_ndc, require_snomed
from agent.payloads import blood_pressure_observation, medication_request, service_request
from graders.state_collector import patient_bundles_for_grade, snapshot_patient
from graders.state_diff_grader import diff_bundles
from analysis.agreement import agreement_counts, format_agreement, load_results, main as agreement_main
from harness.http_client import HttpResult, UrllibHttpClient
from harness.types import AgentAction, TaskCase, ToolObservation, UnrecognizedAction
from proxy.fault_proxy import FaultProxy
from runners.baseline import (
    DEFAULT_LENGTH_RETRY_TOKENS,
    DEFAULT_MODEL,
    GROQ_API_URL,
    GROQ_DEFAULT_MODEL,
    GROQ_RATE_LIMIT_RETRIES,
    GROQ_RATE_LIMIT_WAIT_CAP_S,
    TRANSPORT_RETRIES,
    MissingApiKeyError,
    OpenAIChatClient,
    encode_model_url,
    load_task_ids,
    parse_model_action,
    render_observation,
    run_baseline,
    system_prompt,
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
        if method == "GET" and "?" not in url:
            kind, _, resource_id = url.rstrip("/").rpartition("/")
            kind = kind.rstrip("/").rsplit("/", 1)[-1]
            found = [
                item
                for item in self.resources
                if item.get("resourceType") == kind and item.get("id") == resource_id
            ]
            if found:
                return HttpResult(200, found[0], {}, 1)
            return HttpResult(404, {"resourceType": "OperationOutcome"}, {}, 1)
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


def test_parse_model_action_ignores_a_trailing_finish_and_a_later_post():
    posted = parse_model_action(
        'POST http://fhir.test/fhir/Observation\n'
        '{"resourceType": "Observation", "status": "final"}\n'
        "FINISH\n"
        "[]\n"
    )
    assert posted.kind == "POST"
    assert posted.url == "http://fhir.test/fhir/Observation"
    assert posted.body == {"resourceType": "Observation", "status": "final"}
    read = parse_model_action(
        "GET http://fhir.test/fhir/Patient/1\n"
        "POST http://fhir.test/fhir/Observation\n"
        '{"resourceType": "Observation"}\n'
    )
    assert read.kind == "GET"
    assert read.url == "http://fhir.test/fhir/Patient/1"
    assert read.body is None


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
    assert "max_completion_tokens" not in seen["body"]
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert seen["headers"]["User-Agent"] == f"clinical-agent-bench/{version('clinical-agent-bench-exec')}"


def test_groq_chat_client_omits_reasoning_effort():
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

    client = OpenAIChatClient(
        api_key="groq-key",
        http=FakeHttp(),
        provider="groq",
        reasoning_effort="high",
    )
    text = client.complete([{"role": "user", "content": "hi"}], model=GROQ_DEFAULT_MODEL)
    assert text == "FINISH\n[]"
    assert seen["method"] == "POST"
    assert seen["url"] == GROQ_API_URL
    assert seen["body"]["model"] == "openai/gpt-oss-20b"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["max_completion_tokens"] == 4096
    assert "reasoning_effort" not in seen["body"]
    assert seen["headers"]["Authorization"] == "Bearer groq-key"
    assert seen["headers"]["Content-Type"] == "application/json"
    assert seen["headers"]["User-Agent"].startswith("clinical-agent-bench/")
    assert seen["headers"]["User-Agent"] == f"clinical-agent-bench/{version('clinical-agent-bench-exec')}"


def test_groq_missing_api_key_fails_before_any_request(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "present-but-unused")

    class Boom:
        def request(self, *args, **kwargs):
            raise AssertionError("HTTP was called without a key")

    with pytest.raises(MissingApiKeyError, match="GROQ_API_KEY"):
        OpenAIChatClient(provider="groq", http=Boom())
    with pytest.raises(MissingApiKeyError, match="GROQ_API_KEY"):
        OpenAIChatClient(provider="groq", api_key="  ", http=Boom())
    from runners.baseline import main

    code = main(["--provider", "groq", "--k", "1", "--output", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 2
    assert "GROQ_API_KEY" in err
    assert "OPENAI_API_KEY" not in err
    assert not (tmp_path / "out").exists()


def test_groq_retries_429_using_retry_after(caplog):
    calls = []
    sleeps = []

    class FakeHttp:
        def request(self, method, url, body=None, headers=None, timeout=None):
            calls.append({"method": method, "url": url, "body": body, "headers": headers})
            if len(calls) == 1:
                return HttpResult(429, {"error": "rate_limit"}, {"Retry-After": "2"}, 1)
            return HttpResult(
                200,
                {"choices": [{"message": {"content": "FINISH\n[]"}}]},
                {},
                1,
            )

    client = OpenAIChatClient(
        api_key="groq-key",
        http=FakeHttp(),
        provider="groq",
        sleep=sleeps.append,
    )
    with caplog.at_level(logging.WARNING):
        text = client.complete([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    assert text == "FINISH\n[]"
    assert len(calls) == 2
    assert calls[0]["url"] == GROQ_API_URL
    assert calls[1]["url"] == GROQ_API_URL
    assert "reasoning_effort" not in calls[0]["body"]
    assert calls[0]["body"]["temperature"] == 0
    assert sleeps == [2.0]
    assert "429" in caplog.text
    assert "retrying" in caplog.text


def test_groq_caps_retry_after_and_stops_when_still_limited():
    sleeps = []
    calls = {"n": 0}

    class FakeHttp:
        def request(self, method, url, body=None, headers=None, timeout=None):
            del method, url, body, headers, timeout
            calls["n"] += 1
            if calls["n"] == 1:
                later = datetime.now(timezone.utc) + timedelta(seconds=90)
                return HttpResult(429, {"error": "rate_limit"}, {"Retry-After": format_datetime(later)}, 1)
            return HttpResult(429, {"error": "rate_limit"}, {}, 1)

    client = OpenAIChatClient(
        api_key="groq-key",
        http=FakeHttp(),
        provider="groq",
        sleep=sleeps.append,
    )
    with pytest.raises(RuntimeError, match="429"):
        client.complete([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    assert calls["n"] == GROQ_RATE_LIMIT_RETRIES + 1
    assert GROQ_RATE_LIMIT_WAIT_CAP_S == 30.0
    assert sleeps[0] == 30.0
    assert sleeps[1:] == [1.0, 1.0]


def test_chat_403_error_names_the_status():
    seen = {}

    class FakeHttp:
        def request(self, method, url, body=None, headers=None, timeout=None):
            del method, url, body, timeout
            seen["headers"] = headers
            return HttpResult(403, "error code: 1010", {}, 1)

    client = OpenAIChatClient(api_key="groq-key", http=FakeHttp(), provider="groq")
    with pytest.raises(RuntimeError, match=r"HTTP 403") as caught:
        client.complete([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    message = str(caught.value)
    assert "403" in message
    assert "1010" in message
    assert seen["headers"]["User-Agent"] == f"clinical-agent-bench/{version('clinical-agent-bench-exec')}"


def _gpt_oss_body(content, finish_reason, reasoning, reasoning_tokens=None):
    message = {"role": "assistant", "content": content, "reasoning": reasoning}
    body = {"choices": [{"finish_reason": finish_reason, "message": message}]}
    if reasoning_tokens is not None:
        body["usage"] = {"completion_tokens_details": {"reasoning_tokens": reasoning_tokens}}
    return body


def _groq_client(http, sleep=None):
    return OpenAIChatClient(
        api_key="groq-key",
        http=http,
        provider="groq",
        sleep=sleep,
    )


class _ScriptedHttp:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, body=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "body": body, "headers": headers})
        status, payload, response_headers = self.responses.pop(0)
        return HttpResult(status, payload, response_headers, 1)


def test_gpt_oss_stop_returns_content_and_not_reasoning():
    http = _ScriptedHttp(
        [
            (
                200,
                _gpt_oss_body(
                    "GET http://127.0.0.1:8080/fhir/Patient/1",
                    "stop",
                    "I should look up the patient. STOP_REASONING",
                    reasoning_tokens=12,
                ),
                {},
            )
        ]
    )
    text = _groq_client(http).complete([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    assert text == "GET http://127.0.0.1:8080/fhir/Patient/1"
    assert "STOP_REASONING" not in text
    assert http.calls[0]["body"]["max_completion_tokens"] == 4096


def _length_response(content, reasoning, reasoning_tokens):
    return (
        200,
        _gpt_oss_body(content, "length", reasoning, reasoning_tokens=reasoning_tokens),
        {},
    )


def test_gpt_oss_length_with_empty_or_partial_content_is_not_executed():
    empty = _ScriptedHttp(
        [
            _length_response(
                "",
                "GET http://127.0.0.1:8080/fhir/Patient/REASONING_ONLY",
                4096,
            ),
            _length_response(
                "",
                "GET http://127.0.0.1:8080/fhir/Patient/REASONING_ONLY",
                4096,
            ),
        ]
    )
    with pytest.raises(RuntimeError, match=r"finish_reason=length") as empty_error:
        _groq_client(empty).complete([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    empty_text = str(empty_error.value)
    assert "reasoning_tokens=4096" in empty_text
    assert "REASONING_ONLY" not in empty_text
    assert empty.calls[0]["body"]["max_completion_tokens"] == 4096
    assert empty.calls[1]["body"]["max_completion_tokens"] == DEFAULT_LENGTH_RETRY_TOKENS

    partial = _ScriptedHttp(
        [
            _length_response(
                "GET http://127.0.0.1:8080/fhir/Patient/PARTIAL_ONLY",
                "still thinking PARTIAL_REASONING",
                100,
            ),
            _length_response(
                "GET http://127.0.0.1:8080/fhir/Patient/PARTIAL_ONLY",
                "still thinking PARTIAL_REASONING",
                100,
            ),
        ]
    )
    with pytest.raises(RuntimeError, match=r"finish_reason=length") as partial_error:
        _groq_client(partial).complete([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    partial_text = str(partial_error.value)
    assert "reasoning_tokens=100" in partial_text
    assert "PARTIAL_ONLY" not in partial_text
    assert "PARTIAL_REASONING" not in partial_text
    assert partial.calls[1]["body"]["max_completion_tokens"] == 8192


def test_null_content_names_finish_reason():
    http = _ScriptedHttp(
        [
            (
                200,
                _gpt_oss_body(None, "stop", "NULL_REASONING_NOT_USED"),
                {},
            )
        ]
    )
    with pytest.raises(RuntimeError, match=r"finish_reason=stop") as caught:
        _groq_client(http).complete([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    assert "empty" in str(caught.value)
    assert "NULL_REASONING_NOT_USED" not in str(caught.value)


def test_tool_use_failed_recovers_both_argument_shapes(caplog):
    object_args = json.dumps(
        {
            "name": "assistant",
            "arguments": {
                "action": "GET",
                "url": (
                    "http://127.0.0.1:8080/fhir/Patient"
                    "?name=Peter%20Stafford&birthdate=1932-12-29"
                ),
            },
        }
    )
    string_args = json.dumps(
        {
            "name": "assistant",
            "arguments": "GET http://127.0.0.1:8080/fhir/Patient/9",
        }
    )
    plain = "POST http://fhir.test/fhir/Observation\n" + json.dumps(
        {"resourceType": "Observation"}
    )
    payload_args = json.dumps(
        {
            "name": "assistant",
            "arguments": {
                "action": "PUT",
                "url": "http://fhir.test/fhir/Observation/1",
                "payload": {"resourceType": "Observation", "id": "1"},
            },
        }
    )
    cases = [
        (object_args, "GET", "Peter%20Stafford"),
        (string_args, "GET", "/Patient/9"),
        (plain, "POST", "Observation"),
        (payload_args, "PUT", "Observation/1"),
    ]
    for failed_generation, kind, needle in cases:
        http = _ScriptedHttp(
            [
                (
                    400,
                    {
                        "error": {
                            "code": "tool_use_failed",
                            "message": "Tool choice is none, but model called a tool",
                            "failed_generation": failed_generation,
                        }
                    },
                    {},
                )
            ]
        )
        with caplog.at_level(logging.WARNING):
            text = _groq_client(http).complete(
                [{"role": "user", "content": "hi"}],
                model="openai/gpt-oss-20b",
            )
        action = parse_model_action(text)
        assert action.kind == kind
        assert needle in (action.url or "")
        assert "recovered chat action" in caplog.text
        caplog.clear()


def test_tool_use_failed_without_an_action_is_a_short_http_400(caplog):
    secret = "failed_generation should stay out of the error"
    http = _ScriptedHttp(
        [
            (
                400,
                {
                    "error": {
                        "code": "tool_use_failed",
                        "message": "Tool choice is none, but model called a tool",
                        "failed_generation": secret,
                    }
                },
                {},
            )
        ]
    )
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match=r"HTTP 400: tool_use_failed$") as caught:
            _groq_client(http).complete([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    assert secret not in str(caught.value)
    assert "recovered" not in caplog.text
    assert len(http.calls) == 1


def test_non_finite_retry_after_waits_one_second(caplog):
    for header in ("nan", "inf", "-inf", "NaN", "-5", "Infinity"):
        http = _ScriptedHttp(
            [
                (429, {"error": {"message": "slow", "code": "rate_limit_exceeded"}}, {"Retry-After": header}),
                (200, {"choices": [{"message": {"content": "FINISH\n[]"}}]}, {}),
            ]
        )
        sleeps = []
        text = _groq_client(http, sleep=sleeps.append).complete(
            [{"role": "user", "content": "hi"}],
            model="openai/gpt-oss-20b",
        )
        assert text == "FINISH\n[]"
        assert sleeps == [1.0], header


def test_groq_retries_413_when_the_request_fits_the_limit():
    message = "Tokens per minute exceeded. Limit 8000, Requested 8000."
    http = _ScriptedHttp(
        [
            (
                413,
                {"error": {"code": "rate_limit_exceeded", "message": message}},
                {"Retry-After": "2"},
            ),
            (200, {"choices": [{"message": {"content": "FINISH\n[]"}}]}, {}),
        ]
    )
    sleeps = []
    text = _groq_client(http, sleep=sleeps.append).complete(
        [{"role": "user", "content": "hi"}],
        model="openai/gpt-oss-20b",
    )
    assert text == "FINISH\n[]"
    assert sleeps == [2.0]
    assert len(http.calls) == 2


def test_groq_413_over_the_limit_fails_without_retry():
    message = "Request too large. Limit 8000, Requested 12000."
    http = _ScriptedHttp(
        [
            (
                413,
                {"error": {"code": "rate_limit_exceeded", "message": message}},
                {"Retry-After": "1"},
            )
        ]
    )
    sleeps = []
    with pytest.raises(RuntimeError, match=r"HTTP 413") as caught:
        _groq_client(http, sleep=sleeps.append).complete(
            [{"role": "user", "content": "hi"}],
            model="openai/gpt-oss-20b",
        )
    text = str(caught.value)
    assert "requested 12000 tokens exceeds limit 8000" in text
    assert sleeps == []
    assert len(http.calls) == 1


def test_api_key_with_a_newline_is_rejected_without_quoting_it(monkeypatch):
    class Boom:
        def request(self, *args, **kwargs):
            raise AssertionError("HTTP was called")

    secret = "sk-live-secret-value\r\nX-Injected: yes"
    with pytest.raises(MissingApiKeyError) as caught:
        OpenAIChatClient(api_key=secret, http=Boom(), provider="groq")
    text = str(caught.value)
    assert "sk-live-secret-value" not in text
    assert "X-Injected" not in text
    assert "line feed" in text
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-secret\n")
    with pytest.raises(MissingApiKeyError) as from_env:
        OpenAIChatClient(http=Boom())
    assert "sk-env-secret" not in str(from_env.value)


def test_transport_status_zero_is_not_called_http_zero():
    http = _ScriptedHttp([(0, {"error": "timed out"}, {})] * (TRANSPORT_RETRIES + 1))
    sleeps = []
    with pytest.raises(RuntimeError, match=r"^transport error: timed out$") as caught:
        _groq_client(http, sleep=sleeps.append).complete(
            [{"role": "user", "content": "hi"}],
            model="openai/gpt-oss-20b",
        )
    assert "HTTP 0" not in str(caught.value)
    assert len(http.calls) == TRANSPORT_RETRIES + 1
    assert sleeps == [1.0] * TRANSPORT_RETRIES


def test_openai_rate_limit_is_not_retried():
    calls = {"n": 0}

    class FakeHttp:
        def request(self, method, url, body=None, headers=None, timeout=None):
            del method, url, body, headers, timeout
            calls["n"] += 1
            return HttpResult(429, {"error": "rate_limit"}, {"Retry-After": "0"}, 1)

    client = OpenAIChatClient(api_key="test-key", http=FakeHttp(), sleep=lambda _seconds: None)
    with pytest.raises(RuntimeError, match="429"):
        client.complete([{"role": "user", "content": "hi"}], model="gpt-6-luna")
    assert calls["n"] == 1


def test_cli_groq_uses_its_own_client_and_default_model(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "groq-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured = {}

    def fake_run(tasks, **kwargs):
        del tasks
        captured.update(kwargs)
        return []

    monkeypatch.setattr("runners.baseline.run_baseline", fake_run)
    from runners.baseline import main

    code = main(
        [
            "--provider",
            "groq",
            "--k",
            "1",
            "--output",
            str(tmp_path / "out"),
            "--tasks",
            str(repo_root() / "tasks" / "baseline_100.txt"),
        ]
    )
    assert code == 0
    assert captured["provider"] == "groq"
    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["resume"] is False
    assert captured["ready_timeout_s"] == 180
    assert captured["client"].length_retry_tokens == 8192
    assert captured["client"].api_key == "groq-test"
    assert captured["client"].api_url == GROQ_API_URL
    assert captured["client"].max_completion_tokens == 4096
    assert isinstance(captured["client"].http, UrllibHttpClient)
    assert isinstance(captured["http"], UrllibHttpClient)
    assert captured["client"].http is not captured["http"]
    assert not isinstance(captured["http"], FaultProxy)
    assert not isinstance(captured["client"].http, FaultProxy)

    captured.clear()
    code = main(
        [
            "--provider",
            "groq",
            "--model",
            "openai/gpt-oss-120b",
            "--max-completion-tokens",
            "2048",
            "--k",
            "1",
            "--output",
            str(tmp_path / "large"),
            "--tasks",
            str(repo_root() / "tasks" / "baseline_100.txt"),
        ]
    )
    assert code == 0
    assert captured["model"] == "openai/gpt-oss-120b"
    assert captured["provider"] == "groq"
    assert captured["client"].max_completion_tokens == 2048


def test_results_record_provider_next_to_fault_fields(tmp_path):
    task = load_task_file()["task1_1"]
    chart = Chart(task.eval_mrn)

    def reply_for(assistant_turns, messages, model):
        del assistant_turns, messages, model
        return 'FINISH\n["S6534835"]'

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "groq-row",
        model="openai/gpt-oss-20b",
        provider="groq",
        client=ScriptedChat(reply_for),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
        fault="none",
        seed=4,
        mitigation="off",
    )
    row = rows[0]
    assert row["provider"] == "groq"
    assert row["model"] == "openai/gpt-oss-20b"
    assert row["fault"] == "none"
    assert row["seed"] == 4
    assert row["mitigation"] == "off"
    saved = load_results(tmp_path / "groq-row" / "results.jsonl")
    assert saved[0]["provider"] == "groq"
    assert saved[0]["model"] == "openai/gpt-oss-20b"
    assert saved[0]["fault"] == "none"
    assert saved[0]["seed"] == 4


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
    assert rows[0]["provider"] == "openai"
    assert rows[0]["model"] == DEFAULT_MODEL
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
    assert saved[0]["provider"] == "openai"
    assert saved[0]["model"] == DEFAULT_MODEL
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

    rows = run_baseline(
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
    assert len(rows) == 2
    assert rows[0]["trial"] == 0
    assert rows[0]["state_passed"] is True
    assert rows[1]["state_passed"] is False
    assert rows[1]["contaminated"] is True
    assert "snapshot restore failed" in rows[1]["failure_reason"]
    saved = load_results(tmp_path / "partial" / "results.jsonl")
    assert len(saved) == 2
    assert saved[0]["trial"] == 0
    assert saved[0]["task_id"] == "task2_1"
    assert saved[1]["task_id"] == "task2_1"


def test_stale_search_still_grades_a_write_from_its_location(tmp_path):
    task = load_task_file()["task3_1"]
    posted = blood_pressure_observation(task.eval_mrn, "118/77 mmHg", "BP")

    class StaleSearch:
        def request(self, method, url, body=None, headers=None, timeout=None):
            if method == "GET" and f"/Patient/{task.eval_mrn}" in url and "?" not in url:
                return HttpResult(200, {"resourceType": "Patient", "id": task.eval_mrn}, {}, 1)
            if method == "POST":
                resource = json.loads(json.dumps(body))
                resource["id"] = "99"
                self.resource = resource
                return HttpResult(201, resource, {"Location": url.rstrip("/") + "/99/_history/1"}, 1)
            if method == "GET" and url.rstrip("/").endswith("/Observation/99"):
                return HttpResult(200, self.resource, {}, 1)
            if method == "GET":
                return HttpResult(
                    200,
                    {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []},
                    {},
                    1,
                )
            raise AssertionError(f"{method} {url}")

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "stale",
        model=DEFAULT_MODEL,
        client=ScriptedChat(_post_then_finish(posted, "http://fhir.test/fhir/Observation")),
        fhir_api_base="http://fhir.test/fhir/",
        http=StaleSearch(),
        refsol_path=tmp_path / "absent.py",
        reset=lambda: None,
    )
    assert rows[0]["state_passed"] is True
    assert any(item.startswith("Observation/") for item in rows[0]["created"])


def test_system_prompt_includes_functions_finish_and_no_delete():
    prompt = system_prompt("http://127.0.0.1:8080/fhir/")
    assert "GET {api_base}/Patient" in prompt
    assert "birthdate" in prompt
    assert "given" in prompt
    assert "family" in prompt
    assert "The final answer must start with FINISH" in prompt
    assert "Only GET, POST, and PUT are available" in prompt
    assert "DELETE is not supported" in prompt
    assert prompt.index("functions in JSON format") < prompt.index("note is a list")


def test_parse_model_action_encodes_unsafe_characters_once():
    piped = parse_model_action(
        "GET http://127.0.0.1:8080/fhir/Patient?identifier=MRN|S6415739"
    )
    assert piped.url.endswith("identifier=MRN%7CS6415739")
    zoned = parse_model_action(
        "GET http://127.0.0.1:8080/fhir/Observation?patient=S1"
        "&effective=ge2023-11-12T10:15:00+00:00"
    )
    assert "ge2023-11-12T10" in zoned.url
    assert "%2B00" in zoned.url
    assert "+" not in zoned.url.split("?", 1)[1]
    spaced = parse_model_action("GET http://127.0.0.1:8080/fhir/Patient?name=Peter Stafford")
    assert "name=Peter%20Stafford" in spaced.url
    kept = encode_model_url("http://127.0.0.1:8080/fhir/Patient?name=Peter%20Stafford")
    assert "name=Peter%20Stafford" in kept
    assert "%2520" not in kept


def test_parse_model_action_keeps_only_the_first_of_several():
    action = parse_model_action(
        "GET http://fhir.test/fhir/Patient?name=Peter%20Stafford\n"
        "POST http://fhir.test/fhir/Observation\n"
        '{"resourceType": "Observation"}\n'
    )
    assert action.kind == "GET"
    assert action.url.endswith("name=Peter%20Stafford")
    same_line = parse_model_action("GET http://fhir.test/fhir/Patient/1 POST http://fhir.test/fhir/Observation")
    assert same_line.kind == "GET"
    assert same_line.url == "http://fhir.test/fhir/Patient/1"
    finished = parse_model_action("FINISH([90])")
    assert finished.kind == "FINISH"
    assert finished.finish_payload == "[90]"
    with pytest.raises(UnrecognizedAction, match="unrecognized action"):
        parse_model_action("93.0")
    with pytest.raises(UnrecognizedAction, match="unrecognized action"):
        parse_model_action('{"answer": 1}')


def test_bare_reply_is_returned_to_the_model(tmp_path):
    task = load_task_file()["task2_1"]
    chart = Chart(task.eval_mrn)
    seen = []

    def reply_for(assistant_turns, messages, model):
        del model
        seen.append(messages)
        if assistant_turns == 0:
            return "90"
        if assistant_turns == 1:
            return "-1"
        return "FINISH\n[73]"

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "bare",
        model=DEFAULT_MODEL,
        client=ScriptedChat(reply_for),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
    )
    assert rows[0]["state_passed"] is True
    assert rows[0]["finish_text"] == "[73]"
    feedback = " ".join(message["content"] for message in seen[-1] if message["role"] == "user")
    assert "unrecognized action '90'" in feedback
    assert "unrecognized action '-1'" in feedback
    assert "FINISH" in feedback


def test_delete_is_rejected_without_calling_the_server(tmp_path):
    task = load_task_file()["task2_1"]
    chart = Chart(task.eval_mrn)
    methods = []
    original = chart.request

    def request(method, url, body=None, headers=None, timeout=None):
        methods.append(method)
        return original(method, url, body=body, headers=headers, timeout=timeout)

    chart.request = request

    def reply_for(assistant_turns, messages, model):
        del model
        if assistant_turns == 0:
            return "DELETE http://127.0.0.1:8080/fhir/MedicationRequest/786003"
        text = " ".join(message["content"] for message in messages)
        assert "DELETE is not supported" in text
        return "FINISH\n[1]"

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "delete",
        model=DEFAULT_MODEL,
        client=ScriptedChat(reply_for),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
    )
    assert rows[0]["state_passed"] is True
    assert "DELETE" not in methods


def test_html_error_pages_become_short_text_and_outcomes_stay_json():
    html = (
        "<html><head><title>400 Bad Request</title></head>"
        "<body><h1>HTTP Status 400 – Bad Request</h1>"
        "<p>Invalid character found in the request target</p>"
        + ("<p>pad</p>" * 200)
        + "</body></html>"
    )
    text = render_observation(
        AgentAction(kind="GET", url="http://fhir.test/fhir/Patient"),
        ToolObservation(status=400, body=html, headers={}),
    )
    assert "<html" not in text.lower()
    assert "Bad Request" in text
    assert "Invalid character" in text
    assert len(text) < 800
    outcome = {
        "resourceType": "OperationOutcome",
        "issue": [{"severity": "error", "diagnostics": "Unknown search parameter"}],
    }
    kept = render_observation(
        AgentAction(kind="GET", url="http://fhir.test/fhir/Observation"),
        ToolObservation(status=400, body=outcome, headers={}),
    )
    assert "OperationOutcome" in kept
    assert "Unknown search parameter" in kept


def test_length_retry_uses_the_stopped_completion():
    http = _ScriptedHttp(
        [
            _length_response("GET http://fhir.test/PARTIAL", "thinking", 4094),
            (
                200,
                {"choices": [{"finish_reason": "stop", "message": {"content": "FINISH\n[1]"}}]},
                {},
            ),
        ]
    )
    text = _groq_client(http).complete([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
    assert text == "FINISH\n[1]"
    assert http.calls[0]["body"]["max_completion_tokens"] == 4096
    assert http.calls[1]["body"]["max_completion_tokens"] == 8192


def test_transient_handshake_is_retried():
    http = _ScriptedHttp(
        [
            (0, {"error": "_ssl.c:1067: The handshake operation timed out"}, {}),
            (200, {"choices": [{"message": {"content": "FINISH\n[]"}}]}, {}),
        ]
    )
    sleeps = []
    text = _groq_client(http, sleep=sleeps.append).complete(
        [{"role": "user", "content": "hi"}],
        model="openai/gpt-oss-20b",
    )
    assert text == "FINISH\n[]"
    assert len(http.calls) == 2
    assert sleeps == [1.0]


def test_one_task_error_does_not_stop_the_next(tmp_path):
    tasks = load_task_file()
    first = tasks["task1_1"]
    second = tasks["task2_1"]
    chart = Chart(second.eval_mrn)

    def request(method, url, body=None, headers=None, timeout=None):
        if first.eval_mrn in url:
            raise RuntimeError(f"search exceeded 200 pages: {url}")
        return chart.request(method, url, body=body, headers=headers, timeout=timeout)

    class Split:
        def request(self, method, url, body=None, headers=None, timeout=None):
            return request(method, url, body=body, headers=headers, timeout=timeout)

    def reply_for(assistant_turns, messages, model):
        del assistant_turns, messages, model
        return "FINISH\n[1]"

    rows = run_baseline(
        [first, second],
        k=1,
        output_dir=tmp_path / "isolated",
        model=DEFAULT_MODEL,
        client=ScriptedChat(reply_for),
        fhir_api_base="http://fhir.test/fhir/",
        http=Split(),
        refsol_path=tmp_path / "absent.py",
        reset=lambda: None,
    )
    assert len(rows) == 2
    assert rows[0]["task_id"] == "task1_1"
    assert rows[0]["state_passed"] is False
    assert "search exceeded 200 pages" in rows[0]["failure_reason"]
    assert rows[1]["task_id"] == "task2_1"
    assert rows[1]["state_passed"] is True
    saved = load_results(tmp_path / "isolated" / "results.jsonl")
    assert [row["task_id"] for row in saved] == ["task1_1", "task2_1"]


def test_resume_skips_pairs_already_in_results(tmp_path):
    task = load_task_file()["task2_1"]
    chart = Chart(task.eval_mrn)
    calls = {"n": 0}

    def reply_for(assistant_turns, messages, model):
        del assistant_turns, messages, model
        calls["n"] += 1
        return "FINISH\n[1]"

    output = tmp_path / "resume"
    first = run_baseline(
        [task],
        k=1,
        output_dir=output,
        model=DEFAULT_MODEL,
        client=ScriptedChat(reply_for),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
    )
    assert calls["n"] == 1
    assert first[0]["finish_text"] == "[1]"

    def second_reply(assistant_turns, messages, model):
        del assistant_turns, messages, model
        calls["n"] += 1
        return "FINISH\n[2]"

    calls["n"] = 0
    rows = run_baseline(
        [task],
        k=2,
        output_dir=output,
        model=DEFAULT_MODEL,
        client=ScriptedChat(second_reply),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=chart.reset,
        resume=True,
    )
    assert calls["n"] == 1
    assert rows[0]["trial"] == 0
    assert rows[0]["finish_text"] == "[1]"
    assert rows[1]["trial"] == 1
    assert rows[1]["finish_text"] == "[2]"
    assert chart.reset_count == 2


class LargeHistory:
    """Thousands of observations. Deep pages fail; counts and watermarks do not."""

    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.patient = {"resourceType": "Patient", "id": patient_id}
        self.observations = []
        for index in range(5000):
            self.observations.append(
                {
                    "resourceType": "Observation",
                    "id": f"old-{index}",
                    "status": "final",
                    "code": {"text": "OTHER", "coding": [{"code": "OTHER"}]},
                    "meta": {"lastUpdated": "2020-01-01T00:00:00+00:00", "versionId": "1"},
                    "subject": {"reference": f"Patient/{patient_id}"},
                }
            )
        self.observations.append(
            {
                "resourceType": "Observation",
                "id": "mg1",
                "status": "final",
                "code": {"text": "MG", "coding": [{"code": "MG"}]},
                "effectiveDateTime": "2023-11-13T08:00:00+00:00",
                "valueQuantity": {"value": 1.2, "unit": "mg/dL"},
                "meta": {"lastUpdated": "2020-06-01T00:00:00+00:00", "versionId": "1"},
                "subject": {"reference": f"Patient/{patient_id}"},
            }
        )
        self.medications = []
        self.services = []
        self.histories = {}
        self.urls = []
        self.statuses = []
        self.next_id = 1

    def request(self, method, url, body=None, headers=None, timeout=None):
        del headers, timeout
        self.urls.append(url)
        if "_getpages" in url or "_getpagesoffset" in url:
            result = HttpResult(500, {"error": "deep page"}, {}, 1)
            self.statuses.append(result.status)
            return result
        if (
            "/Observation?" in url
            and "_count=200" in url
            and "_lastUpdated=gt" not in url
            and "_summary=count" not in url
        ):
            result = HttpResult(500, {"error": "full history"}, {}, 1)
            self.statuses.append(result.status)
            return result
        if method == "GET" and f"/Patient/{self.patient_id}" in url and "?" not in url:
            return self._ok(self.patient)
        if "/_history" in url:
            return self._history(url)
        if method == "GET" and "?" not in url:
            return self._read(url)
        if method == "GET" and "_summary=count" in url:
            return self._ok(self._count_bundle(url))
        if method == "GET" and "/Observation?" in url and "_count=1" in url:
            newest = max(self.observations, key=lambda item: item["meta"]["lastUpdated"])
            return self._ok(self._bundle([newest], total=len(self.observations)))
        if method == "GET" and "code=MG" in url:
            labs = [item for item in self.observations if item["code"]["text"] == "MG"]
            return self._ok(self._bundle(labs, total=len(labs)))
        if method == "GET" and ("code=K" in url or "code=A1C" in url):
            return self._ok(self._bundle([], total=0))
        if method == "GET" and "_lastUpdated=gt" in url:
            return self._changed(url)
        if method == "GET" and "/MedicationRequest?" in url:
            return self._ok(self._bundle(self.medications, total=len(self.medications)))
        if method == "GET" and "/ServiceRequest?" in url:
            return self._ok(self._bundle(self.services, total=len(self.services)))
        if method == "POST" and url.split("?")[0].rstrip("/").endswith("/Observation"):
            resource = json.loads(json.dumps(body))
            resource["id"] = "bp-1"
            resource["meta"] = {"versionId": "1", "lastUpdated": "2024-01-01T00:00:00+00:00"}
            self.observations.append(resource)
            location = url.split("?")[0].rstrip("/") + f"/{resource['id']}/_history/1"
            result = HttpResult(201, resource, {"Location": location}, 1)
            self.statuses.append(result.status)
            return result
        if method == "PUT" and "/Observation/" in url:
            resource = json.loads(json.dumps(body or {}))
            resource_id = url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
            resource["id"] = resource_id
            resource["meta"] = {"versionId": "2", "lastUpdated": "2024-01-01T00:00:01+00:00"}
            replaced = False
            kept = []
            for item in self.observations:
                if item["id"] == resource_id:
                    kept.append(resource)
                    replaced = True
                else:
                    kept.append(item)
            if not replaced:
                kept.append(resource)
            self.observations = kept
            return self._ok(resource)
        if method == "POST" and "/MedicationRequest" in url:
            resource = json.loads(json.dumps(body))
            resource["id"] = f"med-{self.next_id}"
            self.next_id += 1
            resource["meta"] = {"versionId": "1", "lastUpdated": "2024-01-01T00:00:00+00:00"}
            self.medications.append(resource)
            location = url.split("?")[0].rstrip("/") + f"/{resource['id']}/_history/1"
            result = HttpResult(201, resource, {"Location": location}, 1)
            self.statuses.append(result.status)
            return result
        result = HttpResult(404, {"resourceType": "OperationOutcome"}, {}, 1)
        self.statuses.append(result.status)
        return result

    def _ok(self, body):
        result = HttpResult(200, body, {}, 1)
        self.statuses.append(result.status)
        return result

    def _bundle(self, resources, total):
        return {
            "resourceType": "Bundle",
            "type": "searchset",
            "total": total,
            "entry": [{"resource": item} for item in resources],
        }

    def _count_bundle(self, url):
        if "/Observation?" in url:
            total = len(self.observations)
        elif "/MedicationRequest?" in url:
            total = len(self.medications)
        else:
            total = len(self.services)
        return {"resourceType": "Bundle", "type": "searchset", "total": total}

    def _changed(self, url):
        from urllib.parse import unquote

        marker = "_lastUpdated=gt"
        raw = unquote(url.split(marker, 1)[1].split("&", 1)[0])
        changed = [
            item
            for item in self.observations
            if item.get("meta", {}).get("lastUpdated", "") > raw
        ]
        return self._ok(self._bundle(changed, total=len(changed)))

    def _read(self, url):
        resource_id = url.rstrip("/").rsplit("/", 1)[-1]
        for collection in (self.observations, self.medications, self.services):
            for item in collection:
                if item["id"] == resource_id:
                    return self._ok(item)
        return HttpResult(404, {"resourceType": "OperationOutcome"}, {}, 1)

    def _history(self, url):
        resource_id = url.split("/_history", 1)[0].rstrip("/").rsplit("/", 1)[-1]
        versions = self.histories.get(resource_id)
        if versions:
            return self._ok(self._bundle(versions, total=len(versions)))
        for item in self.observations:
            if item["id"] == resource_id:
                return self._ok(self._bundle([item], total=1))
        return HttpResult(404, {"resourceType": "OperationOutcome"}, {}, 1)


def test_large_history_grades_a_create_without_paging(tmp_path):
    task = load_task_file()["task5_1"]
    chart = LargeHistory(task.eval_mrn)
    body = medication_request(
        task.eval_mrn,
        require_ndc(task.context),
        "IV",
        2,
        "g",
        rate_value=2,
        rate_unit="h",
    )
    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "large",
        model=DEFAULT_MODEL,
        client=ScriptedChat(_post_then_finish(body, "http://fhir.test/fhir/MedicationRequest")),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=lambda: None,
    )
    assert rows[0]["state_passed"] is True, rows[0]["mismatches"]
    assert 500 not in chart.statuses
    assert not any("_getpages" in url for url in chart.urls)
    assert not any(
        "/Observation?" in url and "_count=200" in url and "_lastUpdated=gt" not in url
        for url in chart.urls
    )
    assert len(chart.urls) < 40
    snapshot = snapshot_patient(chart, "http://fhir.test/fhir/", task.eval_mrn)
    assert len(snapshot.bundle["entry"]) < 20
    assert snapshot.large is True


def test_large_history_grades_a_post_then_put_as_a_create(tmp_path):
    task = load_task_file()["task3_1"]
    chart = LargeHistory(task.eval_mrn)
    posted = blood_pressure_observation(
        task.eval_mrn, first_quoted(task.instruction), require_bp_code(task.context)
    )
    updated = blood_pressure_observation(
        task.eval_mrn,
        first_quoted(task.instruction),
        require_bp_code(task.context),
        "bp-1",
    )

    def reply_for(assistant_turns, messages, model):
        del messages, model
        if assistant_turns == 0:
            return "POST http://fhir.test/fhir/Observation\n" + json.dumps(posted)
        if assistant_turns == 1:
            return "PUT http://fhir.test/fhir/Observation/bp-1\n" + json.dumps(updated)
        return "FINISH\n[]"

    rows = run_baseline(
        [task],
        k=1,
        output_dir=tmp_path / "large-put",
        model=DEFAULT_MODEL,
        client=ScriptedChat(reply_for),
        fhir_api_base="http://fhir.test/fhir/",
        http=chart,
        refsol_path=tmp_path / "absent.py",
        reset=lambda: None,
    )
    assert rows[0]["state_passed"] is True, rows[0]["mismatches"]
    assert rows[0]["updated"] == []
    assert any(item.startswith("Observation/") for item in rows[0]["created"])
    assert 500 not in chart.statuses


def test_large_history_detects_an_update_and_a_delete():
    chart = LargeHistory("S1")
    base = "http://fhir.test/fhir/"
    snapshot = snapshot_patient(chart, base, "S1")
    target = next(item for item in chart.observations if item["id"] == "old-0")
    updated = json.loads(json.dumps(target))
    updated["valueQuantity"] = {"value": 9, "unit": "mg/dL"}
    updated["meta"] = {"lastUpdated": "2024-01-01T00:00:00+00:00", "versionId": "2"}
    chart.observations = [updated if item["id"] == "old-0" else item for item in chart.observations]
    chart.histories["old-0"] = [target, updated]
    pre, post = patient_bundles_for_grade(chart, base, "S1", snapshot, [])
    diff = diff_bundles(pre, post)
    assert [item["id"] for item in diff["updated"]] == ["old-0"]
    assert diff["created"] == []
    assert diff["deleted"] == []


def test_large_history_detects_a_delete():
    chart = LargeHistory("S1")
    base = "http://fhir.test/fhir/"
    snapshot = snapshot_patient(chart, base, "S1")
    assert any(entry["resource"]["id"] == "mg1" for entry in snapshot.bundle["entry"])
    chart.observations = [item for item in chart.observations if item["id"] != "mg1"]
    pre, post = patient_bundles_for_grade(chart, base, "S1", snapshot, [])
    deleted = diff_bundles(pre, post)
    assert any(item["id"] == "mg1" for item in deleted["deleted"])
