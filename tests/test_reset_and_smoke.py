from fhir.hapi_ready import main as ready_main
from fhir.hapi_ready import metadata_ok, recover_ready
from fhir.reset_timing import ResetReceipt, parse_reset_stdout, under_reset_target
from smoke import SmokeRow, format_smoke_table, summarize_resets
from tasks_io import repo_root


def test_parse_reset_receipt_records_a_restart():
    text = (
        'RESET_JSON {"duration_ms": 1, "stop_ms": 1, "restore_copy_ms": 1, "ready_ms": 1, '
        '"under_5s": false, "target_ms": 5000, "restarted": true, "restart_ms": 40, '
        '"ready_timeout_s": 180}\n'
    )
    receipt = parse_reset_stdout(text)
    assert receipt.restarted is True
    assert receipt.restart_ms == 40
    assert receipt.ready_timeout_s == 180


def test_ready_wait_restarts_the_container_once():
    calls = {"wait": 0, "restart": 0}

    def wait(timeout_s):
        calls["wait"] += 1
        calls["timeout"] = timeout_s
        return calls["wait"] == 2

    report = recover_ready(wait, lambda: calls.__setitem__("restart", calls["restart"] + 1), 180)
    assert calls["wait"] == 2
    assert calls["timeout"] == 180
    assert calls["restart"] == 1
    assert report["ready"] is True
    assert report["restarted"] is True
    assert report["attempts"] == 2

    failed = {"restart": 0}
    report = recover_ready(lambda timeout_s: False, lambda: failed.__setitem__("restart", 1), 5)
    assert report["ready"] is False
    assert report["restarted"] is True
    assert failed["restart"] == 1

    skipped = {"restart": 0}
    report = recover_ready(lambda timeout_s: True, lambda: skipped.__setitem__("restart", 1), 5)
    assert report["ready"] is True
    assert report["restarted"] is False
    assert skipped["restart"] == 0


def test_ready_command_restarts_once(monkeypatch):
    calls = {"wait": 0, "restart": 0}

    def wait(url, timeout_s, poll_s=1):
        del url, timeout_s, poll_s
        calls["wait"] += 1
        return calls["wait"] == 2

    monkeypatch.setattr("fhir.hapi_ready.wait_until_ready", wait)
    monkeypatch.setattr(
        "fhir.hapi_ready._restart_container",
        lambda docker, container: calls.__setitem__("restart", calls["restart"] + 1),
    )
    code = ready_main(
        [
            "--timeout",
            "180",
            "--metadata-url",
            "http://127.0.0.1:8080/fhir/metadata",
            "--container",
            "cab-exec-hapi",
            "--docker",
            "docker",
        ]
    )
    assert code == 0
    assert calls["restart"] == 1


def test_metadata_ok_against_a_local_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"resourceType":"CapabilityStatement"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/fhir/metadata"
        assert metadata_ok(url, timeout=1)
    finally:
        server.shutdown()
        server.server_close()
    assert metadata_ok("http://127.0.0.1:9/fhir/metadata", timeout=0.2) is False


def test_restore_script_restarts_after_the_ready_timeout():
    text = (repo_root() / "scripts" / "snapshot_restore.sh").read_text()
    assert "CAB_READY_TIMEOUT_S" in text
    assert "fhir.hapi_ready" in text
    assert "HAPI_CONTAINER" in text
    assert "one container restart" in text


def test_parse_reset_receipt_and_target():
    text = 'noise\nRESET_JSON {"duration_ms": 42000, "stop_ms": 800, "restore_copy_ms": 2100, "ready_ms": 39000, "under_5s": false, "target_ms": 5000}\n'
    receipt = parse_reset_stdout(text)
    assert receipt.duration_ms == 42000
    assert receipt.restore_copy_ms == 2100
    assert receipt.under_5s is False
    assert under_reset_target(4999)
    assert not under_reset_target(5000)


def test_smoke_table_shows_pass_fail_and_reset():
    fast = ResetReceipt(1000, 10, 20, 30, True, 5000)
    slow = ResetReceipt(40000, 10, 20, 30, False, 5000)
    rows = [
        SmokeRow("task1_1", True, None, "GET", '["S6534835"]', fast),
        SmokeRow("task3_1", False, "status 500", "POST,PUT", None, slow),
    ]
    table = format_smoke_table(rows)
    assert "task1_1" in table and "PASS" in table
    assert "task3_1" in table and "FAIL" in table
    assert "POST,PUT" in table
    assert "yes" in table and "no" in table
    summary = summarize_resets(rows)
    assert "1/2" in summary
    assert "5000" in summary
