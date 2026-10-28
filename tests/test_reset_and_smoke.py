from fhir.reset_timing import parse_reset_stdout, under_reset_target
from smoke import SmokeRow, format_smoke_table, summarize_resets
from fhir.reset_timing import ResetReceipt


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
