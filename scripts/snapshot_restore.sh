#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"

if [[ ! -s "$SNAPSHOT_DB" ]]; then
  echo "Missing $SNAPSHOT_DB. Run make load (and make snapshot if you need a fresh capture)." >&2
  exit 1
fi

mkdir -p "$LIVE_DIR" "$ROOT/runs"
TARGET_MS=5000
t0="$(now_ms)"

t_stop0="$(now_ms)"
compose stop -t 8 hapi >/dev/null || true
require_hapi_stopped
t_stop1="$(now_ms)"

t_copy0="$(now_ms)"
own_path "$LIVE_DIR"
own_path "$SNAPSHOT_DB"
rm -f "$LIVE_DIR/test_db.lock.db" "$LIVE_DIR/test_db.trace.db" "${LIVE_DB}.tmp"
cp -a "$SNAPSHOT_DB" "${LIVE_DB}.tmp"
own_path "${LIVE_DB}.tmp"
mv -f "${LIVE_DB}.tmp" "$LIVE_DB"
own_path "$LIVE_DB"
t_copy1="$(now_ms)"

t_ready0="$(now_ms)"
compose up -d hapi >/dev/null
READY_TIMEOUT_S="${CAB_READY_TIMEOUT_S:-180}"
set +e
ready_stdout="$(
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m fhir.hapi_ready \
    --timeout "$READY_TIMEOUT_S" \
    --metadata-url "$FHIR_METADATA_URL" \
    --container "$HAPI_CONTAINER" \
    --docker "${DOCKER[*]}"
)"
ready_code=$?
set -e
t1="$(now_ms)"
export READY_STDOUT="$ready_stdout"
export READY_CODE="$ready_code"
export READY_TIMEOUT_S

python3 - "$t0" "$t_stop0" "$t_stop1" "$t_copy0" "$t_copy1" "$t_ready0" "$t1" "$TARGET_MS" "$ROOT" << 'PY'
import json
import os
import pathlib
import sys

t0, t_stop0, t_stop1, t_copy0, t_copy1, t_ready0, t1, target, root = sys.argv[1:]
t0, t_stop0, t_stop1, t_copy0, t_copy1, t_ready0, t1, target = map(int, (t0, t_stop0, t_stop1, t_copy0, t_copy1, t_ready0, t1, target))
restarted = False
restart_ms = 0
for line in os.environ.get("READY_STDOUT", "").splitlines():
    if line.startswith("READY_JSON "):
        payload = json.loads(line[len("READY_JSON ") :])
        restarted = bool(payload.get("restarted"))
        restart_ms = int(payload.get("restart_ms") or 0)
        break
total = t1 - t0
record = {
    "duration_ms": total,
    "stop_ms": t_stop1 - t_stop0,
    "restore_copy_ms": t_copy1 - t_copy0,
    "ready_ms": t1 - t_ready0,
    "under_5s": total < target,
    "target_ms": target,
    "restarted": restarted,
    "restart_ms": restart_ms,
    "ready_timeout_s": int(float(os.environ.get("READY_TIMEOUT_S", "180"))),
}
line = json.dumps(record)
path = pathlib.Path(root) / "runs" / "reset_timings.jsonl"
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a") as handle:
    handle.write(line + "\n")
print("RESET_JSON " + line)
print(
    f"reset {total} ms (stop {record['stop_ms']} ms, "
    f"copy {record['restore_copy_ms']} ms, ready {record['ready_ms']} ms, "
    f"restarted={str(record['restarted']).lower()}, restart {record['restart_ms']} ms) "
    f"under_{target}ms={str(record['under_5s']).lower()}"
)
PY

if [[ "$ready_code" -ne 0 ]]; then
  echo "FHIR did not become ready after snapshot restore and one container restart." >&2
  "${DOCKER[@]}" logs --tail 40 "$HAPI_CONTAINER" >&2 || true
  exit 1
fi
