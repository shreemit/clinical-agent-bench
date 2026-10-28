#!/usr/bin/env bash
# Restore the Docker bind-mounted H2 file from the clean snapshot, restart
# HAPI, and time the whole reset until GET /fhir/metadata returns 200.
#
# The plan's Week 1 gate is under 5000 ms. This script records the measured
# duration either way. The copy and the JVM restart are timed separately.
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
t_stop1="$(now_ms)"

t_copy0="$(now_ms)"
own_path "$LIVE_DIR"
own_path "$SNAPSHOT_DB"
rm -f "$LIVE_DIR/test_db.mv.db" "$LIVE_DIR/test_db.lock.db" "$LIVE_DIR/test_db.trace.db"
cp -a "$SNAPSHOT_DB" "$LIVE_DB"
own_path "$LIVE_DB"
t_copy1="$(now_ms)"

t_ready0="$(now_ms)"
compose up -d hapi >/dev/null
ready=0
for _ in $(seq 1 120); do
  if curl -sf -m 2 -o /dev/null "$FHIR_METADATA_URL"; then
    ready=1
    break
  fi
  sleep 1
done
t1="$(now_ms)"

if [[ "$ready" -ne 1 ]]; then
  echo "FHIR did not become ready after snapshot restore." >&2
  "${DOCKER[@]}" logs --tail 40 cab-exec-hapi >&2 || true
  exit 1
fi

python3 - "$t0" "$t_stop0" "$t_stop1" "$t_copy0" "$t_copy1" "$t_ready0" "$t1" "$TARGET_MS" "$ROOT" << 'PY'
import json
import pathlib
import sys

t0, t_stop0, t_stop1, t_copy0, t_copy1, t_ready0, t1, target, root = sys.argv[1:]
t0, t_stop0, t_stop1, t_copy0, t_copy1, t_ready0, t1, target = map(int, (t0, t_stop0, t_stop1, t_copy0, t_copy1, t_ready0, t1, target))
total = t1 - t0
record = {
    "duration_ms": total,
    "stop_ms": t_stop1 - t_stop0,
    "restore_copy_ms": t_copy1 - t_copy0,
    "ready_ms": t1 - t_ready0,
    "under_5s": total < target,
    "target_ms": target,
}
line = json.dumps(record)
path = pathlib.Path(root) / "runs" / "reset_timings.jsonl"
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a") as handle:
    handle.write(line + "\n")
print("RESET_JSON " + line)
print(
    f"reset {total} ms (stop {record['stop_ms']} ms, "
    f"copy {record['restore_copy_ms']} ms, ready {record['ready_ms']} ms) "
    f"under_{target}ms={str(record['under_5s']).lower()}"
)
PY
