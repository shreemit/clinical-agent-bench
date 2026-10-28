#!/usr/bin/env bash
# Stop HAPI and copy the live database file over the clean snapshot.
# Run this only while the live database is still the pristine MedAgentBench
# load (or immediately after `make reset`). A snapshot taken after writes
# becomes the new baseline.
source "$(dirname "$0")/lib.sh"

if [[ ! -s "$LIVE_DB" ]]; then
  echo "Live database missing. Run make load first." >&2
  exit 1
fi

echo "Stopping HAPI so the H2 file is quiescent..."
compose stop -t 8 hapi >/dev/null || true

mkdir -p "$(dirname "$SNAPSHOT_DB")"
own_path "$LIVE_DB"
cp -a "$LIVE_DB" "$SNAPSHOT_DB"
own_path "$SNAPSHOT_DB"
echo "Wrote snapshot $(wc -c < "$SNAPSHOT_DB") bytes to $SNAPSHOT_DB"

compose up -d hapi
bash "$ROOT/scripts/wait_for_fhir.sh"
