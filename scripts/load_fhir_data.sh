#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"

mkdir -p "$(dirname "$SNAPSHOT_DB")" "$LIVE_DIR" "$ROOT/runs"

if [[ ! -s "$SNAPSHOT_DB" ]]; then
  echo "Copying /data/test_db.mv.db from the image (about 4.2GB)..."
  cid="$("${DOCKER[@]}" create "$HAPI_IMAGE")"
  cleanup() { "${DOCKER[@]}" rm -f "$cid" >/dev/null; }
  trap cleanup EXIT
  "${DOCKER[@]}" cp "$cid:/data/test_db.mv.db" "$SNAPSHOT_DB"
  trap - EXIT
  cleanup
else
  echo "Snapshot already present at $SNAPSHOT_DB"
fi

own_path "$SNAPSHOT_DB"
own_path "$LIVE_DIR"
cp -a "$SNAPSHOT_DB" "$LIVE_DB"
own_path "$LIVE_DB"

python3 - << PY
import json, os, pathlib
snap = pathlib.Path("$SNAPSHOT_DB")
manifest = {
    "image": "$HAPI_IMAGE",
    "bytes": snap.stat().st_size,
    "path": "docker/snapshots/clean/test_db.mv.db",
}
path = pathlib.Path("docker/snapshots/clean/manifest.json")
path.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"loaded {manifest['bytes']} bytes into {snap}")
PY
