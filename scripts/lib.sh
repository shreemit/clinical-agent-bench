#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HAPI_IMAGE="jyxsu6/medagentbench@sha256:3fb83d7ed71c5476f9eb6212bd440a909ef7505922bbc757dc488a8fc0701966"
HAPI_CONTAINER="cab-exec-hapi"
FHIR_METADATA_URL="${CAB_FHIR_METADATA_URL:-http://127.0.0.1:8080/fhir/metadata}"
SNAPSHOT_DB="$ROOT/docker/snapshots/clean/test_db.mv.db"
LIVE_DIR="$ROOT/docker/live"
LIVE_DB="$LIVE_DIR/test_db.mv.db"

if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif sudo docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
else
  echo "Docker is not available. Install Docker and start the daemon." >&2
  exit 1
fi

compose() {
  "${DOCKER[@]}" compose -f "$ROOT/docker-compose.yml" "$@"
}

now_ms() {
  date +%s%3N
}

require_hapi_stopped() {
  local state
  state="$("${DOCKER[@]}" inspect -f '{{.State.Running}}' "$HAPI_CONTAINER" 2>/dev/null || true)"
  if [[ "$state" == "true" ]]; then
    echo "Container ${HAPI_CONTAINER} is still running; refusing to copy the database." >&2
    exit 1
  fi
}

own_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    return 0
  fi
  if [[ "$(stat -c %u "$path")" != "$(id -u)" ]]; then
    sudo chown "$(id -u):$(id -g)" "$path"
  fi
  chmod a+rwX "$path"
}
