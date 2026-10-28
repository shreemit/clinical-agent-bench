#!/usr/bin/env bash
# Shared docker/compose helpers for the Week 1 FHIR scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Digest of jyxsu6/medagentbench:latest as resolved on 2026-10-28.
HAPI_IMAGE="jyxsu6/medagentbench@sha256:3fb83d7ed71c5476f9eb6212bd440a909ef7505922bbc757dc488a8fc0701966"
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

# docker cp runs as root when the daemon is root. Hand the file back to the
# user who invoked make, and keep it writable by the container uid (65532).
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
