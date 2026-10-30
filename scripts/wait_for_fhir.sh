#!/usr/bin/env bash
source "$(dirname "$0")/lib.sh"

for _ in $(seq 1 90); do
  if curl -sf -m 2 -o /dev/null "$FHIR_METADATA_URL"; then
    echo "FHIR ready at $FHIR_METADATA_URL"
    exit 0
  fi
  sleep 2
done

echo "Timed out waiting for $FHIR_METADATA_URL" >&2
exit 1
