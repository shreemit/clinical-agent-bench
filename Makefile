export PYTHONPATH := src

.PHONY: sync fetch load up snapshot reset smoke test test-docker

sync:
	uv sync --extra dev

fetch:
	bash scripts/fetch_medagentbench.sh

load:
	bash scripts/load_fhir_data.sh

up:
	bash -c 'source scripts/lib.sh && compose up -d hapi && bash scripts/wait_for_fhir.sh'

snapshot:
	bash scripts/snapshot_create.sh

reset:
	bash scripts/snapshot_restore.sh

smoke:
	uv run python -m smoke

test:
	uv run pytest -m "not docker"

test-docker:
	uv run pytest -m docker
