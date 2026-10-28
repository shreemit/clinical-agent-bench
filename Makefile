PYTHON ?= python3
export PYTHONPATH := src

.PHONY: fetch load up snapshot reset smoke test test-docker

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
	$(PYTHON) -m smoke

test:
	$(PYTHON) -m pytest -m "not docker"

test-docker:
	$(PYTHON) -m pytest -m docker
