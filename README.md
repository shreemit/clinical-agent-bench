# ClinicalAgentBench-Exec

Execution harness for a clinical-agent benchmark. A scripted agent sends real FHIR `GET`, `POST`, and `PUT` calls to HAPI FHIR. There are no model API calls.

The tasks and the patient database come from [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) (MIT). See `NOTICE` and `vendor/medagentbench/LICENSE`.

## What this runs

- HAPI FHIR JPA R4 in Docker Compose, using the published MedAgentBench image and its H2 database (`/data/test_db.mv.db`, about 4.2GB).
- Reset by stopping HAPI, copying that database file back from `docker/snapshots/clean/`, starting HAPI, and logging the time until `GET /fhir/metadata` returns 200. The log is `runs/reset_timings.jsonl`. The target is under 5000 ms. The smoke table prints whether each reset met it.
- An execution harness that performs the agent's HTTP calls and records status, URL, request body, and response summary.
- A deterministic mock agent with one script per MedAgentBench v2 task family (`task1`–`task10`).
- `make smoke`: reset, run `task1_1` … `task10_1`, print a pass/fail table and the reset timings.

The fault proxy, idempotency wrapper, state-diff grader, repair-loop grader, prefix runner, autonomy summary, and `DELETE` + `$expunge` reset are not yet implemented. They raise `NotImplementedError` and are not on the smoke path.

## Prerequisites

- Python 3.11+
- Docker Engine with the Compose plugin
- About 10GB of free disk for two copies of the H2 file (snapshot + live)
- Port 8080 free

## Exact local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

make fetch    # docker pull of the pinned image; verifies vendored task checksums
make load     # copy /data/test_db.mv.db out of the image into docker/live and docker/snapshots/clean
make up       # start HAPI and wait until /fhir/metadata is 200
make test     # pytest, no Docker
make test-docker
make smoke    # 10 resets + 10 tasks; prints the table
```

`make reset` restores the snapshot once and prints one timing line. `make snapshot` stops HAPI and replaces the clean snapshot with the current live file. Take that only from a database that has not been written since the last reset.

Smoke task ids are listed in `tasks/smoke_ids.txt`. Override the FHIR base with `CAB_FHIR_BASE` (default `http://127.0.0.1:8080/fhir/`).

A task passes when every FHIR call returns a 2xx status and the agent emits `FINISH`. `task1` rows also compare `FINISH` to the `sol` field in the vendored task file. Other families have no `sol` in that file, so their answers are not graded.

## Layout

```text
docker-compose.yml          HAPI service, bind-mounts ./docker/live at /data
scripts/                    fetch, load, snapshot, restore
vendor/medagentbench/       test_data_v2.json, funcs_v1.json, upstream LICENSE
src/harness/exec_harness.py real GET/POST/PUT
src/agent/mock_agent.py     scripted agent
src/fhir/snapshot_reset.py  calls the restore script
src/smoke.py                10-task command
tests/                      unit tests, plus tests/test_docker_integration.py
```

`runs/` and `docker/live/` are gitignored.

## Open decisions

Each `TODO(decision)` in the source is an open choice:

- Smoke ids are `taskN_1` (first row of each family), not a stratified sample.
- Reset is a full H2 file restore plus HAPI restart. The `DELETE` + `$expunge` controller is unwired. `make smoke` still exits 0 when the 5 second gate is missed.
- Dosing bands are hardcoded from the single task5/task9 context paragraph. Codes (NDC, LOINC, SNOMED, BP flowsheet id) are read from that context.
- Task 1 names are split as "last token = family". The vendored file only has two-token names.
- Writes are valid FHIR R4. MedAgentBench's reference checker expects a string `route` and an object `note`, which HAPI rejects or ignores.
- Only task 3 issues `PUT` (update of the Observation it just created). The benchmark tasks themselves only `POST`.
- `funcs_v1.json` is vendored with the tasks. The mock agent does not interpret it; each family has a hand-written script.
- The harness always forwards writes. A clinician accept/reject check is not yet implemented.
- Saved smoke logs keep full POST/PUT bodies and only a summary of GET search bundles.
- Pass/fail does not run the Stanford reference solution or a chart diff.
