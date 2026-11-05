# ClinicalAgentBench-Exec

Execution harness for a clinical-agent benchmark. A scripted agent sends real FHIR `GET`, `POST`, and `PUT` calls to HAPI FHIR. The smoke path does not call a model API. `runners.baseline` calls the OpenAI chat completions API when you run it.

The tasks and the patient database come from [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) (MIT). See `NOTICE` and `vendor/medagentbench/LICENSE`.

## What this runs

- HAPI FHIR JPA R4 in Docker Compose, using the published MedAgentBench image and its H2 database (`/data/test_db.mv.db`, about 4.2GB).
- Reset by stopping HAPI, copying that database file back from `docker/snapshots/clean/`, starting HAPI, and logging the time until `GET /fhir/metadata` returns 200. The log is `runs/reset_timings.jsonl`. The target is under 5000 ms. The smoke table prints whether each reset met it.
- An execution harness that performs the agent's HTTP calls and records status, URL, request body, and response summary.
- A deterministic mock agent with one script per MedAgentBench v2 task family (`task1`–`task10`).
- `make smoke`: reset, run `task1_1` … `task10_1`, print a pass/fail table and the reset timings.

The fault proxy, idempotency wrapper, repair-loop grader, prefix replay, autonomy summary, and `DELETE` + `$expunge` reset are not yet implemented. They raise `NotImplementedError` and are not on the smoke path. Chart grading and the model baseline are below.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker Engine with the Compose plugin
- About 10GB of free disk for two copies of the H2 file (snapshot + live)
- Port 8080 free

## Exact local run

```bash
uv sync --extra dev

make fetch    # docker pull of the pinned image; verifies vendored task checksums
make load     # copy /data/test_db.mv.db out of the image into docker/live and docker/snapshots/clean
make up       # start HAPI and wait until /fhir/metadata is 200
make test     # uv run pytest, no Docker
make test-docker
make smoke    # uv run python -m smoke; 10 resets + 10 tasks
```

`uv sync` installs the dev group (pytest). `uv sync --extra dev` installs that extra as well. `make test` runs `uv run pytest -m "not docker"`. `make smoke` runs `uv run python -m smoke`.

`make reset` restores the snapshot once and prints one timing line. `make snapshot` stops HAPI and replaces the clean snapshot with the current live file. Take that only from a database that has not been written since the last reset.

Smoke task ids are listed in `tasks/smoke_ids.txt`. Override the FHIR base with `CAB_FHIR_BASE` (default `http://127.0.0.1:8080/fhir/`).

A smoke task passes when every FHIR call returns a 2xx status and the agent emits `FINISH`. `task1` rows also compare `FINISH` to the `sol` field in the vendored task file. Other families have no `sol` in that file, so smoke does not grade their answers. The baseline grades the chart instead.

## State-diff grading

`grade_trial_state` compares one patient's chart before and after a trial. The trial passes when the resources created, updated, and deleted match the expected fields. Server `meta`, narrative `text`, `language`, and `implicitRules` are ignored. Numbers match within 0.1. The finish string is not part of the grade.

Tasks 1, 2, 4, 6, and 7 expect no chart change. Task 3 records a blood-pressure Observation. Tasks 5 and 9 write orders only when the pre-trial lab is low. Task 8 writes one referral. Task 10 orders an HbA1c when the latest result is missing or older than one year. Medication and service payloads use FHIR R4 shapes: `route` is a CodeableConcept and `note` is a list.

`tests/fixtures/state_diffs/` holds 14 golden before/after charts covering those task types. `collect_patient_state` reads Patient, Observation, MedicationRequest, and ServiceRequest from HAPI for a live trial.

## Model baseline

```bash
uv run python -m runners.baseline --k 1 --output runs/baseline
uv run python -m runners.baseline --k 5 --output runs/baseline-k5
uv run python -m runners.baseline --k 5 --no-reset --output runs/baseline-keep
uv run python -m analysis.agreement runs/baseline/results.jsonl
```

The command reads `OPENAI_API_KEY` from the environment and exits with an error if it is missing. It does not run as part of `make test`. `--model` defaults to `gpt-6-luna`. `--tasks` defaults to `tasks/baseline_100.txt`, ten tasks from each of the ten families. Each trial restores the H2 snapshot before the agent runs. `--no-reset` leaves those writes in place so a later trial's pre-chart contains the earlier orders.

The first trial of each task writes `prefixes/<task_id>.json`: messages before the first `POST` or `PUT`, and that pending write. A task that never writes stores the full transcript and a null pending write.

Every trial is graded from the chart diff. Place the authors' `refsol.py` at `vendor/medagentbench/refsol.py`, or pass `--refsol`. The file should define `task1` … `task10(case_data, results, fhir_api_base)`, or a single `eval` function. `results.history` items have `role` (`agent` or `user`) and `content`. `results.result` is the `FINISH` payload. An accepted write is an agent turn `POST <url>` plus JSON, followed by a user turn that contains `POST request accepted`. If the file is absent the run prints `refsol skipped: <path> is not present` and still records the state grade. `analysis.agreement` prints the 2×2 table of transcript pass/fail against state pass/fail.

## Layout

```text
docker-compose.yml          HAPI service, bind-mounts ./docker/live at /data
scripts/                    fetch, load, snapshot, restore
vendor/medagentbench/       test_data_v2.json, funcs_v1.json, upstream LICENSE
src/harness/exec_harness.py real GET/POST/PUT
src/agent/mock_agent.py     scripted agent
src/fhir/snapshot_reset.py  calls the restore script
src/graders/state_diff_grader.py  chart diff and expected fields
src/graders/state_collector.py    FHIR snapshot of one patient
src/graders/refsol_grader.py      optional transcript grader
src/runners/baseline.py     model baseline (k trials, prefixes, resets)
src/analysis/agreement.py   transcript-versus-state table
src/smoke.py                10-task command
tasks/baseline_100.txt      100 task ids for the baseline
tests/fixtures/state_diffs/ 14 golden charts
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
- Smoke pass/fail is HTTP success plus `sol` when the task file has one. The baseline grades the chart diff, and the transcript grader only when `refsol.py` is present.
