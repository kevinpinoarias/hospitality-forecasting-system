# CI/CD: how a change reaches the live service

Everything here lives in `.github/workflows/ci.yml` plus a few small scripts.
The one-time Google Cloud setup is in [CICD_SETUP.md](CICD_SETUP.md).

## The idea in one paragraph

Every change goes through the same automatic route: checked on a clean
machine, the model rebuilt from the raw data, the service packed into a
container image and tried out, and - only on `main` - delivered to Cloud Run.
Nobody runs those steps by hand, and a change cannot reach `main` without
passing the checks.

## The route

| # | Job | What it does | Files |
|---|---|---|---|
| 1 | `lint` | `ruff check .` - unused imports, undefined names, risky comparisons | `ruff.toml` |
| 2 | `secrets` | gitleaks scans the whole git history for keys and passwords | - |
| 3 | `tests` | installs pinned libraries, rebuilds the model from `data/raw/` (or reuses a cached copy), runs `pytest`, then the quality gate | `rebuild_model.py`, `requirements-ci.txt`, `pytest.ini`, `scripts/quality_gate.py`, `config/quality_gate.json` |
| 4 | `image` | builds the API image from the rebuilt model, starts it, runs the smoke test; also builds the assistant image | `Dockerfile`, `Dockerfile.assistant`, `scripts/smoke_test.py` |
| 5 | `deploy` | `main` only: pushes images to Artifact Registry, starts the new revision with no traffic, tests it, then promotes it | `scripts/deploy_service.sh` |

Jobs 1-3 run in parallel; `image` waits for `tests`; `deploy` waits for all.

## Why it is built this way

- **The model is rebuilt in CI, not committed.** `models/` and the engineered
  features are gitignored. The rebuild takes about 85 seconds from the tracked
  raw data, so the pipeline proves on every change that the project can
  rebuild itself from a fresh clone. It is cached under a fingerprint of the
  raw data, the feature/model/evaluation/experiment code and the pinned
  library versions, so unchanged inputs skip the retrain.
- **The quality gate has a tolerance.** A rebuild is not bit-identical (the
  weather service revises old days; CatBoost differs slightly between
  machines). Measured drift was 0.005%; the gate allows 1% either way. Too
  good is a failure as well, because a sudden jump usually means leakage. To
  accept a deliberate change, update `expected` in `config/quality_gate.json`
  in the same pull request.
- **The smoke test uses the real container.** It calls `/health`, `/predict`
  (a date inside the backtest, so no weather download) and `/errors/ranking`.
- **Deploys cannot break the live service.** `deploy_service.sh` deploys with
  `--no-traffic`, tests the revision at its own tagged address and only then
  moves traffic. A failing test leaves the live revision untouched.
- **Only what changed is deployed.** The deploy job compares the last commit
  with the one before: model/data/API changes ship the API, `src/assistant/`
  changes ship the assistant, docs-only changes ship nothing. The Run
  workflow button on the Actions tab redeploys both.
- **No stored credentials.** GitHub proves to Google, per run, that it is this
  repository's `main` branch (Workload Identity Federation); Google lets it act
  as `github-deployer`, which can push to one registry and update two services.
- **`main` is protected.** Changes arrive through pull requests with the
  required checks green (Lint, Secret scan, Tests, Image).
- **Not in the per-push route:** the assistant's 18-case LLM evaluation
  (`python -m src.assistant.evaluate_output`). It needs a Gemini key and a
  live API, and LLM answers are non-deterministic, so it is run on demand.

## Operating it

- **Deploy:** merge a pull request into `main`, or press Run workflow.
- **Approve each deploy by hand:** Settings > Environments > `production` >
  Required reviewers.
- **Stop deploys at once:** delete the `GCP_WIF_PROVIDER` repository variable.
- **Roll back:** send traffic back to an earlier revision, in the Cloud Run
  console or with `gcloud run services update-traffic SERVICE --to-revisions
  REVISION=100`. The newest four revisions and their images are kept for this
  (see Housekeeping).
- **Run the same checks locally:** `ruff check .`, `pytest`, and
  `python rebuild_model.py` for the model.

## Housekeeping

Two things would otherwise only ever grow; both are cleaned automatically:

- **Registry images.** `config/registry_cleanup_policy.json` deletes image
  versions older than 14 days but always keeps the 12 newest of each image
  (about four deploys, since one push leaves three entries). It was set once
  (command in [CICD_SETUP.md](CICD_SETUP.md), step 9) and Google applies it
  by itself.
- **Cloud Run revisions.** They cost nothing, but they pile up.
  `scripts/deploy_service.sh` deletes all but the newest four revisions of a
  service after each successful deploy. Four revisions and four deploys' worth
  of images line up, so every revision that is kept can still start.

## Things that went wrong, and why they are worth knowing

Each of these only showed up on a clean machine, which is what CI is for.

- Folder names differed in case (`Sales` vs `sales`): fine on Windows, fatal on Linux.
- `pytest` on its own does not put the project folder on the import path.
- FastAPI's `TestClient` needs `httpx2`, which had arrived only by accident.
- A branch created from `origin/main` pushes to `main` by default; create
  branches without an upstream and push with `git push -u origin NAME`.

## Known limits

- No drift monitoring or scheduled retraining: the model is retrained only
  when its inputs change.
- There is no staging project; the deploy is tested by the candidate-revision
  step, not by a separate environment.
- The live API is public and unauthenticated (a deliberate demo choice).
