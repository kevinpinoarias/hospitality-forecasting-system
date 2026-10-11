# Changelog

Notable changes, newest first. Dates are when the work was merged or finished.

## 2026-10-11 - CI/CD
- Added GitHub Actions (`.github/workflows/ci.yml`): lint (ruff), secret scan
  (gitleaks), tests, a quality gate on a model rebuilt from the raw data,
  image build with a smoke test, and a deploy to Cloud Run on `main`. See
  `docs/CICD.md`.
- Added `rebuild_model.py` (about 85 s), `scripts/quality_gate.py`,
  `scripts/smoke_test.py`, `scripts/deploy_service.sh`, `ruff.toml`,
  `pytest.ini`, `requirements-ci.txt`, `config/quality_gate.json`,
  `config/registry_cleanup_policy.json`.
- Fixed a Linux-only bug: the splitter wrote `data/interim/sales` and `/labour`
  but the daily builders read `.../Sales` and `.../Labour`.
- Added `httpx2` to the dev requirements (FastAPI's `TestClient` needs it).
- Activated the pre-commit hook (`git config core.hooksPath scripts/git-hooks`)
  and enabled branch protection on `main`.
- Cleanup: deleted old Cloud Run revisions (API 12 to 4, assistant 14 to 4), switched on the Artifact Registry cleanup rule, and made the deploy script prune revisions after each deploy so both stay tidy automatically.
- Deployed the error-ranking endpoint and the assistant's three new tools to
  the live services, through the pipeline.

## 2026-10-10 - Error ranking, cleanup, notebook
- Added `GET /errors/ranking` and three assistant tools for the biggest
  misses (worst, over-forecast, under-forecast), for the model's or the
  manager's forecast.
- Assistant evaluation: fixed a false positive in the scenario check, made
  weather-dependent cases use dates relative to today, and told the assistant
  not to add a closing "if it rains" footnote when the weather is known.
- Removed unused code and imports, moved `xgboost` and `scikit-learn` to the
  dev requirements (API image 791 MB to about 297 MB), added shared
  `src/preprocessing/common.py`, fixed a NumPy deprecation in `build_features.py`.
- Rebuilt and re-executed `notebooks/portofolio_walkthrough.ipynb`.

## 2026-09-14 and earlier
- API (FastAPI), Docker images and Cloud Run deployment of the forecast API
  and the chat assistant; final model promoted to the 25-seed log1p CatBoost
  (Series 14-18). See `docs/EXPERIMENT_LOG.md`.
