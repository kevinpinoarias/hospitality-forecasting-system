# Changelog

Notable changes, newest first. Dates are when the work was merged or finished.

## Unreleased
- Series 19-22 added to `docs/EXPERIMENT_LOG.md`: column order (100 random orders plus a seed-luck control), the LSTM and Transformer on the 38 features (grid search, fresh-seed confirmation), a CatBoost depth-and-iterations grid (60 configurations), and the networks given the predicted day's own features. None improves on the final model. New scripts `run_series19_column_order.py`, `run_series20_neural_v2.py`, `run_series21_catboost_depth_iterations.py`, `run_series22_neural_target_day.py`, a shared resumable harness (`series19_21_common.py`) and a report builder (`report_series19_21.py`); the harness reproduces the final model's £909.57 exactly.
- `build_windows` / `prepare_datasets` in `src/models/neural_common.py` gained an opt-in `include_target_day` switch (off by default; earlier results unchanged).
- Added `src/experiments/run_error_autopsy.py`: an analysis of where the final model misses (the 20 biggest surges and lulls with calendar, holiday and weather context and a blank column for the venue manager, error by kind of day, whether the manager's forecast knew more, and a forward-chained test of weekday bias corrections). Findings: Friday to Sunday carry 55% of the error; Saturdays are usually under-forecast but the bias fades; the first week of a school break and the last week of term carry about 30-40% more error than their share of days.
- Activated `config/config.yaml`, which was previously unused: a strict, validated loader (`src/config.py`, Pydantic) now supplies the paths, split dates, weather thresholds and coordinates, XGBoost/SARIMAX/CatBoost settings, analysis defaults and each model's feature set. About 35 hard-coded path constants and a dozen duplicated settings across 25 files now read from it. The stale values in the old file were replaced with the ones the code actually used, so no result changes (the weather coordinates stay as they were).
- Feature sets: the original 15 features (V1) and all 38 (V2) are each defined once in `src/features/build_features.py` (previously copied into four and two files); `features.use` in the config picks the set per model.
- Removed `data/features/model_features.csv`: it was the V1 selection of `engineered_features.csv`, now `load_model_frame()`.
- Verified by running the whole pipeline before and after: all 585 reproducible output files are byte-identical (features, preprocessing, baselines, SARIMAX, XGBoost, the final model's backtest predictions and metrics, figures). `config/config.yaml` changes now retrain the model in CI and redeploy the API. New tests in `tests/test_config.py`.
- Added `kind=best` to `GET /errors/ranking` and a fourth ranking tool, `get_top_best_days`, so the assistant can list the days the forecast came closest (before, it could only list misses and said it had no tool for the best ones). A forecast that matches to the penny is reported as `exact`.

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
