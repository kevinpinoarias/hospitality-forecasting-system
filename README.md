# Hospitality Forecasting System

[![CI](https://github.com/kevinpinoarias/hospitality-forecasting-system/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinpinoarias/hospitality-forecasting-system/actions/workflows/ci.yml)

## Overview

This repository contains an end-to-end forecasting pipeline built around a real hospitality operations use case. The project was designed to investigate whether machine learning and time-series forecasting methods could improve on manual forecasting and provide more operationally useful demand predictions.

The workflow begins with anonymised rota-style operational exports, separates sales and labour records, aggregates them into daily datasets, engineers forecasting features, benchmarks manual forecasts and baseline models, and then trains both classical and machine learning forecasting models. The system compares manual forecasting, simple baselines, SARIMAX, XGBoost and neural models, then selects its final model - a CatBoost ensemble - through 18 series of rolling-origin backtesting experiments, while also linking forecast performance to operational considerations such as labour efficiency and overstaffing risk.

Beyond the modelling pipeline, the trained model is also served through a small API, containerised with Docker, and deployed to a live public endpoint — turning the notebook-style analysis into a running piece of software. Every change is checked, rebuilt and delivered by an automated CI/CD pipeline (see [CI/CD](#cicd-checks-and-automatic-deploys)).

This project is presented as a portfolio system rather than a production deployment. The public version uses anonymised local data and a modular project structure to demonstrate methodology, system design, and analytical reasoning without exposing proprietary business information.

### 💬 Try the live chat assistant

The easiest way to explore this project is to just ask it questions:

**[hospitality-forecasting-assistant-56220375160.us-central1.run.app](https://hospitality-forecasting-assistant-56220375160.us-central1.run.app)**

A conversational layer (LangChain + Gemini) over the deployed forecasting API and this project's own modelling results - ask it for a forecast, "what if it rains", how a past forecast compared to reality, which day/week/month is strongest, or how the different models tested here (XGBoost, an LSTM, a Transformer, SARIMAX, and baselines) compared. Every answer is grounded in a real tool call, never invented. See [Live chat assistant](#live-chat-assistant) below for more detail, and note the cold-start warning - this runs on Cloud Run's free tier and scales to zero when idle, so the first message can take 20-45 seconds to get a reply.

---

## Business Context

Hospitality forecasting is operationally important because inaccurate demand expectations can have direct effects on staffing, labour cost, service pressure, and decision-making. In this case, the forecasting problem was approached from two perspectives at once:

1. **Predictive accuracy**
   Improving the estimate of daily realised sales relative to the existing manual forecast.

2. **Business relevance**
   Understanding the implications of forecast error for labour planning, especially where overforecasting demand can contribute to unnecessary wage expenditure and overstaffing.

This dual perspective is central to the project. The goal was not only to train a model with lower error, but to understand whether forecasting improvements could be translated into more useful operational decisions.

---

## Project Objectives

The main objectives of the project were:

- build a clean, modular forecasting pipeline from anonymised raw operational files
- benchmark manual forecasts against realised daily sales
- evaluate simple statistical baselines before training more advanced models
- engineer domain-relevant features for hospitality demand forecasting
- compare a classical time-series benchmark (SARIMAX) against a machine learning model (XGBoost)
- examine forecast quality not only with MAE and RMSE, but also with MAPE, bias, and business-oriented labour metrics
- investigate model degradation on later unseen data and assess the implications for generalisation and retraining
- expose the trained model as a real, deployable API rather than a notebook artifact

---

## Repository Structure

```text
hospitality-forecasting-system/
├── README.md
├── main.py
├── rebuild_model.py              # rebuilds just the served model from data/raw/ (what CI runs, ~85 s)
├── CHANGELOG.md
├── ruff.toml                     # linter settings
├── pytest.ini                    # test settings
├── requirements.txt              # runtime deps for the pipeline + API
├── requirements-dev.txt          # notebook, plots, neural models, experiment sweep (LightGBM, Prophet)
├── constraints-experiments.txt   # exact library versions the experiment log was produced with
├── requirements-assistant.txt    # deps for the chat assistant only
├── requirements-mcp.txt          # deps for the MCP server only
├── requirements-ci.txt           # what the CI tests job installs
├── Dockerfile                    # forecast API image
├── Dockerfile.assistant          # chat assistant image
├── .env.example                  # GEMINI_API_KEY, FORECAST_API_BASE_URL
├── LICENSE
├── .github/workflows/ci.yml     # CI/CD: lint, secret scan, tests, quality gate, image, deploy
├── config/
│   ├── config.yaml               # paths, split dates, weather thresholds, model settings, which feature set each model uses
│   ├── quality_gate.json         # expected model error and tolerance for CI
│   └── registry_cleanup_policy.json  # Artifact Registry cleanup rule
├── data/
│   ├── raw/                      # tracked - anonymised source exports
│   ├── raw_private/              # real, unanonymised exports - gitignored, never committed
│   ├── interim/                  # generated by the pipeline, gitignored
│   ├── processed/                # generated by the pipeline, gitignored
│   └── features/                 # generated by the pipeline, gitignored
├── data_cache/                   # weather API cache, gitignored
├── docs/
│   ├── EXPERIMENT_LOG.md         # every experiment series: hypothesis, method, results, conclusion
│   ├── CICD.md                   # how a change reaches the live service, and why
│   └── CICD_SETUP.md             # one-time Google Cloud setup and housekeeping commands
├── models/                       # trained models, gitignored (see "Serving the Model")
├── notebooks/
│   └── portofolio_walkthrough.ipynb
├── reports/                      # generated metrics/figures, gitignored
│   ├── experiments/              # experiment series outputs
│   ├── figures/
│   └── results/
├── scripts/
│   ├── git-hooks/pre-commit      # blocks any commit touching data/raw_private/
│   ├── quality_gate.py           # CI: fails if the rebuilt model's error drifts
│   ├── smoke_test.py             # CI: checks a running API container works
│   └── deploy_service.sh         # CI: deploy a Cloud Run revision with no traffic, test it, then promote
├── tests/                        # pytest suite
└── src/
    ├── config.py                 # loads and validates config/config.yaml (strict: typos and bad values stop start-up)
    ├── ingestion/
    │   └── anonymize_raw.py      # strips venue-identifying fields before anything else runs
    ├── preprocessing/
    │   ├── common.py             # shared helpers for the two builders
    │   ├── split_sales_labour.py
    │   ├── build_daily_sales.py
    │   └── build_daily_labour.py
    ├── features/
    │   ├── build_features.py
    │   └── serving_features.py   # builds the final model's inputs for dates beyond the data
    ├── baselines/
    │   └── run_baselines.py
    ├── models/
    │   ├── evaluate_human_forecast.py
    │   ├── train_sarimax.py
    │   ├── train_xgboost.py
    │   ├── final_model.py        # the served model: 25-seed log1p CatBoost
    │   ├── train_final_model.py  # backtest, refit and save the served model
    │   ├── neural_common.py      # shared windowing/scaling for the neural models
    │   ├── train_lstm.py
    │   └── train_transformer.py
    ├── evaluation/
    │   ├── metrics.py
    │   ├── plots.py
    │   ├── error_analysis.py
    │   ├── experiment_tracking.py  # Weights & Biases logging
    │   ├── model_comparison.py     # single-split comparison across all models
    │   └── assistant_grounding.py  # computes the figures the chat assistant quotes
    ├── experiments/              # rolling-origin backtesting framework and experiment series
    │   ├── splits.py             # walk-forward fold generation
    │   ├── adapters.py           # one fit/predict interface per model family
    │   ├── grids.py, feature_sets.py
    │   ├── run_sweep.py          # Series 1
    │   ├── run_feature_sweep.py  # Series 2
    │   ├── recursive_horizon_test.py  # Series 3
    │   ├── run_series4_sweep.py … run_series18_serving_horizon.py
    │   ├── run_business_impact.py
    │   └── run_all_series.py     # regenerates every result in docs/EXPERIMENT_LOG.md
    ├── api/
    │   ├── main.py               # FastAPI app: /health, /predict, /errors/ranking
    │   ├── error_ranking.py      # ranks the days a forecast missed by the most
    │   ├── schemas.py            # Pydantic request/response models
    │   ├── history.py            # historical-data cache + lookups
    │   ├── weather.py            # live Open-Meteo weather
    │   └── inference.py          # feature construction + model inference
    ├── assistant/                # LangChain + Gemini chat assistant (see "Live chat assistant"); its tools in tools.py
    └── mcp_server/               # MCP server offering the same tools to Claude Desktop (see "MCP server")
```

## Data

The original project used operational exports derived from a private source. For portfolio purposes, the public repository uses anonymised local data files. The structure of the data has been preserved sufficiently to demonstrate the workflow and modelling logic while avoiding disclosure of identifiable business information.

The public dataset includes anonymised daily operational information such as:

- realised sales
- manual forecast sales
- realised labour hours
- realised labour wages
- forecast labour hours
- forecast labour wages

The raw files are processed through a staged workflow:

- raw rota-style files
- split sales/labour files
- daily aggregated sales and labour datasets
- engineered feature datasets
- final model-ready feature table

## Pipeline Summary

The pipeline is organised into several stages.

**1. Anonymisation**

Fresh raw exports land in `data/raw_private/` (gitignored, never committed) and are run through `src/ingestion/anonymize_raw.py` before anything else. This strips real business-identifying information - a `Site` column naming the venue, and a venue-specific brand-code prefix on `Department` values - using an allow-list of permitted output columns rather than a list of known-bad ones, so an unexpected new column in a future export is dropped by default rather than silently let through. Processing is incremental: a manifest tracks which source files (by content hash) have already been anonymised, so re-running after a new export lands only processes what's new. Only the anonymised output in `data/raw/` is ever tracked in git.

**2. Splitting sales and labour**

Weekly rota-style files (now read from `data/raw/`) are separated into sales and labour portions so they can be processed independently.

**3. Daily aggregation**

Sales files are aggregated into daily actual and forecast sales totals. Labour files are aggregated into daily actual and forecast labour hours and wages.

**4. Feature engineering**

A broad feature space is created, including time-based, cyclical, holiday, payday, school holiday, and weather-derived features.

**5. Benchmarking**

Before training more advanced models, manual forecasts and simple baselines are evaluated to establish a credible reference point.

**6. Model training**

SARIMAX and XGBoost are trained as representative classical and machine learning approaches. The final model - the one the API serves - is then backtested, refit on every available day and saved (`src/models/train_final_model.py`).

**7. Evaluation and error analysis**

Performance is assessed numerically and visually, and the project pays special attention to hard days, spikes, drift, and operational implications of forecast error.

## Feature Engineering

Feature engineering was one of the central parts of the project. The original XGBoost model used a selected subset of features, but the broader engineered feature space was intentionally wider - and the final model, chosen by the later experimentation programme, uses all of it.

### Original XGBoost features

The original XGBoost model used the following features:

- forecast_sales
- month_sin
- day_of_week
- day_of_year_cos
- day_of_year_sin
- day_of_year
- month_cos
- month
- day_of_week_sin
- is_bank_holiday
- days_to_bank_holiday
- days_since_payday
- is_payday_window_pm3
- is_long_weekend
- is_heavy_rain

These were chosen because they produced the best generalisation among the explored options and captured a mix of:

- manual operational signal
- cyclical seasonality
- holiday context
- payday behaviour
- limited weather sensitivity

### Feature lists: 15 and 38

The project uses two named feature lists, each defined **once**, in `src/features/build_features.py`:

- **`feature_list_15`** - the original 15 features above. Used by XGBoost, the neural models and the experiment log's baseline.
- **`feature_list_38`** - all 38: the 15 plus the 23 candidates the experimentation programme added, in a fixed order (column order changes CatBoost's seeded fits). Used by the final CatBoost model the API serves, and the base of every piece of the weekly forecast log.

Both are in use; neither replaced the other. Earlier documents (the experiment log up to Series 22) call them V1 and V2.

`config/config.yaml` says which list each model uses (`features.use`), so switching a model's features is one line there, not an edit in several files. The XGBoost and neural models read the 15 straight from `engineered_features.csv` (`load_model_frame`); the old separate `model_features.csv` no longer exists.

### Broader engineered feature set

The broader engineered feature space also included features such as:

- year
- is_weekend
- forecast_error
- lag_7_sales
- rolling_7_sales
- rolling_14_sales
- lag_1_fe
- lag_7_fe
- rolling_7_fe
- days_to_payday
- is_payday
- days_since_bank_holiday
- is_school_holiday
- is_christmas_break
- is_summer_break
- is_easter_break
- max_temp
- rain_mm
- sun_hours
- is_hot_for_scotland
- is_dry_day
- temp_anomaly_14d
- warm_streak_len

The original XGBoost left these out. Rolling-origin backtesting later showed that adding all of them, except `forecast_error` (which directly encodes the target), improves accuracy - so the final model uses those 23 plus the 15 above, 38 in total.

### Candidate future features

A number of additional features were identified as promising areas for future iteration. These include:

- expansion of lag structure, especially shorter-lag effects such as lag_1, lag_2, and lag_3
- more systematic lag families for both sales and forecast error
- richer rolling features across multiple windows
- explicit seasonal-period flags such as Christmas period, Easter period, summer period, and New Year period
- evaluating whether broader seasonal/holiday flags should replace or complement school-holiday proxies
- potential labour-derived features such as wage-to-sales ratio or labour intensity
- more event-driven or anomaly-aware features for capturing sudden spikes in demand

Several of these have since been tested - short lags and Christmas-period flags both made accuracy worse. See [Future Work](#future-work) for what was tested and what remains open.

## Baselines

A proper forecasting project should not jump directly to machine learning. For that reason, several simple baseline models were evaluated first.

The baseline suite includes:

- **Naive baseline** — predict tomorrow using yesterday's realised sales.
- **Seasonal naive baseline** — predict a given day using sales from the same weekday one week earlier.
- **Rolling 7-day mean** — predict using the average of the previous 7 days.
- **Rolling 14-day mean** — predict using the average of the previous 14 days.
- **Rolling 28-day mean** — predict using a wider historical smoothing window.
- **Weekday average baseline** — predict using the average historical sales for that weekday in the training data.

The weekday-average baseline is especially useful in hospitality because it checks whether a more advanced model is genuinely learning more than a simple "typical Monday / typical Friday" weekly pattern.

## Models

### Manual forecast benchmark

The repository first evaluates the existing human/manual forecast against realised daily sales and labour outcomes. This is an important part of the project because it makes the evaluation operationally meaningful.

### SARIMAX

SARIMAX was included as a classical time-series benchmark. It uses autoregressive and seasonal structure with a small exogenous feature set. In this project it serves as a transparent reference model rather than the final deployment choice.

### XGBoost

The first production model was an XGBoost regressor trained on the selected feature subset. The model combines calendar structure, cyclical signal, manual forecast input, holiday context, payday features, and weather information.

XGBoost was chosen because:

- it performs well on structured tabular data
- it can model nonlinear interactions
- it handles mixed feature families naturally
- it is often stronger than classical methods when the data generating process is operationally messy rather than purely statistical

### Neural forecasters and experiment tracking

Beyond SARIMAX and XGBoost, the project also trained an LSTM and a small Transformer (PyTorch) on the same feature set, for a fair comparison against a modern deep-learning approach - full results (including a multi-model leaderboard, spike-day accuracy, and business-impact comparison) are in the [portfolio walkthrough notebook](notebooks/portofolio_walkthrough.ipynb) below.

Every model trained in this project - the six baselines, SARIMAX, XGBoost, the LSTM, the Transformer, and the final CatBoost ensemble - is logged as a run to a public [Weights & Biases](https://wandb.ai/) project:

**[wandb.ai/kevin-pino-arias-three-body/hospitality-forecasting](https://wandb.ai/kevin-pino-arias-three-body/hospitality-forecasting)**

### Systematic experimentation: rolling-origin backtesting

The comparisons above each rest on a single train/validation split. Given the project's own finding that accuracy on one period does not transfer cleanly to another (see [Validation Strategy](#validation-strategy-and-generalisation)), a single split is a weak basis for choosing a model. The next stage replaced it with **10-fold rolling-origin backtesting**: train on history, test on the next 60 days, roll forward, repeat - on 2.5 years of data (January 2024 to September 2026).

On that footing, 18 hypothesis-driven experiment series tested algorithms and hyperparameters (adding CatBoost, LightGBM and Prophet), a 38-feature set, short-lag features, three different encodings of the December trading pattern, post-hoc December corrections, alternative loss functions, a log-transformed target, a CatBoost-Prophet ensemble, seed bagging, and how to serve the result beyond the end of the data. **Every series is recorded in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) with its hypothesis, method, results and conclusion - including the many that failed** - and every figure in it can be regenerated from committed code with `python -m src.experiments.run_all_series`.

**The resulting model**: CatBoost on 38 features, trained on log-transformed sales, averaged across 25 random seeds.

**Four later series tried to improve on it, and did not** ([Series 19-22](docs/EXPERIMENT_LOG.md#series-19--does-the-order-of-the-38-columns-matter), each scored under the model's own protocol and re-checked on fresh seeds): the order of the 38 columns moves the result no more than seed luck does (100 random orders, spread £1.95 against £1.78); shallower trees with more iterations, 60 configurations, never beat the current setting (£909.57); and the LSTM and Transformer - given the 38 features, a hyperparameter grid, and finally the predicted day's own features - stay at least £120 behind (best £1,029 against £907 on the same days).

**A review of the calendar columns found nothing to change** ([Series 44-51](docs/EXPERIMENT_LOG.md#series-44--do-the-month-and-day-of-year-columns-all-earn-their-place)): the month and day-of-year columns were tested as whole encodings rather than one at a time, with 100 fresh seeds, on the 56 days since the backtest ended, and inside the real Friday, Saturday and Sunday models. Day of year carries the seasonality (removing it costs £17-31 a day); the three month columns add nothing once it is there; separate Friday, Saturday and Sunday flags in place of `is_weekend` help a plain model by £11 but add nothing inside the real weekend models, which already see recent same-weekday sales; and dropping `year` is a wash for the full forecast (+£1.8) although it would matter if the model were retrained in the first weeks of a year. Redoing the whole weekend-feature selection with the flags in the base (Series 52) did not help either (+£2 to +£3). The 38 columns stay.

**Against the venue's manual forecast** (Series 17), on 569 matched days:

| Forecast | MAE | MAPE | vs. manual |
|---|---|---|---|
| Manual forecast (the venue's existing process) | £1,086 | 14.9% | — |
| Original portfolio XGBoost, scored the same way | £1,035* | 13.9% | 4.7% better |
| **Final model** | **£919** | **12.0%** | **15.4% better** |

<sub>\* With the pinned library versions. A newer XGBoost release gives £1,022 (5.9% better); the final model's figures don't change.</sub>

The 15% improvement holds over the wider 583-day window (15.35%), so it is not an artefact of the matched days. Scored under one protocol, the experimentation programme lifted the model's advantage over the manual process from roughly 5% to 15%.

Three findings worth highlighting:

- **Seed selection bias.** A log-transformed target first appeared to beat the previous best by £15.39 MAE. Re-running both configurations across 25 random seeds showed the seed used in every experiment until then (42, chosen purely by convention) was an unusually favourable draw that flattered one configuration three times as much as the other. The honest gain was **£3.35**, and that is the figure reported.
- **Explicit December features never helped.** Four separate encodings of the December trading pattern - a flag, a graded intensity index, named phases, and post-hoc multipliers - all made accuracy worse. The model's existing calendar features already capture the pattern, so adding it explicitly double-counts it.
- **The single split had mischaracterised a model.** SARIMAX looked like one of the worst models on the original split (MAE £3,081), because it was trained once and then asked to forecast more than a year ahead without retraining. Retrained every 60 days under backtesting, as it would be in practice, its MAE is £1,180. How a model is evaluated changed the conclusion about it.

**Business impact.** Rotas are planned from the sales forecast, so over-forecasting pays for staff that demand never needed. Staffing to the final model and correcting on site as usual is simulated to save **about £58k a year, 5.5% of the 2025 wage bill**. This counts wage savings only: the model over-forecasts less often but under-forecasts more, and the cost of understaffing (service, lost sales) is not modelled. It is a simulation over historical days, not a measured deployment outcome. See the [business impact entry](docs/EXPERIMENT_LOG.md#business-impact--what-forecast-error-costs-in-labour) for the method and every assumption.

**Deployment, and forecasting further ahead.** The live API and chat assistant serve this model. The 15% figure holds when recent sales are available - forecasting next week with data up to yesterday - because the model reads the last fortnight's sales. The public demo's data stops on 6 September 2026, so its forecasts run weeks or months beyond the data, with those inputs filled from historical weekday and seasonal patterns. Series 18 tested exactly that, out to 180 days: supplied with the manager's own forecast, the served model stayed **about 10% more accurate than the manual forecast** (MAE £978 vs £1,082) and ahead of the original XGBoost served the same way (£1,078); without it, it roughly matched the manual forecast (£1,078) and beat the original XGBoost (£1,132). See [Series 18](docs/EXPERIMENT_LOG.md#series-18--serving-the-final-model-beyond-the-end-of-the-data) for the breakdown by horizon.

## Evaluation Metrics

The project uses several metrics to assess forecasting quality.

**Standard regression metrics**
- MAE — mean absolute error
- RMSE — root mean squared error
- MAPE — mean absolute percentage error
- Bias — average signed forecast error

**Business-oriented labour metrics**

For manual forecast evaluation, the project also estimates operational implications, including:

- total overforecasted wages
- total underforecasted wages
- total overforecasted sales
- overstaffed days
- understaffed days
- average wasted wages on overstaffed days
- estimated wage waste per £1 of overforecasted sales

Because actual wages already reflect staff being sent home once a shift turns out quieter than planned, the metrics above are net of that on-the-day correction. The project therefore also reports **implied labour metrics** built only from planned figures - the wage and hours budget the forecast allocated per £1 of sales, applied to the sales over-forecast - which isolate the forecast's own planning error from how well it was corrected afterwards.

This is a key feature of the project because it connects forecasting accuracy to a practical business narrative rather than treating it as a purely technical exercise.

## Validation Strategy and Generalisation

A particularly important part of the project was the later validation against 2026 data.

Earlier validation on a historical split suggested solid performance from the final XGBoost model. However, when the validation period was extended into later unseen data, performance degraded rather than improving. This raised several important modelling questions:

- whether the model had been tuned too closely to the behaviour of the original training/validation window
- whether the operational system had shifted over time
- whether additional data from a different period was introducing patterns that did not generalise cleanly
- whether the model was relying on feature relationships that were stable in one period but weaker in another

Rather than hiding this degradation, the project treats it as one of its most important findings.

**Why this matters**

In practical forecasting, more historical data does not always improve performance. If the system changes, older periods can become partially misleading rather than helpful. This is especially true in hospitality, where demand can be affected by:

- local events
- changes in customer behaviour
- holidays
- weather
- operational changes
- macroeconomic sentiment
- unexplained spikes or regime shifts

**Portfolio significance**

This validation issue is not presented as a failure of the project. It is presented as evidence that:

- the forecasting problem is real and dynamic
- static models have limits
- feature design and retraining strategy matter
- production forecasting systems need monitoring and adaptation over time

In other words, one of the strongest aspects of the project is that it moved beyond "model accuracy on one split" and into a more realistic discussion of generalisation and drift.

**Follow-up.** This finding is what motivated replacing single-split evaluation with rolling-origin backtesting across the full 2.5 years of data, including January to September 2026 - see [Systematic experimentation](#systematic-experimentation-rolling-origin-backtesting). Scoring every model on repeated, independent forward-looking test windows gives a far sturdier basis for comparison than any one split, and it changed conclusions: SARIMAX, for one, turned out to have been badly mischaracterised by the single split.

## Error Analysis

One of the most important conclusions of the project is that the hardest days to forecast are where the greatest value lies.

The model generally captures recurring structure and weekly demand shape, but it struggles more on:

- spikes
- unusually strong peaks
- difficult transitional periods
- periods where demand appears to shift away from learned patterns

This led to a deliberate emphasis on error analysis.

The repository includes utilities to examine:

- top worst forecast days by absolute error
- largest overpredictions
- largest underpredictions
- grouped error summaries by weekday or other categorical variables

**Why spike days matter**

The most difficult spike days are likely where the current model misses important signal. They are also operationally the most relevant, because errors on unusually large days can have disproportionate effects on staffing and labour decisions.

This means future development should focus not only on reducing average error, but on understanding:

- what makes spike days different
- whether those spikes are linked to holidays, events, weather, or behavioural shifts
- whether feature design should explicitly target these hardest cases

## What the Project Demonstrates

This repository is intended to demonstrate more than just model fitting. It shows:

- ability to structure a multi-stage forecasting pipeline
- handling of messy operational data
- aggregation from raw business exports into modelling datasets
- feature engineering informed by real domain behaviour
- benchmarking discipline
- comparison against manual forecasts
- business-aware evaluation
- awareness of drift, limitations, and future modelling needs
- turning a trained model into a deployed, callable service (API, container, cloud deployment)

This is what makes the project stronger than a standard notebook-style ML exercise on a clean benchmark dataset.

---

## Serving the Model via API

The final model (25-seed CatBoost, see [Systematic experimentation](#systematic-experimentation-rolling-origin-backtesting)) is exposed through a small [FastAPI](https://fastapi.tiangolo.com/) service, containerised with Docker, and deployed to Google Cloud Run.

### Live demo

A live instance is deployed at:

```text
https://hospitality-forecasting-api-56220375160.us-central1.run.app
```

> **Note on the first request:** this service runs on Google Cloud Run's free tier, which scales down to zero instances when idle. If nobody has used it recently, the *first* request has to cold-start a fresh container before it can respond - this can take **20-45 seconds**. This is expected, not a failure; every request after that first one responds in well under a second. If a request seems to hang, give it a moment before assuming something's broken.

```bash
curl https://hospitality-forecasting-api-56220375160.us-central1.run.app/health

curl -X POST https://hospitality-forecasting-api-56220375160.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"requests": [{"date": "2026-09-20", "forecast_sales": 4500}, {"date": "2026-10-01"}]}'
```

### What the API does

- **`GET /health`** — reports whether the model has finished loading, which model it is and the last day it was trained on, and how many historical rows are cached.
- **`POST /predict`** — accepts a batch of `{date, forecast_sales?}` requests.
  - `date` is the only required field.
  - `forecast_sales` (the manual/operational forecast for that day) is optional. If omitted, a past date uses the manager's forecast on record, and a future date a weekday + seasonal-window historical average rather than a blank, since a real prediction still needs *some* value for it.
  - For a date beyond the historical data, the model's recent-sales inputs are filled from the same kind of historical average, and its weather inputs (temperature, rain, sunshine, and the previous fortnight's temperatures) from a live Open-Meteo forecast or observation where one exists, historical averages otherwise - with automatic retry and fallback if the live weather call fails. This is the approach [Series 18](docs/EXPERIMENT_LOG.md#series-18--serving-the-final-model-beyond-the-end-of-the-data) validated, built by the same code (`src/features/serving_features.py`), which reuses the training pipeline's own feature functions.
  - Each result includes a **dry-day scenario**, a **heavy-rain scenario**, and a resolved **best estimate**, plus metadata on where each input came from (`forecast_sales_source`, `rain_data_source`) and how far the requested date sits beyond the training data (`days_beyond_training_data`), so a caller can judge how much to trust it.
  - `prediction_source` says where the best estimate comes from. For a past date covered by the pipeline's backtest it is `made_before_the_day`: the out-of-sample prediction the model made *before* it saw that day, so comparing it with `actual_sales` is a fair test. A past date the served model was trained on is `model_had_seen_the_day`, and a future date `forecast`.
  - When a real weather outlook is available (not the historical-average fallback), the response also includes the actual `weather` figures used - rain, maximum temperature, sunshine, whether it counts as hot for Scotland, how it compares with the previous fortnight, and any warm spell - `null` otherwise, since there's no single real figure to report that far out.
  - `day_context` spells out, as plain facts, what the model took into account about the date: the last and next payday and how many days away they are, the previous and next Scottish bank holiday by name and whether it's a long weekend, any Glasgow school break, whether it's a Friday-to-Sunday weekend trading day, and - for past dates only - the real average daily sales over the week and fortnight before, plus how far actual sales ran from the manager's own forecast yesterday, the same day last week, and on average over the past week. It is built from the same feature row the prediction used (`src/api/day_context.py`).
  - `historical_comparison` returns the real realised sales for the same day the previous week and the same day the previous year, when those exact dates exist in the historical data - `null` for whichever isn't available, never estimated.
- **`GET /errors/ranking?kind=worst|over|under|best&forecast=model|manager&limit=10`** — the days a forecast missed by the most: the biggest misses in either direction (`worst`), the days it was too high (`over`) or too low (`under`) - or, with `best`, the days it came closest (smallest gaps in pounds; a quiet day can rank high simply because there was less to miss). `forecast` chooses whose forecast is judged, the model's or the manager's own manual estimate, and `limit` is capped at 20.
  - Only days that have a forecast made before the day was known (the backtest period) are ranked, for both forecasts, so they are judged on the same days; the response states the period it covers.
  - Christmas Day and New Year's Day (the venue is closed, so £0 sales is not a miss) and days with no recorded sales are left out, and the response counts what was excluded.
  - Each day gives the actual sales, the forecast, the gap as a positive amount plus a `direction` (`over_forecast` / `under_forecast`) and the gap as a percentage of actual sales.

### Running the API locally

```bash
pip install -r requirements.txt
python main.py            # generates data/features/engineered_features.csv and models/final_model/
uvicorn src.api.main:app --reload
```

### Running it in Docker

```bash
docker build -t hospitality-forecasting-api .
docker run -p 8000:8000 hospitality-forecasting-api
```

Note that `models/` is gitignored - the model artifact is regenerated rather than checked into version control, since the live deployment above is the intended way to try the service without training it yourself first. `python rebuild_model.py` builds just what the API needs in about 85 seconds; `python main.py` runs the whole pipeline.

### Live chat assistant

A conversational layer on top of everything above - built with [LangChain](https://www.langchain.com/) (agent + tool-calling, on LangGraph's execution engine) and [Google Gemini](https://ai.google.dev/), served through a [Gradio](https://www.gradio.app/) chat UI, containerised separately (`Dockerfile.assistant`) and deployed to its own Cloud Run service:

```text
https://hospitality-forecasting-assistant-56220375160.us-central1.run.app
```

> Same cold-start note as the API above: this also scales to zero when idle, so the first message can take 20-45 seconds.

> Messages are sent to Google's Gemini API to write the answers, so please don't type personal data - see [Data and privacy](#data-and-privacy).

It has eight tools, all grounded in real, already-computed data - it never invents a figure:

- **`get_forecast`** and **`get_forecast_range`** - the same `/predict` API above, in plain language, for one date or a period of up to 31 days: a forecast, what's special about the day (payday, bank and school holidays, weekend trading, the weather), "what if it rains", and how a forecast compares to what actually happened (it knows the venue is closed on Christmas Day and New Year's Day, so it won't present those as real forecasting misses).
- **`get_model_comparison`** - the real evaluation results: the final model against the manual forecast and the original XGBoost, how its accuracy holds up further ahead, the business-impact simulation, and the earlier comparison of every model tested (XGBoost, an LSTM, a Transformer, SARIMAX, and baselines), including spike days.
- **`get_top_worst_days`**, **`get_top_overpredictions`**, **`get_top_underpredictions`** and **`get_top_best_days`** - the days a forecast was furthest from what actually sold, too high by the most, too low by the most, or came closest, for either the model's forecast or the manager's own (from the `/errors/ranking` endpoint above, at most 20 days, closure days left out).
- **`get_sales_and_forecast_patterns`** - which single day/week/weekend had the highest sales or the best/worst forecast accuracy, and the average pattern by day-of-week and by month.

An automated evaluation suite (`src/assistant/evaluate_output.py`) checks every response against the real tool data behind it - numeric grounding, correct weekday naming, no leaked internal field names, honest "I don't know" answers rather than invented explanations, and correct handling of known closure days.

You can also run the CLI version yourself instead of using the live chat - it talks to the same live `/predict` API above, grounded the same way. Requires a free [Google AI Studio](https://aistudio.google.com/) API key in a local `.env` file (`GEMINI_API_KEY=...`, gitignored):

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m src.assistant.query_forecast "How is the forecast for next Saturday looking?"
```

### MCP server: the same tools for Claude Desktop

The assistant's eight tools are also offered through an [MCP](https://modelcontextprotocol.io/) server (`src/mcp_server/`), so an MCP client such as Claude Desktop can use them directly - for the maintainer and the finance team, with no new app. Every MCP tool calls the same function the Gemini assistant uses (`src/assistant/tools.py`) and returns its result unchanged; the numbers still come only from the forecasting API. Why it exists, the options considered and where it is *not* needed: [docs/MCP_SERVER_2026_11.md](docs/MCP_SERVER_2026_11.md).

**Install and run** (Python 3.11+, from the repository folder):

```bash
pip install -r requirements-mcp.txt
python -m src.mcp_server
```

Run by hand it just waits on stdin - normally the MCP client starts it. By default it uses the live API; set `FORECAST_API_BASE_URL` (e.g. `http://localhost:8000` with a local `uvicorn src.api.main:app`) to use another instance.

**Connect Claude Desktop.** Settings → Developer → Edit Config opens `claude_desktop_config.json`; add the server (Windows paths shown - use the full path to the Python that has the requirements installed, and to this repository):

```json
{
  "mcpServers": {
    "hospitality-forecasting": {
      "command": "C:\Users\<you>\AppData\Local\Python\bin\python.exe",
      "args": ["-m", "src.mcp_server"],
      "env": {
        "PYTHONPATH": "D:\path\to\hospitality-forecasting-system"
      }
    }
  }
}
```

Restart Claude Desktop; the eight tools appear under the tools menu. Try "What's the forecast for Christmas Eve?", "Give me the forecast for every day of the week before Christmas" or "Which ten days did the manager's own forecast miss by the most?". The first answer after a while can take up to a minute: the API on Cloud Run scales to zero when idle.

**The chat assistant through MCP (optional, local).** The Gemini assistant can load the same eight tools through this server instead of importing them: set `ASSISTANT_TOOLS=mcp` before `python -m src.assistant.app` (needs `pip install langchain-mcp-adapters`). The server starts once and one session is shared; a tool call costs about 3-13 ms more than a direct call. The deployed app does not use it. The evaluation runs either way: `ASSISTANT_TOOLS=mcp python -m src.assistant.evaluate_output`. Results and latency: design record, section 5.

**Security choices.**
- **Local only:** the server talks to its client over stdin/stdout (stdio). It opens no network port and is not deployed anywhere; CI never redeploys anything because of it.
- **Read-only:** all eight tools only read, and are marked so for the client. There is no tool that writes or changes anything. (Manager event flags - a propose-only tool with approval outside any AI's reach - are designed but parked; see the design record.)
- **Logs:** one line per tool call on stderr (tool name, duration, success or error); no arguments, results or question text. Claude Desktop keeps it in its MCP log.
- **Data:** answers in Claude Desktop are written by Anthropic's Claude, so forecast figures go to Anthropic rather than Google. The forecasting API itself has no sign-in (see [Limitations](#limitations)); the server reads only what that API already serves.

---

## CI/CD: checks and automatic deploys

Every push and pull request runs `.github/workflows/ci.yml` on a clean Linux machine, and `main` only accepts changes through a pull request whose checks pass:

| Job | What it does |
|---|---|
| Lint | `ruff check .` for unused imports, undefined names and similar mistakes |
| Secret scan | gitleaks over the whole git history |
| Tests | rebuilds the model from the tracked raw data (`rebuild_model.py`, ~85 s, cached when its inputs are unchanged), runs `pytest`, then a quality gate comparing the rebuilt error with `config/quality_gate.json` (tolerance 1%) |
| Image | builds the API image from the rebuilt model, starts it and smoke-tests `/health`, `/predict` and `/errors/ranking`; also builds the assistant image |
| Deploy | on `main` only: pushes the images to Artifact Registry and deploys to Cloud Run. The new revision gets no live traffic until it passes a test at its own address, so a bad build never reaches users |

Login to Google uses short-lived tokens (Workload Identity Federation), not a stored key, and only this repository's `main` branch can use it. Only the service whose inputs changed is redeployed. The assistant's LLM evaluation is not part of the per-push route (it needs a Gemini key and a live API) and is run on demand. See [docs/CICD.md](docs/CICD.md) for the reasoning and how to operate it, [docs/CICD_SETUP.md](docs/CICD_SETUP.md) for the one-time Google setup, and [CHANGELOG.md](CHANGELOG.md) for what changed when.

---

## Configuration

Settings that used to be repeated across the code now live in [`config/config.yaml`](config/config.yaml), loaded and validated by `src/config.py`: file paths, the train/validation split dates, the weather thresholds and coordinates, the XGBoost, SARIMAX and CatBoost settings, and which feature set each model uses. Validation is strict - a misspelt key, a wrong type, a bad date, an unknown feature set or an absolute path stops the program at start-up with a message saying what is wrong, instead of being quietly ignored.

Changing a model or feature setting changes the model's results, so re-run the pipeline afterwards. CI does it for you: `config/config.yaml` is part of the fingerprint that decides whether the model is retrained, and a change to it redeploys the API. The coordinates are the ones every figure in the experiment log was produced with; changing them changes the weather features and so every result.

---

## Data and privacy

This is a portfolio project, not a regulated service. It is built so that no personal data is needed, and this section says what is and isn't collected.

**The dataset.** The tracked files in `data/raw/` hold daily sales, labour hours and labour wages, by department (front of house, kitchen, KPs/cleaners) and stream, with manual forecasts and the venue's own predictions alongside. There are no names, employee IDs, contact details or per-person rows; wages are department totals. `src/ingestion/anonymize_raw.py` strips the venue-identifying fields before anything else runs. Real, unanonymised exports go in `data/raw_private/`, which is gitignored and guarded by a pre-commit hook so it cannot be committed by accident.

**The live API and chat assistant.**

- There are no accounts and no sign-in. The application does not store questions or answers; the chat's memory lives in the running process, keyed to one browser session.
- Messages typed into the assistant are sent to Google's Gemini API to write the answers. Please do not type personal data into the chat.
- The hosting platform (Google Cloud Run) keeps its standard request logs, which can include IP addresses and are under Google Cloud's own retention settings, not the application's.
- The chat page loads its fonts from Google Fonts, so Google can see a visitor's IP address when the page opens, as with any web font.
- Credentials are never committed: the Gemini key lives in Google Secret Manager, `.env` is gitignored and excluded from the container images, and deployment uses short-lived tokens, not a stored cloud key.

**Experiment tracking.** The public Weights & Biases project holds model metrics and plots only.

If you think the repository contains personal data that should not be there, please open an issue and it will be removed.

---

## Limitations

This portfolio version has several important limitations, which are acknowledged deliberately.

**1. Public data constraints**

The public repository uses anonymised local data and does not reproduce the original private ingestion setup.

**2. Diminishing returns from feature engineering**

The deployed model uses 38 features. The experimentation programme found that adding further features on top of them - short lags, or explicit December signals - made accuracy worse, suggesting diminishing returns from further engineering of the same inputs. Genuinely new information (see 3) looks like the more promising route; Easter, summer and New Year period flags remain untested.

**3. Limited external event signals**

The current model includes weather, bank holidays, payday effects, and school-holiday proxies, but it does not yet include richer event calendars or explicit operational anomalies.

**4. Drift on later data**

Validation on newer unseen periods showed degradation, indicating that a one-off model is not enough for a dynamic forecasting environment. Rolling-origin backtesting now measures performance across many periods rather than one, but the deployed model is only retrained when its inputs change (the CI pipeline rebuilds it from the raw data on every change to the data or the modelling code); there is no scheduled retraining or drift monitoring.

**5. Frozen data behind the live demo**

The final model is most accurate with recent sales to read. The public deployment's data ends on 6 September 2026, so every live forecast runs beyond it, where Series 18 measured an MAE of about £978 with the manager's forecast supplied and £1,078 without - against £933 with every real input. Beyond 180 days ahead, accuracy is untested.

**6. Spike-day handling**

The current system still struggles most on the hardest and most volatile days, which are likely where the biggest future gains can be found. December is the clearest case: four different ways of telling the model about the December trading pattern all failed to improve its forecasts.

**7. Two Decembers of history**

Several December-specific ideas could not be properly validated because only two Decembers exist in the data, and each behaved differently (see Series 12 in the experiment log). A third December is needed before a December correction can be tested honestly.

**8. The business-impact figure is a simulation**

The ~£58k annual saving is simulated over historical days, counts wage savings only, and does not cost understaffing on days the model under-forecasts. It is an estimate of potential, not a measured outcome.

**9. No authentication on the deployed API**

The live Cloud Run deployment is public and unauthenticated - anyone with the URL can call `/predict`. This is a deliberate simplification for a portfolio demo rather than a production posture; a real deployment handling genuine business data would add an API key or similar access control before going live.

## Future Work

Several future directions emerged from the project. Items marked **done** or **tested** were addressed by the experimentation programme; details are in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

**Feature expansion**
- ~~test shorter-lag features such as lag_1~~ - **tested** (Series 4): lag_1-3 and 28-day rolling features all made accuracy worse
- ~~re-examine the calendar and cycle columns (month, day of week, day of year, year, is_weekend)~~ - **tested** (Series 44-51): all 38 columns stay; day of month, week of month, week of year and month start/end flags were tested next (Series 54-56) and add nothing
- ~~expand the lag family more systematically~~ - **tested** (Series 2, 4)
- ~~revisit forecast-error lag features~~ - **tested** (Series 2, 4)
- add richer rolling statistics and trend-sensitive features: rolling median, standard deviation, maximum and minimum, and ratio-to-rolling-mean (untested; Series 4 suggests small or negative returns; the manager's forecast against the mean of the sales 14, 21 and 28 days earlier was tested in Series 66 and 69: about £4.4 a day better and reproducible on fresh seeds, but the gain sits in 2025, the intervals include zero and the 56 new days are worse; Series 74 tested the ratio in the Monday to Thursday and Friday pieces only on a third seed set: about £7 to £8 a day better, replicated, but about zero in 2026 and the post model's 56 new days +£2.87, so not supported by the rule set beforehand; it is on the watch list, low priority, with the reason)

**Target decomposition**
- forecast Food and Beverage sales separately and add them up (untested). The raw Rotaready files carry hourly sales and the manager's forecast split into Food and Beverage streams, but the model has only ever seen the daily total. A descriptive look (2026-11-07) compared the 26 Saturdays after a Friday payday with the other Saturdays within five weeks: Food +£388 (higher on 22 of 26, 2.2 standard errors), Beverage +£147 (17 of 26, 1.1), total +£534 (18 of 26), and the drink share and time-of-day shares do not move, so payday seems to lift food and a food-only model might see the payday signal more cleanly. The test would also show whether splitting the target helps at all. About two series of fits; the product would have to serve and maintain two models

**To do next (in order)**
- forecast-log version v1.11 (the hydro model plus the triangle payday intensity in the Saturday pieces only, same training data as v1): fitted and registered on 7 November 2026 (the pre-registration amended first), run beside v1 from the Monday 9 November run as `hydro_v1.11` and scored as Q4 (Saturdays decide; descriptive until 52 Saturdays); first payday Saturday to watch: 28 November 2026
- the triangle payday intensity: live in forecast-log version v1.11 (see above); the signed distance was tested on top of it (Series 75) and adds nothing, so it is closed
- the payday-Saturday correction (Series 65, the split and the stacked versions): parked as a forecasting-method question, not a feature; revisit whenever we discuss how Saturdays are forecast or how the model forecasts overall, and record both versions and v1.11 passively on each Saturday after a Friday payday from 28 November 2026 (Friday 27th pays under the 28th rule), deciding after about 8 to 10 of them if one wins about 70% of the time
- forecast-log versions v1.13 (the year-end specialist models) and v1.14 (the year-ago sales input in Monday to Saturday) were registered on 14 November 2026 (the amendment, the fits and the back-run from 2 November are in the log; details in `docs/FORECAST_LOG_V1_13_V1_14_REGISTRATION.md`); the first live run with them is Monday 16 November, and v1.13's windows (26 December to 8 January) are the first test of the year-end fix
- forecast-log versions v1.131 and v1.141 (v1.13 and v1.14 served with the hybrid stand-in S6 for the manager's number, no refit) were registered on 14 November 2026 (the amendment, the registrations and the back-run from 2 November are in the log); the hybrid is worth about 54 pounds a day alone in the backtest (946 to 892, one December and one January), and Q9 and Q10 read it from the Christmas week. See `docs/FORECAST_LOG_V1_131_V1_141_REGISTRATION.md`.
- December: the closed-day fix and the holiday rule are defined as forecast-log version v1.12 (Series 76-78: the fix is the one thing that helped, an early pay date adds nothing); CLOSED by Kevin on 14 November 2026 ("consider payday family closed"): December pay brought forward and a December correction, so the payday family is closed; still open (watch list): the model cannot learn Boxing Day from two examples, and a Saturday Boxing Day (26 December 2026) and Christmas Day (25 December 2027) are untestable
- IMPORTANT (Series 80, 8 November 2026): holiday-aware recent-sales inputs. Switching off the lag and rolling inputs for 1 to 15 January improved the package by 6.6 pounds a day over the 585 days, beating all ten placebo windows in other months, because the January rows teach the model a wrong lesson that spreads to every month. Series 81 (25 seeds) ruled out the zero-sales closed days as the cause (treating them as missing or filling them makes the package worse; the mask itself keeps its gain), so the lesson is in the post-Christmas transition rows. Series 82 (25 seeds) found where it sits: the first week, the best window is 1 to 8 January (-9.07 a day over the 585 days, the largest step after the weekend pieces), 1 to 15 January gives -6.1, the mask turns into a loss from 22 January, and starting on 26 December adds nothing. Series 83 repeated the two best windows on 175 more seeds and both replicate (1 to 8 January -8.35 on the 100 seeds and -10.09 on fresh seeds; 26 December to 8 January -8.57 and -7.33), 1 to 8 January slightly ahead. Still to do: the remaining ideas on the working list (a repair that keeps the information), leaving the Christmas-break days out of the source series before the averages are taken (a holiday-aware level), then the same for Easter and the summer break; any use needs a refit, a new forecast-log version and Kevin's decision
- IMPORTANT, a final joint feature check across all the pieces, starting from the original one-model recipe (Kevin, 8 November 2026; 14 November: to be run later, kept on this list; the cheaper anchor test of Series 91 was discarded): the pieces were built one at a time and each new feature was tested in one piece, so nobody has asked which features each piece should have in the final system. The staircase (saved predictions, 585 backtest days) shows the weekend pieces are two thirds of the gain over the original model (average daily miss 912 to 873) and the later steps (weekend correction, hydro flag, payday intensity, closed-day fix) about 19 pounds together. It would entail (a) one model with the original 38 features plus everything found since, against the pieces package (cheap; expected worse on weekdays, as Series 26 found for the same-weekday lags, but untested), (b) a per-piece drop-one audit with placebo columns (about 5 to 6 hours at the full protocol), and (c) the closed-day and January-penalty columns in the weekday pieces (a January penalty in the Saturday pieces was tested in Series 79 and does nothing); judge it on about 120 unseen days (about the end of February 2027) and after the 11th-fold decision, since every choice so far used the same 585 days
- forecast Food and Beverage separately (see Target decomposition above)
- decide the 11th-fold question (below, under Post model) before running anything new

**Watch list (waits for a date)**
- recheck the `year` column before any retrain in January to March 2027 (see Adaptive modelling below)
- low priority, inconclusive: the 3-week forecast ratio (Series 66, 69 and 74). Why it sits here: the gain is in 2025 and not in 2026 (the Monday-to-Friday-only version is about £7 to £8 a day better over the 585 days and replicated on three seed sets, but about zero in 2026), and on the 56 newest days it turned against it (+£2.9 post, 0.0 hydro, Fridays +£16; intervals wide). The ratio's link to the model's error is +0.12 in 2025 and about 0 in 2026, while the model's own error on those days fell from £836 to £729: the model may simply have learned what the ratio told it, or the 2025 gain was partly chance. Roadmap: (B, optional, any time) a static split, training once on 2024 only and scoring 2025 and 2026 with and without the ratio in the Monday-to-Friday pieces, to see whether the gain exists only while the model is young; (C) once about 120 unseen days exist after 1 November 2026, rerun Series 74's design on those days only with the rule written beforehand, and also check whether the ratio's link to the model's error came back; if C passes it is a candidate for a later forecast-log version beside v1.11, if not the idea is closed
- low priority: v1.11's gain sits mostly in July and December (61% of it; two Julys and one December in the backtest); read the log's Q4 for July and December Saturdays apart from the other Saturdays once December 2026 and July 2027 have an actual
- medium priority: the Saturday Boxing Day penalty (v1.12's Saturday pieces) and the holiday rule's fixed output (0 on 25 December and 1 January, 3,620 on Boxing Day); Boxing Day 2026 is a Saturday and no Saturday Boxing Day is in the data (Christmas Day 2027 is a Saturday too); read the log's Q5 and Q6 once the actuals for 25 December 2026 to 1 January 2027 are in, then decide the level and whether the model needs more than a penalty The hybrid stand-in of v1.131 and v1.141 meets the same problem: for Saturday 26 December 2026 and Saturday 2 January 2027 it takes the number of the same DATE a year earlier, which was a Friday (4,700 for Boxing Day 2025), so read Q9 and Q10 on those two days too.
- low priority (Kevin, 14 November 2026): untested forms of v1.14's year-ago input (the year-ago value scaled by how much the hotel has grown since) and whether the year-end specialist pieces should carry `sales_364` too (a new version); wait to see how v1.141 and v1.14 do in the December and January actuals before testing either

**CatBoost settings not yet tuned**
- iterations, learning rate and depth have been searched on the final model (Series 1, 8, 14 and 21, 60 combinations in the last: nothing beats 300 / 0.05 / 4, and depths 2-6 are not better), and so have the loss function (Series 13) and the log target (Series 14); these others have never been searched and sit at CatBoost's defaults: `l2_leaf_reg` (regularisation strength, default 3), `rsm` (the share of columns each split may look at, default all), `min_data_in_leaf` (smallest leaf, only with the lossguide and depthwise grow policies), `grow_policy` (symmetric trees by default; depthwise and lossguide are alternatives), `border_count` (how finely a numeric column is cut, default 254), `bagging_temperature` / `subsample` (row sampling), `random_strength` (default 1) and `one_hot_max_size`. The weekly log's Friday, Saturday and Sunday pieces also reuse the final model's settings without a search of their own. Any search needs the 25-seed protocol, a fresh-seed confirmation and the net-effect reading; the Series 21 plateau suggests small gains at best

**Holiday and seasonality redesign**
- ~~build explicit Christmas-period features~~ - **tested** (Series 5-12): a flag, a graded intensity index, named phases and post-hoc corrections all failed to help
- build explicit Easter-period, summer-period, and New-Year-period features
- ~~give the model a smooth seasonal baseline (weekday-specific sine and cosine waves) and let CatBoost learn the leftovers~~ - **tested** (Series 57): £99 to £246 a day worse in every variant, and removing the seasonal columns from CatBoost costs more, so the trees represent the season better than the baseline does; the rescue (a cross-fitted baseline, so the leftovers CatBoost sees are as large as on new days) was tested in Series 58 and changes nothing (−£10 to +£39 against the in-sample version, all intervals include zero), so the idea is closed
- compare these against school-holiday flags
- evaluate whether broad seasonal markers are more robust than council-specific school break proxies
- ~~re-test a December correction once a third December of data exists~~ - **closed** by Kevin on 14 November 2026 with the rest of the payday family (the payday-Saturday correction stays parked as a forecasting-method question)
- ~~revisit the payday feature~~ - **tested** (Series 59-73): the two known flaws (`days_since_payday` counts from the previous month's payday, so it reads 28-33 on payday itself and about 30 rather than 1-2 on the days between an early payday and the month's end; `days_to_payday` is negative on those days) were repaired and the repair is £3 to £5 a day worse than the deployed columns; dropping the four payday columns is £5 to £8 worse; explicit flags for the weekend after an early payday, a mix of the last working day, 25th and 28th, symmetric windows (1, 2 and 5 days), a graded signed distance, the 25th, 28th and 15th as full column sets, and a payday-Saturday correction instead of a feature (split and stacked) all fail to beat the deployed columns. The deployed `days_since_payday` is the day of the month plus 0 to 3 days, but plain day of month is £3 worse (Series 62), so why the deployed columns work is not explained. Also tested without helping: a pay-cycle phase (Series 67), working days to payday (Series 68) and a payday intensity summed over the pay dates (Series 70; the triangle shape is £3.3 to £3.8 better in the backtest with intervals that include zero and better on the new days, so it joins signed distance on the to-do list); payday near a long weekend and December pay were checked without model fits (noise; the two Decembers disagree). What is still to do, and what waits for a date, is under Future Work (To do next, Watch list)
- extend the school-holiday table back to January 2024: the 2024 Easter and summer breaks are missing, so the model learns those breaks from 2025-26 only (a fix changes every backtest figure, so it means re-running the experiment log)

**Event features (to look at when we reach the event features in the feature recheck; not part of the calendar and cycle review)**
- pool the event days: the post model's remaining error sits on named events with one or two occurrences each, too few for a tree to learn one by one (found by the calendar error screen, `run_error_calendar_screen.py`: Mother's Day +27% under-forecast in 2025 and +6% in 2026; Valentine's Day +34% on a Friday in 2025 and +3% on a Saturday in 2026; Father's Day +7% and +6%; Halloween −23% and Bonfire Night −13%, one occurrence each; the start of the school summer break on 28 June 2025 and 28 and 30 June 2026; the month-end excess is these events). Try a single "major event day" flag (and a shared "days to the nearest event" column) so all events share the data and one learned size, instead of one flag per event. Structured event data (the Hydro's show calendar was the one clear event signal found) is worth more than another calendar column. Each event repeats once a year, so more years of data help directly

**Post model (the weekend package: Series 23-27 and the Saturday/Sunday correction)**
- ~~calendar candidates (day of month, week of month, week of year, month start and end flags)~~ - **all tested** (Series 54-56), none helps. ~~week of year and month start/end flags~~ - **tested** (Series 56, same pieces, four encodings): no variant moves either model by more than £2.5, every interval includes zero, and all are worse on the 56 new days. ~~day of month~~ - **tested** (Series 54, in the post and hydro models' pieces, four encodings): no gain, slightly worse the more columns are added. ~~week of month~~ - **tested** (Series 55, same pieces, four encodings): the plain 1-5 version is a hair better (-£0.2 post model, -£1.4 hydro model, 7 of 10 folds) but inside the noise, and the extra columns hurt. A screen of the post model's errors (`run_error_calendar_screen.py`) found no pattern by day of month or week of month, and the month-end excess is mostly event days (the start of the school summer break, Mother's Day). The remaining error sits on event days; that idea has moved out of the calendar review to the Event features group above
- ~~rerun the weekend-feature selection (Series 26-27) with the three single-day flags in the base instead of `is_weekend`~~ - **tested** (Series 52): the flags change which options win (Friday: none, Sunday: A_lags) but the post model is no better (+£2 to +£3, interval about ±£13); the flags help Monday to Saturday by £1 to £9 each and cost Sunday £36
- ~~test the single-day flags in the plain log model as the weekly log serves Fridays~~ - **tested** (Series 53): for the next-week Fridays the log issues, the Friday piece cannot run (the sales 7 days earlier are not yet known) and the plain log model forecasts them; in a replay of the weekly log the flags' Friday gain shrinks from £27-36 to £2-11 (intervals about ±£25) and Monday to Friday gains £4-6 a day (intervals touch zero), so not worth a new model version
- before further backtest series, decide the backtest protocol: once data beyond about 6 November 2026 is ingested there will be 60 days beyond the 10 folds that end on 6 September, so the backtest gains an 11th fold and every 585-day figure changes. Series 44-52 stop with a message when that happens; pin the first 10 folds to compare with earlier results, or move all of them to the new protocol

**Hard-day forecasting**
- test a two-stage model: first classify whether a day is a normal or spike day, then predict sales separately for each (untested; the large misses so far look event-driven, so it may need event data to work)
- isolate and investigate spike days
- classify worst forecast days by type
- inspect whether the worst errors cluster around identifiable contexts
- design features specifically for difficult peak periods

**Neural networks**
- explore the LSTM and Transformer further: they have only been tried on the original 15 features, and on the 38 in Series 20 (worse by about £200-250), so they sit well behind CatBoost (Series 20 and 22: best network about £122 worse on the common days, even when it sees the target day's own features)
- untested: other architectures, features designed for the networks rather than inherited from the tree models, a network on the weekend-package inputs or the log-pound target, blending a network with CatBoost on more than the one leave-one-fold-out check already done (no gain)

**Adaptive modelling**
- ~~test shorter rolling training windows~~ - **tested** (Series 1): no universal winner between a sliding one-year window and all history
- ~~compare models trained on recent-only vs longer historical periods~~ - **tested** (Series 1)
- monitor degradation over time
- evaluate retraining frequency
- **the `year` column at a year rollover:** `year` can only be cut into the years the model has seen, so a model not retrained treats 2027 as 2026 (harmless), but a model retrained in the first weeks of a year learns from a handful of low-season rows of the new year, and `year` can then act as a "quiet day" marker. Series 46 saw this in the backtest fold of January to March 2026 (the plain log model's Saturday error about £1,200 a day larger than without `year`); inside the real pieces it was small (Series 50: the full forecast about £2 a day better without `year` in that fold, about £2 a day worse overall). Before any retrain in January to March 2027, rerun `python -m src.experiments.run_series50_drop_year_in_pieces` on the then-current data (its protocol assertions will need revisiting once the 11th backtest fold exists) and decide whether to drop `year` or replace it with something a tree can extend. Retraining itself stays the owner's decision

**Operational features**
- engineer labour-intensity or wage-efficiency features (note: a labour forecast does not exist at prediction time, since the rota is planned from the sales forecast, so these cannot be model inputs)
- model covers as an alternative or complementary target
- explore whether staffing metrics can be forecast more directly

**Production-style improvements**
- ~~centralise config usage~~ - **done**: `config/config.yaml` is the single, validated source for paths, splits, weather settings, model settings and feature-set choice (see [Configuration](#configuration))
- ~~add experiment tracking~~ - **done**: Weights & Biases, plus the experiment log
- add tests for each stage of the pipeline - **partly done**: the pytest suite covers anonymisation, the experimentation framework, neural data preparation, the labour metrics, the newer feature functions, the final model, the serving features, the API, the error ranking and the assistant's tools and evaluator (tests that need a trained model skip on a fresh clone, and run in CI against a rebuilt model); preprocessing and the core feature functions are not yet covered
- ~~deploy the model from the experimentation programme to the live API~~ - **done** (Series 18 tested how to serve it)
- ~~automate checks and deployment~~ - **done**: GitHub Actions (see [CI/CD](#cicd-checks-and-automatic-deploys))
- add authentication to the deployed API
- scheduled retraining and drift monitoring
- package the system more cleanly for reuse across venues or clients

## How to Run the Project

Clone the repository and install dependencies first.

```bash
git clone <your-repo-url>
cd hospitality-forecasting-system
pip install -r requirements.txt
```

`requirements.txt` covers everything needed to run the pipeline and the API. If you also want to run the notebook or regenerate evaluation plots (matplotlib) and the SARIMAX benchmark (statsmodels), also install the dev extras:

```bash
pip install -r requirements-dev.txt
```

Install the pre-commit safety hook once, so a raw export can never be accidentally committed (see [Anonymisation](#pipeline-summary)):

```bash
git config core.hooksPath scripts/git-hooks
```

Every push and pull request is also checked on GitHub - see [CI/CD](#cicd-checks-and-automatic-deploys). You can run the same checks locally with `ruff check .` and `pytest`, and rebuild just the served model with `python rebuild_model.py`.

The anonymised dataset is already tracked in `data/raw/`, so on a fresh clone you can run the full pipeline straight away. New raw exports, if you have them, go in `data/raw_private/` (gitignored - never committed) first:

```bash
python main.py
```

This will:

- anonymise any not-yet-processed files in `data/raw_private/` into `data/raw/` (skipped when there are none)
- split raw files into sales and labour outputs
- build daily sales and labour datasets
- engineer forecasting features
- run baseline benchmarks
- evaluate the manual forecast
- train SARIMAX
- train XGBoost, saving the model to `models/xgboost_model.json`
- train the final model: re-run its rolling-origin backtest, then refit it on every day and save it, with its out-of-sample backtest predictions, to `models/final_model/` - the model the API serves
- save metrics, predictions, and figures into the `reports/` directory

See [Serving the Model via API](#serving-the-model-via-api) above for running the API itself, either directly or via Docker.

### Reproducing the experiment log

Install with the pinned library versions the log was produced with - gradient-boosting libraries can shift results between releases:

```bash
pip install -r requirements.txt -r requirements-dev.txt -c constraints-experiments.txt
```

Then, with `python main.py` already run, every result in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) can be regenerated in dependency order:

```bash
python -m src.experiments.run_all_series                 # Series 5-18 and business impact
python -m src.experiments.run_all_series --only 16 17    # a subset
python -m src.experiments.run_all_series --include-1-4   # also Series 1-4, which take much longer
```

Outputs are written to `reports/experiments/`. Each series can also be run on its own, e.g. `python -m src.experiments.run_series16_seed_bagging`.

### Running the tests

```bash
ruff check .
pytest
```

Some tests need a trained model and skip themselves on a fresh clone; run `python rebuild_model.py` first to include them (CI does the same).

## Portfolio Walkthrough Notebook

The repository also includes a notebook:

```text
notebooks/portofolio_walkthrough.ipynb
```

This notebook is intended as a guided overview of the project rather than the main execution mechanism. It walks through the business problem, processed data, feature engineering, baseline comparisons, model results, and key conclusions in a format suitable for portfolio review or interview discussion.

## Key Takeaways

This project shows that forecasting in hospitality is not simply a matter of fitting a model once and reporting an error score. It requires:

- careful preprocessing of operational data
- meaningful benchmark comparisons
- domain-aware feature engineering
- attention to drift and changing conditions
- evaluation that survives scrutiny: testing across many forward-looking periods rather than one split, recording failed experiments alongside successful ones, and checking whether an apparent gain is real before reporting it
- interpretation of performance in operational rather than purely statistical terms
- turning a validated model into a real, callable service, not just a notebook result

The strongest result of the project is not just that one model outperformed another. It is that the system exposed where forecasting succeeds, where it breaks, and where future modelling effort is most likely to matter.

## Disclaimer

This repository uses anonymised and modified operational data for demonstration purposes only. No proprietary credentials or identifiable business information are included.

The public version is intended to showcase pipeline design, forecasting methodology, and analytical reasoning in a portfolio context.
