# Holdout test on the new venue days — pre-registration

**Written:** Friday 16 October 2026, before any sales figure from the new days was opened, computed or plotted.
**Branch:** `Product-branch` (private). The public repository does not receive this file or the new data.
**Status of the new data:** the only things looked at so far are file names, sizes, row counts per file, the date column's first
and last day, and hashes (all listed below). No sales, forecast or labour value has been read.

## 1. What is being tested

Everything up to and including **6 September 2026** is the data the project was built on: the final model was trained on it, the
585-day backtest and Series 1-34 were run on it. Six new weekly exports cover **7 September to 15 October 2026** (39 days, no gap;
the last export is a partial week, Monday to Thursday). The models have never seen those days, so they are a true holdout. They can
be used as one **once**: after scoring they become training data and the next holdout has to be new days again.

The questions, in order of importance:

1. **Q1 (the primary claim).** Is the deployed model still better than the manager's own forecast, at the operating point it was built
   for (mode A below, with the manager's forecast as an input)?
2. **Q2.** How does it do at the lead time the rota needs (a week ahead, mode C) and far ahead (mode B)?
3. **Q3.** Does the Track A package (Series 23-27, re-validated in Series 31, "needs confirmation on new venue days") beat the product
   branch's final model on days it has never seen? Exploratory: see section 8.
4. **Q4.** What does withholding the manager's forecast as an input cost?

## 2. What is frozen, and how it is fingerprinted

| Item | Fingerprint |
|---|---|
| Public code (portfolio arm) | `main` at `8cd4b1d78132c0b52cdafe896a55b2ceb274d3a6` |
| Product code (before this file is added) | `Product-branch` at `43134c4fac09f22b3c2a25dbc51b5d6c7dedca95` |
| Tracked anonymised data (140 files, `data/raw/`, files sorted by name and concatenated) | sha256 `0caf20af67853f6b2bda63577af865d53bf5a2b319b66515df622b3cfb5d8a99` |
| Engineered features before the new days (`data/features/engineered_features.csv`, 979 data rows) | sha256 `ad16ffd7aad53f09fb66a22ec62fbc0d3562d1127e9ac0ab69476daf4d9c64b0` |
| Portfolio arm's model (`models/final_model/`, 27 files, sorted and concatenated; trained 2026-09-13 on 2024-01-16 to 2026-09-06, 950 rows, 38 features) | sha256 `b7d347460573956e55fb7f655d5e4d60a17cda5a327ba2c4ec8ec7b4bbbb924c` |
| `Rotaready-CostControlRawData141.csv` (295 rows, 7 Sep to 13 Sep) | `934de404c8629726b6b17486237a8c609d2bdfb998bd451fb53bf0e45e349460` |
| `...142.csv` (298 rows, 14 to 20 Sep) | `b8432ae95da2c1b94e22b0f665fc7ba029e93bcc49f98b6333de500cd9ff5069` |
| `...143.csv` (291 rows, 21 to 27 Sep) | `dee7b5c6a53dbe066b123f5d90593776f90180b52f675af0c46d2130826e8d0f` |
| `...144.csv` (298 rows, 28 Sep to 4 Oct) | `3ef9f924706b534152daa3fcc3ded33bc13126f51d03cfdce980856bebfc8320` |
| `...145.csv` (301 rows, 5 to 11 Oct) | `e0f888cd08fd84ea929a4ca816406bc366fc9728b1aeb81a2f2db68f0c0b0da6` |
| `...146.csv` (174 rows, 12 to 15 Oct) | `080192e2242ccb0ae345727fab499ddae6c4e9ac4913ef1d5e87dba6fed383f5` |

The new files are real, unanonymised exports. They stay in `data/raw_private/` (gitignored, guarded by the pre-commit hook) and are
processed only in an isolated copy of the pipeline (section 9), never into `data/raw/`.

**Rules.** No model, feature, calendar, hyperparameter, seed or threshold in this document changes after the new days are scored. Every
model that is scored is fitted before the new days' features exist, on rows dated **on or before 6 September 2026** only, and its file
fingerprint is recorded in the results before scoring. If a bug is found after scoring, it is fixed and the scoring re-run with the
same settings; both runs are reported.

## 3. The arms

| Arm | What it is |
|---|---|
| **M** | The manager's own forecast on record for the day. The benchmark, and the thing the venue does today. |
| **S** | A seasonal baseline: the median of actual sales for the same weekday within ±10 days of the same date, over all days before the issue date. Needs no model. |
| **P** | The portfolio arm: the deployed model (25-seed log1p CatBoost, 38 features, original school calendar), the frozen files above, served with the public code. |
| **D** | The product final model: the same recipe refitted on rows up to 6 Sep with the verified school calendar. |
| **K** | The package: log model Monday to Thursday and Friday, raw-£ model on Saturday, the weekend options chosen in Series 27 / 31 (Friday `A_all`, Saturday `A_lags`, Sunday `B`) and the Saturday-and-Sunday correction at 8 weeks, strength 0.5. |

All models: depth 4, 300 iterations, learning rate 0.05, **seeds 0-24** (no fresh seeds in this test, to avoid choosing).

## 4. The three modes (what is known when the forecast is made)

| Mode | Forecast issued | What the model may use |
|---|---|---|
| **A. Operational** | Each day, the day before | Actual sales and the manager's forecast errors up to the previous day (the project's rolling-origin protocol, the same construction as the 585-day backtest, as one more test block); observed weather; the manager's forecast for the day. |
| **B. Blind** | Once, on 6 Sep | Data through 6 Sep only. Recent-sales inputs filled the way the live API fills them (weekday and seasonal medians); weather from the live API's historical-average fallback; the manager's forecast for the day. All 39 days. |
| **C. Monday rota run** | Every Monday, for the **following** week (Monday to Sunday, 7 to 13 days ahead) | Data through the Sunday before the issue (nothing from the issue week); recent-sales inputs filled as in B; weather fallback as in B; the manager's forecast for the target days (assumed on record by the Monday, as confirmed by the venue). |

**Mode C issues:** Monday 7, 14, 21 and 28 September and 5 October 2026. Their target weeks are 14 to 20 Sep, 21 to 27 Sep, 28 Sep to 4 Oct, 5 to 11 Oct and 12 to 18 Oct; the last is scored on the four days that have actuals (12 to 15 Oct). **Primary pool: 32 target days.** The first week of the window (7 to 13 Sep) has no mode C run (it would need an issue on 31 August, inside the model's own training period).

**Without-manager variant (every mode, arms P, D and K):** the target day's `forecast_sales` input is replaced by the weekday-and-seasonal median of the manager's forecasts on record before the issue date (what the live API does when none is supplied). Features built from past forecast errors still use the record. Arms M and S are unchanged.

**Weather sensitivity (mode C only, context, not a decision):** the same runs with the observed weather in place of the fallback.

## 5. The package's routing rule at distance (decided here, because it was open)

Some of the package's inputs do not exist weeks ahead. In modes B and C:

- **Monday to Thursday:** the log final model, as arm D.
- **Friday:** the plain log final model. Option `A_all` needs the sales of the same weekday one week earlier, which falls in the unobserved week.
- **Saturday:** the raw-£ model **with** `A_lags` (sales 2 and 3 weeks ago) when both dates are known at the issue date; otherwise the plain raw-£ Saturday model.
- **Sunday:** the log model with option `B` (weekend interactions: calendar and weather only; weather from the fallback).
- **Correction:** the Saturday and Sunday correction as in Series 24, computed from the errors of the 585 backtest days (frozen at 6 Sep, not updated with new weeks).

In mode A the package is used exactly as validated.

## 6. Scoring

- **Scored days:** every day with an actual sales figure in the window, excluding known closure days (none fall in this window).
- **Primary metric:** pooled MAE in £ over the scored days. All arms in a comparison are scored on **the same days**.
- **Also reported:** bias (forecast minus actual), Friday-to-Sunday and Monday-to-Thursday MAE, each week's MAE, each mode C run's MAE.
- **Uncertainty:** the paired difference between two arms, with a 95% moving-block bootstrap interval in 7-day blocks (the method used in Series 23-34).
- **Number of decisions:** one, Q1 below. Everything else is reported as description; no other comparison is used to promote or reject anything.

## 7. Reference bands from the old data (computed before the new days were opened)

From the 583 scored backtest days, over 545 overlapping 39-day windows of the deployed model and the manager:

| | 5th | 25th | median | 75th | 95th |
|---|---|---|---|---|---|
| Model MAE (£) | 734 | 796 | 867 | 966 | 1,333 |
| Manager MAE (£) | 840 | 944 | 1,060 | 1,199 | 1,406 |
| Gain, manager minus model (£/day) | -24 | 76 | 173 | 255 | 357 |

In 91.6% of historical 39-day windows the model beat the manager. So a window in which it does not is unusual (about 1 in 12) but not impossible, and a 39-day window cannot by itself prove a gain: the 95% interval on the gap is roughly ±£270.

## 8. Decision rules

**Q1** (arm P against M, mode A, with the manager's forecast), on the paired gain (manager MAE minus model MAE):

| Outcome | Condition |
|---|---|
| **Confirmed** | the 95% interval is entirely above 0 |
| **Consistent** | the point estimate is above 0 but the interval includes 0 (the expected result with 39 days) |
| **Warning** | the point estimate is 0 or below (about 1 window in 12 historically) |
| **Drift alarm** | the model's window MAE is above £1,333 (the historical 95th percentile) |

**Q2, Q4:** reported against M and S with the same four labels, and against each other; no decision depends on them.

**Q3** (K against D, same days, modes A and C): the change in MAE and its interval, labelled **directionally consistent** if the point estimate is below 0 on all days *and* on Friday to Sunday, **no signal** if mixed, **contradicted** if the point estimate on all days is above +£30. **The package is not promoted on this window.** Promotion needs the prospective log (section 10) to reach 26 weeks and the interval to exclude zero.

## 9. Where it runs

An isolated copy of the pipeline (`FORECAST_CONFIG` pointing at a holdout configuration, all paths under a gitignored `data/holdout/` folder), so nothing in `data/raw/`, `data/features/`, `models/` or `reports/` changes. The new files are anonymised into the isolated area only. Steps: (1) fit and fingerprint every model on rows up to 6 Sep and commit the fingerprints; (2) build the new days' features in the isolated copy; (3) score once. The result is logged as Series 35 in `docs/EXPERIMENT_LOG.md`, whatever it shows.

## 10. The prospective log (from now on)

A forecast made before the week happens cannot be influenced by hindsight, so from here the test continues forward:

- **Every Monday**, starting **19 October 2026** (target week 26 October to 1 November), arms P, D and K issue a mode C forecast for the following week, with and without the manager's forecast. The forecasts are appended to `docs/holdout_log/forecast_log.csv` (issue time, target date, arm, variant, forecast, manager's forecast used, model fingerprint, code commit) and **committed to git before the week begins**; the commit time is the proof.
- **A catch-up run** for 19 to 25 October is issued on Friday 16 October from data through 15 October (a shorter lead, 3 to 9 days). It is labelled, and not pooled with the Monday runs.
- When a week's actuals arrive they are appended to `actuals_scored.csv` and the running tally is updated. The decision rule for the running tally is the same as Q1 and Q3, applied to the accumulated days.

## 11. Limits, stated in advance

- 39 days cannot confirm a gain smaller than about £270/day; the model's known gain over the manager (£166/day) needs about 31 weeks of days to confirm, a £44/day improvement (the package's size) several years. This test can catch breakage, bias and drift, and start the prospective record; it cannot settle small differences.
- The new days include whatever the autumn brought, any school break and any unusual event included. None of it is corrected for or excluded after the fact.
- The manager's forecast for the following week is assumed to have been on record by the Monday.
- Mode A uses observed weather and known recent sales, as the project's backtest always has; B and C are the honest far-ahead conditions.
