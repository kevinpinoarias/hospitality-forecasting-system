# Experiment Log — Hospitality Sales Forecasting

This is the canonical, cumulative record of every systematic model/feature
experiment run on this project's forecasting problem: what was tried, the
exact configuration of each trial, the hypothesis it was testing, and the
result. It exists so that "what have we already tried, and what did it
show" is answerable from one document rather than reconstructed from
memory or scattered CSVs.

**Conventions used throughout this log**

- **Target**: `total_sales`, daily, in GBP (£).
- **Primary metric**: MAE (mean absolute error), £, averaged across
  rolling-origin folds. RMSE, MAPE, and Bias are also tracked per trial —
  see `src/evaluation/metrics.py` for exact definitions.
- **Validation scheme**: rolling-origin (walk-forward) backtesting, not a
  single fixed train/test split — see `src/experiments/splits.py`. A
  model is trained on a window of history, tested on the following block,
  then the whole thing rolls forward and repeats, producing multiple
  independent (train, test) snapshots. This directly tests the project's
  own documented finding that performance on one period does not
  guarantee performance on another (see the main `README.md`'s
  "Validation Strategy and Generalisation" section).
- **Every trial is logged whether it wins or loses.** A negative result
  (a feature that made things worse, a hyperparameter that didn't help)
  is recorded with the same rigor as a positive one — that is itself
  useful knowledge, and omitting it would make this log a highlight reel
  rather than a record.
- **Raw data**: every trial's fold-level result is written to CSVs in
  `reports/experiments/`; this document is the narrative index into them,
  not a replacement for them. Those CSVs are generated output (gitignored,
  like the rest of `reports/`), so each entry below names the script that
  regenerates it.

**Reproducing every result**

```bash
pip install -r requirements.txt -r requirements-dev.txt -c constraints-experiments.txt
python main.py                               # builds data/features/engineered_features.csv
python -m src.experiments.run_all_series     # Series 5-17 and business impact, in dependency order
python -m src.experiments.run_all_series --include-1-4   # also Series 1-4 (much slower)
```

Series 1-4 were scripted as they were run. Series 5-16 were first run
interactively during the same working session, and their scripts were
reconstructed afterwards from the exact code that produced each logged
result. Every reconstructed script was then re-run and checked against
this log: **all twelve reproduce their logged figures exactly**, as do
Series 17's headline comparison and the business-impact simulation. The
whole chain was then re-run from a fresh clone of the repository, with a
new environment installed from the requirements files: every figure
matched again except one.

**Library versions matter.** The requirements files set only minimum
versions, and gradient-boosting libraries do not guarantee identical
results across releases. In the fresh-clone check, a newer XGBoost
(3.4.1 rather than 3.2.0) moved Series 17's portfolio-XGBoost figure from
£1,034.82 to £1,022.17; every CatBoost result, including all headline
figures, was unaffected. `constraints-experiments.txt` pins the exact
versions this log was produced with - install with it to reproduce the
figures exactly. Every fit also uses a fixed seed, but different hardware
or thread counts can still introduce tiny floating-point differences.

---

## Series 1 — Algorithm & Hyperparameter Sweep

**Run**: 2026-09-09 · **Script**: `src/experiments/run_sweep.py` ·
**Raw data**: `sweep_results_full.csv` (3,880 rows), `sweep_leaderboard.csv`
(388 rows), `sweep_summary.md`

### Hypothesis

The production model (XGBoost, evaluated on one fixed 2024-12-31 train/test
split) has never been compared against other algorithm families under a
methodologically matched, multi-period evaluation. Given the drift already
documented in this project on later 2026 data, a single fixed split cannot
distinguish "this model generalises" from "this model got lucky on this
one split." **H1**: a systematic, rolling-origin comparison across
algorithm families and hyperparameters will identify a stronger and/or
more reliably-generalising model than the incumbent.

### Method

- **Data**: `data/features/model_features.csv`, 978 rows, 2024-01-02 to
  2026-09-06.
- **Folds**: 10 rolling-origin folds. Fold 0 trains on the first 365 days
  (2024-01-02 → 2025-01-01) and tests on the following 60 days; each
  subsequent fold's test window starts immediately after the previous
  one's, non-overlapping, through fold 9 (test: 2026-06-25 → 2026-08-24).
- **Training-window mode** (applied on top of every fold): **expanding**
  (all history before the test window) vs. **sliding 365-day** (only the
  most recent year of history). Both tested for every configuration below.
- **Families and grids tested** (full grid search, every combination):

| Family | Hyperparameter grid | Combinations |
|---|---|---|
| `naive` | none | 1 |
| `seasonal_naive` | `lag_days=7` (fixed) | 1 |
| `rolling_mean` | `window ∈ {7, 14, 21, 28, 45}` | 5 |
| `weekday_average` | none | 1 |
| `sarimax` | `order ∈ {(1,1,1),(2,1,1),(0,1,1)}` × `seasonal_order ∈ {(0,1,1,7),(1,1,1,7)}` | 6 |
| `xgboost` | `n_estimators∈{100,300}` × `learning_rate∈{0.03,0.05,0.1}` × `max_depth∈{3,4,6}` × `subsample∈{0.8,1.0}` × `colsample_bytree∈{0.8,1.0}` | 72 |
| `lightgbm` | same grid shape as xgboost | 72 |
| `catboost` | `iterations∈{100,300}` × `learning_rate∈{0.03,0.05,0.1}` × `depth∈{4,6}` | 12 |
| `prophet` | `changepoint_prior_scale∈{0.01,0.05,0.5}` × `seasonality_prior_scale∈{1.0,10.0}` × `seasonality_mode∈{additive,multiplicative}` | 12 |
| `lstm` | `hidden_size∈{16,32}` × `num_layers∈{1,2}` × `window∈{14,30}`, 40 epochs | 8 |
| `transformer` | `d_model∈{16,32}` × `nhead=2` × `window∈{14,30}`, 40 epochs | 4 |

Total: 194 unique (family, hyperparameter) configurations × 2 training-window
modes × 10 folds = **3,880 fits**. All 3,880 completed successfully (0
failures) after one bug fix during the pilot phase (see Deviations below).

### Results — best configuration per family

| Family | Best hyperparameters | Window | Mean MAE (£) | Mean RMSE | Mean MAPE | Mean Bias |
|---|---|---|---|---|---|---|
| **catboost** | `iterations=300, learning_rate=0.03, depth=4` | sliding 365d | **975.59** | 1263.88 | 13.14% | 237.89 |
| xgboost | `n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8` | expanding | 993.25 | 1269.62 | 13.64% | 214.39 |
| prophet | `changepoint_prior_scale=0.01, seasonality_prior_scale=1.0, seasonality_mode=additive` | expanding | 1008.32 | 1295.74 | 14.25% | 282.19 |
| lightgbm | `n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8` | expanding | 1009.53 | 1313.15 | 13.56% | 218.56 |
| transformer | `d_model=32, nhead=2, window=30, epochs=40` | expanding | 1168.06 | 1588.68 | 15.70% | 163.99 |
| sarimax | `order=(2,1,1), seasonal_order=(0,1,1,7)` | sliding 365d | 1179.90 | 1497.36 | 17.35% | 288.20 |
| lstm | `hidden_size=32, num_layers=1, window=14, epochs=40` | expanding | 1217.76 | 1598.59 | 15.87% | −10.66 |
| weekday_average | n/a | expanding | 1252.29 | 1638.11 | 17.59% | 88.49 |
| seasonal_naive | `lag_days=7` | expanding | 1369.54 | 1883.13 | 20.05% | 31.00 |
| rolling_mean | `window=7` | sliding 365d | 2795.42 | 3494.98 | 36.21% | 22.87 |
| naive | n/a | sliding 365d | 3054.40 | 3953.73 | 38.18% | 6.44 |

### Additional finding: expanding vs. sliding training window

Averaged across all 388 configurations: expanding = £1,124.44 mean MAE,
sliding 365d = £1,150.64 mean MAE — a ~2% difference, not a clear or
consistent effect. Per-family preference is genuinely mixed (catboost,
sarimax, rolling_mean, and naive slightly prefer the sliding window;
xgboost, prophet, lightgbm, and both neural families slightly prefer
expanding). **Conclusion: no universal answer** — this project's own
open question about whether more historical data helps or hurts does not
have a single answer across model families.

### Conclusions

1. **CatBoost is the strongest model found**, beating the incumbent
   XGBoost by ~1.8% MAE, and every other family tested.
2. **Prophet is a legitimate, cheap alternative** — 3rd overall, ahead of
   LightGBM, and with the *lowest* variance across folds (`MAE_std ≈
   98.7`, versus ~120–150 for the gradient-boosted models) — flagged for
   inclusion in Series 2 as "very promising."
3. **SARIMAX was previously mischaracterised.** Under the original
   single-split evaluation it scored MAE ≈ £3,080 (worst of all models
   tested at the time). Under proper rolling-origin backtesting — i.e.
   retrained periodically instead of forecasting ~630 days from one 2024
   fit — it scores £1,179.90, solidly mid-pack. The earlier bad score
   reflected an evaluation-methodology artifact (never retraining it),
   not a real model weakness.
4. **Neural forecasters (LSTM, Transformer) do not earn their complexity**
   on this data volume — both underperform every gradient-boosted model
   and Prophet. Transformer beats LSTM, consistent with the original
   portfolio notebook's finding.
5. **Training-window mode (expanding vs. sliding) is not a meaningful
   lever** on its own — the effect is small and inconsistent across
   families.

### Deviations from plan / bugs caught

- **SARIMAX bug, caught in pilot mode, fixed before the full run.** The
  first implementation resolved missing-calendar-day gaps separately on
  each fold's train and test slice; `dropna()` followed by `asfreq("D")`
  can reintroduce a dropped date as a fresh NaN row when the reindex
  fills the gap back in, causing every SARIMAX fit to fail
  (`MissingDataError: exog contains inf or nans`). Fixed by resolving
  gaps once on the combined train+test series before splitting, matching
  `src/models/train_sarimax.py`'s original approach. Verified after the
  fix: 0 SARIMAX failures across all 120 fits.
- **Metadata bug, caught after the full run, corrected via patch (not a
  re-run — the actual training data used per fold was always correct;
  only a reporting column was wrong).** `run_one_fit`'s logged
  `train_start` initially recorded the fold's nominal (always
  dataset-start) boundary rather than the actual post-sliding-window
  start date, understating what "sliding 365d" configurations had
  actually trained on when read back later. `sweep_results_full.csv` was
  patched in place to record the true per-row training start; the
  leaderboard and all reported metrics were unaffected since they were
  computed from real model output, not from this metadata field.

---

## Series 2 — Feature Engineering Sweep

**Run**: 2026-09-09 · **Script**: `src/experiments/run_feature_sweep.py` ·
**Raw data**: `feature_sweep_results_full.csv`, `feature_sweep_leaderboard.csv`,
`feature_sets_registry.json`

### Hypothesis

The production feature set (15 features) was selected during earlier,
less systematic exploration, from a broader engineered pool
(`data/features/engineered_features.csv`, 23 additional already-computed
columns never selected in). **H2**: some of these unused columns —
particularly short-lag sales/forecast-error history, which is
conspicuously absent from the current 15 (only a 7-day lag exists; there
is no 1-day lag at all) — carry real predictive signal the current model
is not using. A secondary question: **H3**, is every one of the current
15 features actually earning its place, or are some carried over from
earlier work without real justification?

### Method

This is a design-of-experiments, not a search over all `2^23` subsets
(computationally infeasible, and uninterpretable even if it weren't — a
best subset found by brute force doesn't explain *why* it's best). See
`src/experiments/feature_sets.py` for the full rationale. Five kinds of
trial, all evaluated on the identical row set (rows where every candidate
*and* baseline column is simultaneously non-null — 950 of 979 rows,
2024-01-16 to 2026-09-06 — so no comparison between feature sets is ever
confounded by different feature sets seeing different amounts of data):

1. **Baseline** — the current 15-feature production set (control).
2. **Leave-one-out** (15 trials) — remove each current feature
   individually; tests H3.
3. **Add-one** (23 trials) — add each candidate individually to the
   baseline; tests each candidate's own marginal contribution.
4. **Add-group** (7 trials) — add each thematic group of candidates as a
   whole (sales/forecast-error history, calendar extras, payday extras,
   bank-holiday extra, school-holiday flags, raw weather, extra derived
   weather); tests for synergy beyond individual effects.
5. **Add-all** (1 trial) — every candidate at once; upper-bound /
   overfitting-risk check.
6. **Add-top-K combination** (built from Phase 1's own add-one ranking,
   per family — the best individual performers are not guaranteed to
   combine additively) — top 3, top 5, and top 8 candidates by that
   family's own individual ranking, combined and retested.

**Models**: `catboost` and `prophet` — Series 1's #1 (CatBoost) and its
most promising alternative (Prophet: 3rd overall, lowest fold-to-fold
variance). Each held fixed at its Series 1 winning hyperparameters
(`catboost`: `iterations=300, learning_rate=0.03, depth=4`; `prophet`:
`changepoint_prior_scale=0.01, seasonality_prior_scale=1.0,
seasonality_mode=additive`) so this experiment isolates the feature
effect specifically, not a joint feature-and-hyperparameter search.

**Candidate feature groups tested**:

| Group | Features |
|---|---|
| `sales_error_history` | `lag_7_sales`, `rolling_7_sales`, `rolling_14_sales`, `lag_1_fe`, `lag_7_fe`, `rolling_7_fe` |
| `calendar_extra` | `year`, `is_weekend`, `day_of_week_cos` |
| `payday_extra` | `is_payday`, `days_to_payday` |
| `bank_holiday_extra` | `days_since_bank_holiday` |
| `school_holiday` | `is_christmas_break`, `is_summer_break`, `is_easter_break`, `is_school_holiday` |
| `weather_raw` | `max_temp`, `rain_mm`, `sun_hours` |
| `weather_derived_extra` | `is_hot_for_scotland`, `is_dry_day`, `temp_anomaly_14d`, `warm_streak_len` |

**Excluded from every candidate set**: `forecast_error` itself
(`= total_sales − forecast_sales`) — this directly encodes the target and
would be leakage, not a feature. Its *lagged* values (`lag_1_fe`,
`lag_7_fe`, `rolling_7_fe`) reference only already-known history and are
legitimate candidates. Also excluded: `payday`, `prev_payday`,
`next_payday` — date-typed intermediate columns that exist only to
*derive* other features, not standalone model inputs.

Same rolling-origin fold scheme as Series 1 (10 folds, 60-day test
windows) and both training-window modes (expanding, sliding 365d), for
full methodological parity.

Total planned: (1 baseline + 15 leave-one-out + 23 add-one + 7 add-group +
1 add-all) = 47 Phase-1 sets × 2 families × 2 window modes × 10 folds =
1,880 fits, plus Phase-2 top-K combinations (3 combos × 2 families × 2
window modes × 10 folds = 120 fits) = **2,000 fits total**.

### Results

All 2,000 fits completed successfully (0 failures).

**Baseline reference** (current 15-feature production set, averaged across
both training-window modes): CatBoost £973.03 MAE, Prophet £1,047.43 MAE.

#### Top 15 overall (across every category, family, and window mode)

| Feature set | Category | Family | # features | Window | Mean MAE (£) |
|---|---|---|---|---|---|
| `add_all_candidates` | add_all | catboost | 38 | expanding | **907.77** |
| `add_top5_individual` | add_combo | catboost | 20 | sliding 365d | 908.49 |
| `add_top3_individual` | add_combo | catboost | 18 | expanding | 908.61 |
| `add_top8_individual` | add_combo | catboost | 23 | expanding | 910.03 |
| `add_top5_individual` | add_combo | catboost | 20 | expanding | 912.68 |
| `add_all_candidates` | add_all | catboost | 38 | sliding 365d | 924.09 |
| `add_top8_individual` | add_combo | catboost | 23 | sliding 365d | 924.21 |
| `add_group__sales_error_history` | add_group | catboost | 21 | expanding | 926.31 |
| `add_top3_individual` | add_combo | catboost | 18 | sliding 365d | 931.33 |
| `add_one__lag_1_fe` | add_one | catboost | 16 | expanding | 932.47 |

**Every one of the top 10 results beats both the Series 1 winner (£975.59)
and this sweep's own CatBoost baseline (£973.03).** The best single change
— adding every candidate feature at once — improves MAE by **£65.26 (6.7%)**
over the Series 1 production configuration.

#### Add-one ranking (individual marginal contribution, averaged across window modes)

| Rank | CatBoost candidate | Δ MAE vs. baseline | Prophet candidate | Δ MAE vs. baseline |
|---|---|---|---|---|
| 1 | `lag_7_sales` | −32.56 | `rolling_7_sales` | −52.83 |
| 2 | `lag_1_fe` | −27.88 | `lag_7_sales` | −46.32 |
| 3 | `rolling_7_fe` | −21.81 | `rolling_7_fe` | −39.08 |
| 4 | `year` | −18.18 | `lag_1_fe` | −35.47 |
| 5 | `is_weekend` | −10.70 | `is_dry_day` | −34.48 |
| 6 | `day_of_week_cos` | −9.61 | `day_of_week_cos` | −32.12 |
| 7 | `lag_7_fe` | −9.21 | `lag_7_fe` | −31.61 |

The four sales/forecast-error history features (`lag_7_sales`, `lag_1_fe`,
`rolling_7_fe`, `lag_7_fe`) rank in the **top 7 for both model families
independently** — strong convergent evidence this is a real, model-agnostic
signal, not an artifact of one algorithm's inductive bias. Every single
add-one candidate improved on baseline for both families **except**
`temp_anomaly_14d`, `days_to_payday`, `sun_hours`, and
`days_since_bank_holiday` (all four showed a small, consistent *negative*
effect for both families — see full ranking in
`feature_sweep_leaderboard.csv`).

#### Add-group ranking (mean MAE, £, lower is better)

| Group | CatBoost | Prophet |
|---|---|---|
| **sales_error_history** | **931.20** | **980.34** |
| calendar_extra | 941.26 | 1029.29 |
| payday_extra | 969.94 | 1048.67 |
| school_holiday | 971.00 | 1011.79 |
| weather_derived_extra | 973.27 | 1059.61 |
| weather_raw | 976.85 | 1049.95 |
| bank_holiday_extra | 978.25 | 1062.23 |

`sales_error_history` is the single strongest group for **both** families
by a clear margin, consistent with the add-one ranking above.

#### Leave-one-out (CatBoost, averaged across window modes) — is every current feature earning its place?

| Feature removed | MAE with it removed | Δ vs. baseline | Reading |
|---|---|---|---|
| `month_sin` | 963.65 | −9.37 | removing it **helps** |
| `day_of_year_cos` | 966.40 | −6.63 | removing it **helps** |
| `days_to_bank_holiday` | 966.58 | −6.44 | removing it **helps** |
| `day_of_year` | 967.25 | −5.78 | removing it **helps** |
| `month_cos` | 967.25 | −5.77 | removing it **helps** |
| `days_since_payday` | 968.12 | −4.90 | removing it **helps** |
| `month` | 971.13 | −1.89 | mildly helps |
| `is_bank_holiday` | 971.30 | −1.72 | mildly helps |
| `is_payday_window_pm3` | 973.01 | −0.02 | neutral |
| `day_of_year_sin` | 973.84 | +0.81 | neutral |
| `is_long_weekend` | 974.63 | +1.60 | neutral |
| `is_heavy_rain` | 978.51 | +5.48 | removing it **hurts** |
| `day_of_week_sin` | 981.03 | +8.00 | removing it **hurts** |
| `day_of_week` | 1015.44 | +42.41 | removing it **hurts substantially** |
| `forecast_sales` | 1047.93 | +74.90 | removing it **hurts substantially** |

Only 4 of the current 15 features (`forecast_sales`, `day_of_week`,
`day_of_week_sin`, `is_heavy_rain`) show a clear, individually-attributable
positive contribution. Eight — mostly the calendar/cyclical-encoding
cluster (`month`, `month_sin`, `month_cos`, `day_of_year`,
`day_of_year_cos`) plus `days_to_bank_holiday` and `days_since_payday` —
show *improvement* when individually removed.

**Important methodological caveat, stated explicitly rather than acted on
blindly**: several of these are redundant *encodings of the same
underlying signal* (e.g. `month`, `month_sin`, and `month_cos` all encode
calendar month; `day_of_year`, `day_of_year_sin`, `day_of_year_cos` all
encode day-of-year). A one-at-a-time leave-one-out test cannot distinguish
"this feature is genuinely useless" from "this feature's information is
still present via its redundant sibling, so removing just one of three
barely changes anything, or even helps by reducing noise/dimensionality
slightly." **This result says the current *encoding scheme* for calendar
seasonality is carrying more redundant dimensions than necessary — it does
not say calendar seasonality itself is unhelpful.** A proper follow-up
would test removing each *redundant group* as a whole (e.g. drop all of
`month`/`month_sin`/`month_cos` together and see if a single cyclical
pair suffices), not one column at a time. This is flagged as unfinished
work below, not concluded here.

### Conclusions

1. **H2 is confirmed, strongly and consistently.** The unused
   `sales_error_history` features — short-lag actual sales and
   forecast-error history — are the single strongest lever found in this
   entire project to date, for both CatBoost and Prophet independently.
   `lag_7_sales`, `lag_1_fe`, `rolling_7_fe`, and `lag_7_fe` are the
   standout individual contributors.
2. **The best result found so far, project-wide, is CatBoost + every
   candidate feature added (`add_all_candidates`) on the expanding
   window: £907.77 mean MAE** — a 6.7% improvement over the Series 1
   winner, and better than any single top-K combination tested. This
   suggests the "kitchen sink" risk (multicollinearity/overfitting from
   adding all 23 candidates) did not materialise here — more likely
   because tree-based CatBoost handles correlated/irrelevant features
   comparatively gracefully via its own feature-importance-driven
   splitting.
3. **Prophet improved even more dramatically in relative terms**
   (baseline £1,047.43 → best `add_group__sales_error_history` £980.34,
   a 6.4% improvement), closing much of the gap to CatBoost. Combined
   with Series 1's finding that Prophet has the lowest fold-to-fold
   variance of any model tested, it remains a genuinely strong second
   option, not just a curiosity.
4. **H3 is partially supported, with an important caveat.** Several
   current features look individually removable, but this is very
   likely a redundant-encoding artifact (see caveat above) rather than
   proof those features carry no signal. This needs a follow-up
   experiment (remove redundant encoding groups together, not
   one-by-one) before any feature is actually dropped from production.
5. **Not yet answered by this experiment**: whether `add_all_candidates`
   generalises, or is itself an artifact of this specific 950-row,
   single-venue dataset — the same caution this project has consistently
   applied to every other result (see Series 1's SARIMAX finding, and the
   project's own documented 2026 drift). The next production feature set
   should not be finalised without validating this on genuinely new data
   as it arrives.

### Recommended next steps (not yet executed — logged as open items)

- Retest with the redundant-encoding-group leave-out design described in
  the leave-one-out caveat above, before touching the production feature
  list.
- Re-tune CatBoost's hyperparameters (Series 1's grid) on top of the
  `add_all_candidates` / `sales_error_history`-group feature set — Series
  1's winning hyperparameters were chosen for the old 15-feature set and
  are not guaranteed optimal for a materially different feature space.
- The new features requested but deliberately not engineered in this
  series (short lags 1–3, a 28-day rolling window, explicit fixed-calendar
  seasonal periods) remain open — see Appendix.

---

## Series 3 — Recursive Multi-Day-Ahead Horizon Test

**Run**: 2026-09-09 · **Script**: `src/experiments/recursive_horizon_test.py` ·
**Raw data**: `recursive_horizon_results.csv` (day-by-day predictions),
`recursive_horizon_summary.csv`

### Hypothesis

Series 2 found `lag_7_sales`, `lag_1_fe`, `rolling_7_fe`, and `lag_7_fe` to
be the strongest levers found in this project. None of these reference
same-day or future data - verified directly against
`src/features/build_features.py`'s `shift()`/`rolling()` calls, so this is
not target leakage. But Series 2's rolling-origin evaluation always had
access to *true* actuals for the days immediately preceding whatever date
it was scoring - an assumption that only holds for a forecast generated
shortly before its target date with fresh data, not for the longer-horizon
forecasts this project's own API explicitly supports (`days_beyond_
training_data` exists precisely to flag this kind of trust gap). **H4**:
Series 2's reported improvement from lag features will shrink, and
possibly reverse, once forecasts are generated genuinely recursively -
using the model's own prior predictions, not real actuals, for any lag
input whose reference date falls inside the forecast batch itself.

### Method

For each of the 10 Series-2 rolling-origin folds, `catboost` and `prophet`
(each at their Series 1 winning hyperparameters, expanding training
window, matching Series 2's own winning configuration for
`add_all_candidates`) are fit **once** on the fold's training data, then
used to predict forward day-by-day through the first 30 days of the
fold's test window. At each step, the six buffer-dependent features
(`lag_7_sales`, `rolling_7_sales`, `rolling_14_sales`, `lag_1_fe`,
`lag_7_fe`, `rolling_7_fe`) are recomputed from a running buffer seeded
with real history and then **overwritten with the model's own prediction**
for any date that falls inside the batch already forecast - never with
the real actual, even though it exists in the backtest data (used only
for scoring, never fed back in). Every other feature (calendar, weather,
payday, holiday, `forecast_sales`) is unaffected by horizon and stays as
originally computed, since none of them depend on realised sales.

Two feature sets compared: `baseline_15` (the current production set - no
buffer-dependent features at all, and therefore horizon-invariant by
construction, serving as the reference line) vs `add_all_candidates`
(Series 2's overall winner, includes all six buffer-dependent features).

**Verification before trusting any result**: the buffer/lag recomputation
logic was checked two ways before running the real experiment - (1) unit
tests (`tests/test_recursive_horizon.py`) confirming the lag/rolling
arithmetic exactly matches `build_features.py`'s real semantics, and (2)
a direct spot-check that, at horizon day 1 (where every referenced date
is still real history), the recomputed feature values are bit-for-bit
identical to the precomputed values already in `engineered_features.csv`
for the same date (confirmed: exact match on all six features).

### Deviations / data issues found and handled

- **A genuine, pre-existing single-day data gap**: `total_sales` is
  missing entirely for 2025-01-24 (the same date `model_features.csv`
  silently drops via its own `dropna()` elsewhere in this project - a
  real reporting gap in the underlying export, not a bug introduced
  here). Left as NaN, this would propagate into every lag/rolling feature
  referencing it for up to 14 days afterwards. Linearly interpolated for
  recursion-buffer continuity only, with the interpolation explicitly
  logged at runtime - it does not affect which dates are scored.
- **Fold 0 excluded.** Its 30-day horizon window (2025-01-01 → 2025-01-30)
  crosses a *separate* 15-day span (2025-01-24 → 2025-02-07) where the
  feature-selection dataset (`df`, used in Series 2) has no rows at all,
  because some unrelated column (most likely weather) was occasionally
  missing there - rows Series 2's own `dropna()` already silently
  excluded. There is no clean feature row to predict from or score
  against on those calendar days, so fold 0 cannot be evaluated cleanly
  for this experiment. The other 9 folds are unaffected and were run in
  full. This is disclosed rather than worked around with a more invasive
  imputation, since it affects only 1 of 10 folds.

### Results

All predictions completed successfully. Because a single specific
calendar day (e.g. "day 1 of each fold") is only 9 data points and is
additionally confounded by which weekday happens to land on that offset
in each fold (folds are spaced 60 days apart, and 60 is not a multiple of
7, so "day 1" is a different weekday in different folds), the primary
result here is the **horizon-bucketed average** (pooling multiple days
per bucket, 9 folds x days-per-bucket data points), not the raw
single-day checkpoints in isolation.

**MAE by horizon bucket (£), and the delta (`add_all_candidates` −
`baseline_15`; negative means the lag-feature set is still better):**

| Family | Bucket | `add_all_candidates` | `baseline_15` | Delta |
|---|---|---|---|---|
| catboost | days 1–7 | 861.83 | 953.95 | **−92.12** |
| catboost | days 8–14 | 788.92 | 844.60 | **−55.68** |
| catboost | days 15–30 | 899.80 | 945.19 | **−45.39** |
| prophet | days 1–7 | 1044.53 | 896.78 | **+147.75** |
| prophet | days 8–14 | 856.32 | 932.10 | −75.78 |
| prophet | days 15–30 | 943.10 | 966.74 | −23.64 |

**Overall MAE, full 30-day horizon pooled (270 data points per cell):**

| Family | `add_all_candidates` | `baseline_15` |
|---|---|---|
| catboost | **865.07** | 923.76 |
| prophet | 946.52 | 942.33 |

### Addendum: matched real-data-vs-predicted-data comparison

Series 2's £907.77 (10 folds, full 60-day window, always real data) and
this series' £865.07 (9 folds, first 30 days only, predicted data fed
back) are **not directly comparable numbers** - they differ in fold
count and day-range scored, not just in the recursion treatment. Before
drawing any conclusion from comparing them directly, a third run was
added: `add_all_candidates`, identical 9 folds, identical first 30 days,
but with `use_true_actuals=True` (the buffer is fed the real actual each
day instead of the model's own prediction - reproducing Series 2's
assumption on the exact same subset Series 3 used).

| | CatBoost | Prophet |
|---|---|---|
| With real data (oracle) | **838.12** | **901.67** |
| Without real data (recursive) | 865.07 | 946.52 |

On a genuinely matched subset, real data beats predicted data for both
models, as expected - confirming the £907.77 vs. £865.07 comparison was
an artifact of non-matched fold/day counts, not evidence that recursion
somehow helps. Raw data: `recursive_horizon_oracle_results.csv`.

### Conclusions

1. **H4 is confirmed for CatBoost, but only partially - the advantage
   shrinks, it does not vanish.** The lag-feature set beats the baseline
   in every single bucket tested (day 1 through day 30), though the
   margin roughly halves from the near-term bucket (−92) to the far
   bucket (−45). Series 2's CatBoost finding survives genuinely recursive
   testing, at a reduced but still real magnitude, across the entire
   30-day horizon tested.
2. **H4 is confirmed more strongly for Prophet, and with a real
   reversal.** Under genuine recursion, Prophet's lag-feature advantage
   is *negative* in the near term (days 1–7: +147.75, i.e. adding the lag
   features makes Prophet's near-term forecasts *worse* once they can no
   longer cheat with fresh actuals), only turning net-positive from day 8
   onward, and the full-horizon pooled result is essentially a wash
   (946.52 vs. 942.33 - not a meaningful difference). **Series 2's
   headline finding that Prophet benefited even more than CatBoost from
   the lag features does not survive this test.** That finding was
   substantially an artifact of Series 2's "always-fresh-actuals"
   evaluation assumption, not a durable property of Prophet with these
   features.
3. **Practical implication for production adoption**: if
   `add_all_candidates` / CatBoost is promoted to production, it should
   be understood as roughly halving its edge over the current baseline
   for forecasts requested further in advance, not losing it entirely, at
   least out to a 30-day horizon (untested beyond that). If Prophet is
   ever reconsidered as the production model, the lag-feature additions
   from Series 2 should **not** be carried over without this correction -
   they help only from roughly a week out, and actively hurt near-term
   accuracy, the opposite of the ordering Series 2 implied.
4. **Not yet answered**: *why* Prophet specifically shows a near-term
   regression rather than a small, CatBoost-like shrinkage - plausibly
   related to how Prophet's own trend/changepoint component interacts
   with a freshly-buffer-filled (imperfect, model-generated) regressor
   value right after a fold boundary, but this is a hypothesis, not
   verified. Flagged as an open item, not concluded here.
5. **Methodological note for any future horizon experiment**: fold
   spacing (60 days) is not a multiple of 7, so any single fixed
   horizon-day checkpoint mixes different weekdays across folds - a
   confound this analysis avoided by bucketing (days 1-7, 8-14, 15-30)
   rather than trusting any single day's number in isolation. A cleaner
   future version would align fold boundaries to a fixed weekday.

## Series 4 — Short-Lag/Rolling Features on Top of the Winning Configuration

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series4_sweep.py` ·
**Raw data**: `series4_results_full.csv`, `series4_leaderboard.csv`

### Hypothesis

The 6 features engineered on 2026-09-10 (`lag_1_sales`, `lag_2_sales`,
`lag_3_sales`, `lag_2_fe`, `lag_3_fe`, `rolling_28_sales`) fill a gap
Series 2's own conclusions identified: plain "yesterday's sales" was not
in the engineered pool at all, despite `lag_7_sales` and `lag_1_fe` being
the two strongest individual features found in that series. **H5**: these
should provide further improvement when added to `add_all_candidates`
(Series 2's 38-feature winner), for the same reason `lag_7_sales` helped
so much when added to the original 15-feature baseline.

### Method

CatBoost only, fixed at the Series 1/2 winning hyperparameters
(`iterations=300, learning_rate=0.03, depth=4`). Same 10 rolling-origin
folds and same two training-window modes as Series 2 - deliberately the
*non-recursive* evaluation (Series 2's methodology, not Series 3's), for
direct comparability with the £907.77 figure. `add_all_candidates` is
re-run here as the reference point rather than only cited, so it sits in
the same directly-comparable table. Two questions tested: adding all 6
new features to `add_all_candidates` at once, and adding each
individually.

### Results

All 160 fits completed successfully (0 failures). The reference re-run of
`add_all_candidates` reproduced £907.77 exactly, confirming methodology
parity with Series 2.

| Feature set added (expanding window) | MAE (£) | Δ vs. `add_all_candidates` |
|---|---|---|
| `add_all_candidates` (reference) | 907.77 | — |
| + `lag_1_sales` | 914.19 | +6.41 |
| + `lag_2_sales` | 914.77 | +7.00 |
| + `lag_3_sales` | 918.00 | +10.23 |
| + `lag_3_fe` | 920.66 | +12.89 |
| + `lag_2_fe` | 921.19 | +13.41 |
| + `rolling_28_sales` | 922.23 | +14.46 |
| + all 6 together | 926.94 | +19.17 |

**Every single addition made MAE worse**, individually and more so
combined (all 6 together is worse than any single one - the effect
compounds rather than cancels).

### Conclusions

1. **H5 is rejected.** None of the 6 new short-lag/rolling features
   improve on `add_all_candidates`; all degrade it, in the same direction
   for both window modes.
2. **This does not contradict Series 2** - it refines it. `lag_7_sales`
   and `lag_1_fe` were transformative additions to the *raw 15-feature
   baseline*, which had almost no recent-history signal. `add_all_candidates`
   already includes `lag_7_sales`, `rolling_7_sales`, `rolling_14_sales`,
   `lag_1_fe`, and `rolling_7_fe` - the short-term-level and weekly-pattern
   information the 6 new features would add is very likely already
   captured by that existing cluster. Adding more highly-correlated lag
   variants on top introduces redundant dimensions for CatBoost to
   potentially overfit on, without new information to justify it - exactly
   the risk Series 2's own leave-one-out caveat about correlated
   encodings predicted, now observed directly rather than only inferred.
3. **These 6 features are not concluded to be worthless** - only
   redundant *given what add_all_candidates already contains*. A
   different test (e.g. added to a feature set that excludes
   `lag_7_sales`/`rolling_7_sales`/`rolling_14_sales`/`lag_1_fe`/
   `rolling_7_fe`) could show a different result; not run here.
4. **Practical implication**: `add_all_candidates` (38 features) remains
   the best configuration found across all four series to date. The new
   features engineered this session should not be added to the
   production candidate set on current evidence.

## Series 5 — Fixed-Calendar Christmas Period Flag

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series5_christmas_period.py` ·
**Raw data**: `series5_christmas_period_results.csv`

### Hypothesis

Hospitality demand may shift as soon as December begins (works parties,
festive bookings), not only in the narrower council-specific school-break
window already in the model (`is_christmas_break`, ~Dec 22 - Jan 5). **H6**:
a whole-month `is_christmas_period` flag (December 1st-31st, binary)
captures this and improves on `add_all_candidates`.

### Method

Same methodology as Series 4: CatBoost at Series 1/2 hyperparameters, same
10 folds, same two training-window modes, `add_all_candidates` re-run as
the reference point.

### Results

| Feature set | Window | MAE (£) |
|---|---|---|
| `add_all_candidates` (reference) | expanding | 907.77 |
| + `is_christmas_period` | expanding | 919.33 |
| + `is_christmas_period` | 365d | 923.36 |
| `add_all_candidates` (reference) | 365d | 924.09 |

### Conclusions

1. **H6 is rejected.** MAE worsens by +11.56 on the expanding window -
   the same direction and a similar magnitude to Series 4's rejected
   features.
2. **Likely the same redundancy story as Series 4**: `add_all_candidates`
   already includes `is_christmas_break` (a narrower but real December
   signal), `is_school_holiday`, `is_long_weekend`, and the month/
   day-of-year cyclical encodings - a whole-month binary flag most likely
   duplicates signal already present rather than adding new information.
3. **DEPRECATED** on the same terms as Series 4's features: code stays
   in `build_features.py` (see `add_christmas_period_feature`), not to be
   used in a production or recommended feature set on current evidence.
4. This does not settle the broader question of whether a *graded*
   Christmas signal (intensity increasing toward a real peak, not a flat
   window) would do better - see the discussion following this log entry
   for the empirical December pattern examined before attempting that.

## Series 6 — Empirical Graded December Intensity Index

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series6_december_intensity.py` ·
**Raw data**: `series6_december_intensity_results.csv`

### Hypothesis

Series 5's binary flag was rejected, but the real December pattern
(examined before building anything - see discussion above) is a genuine
twin-peak shape: a pre-Christmas peak (~22nd-23rd, up to ~2.0x a typical
weekday), a real trough on the 25th-26th (closure + Boxing Day lull), and
a second, *larger* peak around the 29th-30th (New Year/Hogmanay run-up,
up to ~2.2x). **H7**: a graded, empirically-derived signal that reflects
this actual shape - rather than either a flat binary window (H6, rejected)
or an assumed single-peak decay curve - improves on `add_all_candidates`.

### Method

`december_intensity_index` (see `add_december_intensity_feature` in
`src/features/build_features.py`): weekday-adjusted relative sales
intensity, averaged by day-of-December, computed walk-forward (each
year's index built only from *strictly earlier* years' December data -
with only two Decembers in the data, 2024 has no prior year and is
NaN; only 2025 gets a real, non-leaked value). Verified with 4 unit tests
(`tests/test_december_intensity.py`) including a dedicated leakage guard
that sabotages a year's own December data with an absurd value and
confirms it never appears in that year's own computed index. Same
methodology as Series 4/5 otherwise: CatBoost at Series 1/2
hyperparameters, same 10 folds, same two window modes,
`add_all_candidates` re-run as reference.

### Results

| Feature set | Window | MAE (£) |
|---|---|---|
| `add_all_candidates` (reference) | expanding | 907.77 |
| + `december_intensity_index` | expanding | 916.20 (+8.42) |
| `add_all_candidates` (reference) | 365d | 924.09 |
| + `december_intensity_index` | 365d | 924.61 (+0.52) |

### Conclusions

1. **H7 is rejected, but more narrowly than H5/H6.** The expanding-window
   result is still worse (+8.42), though a smaller degradation than the
   binary flag (+11.56) or any of Series 4's features (+6.41 to +19.17).
   The sliding-365d result is close to neutral (+0.52) - within noise.
2. **Getting the empirical shape right did not overcome the same
   underlying issue as Series 4/5**: `add_all_candidates` already carries
   December-adjacent signal through `is_christmas_break`,
   `is_school_holiday`, `is_long_weekend`, and the month/day-of-year
   cyclical encodings. A more accurately-shaped December feature is still
   competing with, not adding to, information already present.
3. **DEPRECATED** on the same terms as Series 4/5: `add_december_
   intensity_feature` stays in `build_features.py` for reference, marked
   deprecated, not to be used in a production or recommended feature set
   on current evidence.
4. **Worth stating plainly**: three independent attempts to encode a
   December/Christmas signal (Series 4's short lags, Series 5's binary
   flag, Series 6's graded index) have now all failed to beat
   `add_all_candidates`. That configuration's existing seasonal features
   appear to already capture what's extractable from this signal family
   with the current ~2.5 years of data - a further attempt in this same
   direction should have a clear reason to expect a different outcome.

## Series 7 — Named December Phase Flags (Direct Encoding)

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series7_december_phases.py` ·
**Raw data**: `series7_december_phases_results.csv`

### Hypothesis

Series 6's continuous index estimates 31 separate day-of-month values
from just 2 years (2 real observations per day) - a plausible source of
noise. **H8**: grouping December into a small number of named phases,
each directly matching the human-identified real pattern (`is_party_season`
Dec 1-21, `is_pre_christmas_peak` 22-23, `is_christmas_eve_dip` 24,
`is_boxing_day_lull` 26, `is_post_christmas_recovery` 27-28,
`is_hogmanay_peak` 29-30, `is_new_years_eve` 31), is more robust - each
phase is backed by many more training rows than a single calendar day
ever could be, and needs no historical-outcome derivation at all (fixed
calendar rules only, unlike Series 6).

### Method

Same as Series 4/5/6: CatBoost at Series 1/2 hyperparameters, same 10
folds, same two window modes, `add_all_candidates` re-run as reference.
Tested the 7 phase flags added all together, and each individually.
Verified with 10 unit tests (`tests/test_december_phases.py`) confirming
exact boundaries and that no two phases ever overlap on the same day.

### Results (expanding window)

| Feature set added | MAE (£) | Δ vs. `add_all_candidates` |
|---|---|---|
| `add_all_candidates` (reference) | 907.77 | — |
| all 7 phases together | 916.13 | +8.35 |
| + `is_pre_christmas_peak` | 918.24 | +10.47 |
| + `is_hogmanay_peak` | 921.70 | +13.93 |
| + `is_new_years_eve` | 922.34 | +14.57 |
| + `is_boxing_day_lull` | 922.62 | +14.84 |
| + `is_christmas_eve_dip` | 922.80 | +15.03 |
| + `is_post_christmas_recovery` | 923.04 | +15.27 |
| + `is_party_season` | 923.13 | +15.36 |

**Every single addition made MAE worse, again** - no exceptions, across
7 individual flags and the combined set.

### Conclusions

1. **H8 is rejected.** DEPRECATED on the same terms as Series 4/5/6
   (`add_december_phase_features` stays in `build_features.py`, marked
   deprecated, not for production use).
2. **This is now the fourth independent, methodologically distinct
   encoding of December/Christmas signal to fail in a row**: short lags
   (Series 4), a binary window (Series 5), a continuous empirical index
   (Series 6), and now direct named phases matching the exact
   human-identified pattern (Series 7). The consistency itself is now
   the more interesting finding than any single result.
3. **An alternative explanation worth taking seriously before concluding
   "this signal is truly exhausted"**: every one of these four series
   held CatBoost's hyperparameters fixed at the Series 1/2 values
   (`iterations=300, learning_rate=0.03, depth=4`), which were tuned for
   the original 15-feature space, not for a 38-45-feature one. A model
   with limited depth and a fixed tree count may simply lack the
   capacity to exploit a growing feature space well, which could produce
   exactly this kind of small, consistent penalty for *any* added
   feature - useful or not - rather than each December encoding
   specifically being redundant. This has not been tested: no series so
   far has re-tuned hyperparameters on top of an expanded feature set.
   **Recommended next step, not yet run**: repeat Series 1's
   hyperparameter grid search, but on `add_all_candidates` (or
   `add_all_candidates` + the best December encoding) rather than the
   original 15 features, before treating the December-signal question as
   closed.

## Series 8 — Hyperparameter Re-Tune on `add_all_candidates`

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series8_hyperparameter_retune.py` ·
**Raw data**: `series8_hyperparameter_retune_results.csv`

### Hypothesis

Series 7's conclusions raised an alternative explanation for the
4-for-4 pattern of failed December features: CatBoost's hyperparameters
were fixed at the Series 1 values throughout Series 4-7, tuned for the
*original 15-feature* space, not the 38-45-feature space every test since
Series 2 actually uses. **H9**: re-running the Series 1 hyperparameter
grid search on `add_all_candidates` finds a better configuration than
`iterations=300, learning_rate=0.03, depth=4`, indicating the model was
under-capacity rather than the added features being genuinely redundant.

### Method

The exact `CATBOOST_GRID` from `src/experiments/grids.py` (12 combinations:
`iterations∈{100,300}` × `learning_rate∈{0.03,0.05,0.1}` × `depth∈{4,6}`),
on `add_all_candidates`, same 10 folds, same two window modes = 240 fits.

### Results (top 3)

| Hyperparameters | Window | MAE (£) |
|---|---|---|
| `iterations=300, learning_rate=0.03, depth=4` | expanding | **907.77** |
| `iterations=300, learning_rate=0.05, depth=4` | expanding | 919.05 |
| `iterations=300, learning_rate=0.03, depth=6` | expanding | 923.55 |

### Conclusions

1. **H9 is rejected, cleanly.** The original Series 1 hyperparameters
   remain the best of all 12 combinations tested, even when re-searched
   specifically on the larger 38-feature space. Nothing in the grid beats
   £907.77.
2. **This closes off the hyperparameter-capacity explanation for Series
   4-7's pattern.** The model was not under-capacity; adding trees,
   depth, or a different learning rate does not help. The more direct
   explanation stands: the December/Christmas signal these features carry
   is most likely genuinely redundant with what `add_all_candidates`
   already extracts via `is_christmas_break`, `is_school_holiday`,
   `is_long_weekend`, and the month/day-of-year cyclical encodings, not
   an artifact of insufficient model capacity.

## Series 9 — Post-Hoc Multiplicative December Correction

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series9_multiplicative_correction.py` ·
**Raw data**: `series9_raw_predictions.csv`,
`series9_corrected_predictions.csv`, `december_ratio_lookup_pooled.json`

### Hypothesis

Series 4-7 all tried giving the model the December pattern as a training
*feature*, and all failed. **H10**: applying the same known pattern as a
*post-hoc multiplier* on the model's final prediction - `corrected_pred =
raw_pred × ratio[day_of_December]`, using the pooled (both years, not
walk-forward-restricted) ratio table shown earlier in this project -
succeeds where feeding it in as a feature did not, since it does not
require the model to discover or weight the pattern itself. The pooled
(not walk-forward-safe) ratio is used deliberately here, per an explicit
argument that this is a genuine, real, recurring calendar pattern (not
something to treat as unknown for past Decembers) - unlike Series 6,
where the same choice would have been leakage for backtesting a *learned*
feature.

### Method

`add_all_candidates`, CatBoost at Series 1/2/8 hyperparameters (confirmed
optimal in Series 8), expanding window, raw per-day predictions generated
across all 10 folds (585 days, 2025-01-15 to 2026-09-06). December rows
(31 days, all in 2025) multiplied by the pooled ratio table; all other
days left unchanged.

### Results

| | Overall MAE (585 days) | December-only MAE (31 days) |
|---|---|---|
| Raw prediction | 907.42 | 1,378.38 |
| **Corrected (×ratio)** | **1,127.31** | **5,527.95** |

**Overall MAE degrades by 24%; December MAE degrades 4x.** Every single
December day's error got worse after correction, several dramatically
(e.g. 23rd: raw error £2,119 -> corrected error £14,375; 30th: £635 ->
£12,952).

### Conclusion: double-counting, confirmed directly in the numbers

This is not a subtle effect. On 2025-12-23, the model's *raw* prediction
was already £11,973 against an actual of £9,854 - i.e. the model, using
`is_christmas_break`/`is_school_holiday`/the month cyclical encodings
already in `add_all_candidates`, had **already learned to elevate its
December prediction** before any multiplier was applied. Multiplying that
already-elevated prediction by the *same* real-world effect (2.02x)
double-counts it, pushing the corrected figure to £24,228 - roughly 2.5x
the real outcome. The mechanism is confirmed, not just inferred: the raw
predictions are visibly already above a "typical" (non-December) level
throughout the month, before the multiplier is even applied.

**H10 is rejected as implemented.** This does not mean the underlying
idea (a post-hoc multiplicative correction) is unworkable - it means it
cannot be applied on top of a model whose own features already encode
part of the same effect. The natural next test, not yet run: apply the
same multiplier to a version of the model with the redundant
December-adjacent features removed first (`is_christmas_break`,
`is_school_holiday`, and possibly the month cyclical encodings), so the
multiplier has a genuinely "clean" baseline to correct rather than
double-counting on top of an already-adjusted one.

## Series 10 — Multiplicative Correction on a "Clean Slate" Feature Set

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series10_clean_slate_correction.py` ·
**Raw data**: `series10_clean_slate_results.csv`

### Hypothesis

Series 9's multiplicative correction failed because the model's raw
prediction already partially encoded the December effect via
`is_christmas_break`/`is_school_holiday` (among others), so multiplying
by the full real-world ratio double-counted it. **H11**: removing those
two most directly December-specific features first gives the same
multiplier a "clean" baseline to correct, avoiding the double-count.

### Method

`add_all_candidates` minus `is_christmas_break` and `is_school_holiday`
(36 features), same hyperparameters/folds/window as Series 9, same
pooled ratio table applied to December predictions.

### Results

| | Overall MAE | December MAE |
|---|---|---|
| Clean-slate, raw (uncorrected) | 918.58 | 1,378.56 |
| Clean-slate, corrected (×ratio) | 1,131.60 | 5,398.37 |
| *For reference: full add_all_candidates, raw* | *907.42* | *1,378.38* |

### Conclusion

**H11 is rejected - removing 2 features barely moved the outcome.**
Corrected December MAE is still ~4x worse (5,398 vs 1,379), almost
identical in magnitude to Series 9's result with the full feature set
(5,528). Also notable: removing just those two flags made the *raw*
(uncorrected) prediction slightly worse too (918.58 vs 907.42), confirming
they do carry real, non-redundant value on their own - but their removal
barely dented the double-counting problem. **This means the double-count
is not coming primarily from those two explicit flags** - it is far more
likely embedded in the broader, continuous calendar features already in
the *original 15-feature baseline* (`month`, `month_sin`/`cos`,
`day_of_year`, `day_of_year_sin`/`cos`), which give the model a smooth,
whole-year route to learning "this time of year is different" with or
without any December-specific flag. Removing those would gut the model's
general seasonality awareness for the whole year, not just December, and
was not attempted here as a result - a "clean slate" multiplicative
approach does not appear practically achievable without a much larger,
harder-to-justify feature removal.

## Series 11 — Multiplicative Correction Using a Genuinely Out-of-Sample Calibration Ratio

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series11_calibration_ratio.py` ·
**Raw data**: `series11_calibration_ratio_derivation.csv`,
`series11_calibration_corrected_predictions.csv`

### Hypothesis

Series 9's ratio (`actual ÷ typical-weekday-level`) measures the raw
real-world effect, not the model's own residual gap - part of why it
double-counted. **H12**: a ratio computed as `actual ÷ the model's own
prediction`, for a genuinely held-out December, corrects only the
*remaining* gap after the model's existing features, avoiding the
double-count. Critically, this ratio must come from a December the model
never trained on - computing and re-applying it on the *same* December
would be circular (trivially forcing corrected = actual, testing
nothing).

### Method

Trained `add_all_candidates`/CatBoost on **only** data before 2024-12-01
(320 days - everything before the dataset's first December), predicted
the held-out December 2024 (a real out-of-sample test - this exact
period was never part of any of the 10 rolling-origin folds, all of
which start testing in 2025), and computed
`calibration_ratio[day_of_month] = actual / pred` from those genuine
residuals. That ratio (derived entirely from 2024) was then applied to
December 2025's predictions from Series 9's normal 10-fold run.

### Results

| | Overall MAE | December MAE |
|---|---|---|
| Raw (uncorrected) | 907.42 | 1,378.38 |
| Calibration-corrected | 1,046.53 | 4,003.53 |
| *For reference: Series 9's naive ratio-corrected* | *1,127.31* | *5,527.95* |

### Conclusion

**H12 is rejected, but the reasoning behind it was directionally
correct** - this calibration approach degrades results *less severely*
than Series 9's naive ratio (December MAE 4,004 vs 5,528; overall 1,047
vs 1,127). The remaining failure has a clear, specific cause, visible in
`series11_calibration_ratio_derivation.csv`: the held-out December-2024
model had **no prior December in its training data at all** (it is the
very first December in the dataset), so its residual gaps are unusually
large (ratios up to 1.6-1.7x) - representing "how wrong a
December-blind model is," not "how wrong a model with one December of
experience already is." December 2025's predictions, by contrast, come
from a model that already had December 2024 in training - already
partially calibrated - so applying the *larger* 2024-derived correction
over-corrects it, just less severely than the still-larger raw real-world
ratio did. **The core problem is structural, not a matter of which ratio
formula is used**: with only 2.5 years of data, there is exactly one
December a calibration ratio could ever be derived from out-of-sample,
and it does not represent the "experience level" of the model being
corrected. This would need either more years of data (a
genuinely-out-of-sample December where the model already has ≥1 prior
December in training) or a fundamentally different validation design to
test properly - not attempted further here.

## Series 12 — Pooled Calibration Table (Both Years Combined)

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series12_pooled_calibration.py` ·
**Raw data**: `series12_pooled_calibration_table.csv`,
`december_calibration_pooled.json`

### Context

Following up on Series 11: rather than treating "not independently
validated" as disqualifying, this pools both Decembers' real
model-residual ratios (2024's held-out-blind-model gap and 2025's
normal-model gap) into one averaged per-day calibration table, on the
explicit basis that this is a genuine recurring pattern and every future
year will have all prior years' data available to build it from - the
same reasoning already applied in Series 9. Note this is a *different*
concern from Series 9's leakage question: computing a ratio from a
December and re-applying it to that *same* December is a literal
algebraic identity (`pred × (actual/pred) = actual`, exactly, regardless
of any real pattern) - not a leakage judgement call, just uninformative
by construction. Pooling both years and reporting the table is
meaningful; "testing" it against either of the two years it was built
from is not, and is shown below only for completeness.

### Result: the two years disagree about how much correction is needed - and not by a little

| Day | 2024 ratio (blind model) | 2025 ratio (experienced model) | Pooled |
|---|---|---|---|
| 1 | 1.329 | 0.952 | 1.140 |
| 11 | 1.628 | 1.031 | 1.329 |
| 23 | 1.663 | 0.823 | 1.243 |
| 30 | 1.518 | 1.054 | 1.286 |

2025's ratios are consistently, substantially lower than 2024's across
nearly every day - several (days 9, 15, 21, 24) are even *below* 1.0,
meaning the experienced model slightly over-predicted those specific
days. Applying the pooled table back to each year for reference only
(not a fair test - both years contributed to the table):

| | Raw MAE | Pooled-corrected MAE |
|---|---|---|
| 2024 (blind model) | 3,058.58 | 1,500.14 (better) |
| 2025 (experienced model) | 1,378.38 | 2,001.76 (worse) |

### Conclusion

**Pooling helps the blind-model year and hurts the experienced-model
year** - because they need genuinely different amounts of correction, not
because the pooling was done incorrectly. This matters for what to do
next: every real future December will be forecast by a model that
already has *at least* one prior December in training (2024's
zero-experience scenario can only ever happen once, for the very first
December in this dataset's history, and will never recur). That means
**2024's data point is not representative of any future deployment
scenario** - pooling it in dilutes the estimate with a regime that is
now permanently in the past. The single most relevant real evidence for
what a correction should look like going forward is 2025's ratio alone,
not the pooled average - though with only one such data point, it still
cannot be validated without a third, independent December, which does
not yet exist. **Recommendation**: if a multiplicative correction is
revisited once a third December of real data exists, weight recent years'
ratios far more than the 2024 baseline, or discard 2024's data point
entirely rather than pooling it in unweighted.

## Series 13 — Training Directly on MAE Loss

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series13_mae_loss.py` ·
**Raw data**: `series13_mae_loss_grid_results.csv`

### Hypothesis

Every CatBoost run in this log (Series 1-12) used CatBoost's default
regression loss (RMSE/squared error) despite every result being judged
on MAE. **H13**: training with `loss_function='MAE'` directly - matching
the objective to the evaluation metric - improves on £907.77.

### Method

`add_all_candidates`, expanding window, all 10 folds. First a direct
swap (Series 1/2/8's winning hyperparameters, `loss_function='MAE'`
added, nothing else changed), then a dedicated 18-combination grid
(`iterations∈{100,300,500}` × `learning_rate∈{0.03,0.05,0.1}` ×
`depth∈{4,6}`, all with `loss_function='MAE'`) - since MAE's gradient is
constant-magnitude rather than shrinking near the optimum (unlike squared
error), the RMSE-tuned hyperparameters were not assumed to transfer, and
weren't given a chance to be blamed for a bad result.

### Results

| Configuration | MAE (£) |
|---|---|
| RMSE loss, Series 1/2/8 hyperparameters (reference) | **907.77** |
| MAE loss, same hyperparameters | 940.50 |
| MAE loss, best of 18-combination dedicated grid (`iterations=500, learning_rate=0.05, depth=4`) | 915.65 |

### Conclusion

**H13 is rejected.** Even after a dedicated hyperparameter search
specifically for the MAE loss function (not just reusing RMSE-tuned
settings), the best MAE-loss configuration still falls short of the
RMSE-loss winner by £7.88. "Optimise for the metric you're evaluated on"
is sound general advice, but does not hold here - RMSE's loss surface
evidently produces splits that generalise better on this data even though
the final evaluation is in MAE terms. Not adopted; `add_all_candidates`
trained with default (RMSE) loss remains the best configuration found
across all thirteen series.

## Series 14 — Log1p Target Transform

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series14_log1p_target.py` ·
**Raw data**: `series14_log_transform_grid_results.csv`,
`series14_best_config_per_fold.csv`

### Hypothesis

Daily sales are strictly non-negative and right-skewed (occasional very
high days, a floor near zero on the 4 closure days), a shape CatBoost's
default squared-error loss does not assume. **H14**: training on
`log1p(total_sales)` and inverting with `expm1` at prediction time - a
standard transform for this kind of target - improves on £907.77.
`log1p`/`expm1` (not plain `log`/`exp`) specifically to keep the 4 known
zero-sales closure days finite rather than producing `-inf`.

### Method

`add_all_candidates`, expanding window, all 10 folds, CatBoost. First a
direct swap (Series 1/2/8's winning hyperparameters, target replaced by
`log1p(total_sales)`, predictions inverted with `expm1`, nothing else
changed). Then, following the same logic as Series 13 - a transformed
loss surface has no reason to favour the same hyperparameters as the
untransformed one - a dedicated 18-combination grid
(`iterations∈{100,300,500}` × `learning_rate∈{0.03,0.05,0.1}` ×
`depth∈{4,6}`) trained and evaluated entirely in log space. Finally, the
best log-space configuration found was re-checked on the sliding-365-day
window to confirm the expanding window is still preferred under the
transform too.

### Results

| Configuration | MAE (£) | MAE std (£) |
|---|---|---|
| Raw scale, Series 1/2/8 hyperparameters (reference) | **907.77** | 182.08 |
| Log1p, same hyperparameters (`iterations=300, learning_rate=0.03, depth=4`) | 919.55 | 148.93 |
| Log1p, best of 18-combination dedicated grid (`iterations=300, learning_rate=0.05, depth=4`) | **892.38** | **153.52** |
| Log1p, best config, sliding-365 window (vs. expanding) | 933.56 | 159.63 |

Full 18-combination grid (mean MAE, £, ascending):

| iterations | learning_rate | depth | MAE mean | MAE std |
|---|---|---|---|---|
| 300 | 0.05 | 4 | 892.38 | 153.52 |
| 500 | 0.03 | 4 | 906.66 | 164.00 |
| 500 | 0.05 | 4 | 909.52 | 168.25 |
| 300 | 0.03 | 4 | 919.55 | 148.93 |
| 100 | 0.1 | 4 | 920.06 | 131.46 |
| ... (remaining 13 combinations, all worse, down to 1316.21 for `iterations=100, learning_rate=0.03, depth=4`) | | | | |

Per-fold breakdown of the winning configuration (`iterations=300,
learning_rate=0.05, depth=4`, expanding window):

| fold_id | MAE | RMSE | MAPE % | Bias |
|---|---|---|---|---|
| 0 | 931.13 | 1168.84 | 12.61 | -239.00 |
| 1 | 880.97 | 1151.76 | 10.52 | -106.16 |
| 2 | 1164.23 | 1626.91 | 12.48 | -155.06 |
| 3 | 760.81 | 1030.98 | 10.61 | -73.45 |
| 4 | 829.43 | 1034.77 | 10.90 | 67.36 |
| 5 | 1164.17 | 1571.09 | 16.52 | 229.58 |
| 6 | 789.68 | 1035.82 | 11.49 | -135.42 |
| 7 | 815.31 | 1019.41 | 11.94 | 292.20 |
| 8 | 846.90 | 1131.40 | 10.84 | 83.84 |
| 9 | 741.15 | 980.54 | 9.90 | -7.83 |

No single fold dominates the improvement - the gain is spread across
most of the 10 folds, not an artefact of one lucky window.

### Conclusion

**H14 is confirmed.** Unlike every attempt since Series 2 (hyperparameter
retune, three multiplicative Christmas/December corrections, pooled
calibration, MAE loss), the log1p transform - with hyperparameters
re-tuned specifically for log space, not reused from the raw-scale
winner - produces a genuine improvement: **£892.38 MAE, down £15.39
(1.7%) from £907.77**, and also lower variance across folds (MAE std
153.52 vs. 182.08, a 16% reduction). As with every other configuration
tested in this log, the expanding window beats the sliding-365 window
(892.38 vs. 933.56), so that preference is unaffected by the transform.

This becomes the new leading candidate configuration:
**`add_all_candidates` (38 features), CatBoost with `loss_function`
default (RMSE), target `log1p(total_sales)` inverted with `expm1`,
`iterations=300, learning_rate=0.05, depth=4`, expanding window.**

## Series 15 — CatBoost + Prophet Ensemble

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series15_prophet_ensemble.py` ·
**Raw data**: `series15_prophet_predictions.csv`,
`series15_ensemble_data.csv`

### Hypothesis

Series 1 found Prophet the third-best model overall and, notably, the
model with the **lowest fold-to-fold variance** of anything tested -
often a sign a model is making a different *kind* of error than the
leader, which is exactly the condition under which even a naive average
of two models' predictions beats either alone. **H15**: blending the
Series 14 CatBoost (log1p) winner with Prophet improves on £892.38.

### Method

Prophet trained with Series 1's winning hyperparameters
(`changepoint_prior_scale=0.01, seasonality_prior_scale=1.0,
seasonality_mode=additive`) but with the `add_all_candidates` (38)
feature set as regressors - the same feature set behind every winning
result since Series 2/4, for parity with the CatBoost side - on the
same 10 expanding-window folds. Sanity check: this reproduces Series 4's
logged Prophet + `add_all_candidates` result almost exactly (954.45 here
vs. 954.23 logged; the ~£0.2 difference is ordinary Prophet fit noise).

Three tests, each more rigorous than the last:
1. A **naive 50/50 average** of the CatBoost(log1p) and Prophet
   predictions, per day.
2. A **blend-weight sweep** (weight on CatBoost from 0.0 to 1.0 in steps
   of 0.05) evaluated on the same 10 folds, to check whether some
   non-1:1 mix does better than either the 50/50 blend or pure CatBoost.
3. An **honest out-of-sample check** on that sweep: split the 10 folds
   in half (folds 0-4 / 5-9), pick the best blend weight on one half,
   apply that fixed weight to the *other* half, and compare to pure
   CatBoost on that same held-out half - since picking a weight and
   grading it on the identical data it was picked from is the same
   kind of circularity flagged in Series 11's calibration-ratio critique.
4. The residual correlation between the two models' errors, to check
   whether they are actually making different mistakes (the premise the
   whole idea rests on) or largely the same ones.

### Results

| Configuration | MAE (£) | MAE std (£) |
|---|---|---|
| CatBoost log1p (Series 14 winner, reference) | **892.38** | 153.52 |
| Prophet, `add_all_candidates` (standalone) | 954.23 | 168.19 |
| Ensemble: 50/50 CatBoost(log1p) + Prophet | 903.90 | 154.72 |
| Ensemble: best weight found by sweep on all 10 folds (90% CatBoost / 10% Prophet) | 891.19 | 153.55 |

Blend-weight sweep (selected points, weight = share on CatBoost log1p):

| Weight on CatBoost | 0.0 | 0.5 | 0.8 | 0.9 | 0.95 | 1.0 |
|---|---|---|---|---|---|---|
| MAE (£) | 954.23 | 903.90 | 892.02 | **891.19** | 891.48 | 892.38 |

Honest out-of-sample check (weight tuned on one half of the folds,
evaluated on the other):

| Weight tuned on | Weight chosen | Evaluated on | Blend MAE (£) | Pure CatBoost MAE (£) on same held-out half |
|---|---|---|---|---|
| Folds 0-4 | 0.80 | Folds 5-9 | 876.17 | **871.44** |
| Folds 5-9 | 1.00 (i.e. no Prophet at all) | Folds 0-4 | 912.38 | 912.38 |

Residual correlation between CatBoost(log1p) and Prophet errors: **0.892**
(strongly positive - the two models are wrong on the same days, in the
same direction, most of the time).

### Conclusion

**H15 is rejected.** The naive 50/50 blend is worse than CatBoost alone
(903.90 vs. 892.38), consistent with Prophet's meaningfully higher
standalone error (954.23) simply dragging the stronger model down. A
full weight sweep does find a marginally better point (90/10, £891.19),
but that £1.19 "improvement" evaporates under honest cross-validation:
picking the best weight on one half of the folds and applying it to the
other half performs *worse* than pure CatBoost in one split and exactly
matches it (weight collapses to 1.0, i.e. no Prophet) in the other. This
is the same circularity trap flagged in Series 11 - the on-sample "best
weight" is not a real, generalisable finding, just noise the search
happened to fit.

The root cause is visible directly in the residual correlation: **0.892**.
Series 1's "lowest fold-to-fold variance" observation about Prophet was
real, but low variance in aggregate MAE across folds is not the same
thing as making *different daily errors* than CatBoost - and it's the
latter, not the former, that an ensemble needs to gain anything.
Empirically, Prophet and CatBoost(log1p) are wrong about the same days
in the same direction almost 90% of the time (likely because both are
driven by the same underlying calendar/regressor signal - `add_all_candidates`
- rather than genuinely independent modelling approaches). Not adopted;
the Series 14 log1p CatBoost model remains the best configuration found
across all fifteen series.

## Series 16 — Seed Bagging, and the Discovery of Seed Selection Bias

**Run**: 2026-09-10 · **Script**: `src/experiments/run_series16_seed_bagging.py` ·
**Raw data**: `series16_bagging_raw_predictions.csv`,
`series16_reference_seed_predictions.csv`

### Hypothesis

**H16 (original)**: averaging predictions from the Series 14 log1p
CatBoost winner trained with multiple random seeds (bagging) reduces
variance further and improves on £892.38 - a standard, low-risk
variance-reduction technique, distinct from the Series 15 cross-model
ensemble (whose failure was explained by high error correlation between
*different* algorithms; same-model seed bagging draws on training-time
randomness instead, which has no such correlation problem by
construction).

### Method

25 additional CatBoost models trained per fold (seeds 0-24, everything
else identical to the Series 14 winner: `add_all_candidates`,
`iterations=300, learning_rate=0.05, depth=4`, log1p/expm1, expanding
window), predictions averaged per day at increasing bag sizes (2, 3, 5,
10, 15, 20, 25 seeds) to check convergence. The same 26-seed sweep
(0-24 plus 42) was then repeated for the **raw-scale reference model**
(Series 1/2/8 winner) specifically to test a question the bagging result
raised: is seed 42 - the fixed seed used in every fit in this log since
Series 1 - an unusually favourable draw for this dataset, and if so,
does it favour one configuration over another unequally?

### Results

**Bagging convergence (log1p model):**

| Bag size (seeds) | MAE (£) | MAE std (£) |
|---|---|---|
| 1 (seed=42, Series 14 reported result) | **892.38** | 153.52 |
| 2 | 916.34 | 162.05 |
| 5 | 913.34 | 156.57 |
| 10 | 912.62 | 153.93 |
| 15 | 910.95 | 151.11 |
| 20 | 909.59 | 152.95 |
| 25 | 910.78 | 152.82 |

Bagging converges to ~£910-914 and stays there - it does not approach,
let alone beat, the £892.38 figure.

**Per-seed MAE distribution, 25 independent seeds (0-24), identical data
and hyperparameters, log1p model:**

min=901.33, mean=922.01, median=921.38, max=944.16, std=9.51 (range
across seeds: £42.83). **Seed 42 (892.38) falls below every one of
these 25 values** - not within the distribution, below its minimum.

**Same check for the raw-scale reference model**, 25 seeds (0-24) plus
seed 42 itself:

min=909.34, mean=917.70, max=926.66, std=4.63 (range: £17.32). **Seed 42
(907.77) is again below every one of the 25** - also an outlier low, but
by a smaller margin relative to its own distribution than log1p's.

**Seed-noise-controlled comparison** (25-seed bagged average for both
configurations, seed 42 excluded from both so neither benefits from a
lucky draw):

| Configuration | Bagged MAE (25 seeds, £) |
|---|---|
| Raw-scale reference | 914.12 |
| Log1p transform | **910.78** |
| Gap | **3.35** |

Compare to the originally reported, single-lucky-seed gap of
907.77 − 892.38 = **15.39**.

### Conclusion

**H16 is rejected as stated** - bagging does not improve on £892.38; it
makes the point estimate worse (~£910-914), because £892.38 was never a
representative expectation of the model's performance to begin with.

**The more important finding is a methodological one.** Seed 42 - used
as the fixed `random_state` in every single fit across this entire log,
Series 1 through 15, purely because it is the common sklearn/CatBoost
convention, never because it was searched for - happens to sit at the
extreme favourable tail of the seed distribution for this specific set
of 10 rolling-origin test folds, for *both* the raw-scale and log1p
configurations. Critically, it does not favour them equally: it boosts
log1p by roughly triple what it boosts the raw-scale model relative to
each one's own 25-seed average (≈30 vs. ≈10). This means single-seed
comparisons between configurations - exactly the method used for every
comparison in Series 1-15 - can systematically mis-state the size of a
genuine effect, even without anyone deliberately searching over seeds.
It is the same underlying failure mode as Series 11's calibration-ratio
circularity and Series 15's blend-weight overfitting: information about
the exact evaluation folds "leaking" into a reported number through an
unexamined researcher degree of freedom.

Under the honest, seed-noise-controlled comparison, **log1p still beats
raw-scale (910.78 vs. 914.12) - the qualitative conclusion of Series 14
survives** - but the effect shrinks from £15.39 to **£3.35**, roughly a
fifth of what was reported. This does not overturn any series whose
effect size was large relative to the ~£10-15 seed-noise band identified
here (Series 2/4's £65.26 gain from `add_all_candidates`, for instance,
is safely outside this noise floor), but it means the exact margins
reported for closer results in this log (Series 8's hyperparameter
retune, Series 13's MAE-loss gap, and Series 14 itself) should be read
as directionally suggestive rather than precisely accurate, unless
re-verified with a multi-seed average.

**Recommendation for production**: use the 15-25-seed bagged ensemble of
the log1p configuration, not the single seed=42 model. Its point
estimate (~£910) is a small, honest step back from the headline £892.38,
but it is a genuine, defensible expectation of future performance rather
than one arbitrary favourable roll of the dice - and it removes exactly
the kind of seed-lottery risk this series just demonstrated is real.
Not logged as a new "winning" MAE figure; logged instead as a correction
to how every prior figure in this document should be interpreted.

## Series 17 — The Final Model Against the Venue's Manual Forecast

**Run**: 2026-09-12 (Section A), 2026-09-13 (Sections B-C) · **Script**:
`src/experiments/run_series17_manual_forecast_comparison.py` ·
**Raw data**: `series17_manual_comparison_matched_days.csv`,
`series17_summary.csv`, `catboost_winner_predictions.csv`,
`wage_comparison_ai_vs_manual_vs_actual.csv`

### Context

Series 1-16 compare models with each other. None of them answers the
question that matters commercially: is the final model better than what
the venue actually does today, which is a manager's manual sales
forecast? This series makes that comparison, for the Series 16
recommended model (the 25-seed bagged log1p CatBoost).

All metrics here are **pooled over individual days**, not averaged per
fold as elsewhere in this log, because the comparison runs over a set of
matched calendar days rather than over folds. This is why the final model
reads £919 here but ~£910 in Series 16: same model, different averaging
and a different set of days.

### Method

**Section A** (the headline figure): the Series 1 CatBoost winner's
out-of-fold predictions (15 production features, sliding 365-day window,
Series 1's own folds) are joined to labour data - closure days have no
labour records and drop out, leaving 596 days - and then intersected with
the final model's test days: **569 matched days, 2025-01-15 to
2026-08-23**. The manual forecast, the Series 1 winner and the final model
are all scored on exactly those days.

The 569-day window is the overlap of two different fold setups (Series
1's, and the one used by Series 2-16), not a window chosen on its own
merits. Section B exists to check it does not flatter the result.

**Section B** (new measurement): the same final-vs-manual comparison over
every final-model test day that has labour data, not only the days that
overlap Series 1.

**Section C** (new measurement): the original portfolio XGBoost (15
features, its production hyperparameters from `train_xgboost.py`)
retrained on the same rolling-origin folds and scored on the same 569
days. Until now the portfolio model was only ever scored on a single
train/test split (MAE £1,050 over 339 days), which is not comparable
with a backtested figure.

### Results

| Section | Model | Days | MAE (£) | MAPE | vs. manual |
|---|---|---|---|---|---|
| A | Manual forecast | 569 | 1,085.97 | 14.86% | — |
| A | Series 1 CatBoost winner | 569 | 967.11 | 12.95% | 10.94% better |
| A | **Final bagged log1p CatBoost** | 569 | **919.17** | **11.97%** | **15.36% better** |
| B | Manual forecast | 583 | 1,078.14 | 14.76% | — |
| B | Final bagged log1p CatBoost | 583 | 912.63 | 11.89% | 15.35% better |
| C | Portfolio XGBoost, same folds | 569 | 1,034.82 | 13.94% | 4.71% better |

Section C is the one figure here sensitive to library version: with
XGBoost 3.4.1 instead of 3.2.0 it reads £1,022.17 (13.78% MAPE, 5.87%
better than manual). The table uses the pinned version.

### Conclusion

**The final model is about 15% more accurate than the venue's manual
forecast, and that figure is robust**: widening the comparison from the
569-day overlap to all 583 available test days changes it from 15.36% to
15.35%.

Section C puts the whole project on one protocol. Scored the same way, the
original portfolio XGBoost is only **about 5-6%** better than the manual
forecast (4.7% or 5.9%, depending on XGBoost version), the Series 1
CatBoost winner 10.9%, and the final model 15.4% - so the rolling-origin
experimentation programme lifted the model's real-world advantage over
the process it would replace from roughly 5% to 15%. Stated like-for-like,
the model improved from **about £1,020-1,035 to £919 MAE**; the £1,050
single-split figure should not be set against £919, since the two were
measured differently.

## Business Impact — What Forecast Error Costs in Labour

**Run**: 2026-09-09 (Sections 1-2), 2026-09-13 (Section 3) · **Script**:
`src/experiments/run_business_impact.py` · **Raw data**:
`business_impact_manual_planning_by_year.csv`, `business_impact_summary.csv`

### Context

Accuracy only matters to the venue through staffing: rotas are planned
from the sales forecast, so an over-forecast day means paying for staff
the demand never needed. This analysis translates forecast error into
wages and hours.

**Why forecast-side ratios, not actual wages.** Actual wages already
reflect on-the-day correction - staff sent home once a quiet shift is
obvious. Any measure built on actual wages therefore mixes up two
different things: the forecast's own error, and how well the venue
corrected for it afterwards. The more aggressively the venue corrects,
the less a forecast's mistakes show up in what was actually paid.
Instead, a wage-per-£-sale and hours-per-£-sale ratio
is derived purely from planned figures (what the planning process itself
believes it needs), then applied to the sales over-forecast, which cannot
be corrected after the fact: a customer who never arrives cannot be sent
home. See `calculate_implied_labour_metrics` in
`src/models/evaluate_human_forecast.py`.

### Method

**Section 1**: the manual forecast's own labour planning, by year.

**Section 2** (reproduced from the original analysis): for the Series 1
CatBoost winner over its 596 days, the wages implied by staffing to each
forecast; the overstaffing cost (planned wages attributable to days a
forecast exceeded actual sales); and a deployment simulation. The
**correction factor** (actual wages ÷ planned wages, per day) captures how
far below plan the venue historically ran after on-the-day correction.
Applying it to the model's plan simulates "staff to the model, then keep
correcting on site as usual" - the realistic scenario, since a better
forecast would not stop managers correcting in real time.

**Section 3** (new measurement): the same simulation for the final model
on Series 17's 569 matched days.

### Results

**Section 1 — manual forecast, by year**

| | 2024 | 2025 | 2026 (to Sep) | Total |
|---|---|---|---|---|
| Days | 364 | 362 | 248 | 974 |
| Over-forecast sales (£) | 191,384 | 252,460 | 206,227 | 650,070 |
| Planned wage per £1 of sales | 0.3324 | 0.3282 | 0.3446 | 0.3338 |
| Implied wages planned for sales that never happened (£) | 63,617 | 82,869 | 71,068 | 217,009 |
| Over-forecast wages net of correction (£) | 58,297 | 49,707 | 33,564 | 141,567 |
| Days actual hours exceeded the planned rota | 98 | 100 | 75 | 273 |

The two money rows measure different things and should not be subtracted
from each other. The implied figure is the wage budget the manual
forecast allocated to demand that never materialised. The net-of-
correction figure is how far actual wages came in under plan, summed over
under-budget days - and on-the-day correction makes it *larger*, not
smaller, which is exactly why it cannot be read as waste on its own.

**Sections 2-3 — wage simulations**

| | Series 1 CatBoost winner | Final bagged log1p CatBoost |
|---|---|---|
| Days | 596 | 569 |
| Manual forecast overstaffing cost (£) | 151,576 | 145,483 |
| Model overstaffing cost (£) | 119,122 | 81,338 |
| Over-forecast days, manual → model | 376 → 346 | 362 → 283 |
| Simulated savings vs. actual wages spent (£) | 36,147 | 90,194 |
| Annualised (£ per year) | 22,137 | 57,858 |
| As % of 2025 wage bill | 2.12% | 5.55% |
| As % of 2025 revenue | 0.70% | 1.83% |

(2025 context: revenue £3.17m, wage bill £1.04m, wages 32.9% of revenue.)

### Conclusion

Staffing to the final model and correcting on site as usual is simulated
to save **about £58k a year, 5.5% of the wage bill**, against the £22k the
earlier Series 1 model would have saved. Most of the gain comes from
over-forecasting far less often (283 days rather than 362).

**Assumptions and limits - read these before quoting the figure:**

- **Understaffing is not costed.** Fewer over-forecast days means more
  under-forecast days, where the real cost is service quality and lost
  sales, not wages. This simulation counts wage savings only, so it
  overstates net benefit by however much that understaffing costs.
- **Staffing is assumed proportional to forecast sales**, at the planned
  wage-per-sale ratio of this window.
- **On-the-day correction is assumed unchanged** when staffing to the
  model. Managers might correct differently when trusting a better
  forecast, in either direction.
- The figures are a simulation over historical days, not a measured
  outcome of deploying the model.

## Appendix: full candidate pool reference

**In the current production model (15):** `forecast_sales`, `month_sin`,
`day_of_week`, `day_of_year_cos`, `day_of_year_sin`, `day_of_year`,
`month_cos`, `month`, `day_of_week_sin`, `is_bank_holiday`,
`days_to_bank_holiday`, `days_since_payday`, `is_payday_window_pm3`,
`is_long_weekend`, `is_heavy_rain`.

**Engineered but not yet tested in the model, tested in Series 2 (23):**
see the group table above.

**Engineered 2026-09-10, tested, DEPRECATED (15):**
- `lag_1_sales`, `lag_2_sales`, `lag_3_sales`, `lag_2_fe`, `lag_3_fe`,
  `rolling_28_sales` (Series 4) — built to fill the short-lag/rolling gap
  Series 2's own conclusions identified, but every one degraded
  `add_all_candidates` when added individually or all together.
- `is_christmas_period` (Series 5) — fixed-calendar December 1st-31st
  binary flag; degraded `add_all_candidates`.
- `december_intensity_index` (Series 6) — empirically-derived, graded,
  walk-forward-safe December signal reflecting the real twin-peak shape;
  still degraded `add_all_candidates`, though more narrowly than the
  binary version.
- `is_party_season`, `is_pre_christmas_peak`, `is_christmas_eve_dip`,
  `is_boxing_day_lull`, `is_post_christmas_recovery`, `is_hogmanay_peak`,
  `is_new_years_eve` (Series 7, `DECEMBER_PHASES`) — named phases
  directly encoding the real pattern; every one degraded
  `add_all_candidates` too, individually and combined.

All fifteen are still computed in `build_features.py` (each marked
`# DEPRECATED` at its assignment line, not deleted, for reference/
reproducibility) but must not be added to any production or recommended
feature set on current evidence. See Series 7's conclusions for an open
question about whether this consistent pattern reflects real redundancy
or CatBoost's hyperparameters (fixed since Series 1, tuned for a smaller
feature space) being unable to exploit any added feature well.

**Never a candidate (leakage or non-feature intermediate):**
`forecast_error`, `payday`, `prev_payday`, `next_payday`.

**Not yet engineered at all**: explicit fixed-calendar New Year/Easter/
summer period flags (Christmas has now been tried twice, see above; these
three siblings from the original idea have not), and any labour-derived
feature (wage-to-sales or hours-to-sales ratio) — the latter is excluded
on a structural ground, not just an unbuilt one: at inference time for a
future date, no labour forecast exists yet, since the rota is planned
*from* this system's sales forecast, not before it.
