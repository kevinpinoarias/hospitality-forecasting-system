# Forecast-log versions v1.13 and v1.14: built, not registered (14 November 2026)

Status: the code, the tests and the dry-run fits are done. **Nothing is registered**: no amendment is in the log's `PREREGISTRATION.md`, `models/versions.json` has no v1.13 or v1.14, and no model file of either exists
outside a scratch folder. Registration is a separate step (the amendment first, then the fits), and it is Kevin's decision. This file holds what was built, how it was checked, the amendment text ready to paste, and the
steps to take.

## What was built

| Version | What it is | Code |
|---|---|---|
| **v1.13**, the year-end fix | v1.12's five hydro-arm pieces, copied unchanged, plus two sets of the same five pieces used on two stretches only: the **companion** (refitted without the ten recent-sales inputs) on 26 to 31 December, and the **masked** model (ten inputs missing on the training rows of 1 to 8 January, and missing on those forecast days) on 1 to 8 January. Differs from v1.12 on 14 days a year. | `src/forecast_log/v1_13.py`, routing in `src/forecast_log/predict.py` (`with_year_end_specialists`, `route_post(prefix, masked)`) |
| **v1.14**, the year-ago input | v1.13 plus `sales_364` (the sales on the same weekday 364 days earlier), appended last to the Monday-to-Thursday (hydro), Friday and Saturday pieces (`log_hydro_thu`, `fri_A_all`, `sat_A_lags`). The Sunday piece, the raw Saturday fallback and the ten specialist pieces are v1.13's files, copied unchanged. | `src/forecast_log/v1_14.py`, `src/features/year_ago.py` |

Scoring: `src/forecast_log/score_forecasts.py` gains **Q7** (v1.13 hydro against v1.12 hydro: all days, 26 to 31 December, 1 to 8 January, all other days) and **Q8** (v1.14 hydro against v1.13 hydro: all days, Monday to
Saturday, Saturdays, Sundays), each appearing once its rows exist. Arm names in the log: `hydro_v1.13` and `hydro_v1.14`. Tests: `tests/test_v1_13_v1_14.py` (14 tests; the whole suite is 392 passing).

Two decisions inside v1.14 that were not tested as such, to be confirmed or changed before the fit:

1. **The raw Saturday fallback (`raw`) does not get the input.** Series 93 tested the form "Monday to Saturday" with the three pieces that serve those days when their inputs are known; `raw` is used only when the Saturday
   lags are not known (a Saturday more than about two weeks beyond the last known sales day), and was not part of it. In a Monday run's two weeks the Saturday of the second week may use it. If `raw` should carry the input,
   that is a small untested addition.
2. **The specialist pieces do not get the input** (Series 93 composed the package that way; the input does not move late December). On 26 December to 8 January v1.14 forecasts what v1.13 does.

## How it was checked (nothing here writes to the log)

* **Same recipe as the backtest.** The companion and masked input lists, the mask window values and `sales_364` were compared with the experiment code they come from (`run_series82_mask_windows`, `run_series85_companion_model`,
  `run_series90_year_end_candidates`): the input lists are identical in content and order for all four backtest pieces, the masked training columns are identical to Series 82's `j1_8` on the 16 masked training rows,
  and `sales_364` is identical to Series 90's on all 1,006 training rows (655 have a value).
* **Dry-run fits** into a scratch folder (15 pieces for each version, 25 seeds each, the frozen features file, training rows to 1 November 2026).
* **Served through the log's own `forecast_days`** for the logged 2 and 9 November runs, beside v1.12: v1.13 equals v1.12 on every one of those days (the windows are far away); v1.14 differs from v1.13 on Monday to Saturday
  and not on Sundays. A past December (served from the sales known on 14 December 2025, five weeks ahead) shows the companion on 26 to 31 December and the masked pieces on 1 to 8 January, v1.12 elsewhere.
* **Not done:** the real fits, the registration, the back-run rows. A backtest of the built code was not repeated; its numbers are the series' (below).

## What the backtest says to expect (Series 85 to 94, 585 days, one December and one January)

* **Q7, v1.13 against v1.12:** about -5.9 pounds a day over all 585 days served with the stand-in the log uses today (Series 87), which is about 250 pounds a day on the 14 days the two differ on; about -11 with the better
  stand-in (Series 92: -10.95 on 100 seeds, -10.65 on fresh seeds). The companion rests on six days in the backtest (one December), the masked model on eight.
* **Q8, v1.14 against v1.13:** about -8 to -10 pounds a day over the 585 days (Series 93: -8.03 on 100 seeds, -9.73 on fresh seeds; Series 94: -5.16 on 25 seeds), measured **served with the better stand-in**, not with
  today's; uneven (Sundays get worse, which is why the Sunday piece is left out; early 2025 and spring 2026 worse; winter and summer better).

## Draft amendment for `PREREGISTRATION.md` (to be committed to the log repository BEFORE either fit)

> **Amendment, [date], before the v1.13 fit.** Two changes: a fourth and a fifth model version, `v1.13` and `v1.14`, are added and run beside `v1`, `v1.11` and `v1.12`. Where they conflict with the text below, this
> amendment wins. Written before the fits, as rule 1 requires.
>
> 1. **What v1.13 is.** The hydro arm of `v1.12` with two sets of specialist pieces used on two stretches of the year only; everything else is `v1.12`'s own files, copied unchanged. On **26 to 31 December** the
>    *companion* forecasts: each of the five pieces (`log_hydro_thu`, `fri_A_all`, `sat_A_lags`, `raw`, `sun_B`) refitted without the ten recent-sales inputs (`lag_7_sales`, `rolling_7_sales`, `rolling_14_sales`,
>    `lag_1_fe`, `lag_7_fe`, `rolling_7_fe`, and for the Friday and Saturday pieces `lag_14_sales`, `lag_21_sales`, `same_wd_mean4`, `same_wd_max4`), which carry Christmas into the days after it. On **1 to 8 January**
>    the *masked* model forecasts: each piece refitted with the same ten inputs set to missing on every training row dated 1 to 8 January, and with them set to missing on the forecast days of that stretch. The
>    specialist pieces are saved beside the main ones as `comp_<piece>` and `mask_<piece>`. The training data, cut-off (rows to 1 November 2026, the frozen features file), the weekend options and the Saturday and
>    Sunday correction history are `v1`'s. Only a hydro arm is issued; its rows have model `hydro_v1.13` and model version `v1.13`. It is fitted once, fingerprinted in `models/versions.json` and never refitted.
>    `v1.13` forecasts exactly what `v1.12` does on every day outside 26 December to 8 January.
> 2. **What v1.14 is.** The hydro arm of `v1.13` with one new input, `sales_364` (the actual sales on the same weekday 364 days earlier, a missing value where there is none: 655 of the 1,006 training rows have one,
>    none before 2025), appended last to the inputs of the Monday-to-Thursday piece with the hydro flag (`log_hydro_thu`), the Friday piece (`fri_A_all`) and the Saturday piece (`sat_A_lags`). The Sunday piece, the raw
>    Saturday fallback and the ten specialist pieces are `v1.13`'s own files, copied unchanged, so `v1.14` forecasts what `v1.13` does on Sundays and on 26 December to 8 January, apart from the weekend correction's
>    reaction to the other days. Same data, cut-off and correction history. Rows: model `hydro_v1.14`, model version `v1.14`. Fitted once, fingerprinted, never refitted.
> 3. **Run beside, not instead.** All versions are issued in the same weekly run, for the same days, with the same inputs, weather and stand-in for the manager's number. Nothing about the stand-in changes with these
>    versions.
> 4. **How they are scored.** **Q7:** the `v1.13` hydro arm against the `v1.12` hydro arm, **Q8:** the `v1.14` hydro arm against the `v1.13` hydro arm, each on matched on-time days with the no-manager variant, paired as in
>    section 7 (error of the first minus error of the second, in pounds, 95% moving-block bootstrap of 7-day blocks). Q7 is reported on all days, on 26 to 31 December, on 1 to 8 January and on all other days (where the
>    forecasts are identical before the weekend correction); the decision is the paired change over the 14 days of 26 December to 8 January, **descriptive** after the first window (a reading is possible once the actuals
>    to 8 January 2027 are in the log) and again after December 2027; fourteen days a year have little power and no formal threshold is set. Q8 is reported on all days,
>    Monday to Saturday, Saturdays and Sundays; descriptive until **52 weeks** of forecasts have actuals, and then the formal reading is the paired change over the Monday-to-Saturday days, "better" meaning an interval that
>    excludes zero in favour of `v1.14`. Replayed rows are scored apart, as the amendment of 9 November says.
> 5. **Back-run.** Both versions are fitted on data to 1 November 2026, so they are clean for every Monday run since 2 November and are replayed for each (`replay_run.py --backfill v1.13 v1.14`), the replay having to
>    reproduce the logged `v1` numbers to the penny. The windows of `v1.13` lie weeks ahead, so its replayed rows equal `v1.12`'s.
> 6. **Expected from the backtest.** `v1.13`: about 5.9 pounds a day better than `v1.12` over the 585 days of Series 87 served with today's stand-in (about 250 pounds a day on the 14 days it differs on; -10.95 and
>    -10.65 pounds a day on 100 and fresh seeds with a better stand-in, Series 92). `v1.14`: about 8 to 10 pounds a day better than `v1.13` over the 585 days (Series 93: -8.03 and -9.73), measured with the better
>    stand-in and not with the log's current one.
> 7. **Disclosure.** Both were chosen after Series 79 to 94 in `forecasting-product`'s `docs/EXPERIMENT_LOG.md`, which look at the same 585 backtest days as every earlier experiment: one December and one January. The
>    window 1 to 8 January was the best of six windows tried on those days (a selection effect); the companion's gain on 26 to 31 December rests on six days of one December; the specialists replicated on 100 and fresh
>    seeds, but the days do not change. The year-ago input's gain is uneven: Sundays get worse (+14 and +11 a day when the Sunday piece has it, so it is left out), the early-2025 folds (little year-ago history) and
>    the spring of 2026 get worse, winter and summer improve; its exact form was tested against five alternatives (Series 94) and nothing beat the plain same weekday. The Series 94 test and the gains above use a
>    better stand-in for the manager's number than the one the log uses, which is not holiday-aware (it can take 25 December's closed-day figures as the stand-in for the Wednesday or Thursday before it); how `v1.13`
>    and `v1.14` behave with the log's stand-in around Christmas is part of what Q7 will show.
> 8. **When it starts.** Issued from the first Monday run after registration. The first days `v1.13` changes are 26 and 27 December 2026 (the Monday 14 December run) and 28 December to 3 January (the Monday 21 December
>    run).

## Steps to register (when Kevin decides; the order matters)

1. Commit and push this build (the code is the thing the fits record as `code_commit`); update the runner's checkout to that commit.
2. Copy the correction history: `data/training/v1.12_correction_history.csv` to `v1.13_correction_history.csv` and `v1.14_correction_history.csv` (same cut-off, so the same file; the fit checks the hash). Both the development and the runner `data/training/` folders.
3. Commit the amendment above (with its date, and Kevin's edits to the Q7 and Q8 readings) to `forecasting-log` `PREREGISTRATION.md` and push.
4. In the runner folder (clean checkout, the frozen features file there): `python -m src.forecast_log.v1_13 fit --log-repo D:\Projectos\DRG\forecasting-log --training-features <runner>\data\training\v1_engineered_features.csv`,
   commit the log's `models/versions.json`, then the same with `v1_14`. The fit of v1.14 copies v1.13's pieces from `models/v1.13` and checks them against the registered fingerprints.
5. `config/forecast_log.local.yaml` (untracked): `also_run: [v1.11, v1.12, v1.13, v1.14]`.
6. `python -m src.forecast_log.replay_run --backfill v1.13 v1.14` without `--write` to see what it would add, then with `--write`.
7. A dry run of the Monday run (`run_forecasts --trigger monday --dry-run`) before the next scheduled one (Monday 16 November, 12:30).

A fit is irreversible in the log's terms (a refit is a new version), which is why none was made. The dry-run fits in the scratch folder are not the registered models and are not kept.
