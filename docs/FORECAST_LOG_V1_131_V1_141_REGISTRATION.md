# Forecast-log versions v1.131 and v1.141: built and registered (14 November 2026)

Status: **registered on 14 November 2026.** The amendment is in the log's `PREREGISTRATION.md` (log commit `35c1765`), v1.131 is registered (`89d83cb`) and v1.141 is registered (`fb3b008`), both from the runner's clean
checkout at code commit `728f148` (copies of `v1.13`'s and `v1.14`'s 15 pieces, each fingerprint equal to its parent's). The back-run rows are in the log (`09bd38c`: 28 rows for each version, one replay for each of the Monday
runs of 2 and 9 November, each reproducing the logged v1 numbers to the penny before anything was written). The runner's untracked `config/forecast_log.local.yaml` now has `also_run: [v1.11, v1.12, v1.13, v1.14, v1.131, v1.141]`,
so the Monday 16 November run issues all seven arms (checked by building that run in memory: 140 rows, nothing written). On the 28 replayed days (2 to 22 November) v1.131 equals v1.13 and v1.141 equals v1.14 to the penny:
the two stand-ins agree on ordinary days, and the Christmas week is weeks away. The sections below were written before the registration and are kept as they were; the steps at the end have been carried out.

## What they are

| Version | What it is | Code |
|---|---|---|
| **v1.131** | `v1.13` served with the hybrid stand-in (S6). **No refit**: all 15 fitted pieces are byte-for-byte copies of `models/v1.13` (the fingerprints are checked against `v1.13`'s registration). | `src/forecast_log/v1_131_141.py` (registration), `src/forecast_log/standin.py` (the stand-in), `Models.standin` and `forecast_days` in `src/forecast_log/predict.py` |
| **v1.141** | `v1.14` served with the same stand-in; 15 pieces copied from `models/v1.14`. | same |

The stand-in is the number given to the models in the place of the manager's forecast for the day, which reaches us only after the day (Series 42-43). Every version up to `v1.14` is given **S0**, the median of the manager's
past numbers on the same weekday within ten days of the same day of the year, closed days included, so 25 December's zero can become the number for the Wednesday or Thursday before it (the live path forecasts about
£6,200 for Thursday 24 December 2026 with S0, and about £8,700 with S6). The hybrid **S6** (Series 89) is:

* **24 December to 2 January:** the manager's number on the same date one year earlier (365 days), when there is one;
* **every other day, and any day with no year-ago number:** **S1**, the same median with the closed days (manager number 0) left out, and 0 on 25 December and 1 January.

It uses the numbers of the days known when the run is issued and nothing after them (tested). It replaces the manager's-number input of the target rows only, as the backtest did (Series 87-89); the history inputs built from
the stand-in on later days are unchanged. The log's column `manager_forecast_used` for a `hydro_v1.131` or `hydro_v1.141` row is the number **that arm** was given, so it differs from the main arm's on the days S6 differs.

Because the pieces are the parents', `v1.131` differs from `v1.13` (and `v1.141` from `v1.14`) on every day the two stand-ins differ: mostly the Christmas week, small differences in the weeks around the other closed days,
and almost nothing the rest of the year. Seen on the served path from the sales known on 1 November 2026: the two forecasts are identical on 16 and 21 November, 22 December and 4 January, and differ on 17 and 23 December and 24 December to 2 January.

Scoring: `score_forecasts.py` gains **Q9** (v1.131 hydro against v1.13 hydro) and **Q10** (v1.141 hydro against v1.14 hydro), each reported on all days, 24 December to 2 January and all other days, appearing once its rows
exist. Tests: `tests/test_v1_131_141.py` (8 tests) and a changed fixture in `tests/test_replay_run.py`; the whole suite is 400 passing.

## What the backtest says to expect (Series 88 and 89, 585 days, one December and one January)

* **The stand-in alone is worth about 54 pounds a day** on the unmasked package (Series 89: average daily miss 892 with S6 against 946 with S0, 100 and fresh seeds alike; the real manager's number would give 854). Over the
  24 holiday-window days: 1,441 against 2,755. S1 alone gives 903 and S3 (the date a year ago inside the holiday window) 904; the hybrid beat both by about 11 pounds a day on both seed sets.
* **For `v1.131` against `v1.13`** the same comparison with the year-end pieces on both sides: about -59 pounds a day over the 585 days (S6 with the specialist windows -64.9 against today's -5.9 for the windows alone,
  both measured against today's 945.7).
* **For `v1.141` against `v1.14`** the stand-in effect with the year-ago input was **not measured on its own** (Series 93 and 94 measured v1.14 only with S6). It is expected to be of the same order as for `v1.131`; that
  expectation is a guess and is stated as one.

## Disclosure (goes into the amendment)

All of it rests on one December and one January (24 holiday-window days), the same 585 days as every earlier experiment. The hybrid was designed after seeing where S1 and S3 failed on those same days, so its extra 11
pounds over the better of them is probably optimistic; the 43 pounds each of S1 and S3 gave over S0 replicated on 175 more seeds but not on more days. The real test is December 2026. Two known weak spots: the same-date
rule puts the wrong weekday on a weekend day (in 2026 Boxing Day and 2 January are Saturdays, the dates last year were Fridays, so the stand-in for them is a Friday's number; the backtest had the same one-day shift, but only once),
and the stand-in is built from the log's history, which is fresher than the backtest's but not otherwise different.

## Draft amendment for `PREREGISTRATION.md` (to be committed to the log repository BEFORE the registration)

> **Amendment, [date], before the v1.131 and v1.141 registrations.** Two further model versions, `v1.131` and `v1.141`, are added and run beside the others. Where this conflicts with the text below, this amendment wins.
> Written before the registrations, as rule 1 requires.
>
> 1. **What they are.** `v1.131` is `v1.13` and `v1.141` is `v1.14`, **with no refit**: every fitted piece is a copy of the parent's own file, checked against the parent's registered fingerprint; same training data and
>    cut-off, same weekend options and correction history. The only difference is the **stand-in for the manager's number**, the figure given to the models where the manager's forecast would be, because it reaches us
>    only after the day. Every earlier version is given the weekday-and-season median of the manager's past numbers (S0). These two are given the hybrid S6: from 24 December to 2 January the manager's number on the
>    same date 365 days earlier, when there is one; every other day, and any day without a year-ago number, the same median with the closed days (manager number 0) left out and 0 on 25 December and 1 January. It is
>    computed from the manager's numbers of the days known when the run is issued. It replaces the manager's-number input of the forecast days only. The registry marks the versions with `"standin": "S6"`.
> 2. **Rows.** Models `hydro_v1.131` and `hydro_v1.141`, model versions `v1.131` and `v1.141`; `manager_forecast_used` is the number that arm was given. A hydro arm only. Fitted (copied) once, fingerprinted in
>    `models/versions.json`, never changed: a different stand-in is a new version.
> 3. **Run beside, not instead.** Issued in the same weekly run, for the same days, with the same inputs and weather as the other versions. The main `v1` arms and `v1.11` to `v1.14` keep S0, so the comparison is clean.
> 4. **How they are scored.** **Q9:** the `v1.131` hydro arm against the `v1.13` hydro arm; **Q10:** the `v1.141` hydro arm against the `v1.14` hydro arm; each on matched on-time days with the no-manager variant,
>    paired as in section 7 (error of the first minus error of the second, in pounds, 95% moving-block bootstrap of 7-day blocks). Each is reported on all days, on 24 December to 2 January and on all other days.
>    **Descriptive** until the Christmas week has passed (readable once the actuals to 2 January 2027 are in the log), and again after December 2027; ten days a year have little power and no formal threshold is set.
>    Replayed rows are scored apart, as the amendment of 9 November says.
> 5. **Back-run.** Both are clean for every Monday run since 2 November 2026 (the pieces are the parents', fitted on data to 1 November) and are replayed for each (`replay_run.py --backfill v1.131 v1.141`); the replay
>    must reproduce the logged `v1` numbers to the penny. Until the Christmas week enters a run's horizon, the replayed rows differ from the parents' only where the closed-day-aware median differs from S0.
> 6. **Expected from the backtest.** Series 89 (585 days, 100 and fresh seeds): the hybrid alone cuts the average daily miss from 946 to 892 pounds, the specialist windows add about 11; for `v1.131` against `v1.13`
>    about 59 pounds a day over the 585 days. For `v1.141` against `v1.14` the effect was not measured on its own and is expected to be of the same order; this is a guess.
> 7. **Disclosure.** As above: one December and one January, the hybrid designed after seeing the failures of S1 and S3 on those same days (its extra 11 pounds is probably optimistic), the weekday shift of the
>    year-ago date on weekend days in 2026, the live history being different from the backtest's. The real test is December 2026.
> 8. **When it starts.** Issued from the first Monday run after registration, and replayed from Monday 2 November. The days it changes most are 24 December 2026 to 2 January 2027 (the Monday 14 December run is the
>    first to reach them, with a horizon of about two weeks, and the Monday 21 December run the rest); on other days the two stand-ins differ only slightly.

## Steps to register (when Kevin gives the go-ahead; the order matters)

1. Commit and push this build to the private remote (the code is what the registrations record as `code_commit`); update the runner's checkout to that commit.
2. Copy the correction histories in the runner (and here): `v1.13_correction_history.csv` to `v1.131_correction_history.csv` and `v1.14_correction_history.csv` to `v1.141_correction_history.csv` (same cut-off; the registration checks the hash).
3. Commit the amendment above (with its date) to `forecasting-log` `PREREGISTRATION.md` and push.
4. In the runner folder (clean checkout): `python -m src.forecast_log.v1_131_141 register --version v1.131 --log-repo <path-to>\forecasting-log`, commit the log's `models/versions.json`, then the same with `--version v1.141`.
5. `config/forecast_log.local.yaml` (untracked): `also_run: [v1.11, v1.12, v1.13, v1.14, v1.131, v1.141]`.
6. `python -m src.forecast_log.replay_run --backfill v1.131 v1.141` without `--write` to see what it would add (the standing rule: every registered model runs from Monday 2 November 2026), then with `--write`.
7. A dry run of the Monday run before the next scheduled one (Monday 16 November, 12:30).
