# Training for the day the manager's number is missing (Series 42)

Rules written on 1 November 2026, **before any code or run**. Status: draft until Kevin approves it; the commit that records it is the proof of order.

## 1. Why this test

The weekly export reaches us after the day, and Rotaready cannot export a forecast for future dates. So every forecast the weekly log issues is a **no-manager** forecast. The log model was trained with the manager's real number as an input (`forecast_sales`) and is served, in the log, with a stand-in in its place: the weekday-and-season median of the manager's past forecasts.
Series 18 measured what that costs: about £100 a day (MAE 978 with the manager's number, 1,078 without). The model has never been trained for the situation it is now always in.

## 2. Question and hypothesis

**H42**: a log model trained for the no-manager situation (stand-in in training, or no manager input at all) has lower error, when served without the manager's number, than the log model as deployed.

## 3. Arms

All three are the log model (the final model on the log1p target: CatBoost, the 38 features, the final model's settings, seeds 0 to 24; replicate seeds 100 to 124), refitted on the days before each fold. No weekend package, no Hydro flag: the question is about the manager input only.

| Arm | Trained with | Served with |
|---|---|---|
| **N0 as deployed** | the manager's real number | the stand-in |
| **N1 stand-in training** | the stand-in in place of the real number | the stand-in |
| **N2 input dropped** | no `forecast_sales` column (37 features) | no `forecast_sales` column |

The stand-in is the one the live API and the log use (`seasonal_medians` on `forecast_sales`). For a training row it is computed from the **other training rows only** (the row itself excluded). A training row that cannot get one is dropped from **all three** arms, so they train on identical rows. The three `_fe` features (lagged forecast errors) are left as they are in every arm.

## 4. Protocol

- The project's 585-day rolling-origin backtest (the same folds as Series 1 to 41). Each arm is refitted on the days before each fold's test block.
- Scored as the log serves a week: for every issue Monday whose target week lies inside a fold's test block, the last known sales day is the Sunday before the issue, history after it is filled with weekday-and-season medians, weather is the historical-average fallback, nothing from the target week is read (`served_rows`, the function the log uses).
- **Primary scoring unit**: next-week days (NW, Monday+7 to Monday+13, 8 to 14 days after the last known day). **Secondary**: this-week days (TW, 1 to 7 days after).
- **Sanity check** (must pass before any result is read): N0 served with the real manager inputs and real history reproduces the stored final-model backtest predictions to £0.05.
- **Context only, no rule**: the 39 days of 7 September to 15 October 2026, scored the same way by models fitted on rows before 7 September, if the holdout files load. 39 days cannot confirm a gain smaller than about £270 a day.

## 5. Statistic

Mean absolute error of the arm minus that of N0, on the same days (negative is a gain), with a 95% moving-block bootstrap interval (block of 10 days, as in the earlier series). N1 and N2 are judged separately, each against N0.

## 6. Rules (an arm is a candidate only if all four hold)

1. *Gain*: on NW days the interval's upper end is below zero, on **both** seed sets.
2. *Size*: the pooled NW point estimate is at least £15 a day better than N0 (about 1.5% of the typical error); anything smaller is "no practical change" however narrow the interval.
3. *Consistency*: better than N0 in at least 7 of the 10 folds, on NW days.
4. *No harm*: the interval for TW days is not wholly above zero, and the change on the busiest quarter of days (by actual sales) is not above +£25.

## 7. Reading, fixed in advance

- All four rules hold: that arm is **a candidate for the log's `v2`**, added beside `v1` as a labelled extra arm. Not promoted, and not a replacement for the frozen arms.
- Rule 1 fails for both N1 and N2: **no detectable change**; the log keeps `v1` as it is.
- Rule 1 holds but 2, 3 or 4 fails: recorded as unclear; no action.
- If N1 and N2 are within £5 of each other, they are one finding ("the stand-in carries no information beyond the other inputs") and N2, the simpler one, is the one named.

## 8. Limits and expectation, stated in advance

- A fresh hypothesis on days used by every earlier series (a 42nd look at the same 585 days); the earlier-series warnings about reusing them apply.
- The stand-in is a calendar median, so it adds little to what the calendar features already say. **Expectation**: N1 and N2 both close to N0, a gain of £0 to £25 a day, the interval likely to include zero. A gain near the £100 of Series 18 would be a surprise, because that gap is mostly the real number's own information, which no training can restore.
- What would settle it for real is a source for the manager's number before the day, which does not exist. This test only asks whether we are leaving accuracy on the table in the situation we cannot change.

## 9. How the rules are read in the code (fixed before the full run)

Written when the script and its tests were finished, after a rehearsal and before the full run. Nothing in sections 1 to 8 is changed.

- **Days scored.** Every week that lies wholly inside a fold's 60-day test block, for days that have an actual. NW: the last known day is the Sunday eight days before the week's Monday; TW: the Sunday before it. Weeks cut by a block's edge are not scored.
- **"Pooled".** The change for a day is averaged over the two seed sets. Rule 1 is judged on each seed set separately; rule 2, rule 3 and the busiest-quarter part of rule 4 use the pooled change; the TW part of rule 4 holds if neither seed set's interval lies wholly above zero.
- **Busiest quarter.** The 25% of NW days with the highest actual sales.
- **Folds (rule 3).** Folds whose pooled mean NW change is below zero, out of the 10 folds.
- **The stand-in.** For a training row, the median over the other rows of that fold's training window (the row itself left out); when serving, over all days known at the last known day. Training rows that cannot get a stand-in would be dropped from all three arms; none is expected.
- **Sanity check.** N0 (seed set A) predicting the test days with their real inputs and real history must match `models/final_model/backtest_predictions.csv` to £0.05; if it does not, no result is read.
- **Rehearsal.** One fold (the last) with 2 seeds per set was run to test the machinery. Its figures were seen. They are part of the 585 days and no choice was made from them; the full run is the test.
- **Not built yet.** The 39-day context check on the new data (section 4) is a separate step, run after the main result.
