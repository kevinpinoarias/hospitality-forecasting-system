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

## 10. Correction to the sanity check (found on the first full run)

The full run finished on 1 November 2026. Its sanity check **failed** (largest difference £173.9 against £0.05) because sections 4 and 9 named the wrong reference: `models/final_model/backtest_predictions.csv` is the **deployed model on the original school calendar**, while the arms here are fitted on this branch's verified-calendar features. A fresh refit with the repository's own `BaggedCatBoost.fit` reproduces N0 exactly (difference 0.000) and also differs from that file by up to £118, so the fault is the reference, not the arms.

The reference the check should have named is the stored log-model backtest on the verified calendar (`reports/experiments/series19_21_cal2`, variant `log1p_final`, seeds 0 to 24). Against it N0 differs by at most **£0.0022** on all 585 days (tolerance £0.05): **pass**. The script now uses that reference.

**What was seen when.** The script printed its summary table in the same command that reported the failed check, and I read the table before I understood why the check had failed, which is the opposite of what section 9 says ("no result is read"). The cause was found and the correct check passes without changing any arm, rule or threshold, so the rules' outcome is not affected, but the order should be on record.

## 11. Result (run 1 November 2026, after the rules in `b9c71cd`, the machinery in `b1e3c5e` and the correction in `6644259`)

Three arms, 10 folds, two seed sets (0 to 24 and 100 to 124), 524 next-week (NW) days and the same 524 this-week (TW) days. Sanity check (section 10): N0 against the stored verified-calendar log-model backtest, largest difference £0.0022, **pass**.

Mean absolute error, £ a day (seed set A / seed set B):

| | NW | TW |
|---|---|---|
| N0 as deployed | 1,042.1 / 1,037.0 | 1,018.8 / 1,016.9 |
| N1 stand-in training | 1,041.8 / 1,034.7 | 1,037.5 / 1,036.4 |
| N2 input dropped | 1,077.0 / 1,073.1 | 1,073.6 / 1,073.9 |

Change against N0 (negative is a gain), 95% moving-block bootstrap interval:

| | NW set A | NW set B | NW pooled | TW set A | TW set B | Folds better (NW) | Busiest quarter (NW) |
|---|---|---|---|---|---|---|---|
| **N1** | −0.3 (−78.6, +51.3) | −2.3 (−79.9, +49.8) | **−1.3** (−79.5, +50.3) | +18.7 (−37.8, +59.8) | +19.5 (−33.0, +57.6) | 3 of 10 | +73.3 |
| **N2** | +34.9 (−30.9, +83.9) | +36.2 (−34.9, +88.2) | **+35.5** (−32.6, +85.7) | +54.8 (+1.9, +100.5) | +57.0 (+2.7, +103.5) | 4 of 10 | +182.9 |

**Verdict against the rules.** Neither arm is a candidate.

| Rule | N1 | N2 |
|---|---|---|
| 1 gain, upper end below zero on both seed sets | no | no |
| 2 pooled gain of at least £15 | no (−1.3) | no (+35.5, a loss) |
| 3 better in at least 7 of 10 folds | no (3) | no (4) |
| 4 no harm | no (busiest quarter +73 against a limit of +25) | no (TW interval wholly above zero; busiest quarter +183) |

N1 minus N2 is −36.8, so they are not one finding.

**Reading, as fixed in section 7.** Rule 1 fails for both arms: **no detectable change; the log keeps `v1`.** No `v2` arm is added on this evidence.

**What the numbers say beyond the rules.**
1. Training with the stand-in (N1) leaves next-week error where it was (−1.3) and is a little worse on this-week days (+19, interval includes zero). It does not repair the mismatch between training with the number and serving without it.
2. Dropping the input (N2) is worse, not better: +35 on next-week days (interval includes zero) and +55 on this-week days (both intervals lie above zero), and +183 on the busiest quarter. The real number in training is not a liability even when it is replaced by a median in service.
3. The expectation in section 8 held for N1 (close to N0, interval including zero) and not for N2 (a loss, where a small gain had been expected). The intervals are about ±65, so gains under roughly £50 a day could not have been seen; "no detectable change" is not "no change".
4. Not investigated: why N2 is worse (the column is removed from the model's tie-breaking and splits, but a loss of £35 is more than the £2 column-order effects of Series 19), and what next-week error would be with the real number supplied.

**Limits.** A 42nd look at the same 585 days; weeks cut by a block's edge are not scored (524 of 585 days); the 39-day context check of section 4 has not been run.
