# What the missing manager number costs the post model, by distance (Series 43)

Rules written on 1 November 2026, **before any code or run**. Status: draft until Kevin approves it; the commit that records it is the proof of order.

## 1. Why this test

Every forecast the weekly log issues is made without the manager's number (it arrives after the day). Series 18 measured what losing it costs, by how far ahead the forecast is, but for the final model on the original school calendar, before the post model existed. The log issues the post model. Series 35 has the post model with and without the number on 32 new days only, too few to say anything (interval about ±£200).

This test repeats Series 18 with the post model, so the cost can be read by distance for the model the log actually uses.

## 2. Protocol: Series 18's, unchanged where possible

- **Origins and days.** Series 18's: every rolling-origin fold whose origin has 180 days of data after it (8 origins), each model refitted on every day before the origin, every day of the 180 that is in the experiment dataset scored (1,425 day-scores). Day 1 is the origin itself (the first day after the last known one).
- **Inputs after the origin, as the API sees them.** History filled with weekday-and-season medians (Series 18's seasonal fill, the one the API and the log use; the recursive fill is not repeated). Weather: what happened for the first 16 days (Series 18's stand-in for a live forecast), weekday-and-season medians after that. Calendar features from the verified school calendar.
- **With the number**: the manager's real forecast on every day. **Without**: the weekday-and-season median of the manager's past forecasts (the stand-in the API and the log use).
- **Distance buckets.** Series 18's: days 1-7, 8-16, 17-30, 31-60, 61-90, 91-180, and all 180.

## 3. What is scored

| Column | Model | Inputs |
|---|---|---|
| Manager alone | the manager's forecast | - |
| Log model, every real input | the final model on the log target, verified calendar (seeds 0 to 24) | real history, real weather, real number (the ceiling) |
| Log model, with the number | same | as section 2, with the number |
| Log model, without it | same | as section 2, without it |
| Post model, every real input | the post model (below) | real everything (the ceiling) |
| Post model, with the number | same | as section 2, with the number |
| Post model, without it | same | as section 2, without it |

**The post model** is the weekly log's, without the Hydro flag: Monday to Thursday the log model; Friday the `A_all` piece if the sales of the same weekday 1 to 4 weeks earlier are all known, otherwise the log model; Saturday the `A_lags` piece if the sales 2 and 3 weeks earlier are known, otherwise the raw-pound piece; Sunday the `B` piece; then the Saturday and Sunday correction, learnt only from the post model's own out-of-sample weekend errors before the origin. Each piece is refitted on the days before each origin (seeds 0 to 24). At a distance the Friday and Saturday pieces fall back to the plain models, exactly as they will in the log. The hydro arm is left out (it adds a Thursday flag and needs the Hydro list for every day).

## 4. Sanity checks (both must pass before any table is produced)

1. The log model with every real input reproduces the stored verified-calendar log-model backtest (`series19_21_cal2`, `log1p_final`) to £0.05 on the first 60 days after each origin (each origin's test block in that backtest).
2. The post model with every real input reproduces the stored post-model backtest (the package's final step in `run_package_validation`, verified calendar) to £0.05 on the same days.

The script checks both first and, if either fails, prints only the check and stops. No result table exists until both pass. (The lesson of Series 42, section 10.)

## 5. Figures, fixed in advance

- MAE (£ a day) for every column and every bucket, the same table as Series 18.
- **The cost of losing the number** = MAE without it minus MAE with it, for the log model and the post model, per bucket, with a 95% moving-block bootstrap interval (block of 10 days, days ordered by origin and date).
- **Primary figures**: the post model's cost at days 8 to 16 (the log's next-week distance) and over all 180 days.
- **One comparison**: post model's cost minus log model's cost over all 180 days. The post model "loses less without the number" only if that interval lies wholly below zero; "loses more" only if wholly above; otherwise "the same cost".

This is a measurement, not a promotion test: nothing changes in the log whatever the result.

## 6. Limits and expectation, stated in advance

- The 180-day windows of neighbouring origins overlap (an origin every 60 days), so most days are scored up to three times and the intervals are narrower than the evidence really is.
- The real weather for the first 16 days makes the short buckets a little optimistic (kept for comparability with Series 18; the log uses average weather).
- The short buckets are small (about 56 to 65 day-scores each), so their costs are noisy.
- The 39 new days are not included (the models stop at 6 September).
- **Expectation**: the post model's cost about the same as the log model's (around +£70 at days 8 to 16 and +£100 overall in Series 18), because every piece of the post model takes the number as an input and the weekend pieces fall back to the plain models beyond one or two weeks. The post model itself should be somewhat better than the log model at days 1 to 14, where its weekend pieces are used.
