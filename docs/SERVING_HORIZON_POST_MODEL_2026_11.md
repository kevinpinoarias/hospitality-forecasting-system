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

## 7. How the code reads the rules, and one fix to the sanity check (before any table existed)

- **Fix.** The first full run stopped at the gate: the post-model check failed (largest difference £42.6) while the log-model check passed. Every piece matched its stored backtest to £0.002; the difference was only on four Fridays of the first origin (31 January to 21 February 2025), because the data has no sales on Friday 24 January 2025 and the script's routing treated a missing day among the 1-to-4-week lags as "not known" and switched those Fridays to the plain model. The validated backtest routes by weekday alone. The "every real input" columns now route by weekday alone (as the validated backtest does); the "with" and "without" columns keep the log's availability rule (as the log itself would). With that, the check passes (largest difference £0.0022). The gate worked: no table was written or read before both checks passed.
- **Correction history.** The first origin (15 January 2025) has no earlier post-model errors, so its Saturday and Sunday forecasts are not corrected, as in the validated backtest.
- **Order of commits.** The script was run before it was committed (the rules were committed first, in `bcd6ac4`); it is committed with the results.

## 8. Result (run 1 November 2026)

8 origins (15 January 2025 to 11 March 2026), 1,425 day-scores, the same days and buckets as Series 18. Sanity checks: both pass (£0.0022).

MAE by distance, £ a day:

| Days ahead | Scores | Manager alone | Log model, real inputs | Log model, with the number | Log model, without it | Post model, real inputs | Post model, with the number | Post model, without it |
|---|---|---|---|---|---|---|---|---|
| 1-7 | 56 | 1,104 | 832 | 863 | 888 | 794 | 821 | 834 |
| 8-16 | 65 | 949 | 913 | 966 | 1,036 | 804 | 879 | 928 |
| 17-30 | 104 | 1,038 | 914 | 1,027 | 1,077 | 867 | 986 | 1,037 |
| 31-60 | 240 | 1,118 | 978 | 1,013 | 1,199 | 944 | 974 | 1,155 |
| 61-90 | 240 | 1,022 | 868 | 911 | 953 | 832 | 892 | 920 |
| 91-180 | 720 | 1,108 | 947 | 988 | 1,096 | 932 | 963 | 1,077 |
| **1-180** | **1,425** | **1,082** | **930** | **976** | **1,077** | **901** | **945** | **1,045** |

Cost of losing the number (without minus with), £ a day, 95% block-bootstrap interval:

| Days ahead | Log model | Post model | Post cost minus log cost |
|---|---|---|---|
| 1-7 | +24.8 (−66.7, +151.7) | +12.5 (−86.2, +148.1) | −12.2 (−41.2, +15.4) |
| **8-16** | +70.7 (−18.1, +164.5) | **+48.2 (−39.9, +150.4)** | −22.5 (−57.9, +9.0) |
| 17-30 | +50.4 (−58.8, +152.4) | +50.9 (−58.5, +154.2) | +0.5 (−12.1, +11.1) |
| 31-60 | +186.0 (+32.1, +428.5) | +181.2 (+26.3, +425.3) | −4.8 (−23.2, +11.6) |
| 61-90 | +42.3 (−21.7, +101.6) | +28.1 (−33.4, +84.4) | −14.2 (−32.1, −0.1) |
| 91-180 | +108.0 (+22.9, +218.5) | +114.1 (+29.8, +223.3) | +6.1 (−2.5, +14.3) |
| **1-180** | +100.9 (+45.7, +169.4) | **+99.3 (+44.7, +168.3)** | **−1.6 (−7.9, +4.5)** |

**Reading, by the rule in section 5.** Post cost minus log cost over all 180 days is −1.6 (−7.9, +4.5): **the same cost.** Losing the number costs the post model about £99 a day over 180 days and about £48 a day at 8 to 16 days ahead, with wide intervals in the short buckets.

**What the numbers say.**
1. The expectation in section 8 held: the post model loses about what the log model loses (£99 against £101 over 180 days), because every one of its pieces takes the number.
2. The post model is better than the log model in every bucket and every condition (without the number, over 180 days, 1,045 against 1,077; at 8 to 16 days, 928 against 1,036).
3. For the weekly log's next-week forecasts (8 to 14 days ahead, without the number), the post model scores 928 against the manager's 949: level with the manager, where the log model was about £90 worse. At 1 to 7 days it is far better than the manager (834 against 1,104).
4. The log model on the verified calendar reproduces Series 18's original-calendar figures closely (without the number over 180 days 1,077 against 1,078), as expected from Series 29.

**Limits.** Overlapping windows (most days scored up to three times) make the intervals too narrow; real weather for the first 16 days makes the short buckets a little optimistic (the log uses average weather); the short buckets have 56 and 65 day-scores; the hydro arm and the 39 new days are not included.
