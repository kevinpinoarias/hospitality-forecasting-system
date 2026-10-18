# Events exploration (Series 36) — football and concerts, October 2026

**Written:** Sunday 18 October 2026. **Branch:** `Product-branch` (private). **Status:** exploratory; nothing here is in the model, the
feature pipeline or the API. Code: `src/features/events.py`, `src/experiments/run_series36_event_checks.py`, `tests/test_events.py`.
Curated lists: `config/events/football_matches.csv`, `config/events/concert_days.csv`.

## 1. Why

The holdout test on the new venue days (`docs/HOLDOUT_2026_10_PREREGISTRATION.md`, Series 35) left several days with large misses that
neither the deployed model nor the manager anticipated. Looking those days up afterwards found a likely cause on four of five: a concert
or race in Glasgow (27 Sep Megan Moroney at the O2 Academy; 30 Sep and 1 Oct Westlife at the OVO Hydro; 4 Oct the Great Scottish Run). This
series asks whether such events explain the misses **across the old data**, not only the days we happened to look at.

**Hindsight warning.** Those five days were chosen after seeing their errors. The checks below use the 585-day backtest, which is separate
from them, but the idea itself came from them. The 39 new days are therefore *context*, not clean evidence, for any event feature.

## 2. The lists, and what is wrong with them

| List | Source | Coverage | Known gaps |
|---|---|---|---|
| Celtic and Rangers matches, all competitions, 2024-01 to 2026-10 (285 rows) | Wikipedia season pages (club seasons, Scottish Cup, League Cup), read on 16 Oct 2026 through a page summariser | Celtic nearly complete; Rangers league complete, cups added from the cup pages | Rangers' European home legs are partly missing (for example the March 2025 Europa League home leg); kick-off times are not stored; the Scotland team's matches are not collected |
| Shows at the OVO Hydro, O2 Academy and SEC Armadillo, 2024-01 to 2026-10 (677 venue-days) | Songkick venue gigography pages (Hydro 2443719, O2 Academy 38722, Armadillo 56688), read page by page on 16 Oct 2026 | Hydro 276, O2 Academy 307, Armadillo 94 show-days | Hydro entries are about 6–12% below Concert Archives' counts (123 against 131 in 2024, 107 against 121 in 2025); non-music events are absent (WWE Clash at the Castle, 15 Jun 2024, and PFL Europe 3, 28 Sep 2024, are not on Songkick); other venues (Royal Concert Hall, Barrowland, SWG3, Glasgow Green, Hampden) are not collected |
| Great Scottish Run | Traffic Scotland and charity pages | 5 Oct 2025 and 4 Oct 2026 (2024: 6 Oct) | Only two dates in the 585 days' span: no effect can be learned |

Capacities used for the capacity sum: OVO Hydro 14,300 (the venue's own figure), SEC Armadillo 3,000, O2 Academy 2,550 (standing). A capacity
is not attendance.

Spot checks against independent listings agreed (Westlife 30 Sep and 1 Oct; Megan Moroney 27 and 28 Sep; Kevin Bridges 3 and 4 Oct).
Songkick entries without their own date header were assigned to the date above them by the page reader; duplicates for one venue-day
are collapsed, so a three-show day at the Armadillo counts once.

## 3. Check 1 — do the errors line up with event days? (`residuals`)

Least squares of (actual − forecast) on the flags with weekday dummies, 95% interval from a 7-day moving-block bootstrap, on the 585
backtest days. Positive means sales came in above the forecast.

| Flag (days) | vs manager (£/day) | vs deployed model (£/day) |
|---|---|---|
| Celtic or Rangers home match (84) | −135 (−499 to +216) | −184 (−514 to +122) |
| Old Firm (8) | −783 (−1,962 to +528) | −1,051 (−2,412 to +335) |
| Celtic or Rangers away (39 / 36) | +58 / −44, nothing clear | −1 / −69, nothing clear |
| Any show at the three venues (284) | +135 (−77 to +352) | +216 (+49 to +380) |
| **OVO Hydro show (153)** | **+324 (+63 to +602)** | **+366 (+167 to +587)** |
| O2 Academy show (177) | −51 (−308 to +206) | −53 (−275 to +177) |
| SEC Armadillo show (59) | −214 (−580 to +173) | −121 (−463 to +211) |
| Summed capacity, per 1,000 seats | +17 (−1 to +37) | +21 (+7 to +36) |

Context on the 39 new days: 13 Hydro days, actual minus the model's forecast +£1,095 on those and +£107 on the other days; against the
manager +£1,164 and −£138.

Reading: **football home matches show no extra trade** (if anything a little less than forecast, not clearly). **A show at the Hydro lifts
sales by about £320–370 a day beyond what both forecasts expect**, the interval excluding zero for both; the O2 Academy and the Armadillo
show nothing. The capacity sum is a weaker input than a Hydro flag because it mixes the venues that do nothing with the one that does.

## 4. Check 2 — football features in the model (`football`, `saturday`)

### 4.1 Plain football features, all days

The final model's recipe (log1p target, 38 features, depth 4, 300 iterations, rate 0.05), the 585-day rolling-origin backtest, seeds 0–24, with
features added: `plays` (celtic_plays, rangers_plays, old_firm) and `home_away` (home and away flag per club, old_firm).

| | MAE | Change vs baseline (95% interval) |
|---|---|---|
| Baseline | 908.2 | — |
| plays (3 features) | 899.3 | −8.9 (−20.8 to +3.2) |
| home_away (5 features) | 899.5 | −8.7 (−20.9 to +2.9) |

Not clear. (A first run on a calendar still missing six Rangers cup ties gave −7.4 and −7.1; the numbers above are from the final lists in
`config/events/`.) On the 39 new days both variants were worse than the baseline (916 and 914 against 905; that check used the first, slightly
less complete calendar and was not repeated). Saturday was the only weekday whose interval excluded zero (−£65 in both variants), one of 14 slices.

### 4.2 The Saturday flag, rules written before the run

Written 16 Oct 2026 22:30 before the run (the file's sha256, taken before running, was `679bafd27d238c9e71dba812c3de9e1464255bfca90b11352eff4f3aca48dd2f`):

> **Disclosure:** the earlier exploratory run (3- and 5-feature variants, all weekdays) showed Saturday MAE falling by about 58–65 GBP with
> intervals excluding zero, one of 14 weekday slices. That is why this test is run; it is a lead, not a result.
>
> **Feature:** `sat_big_home` = 1 on a Saturday when Celtic or Rangers play at their own ground (any competition), else 0. 53 of 145 Saturdays
> 2024-01-01 to 2026-10-15 are flagged. Neutral-ground games are not flagged.
> **Models:** raw-pound CatBoost (the package's Saturday model recipe) = primary; log1p CatBoost = secondary. Same settings as the final model,
> FINAL_FEATURES (+ the flag), 585-day rolling backtest, seeds 0–24; replicate on seeds 100–124.
> **Metric:** MAE on the 84 backtest Saturdays, change = with flag minus without, 95% moving-block bootstrap.
> **Success: all of** (1) primary change < 0 with the 95% interval excluding 0 on seeds 0–24; (2) the replicate point estimate also < 0;
> (3) the improvement is concentrated in flagged Saturdays (their change is more negative than the unflagged Saturdays').
> **Not a promotion:** even if all three hold, the flag is not added to the product model on this evidence; it enters the prospective weekly
> log and is promoted only with the package's rule (26 weeks and an interval excluding zero).

Result (average miss on the 84 Saturdays; 30 flagged, 54 unflagged):

| Model | Seeds | Without → with flag | Change (95% interval) | Flagged | Unflagged |
|---|---|---|---|---|---|
| Raw (primary) | 0–24 | 1,177 → 1,176 | −2 (−13 to +10) | −4 | −0 |
| Raw | 100–124 | 1,184 → 1,188 | +4 (−10 to +18) | −22 | +18 |
| Log | 0–24 | 1,293 → 1,319 | +26 (−8 to +62) | −48 | +68 |
| Log | 100–124 | 1,303 → 1,316 | +14 (−21 to +52) | −52 | +50 |

**The rules fail:** the primary change is −£2 with an interval spanning zero and the replicate has the wrong sign. The earlier Saturday gain
was most likely chance. On the flagged Saturdays alone the log model's bias falls from −£346 to about −£150 (match Saturdays may sell a little more
than it expects), but the error does not improve overall and the unflagged Saturdays get worse (the model fitting noise).

## 5. What follows

1. **Football is not pursued as a model input.** It stays in the weekly forecast log as a recorded fact (home matches on Saturdays) and can be
   re-checked after months of new days with a complete fixture list and kick-off times.
2. **A Hydro flag is the next test**, not yet run. It needs its rules written first. Candidates: a yes/no `hydro_show`; the count of consecutive Hydro
   nights; music against comedy. It must pass the same kind of pre-registered test on the 585 days with a replicate, and then be checked on the
   new days and the weekly log, before it goes near the product model.
3. **Completing the concert list** is worth doing before that test: the Hydro's non-music events, sport and wrestling especially, and the
   Hydro's missing music entries.
4. **At serving time** the Hydro's programme is public and published weeks ahead, so a Hydro flag, unlike many features, would be available for
   the forecast horizon. The Ticketmaster Discovery API (free key) is a candidate source for upcoming dates; Songkick has no venue-history endpoint
   in its API and is a paid licence.

## 6. Reproducing it

```
python -m src.experiments.run_series36_event_checks residuals
python -m src.experiments.run_series36_event_checks football      # about 2 minutes on 11 workers
python -m src.experiments.run_series36_event_checks saturday      # about 3 minutes
python -m pytest tests/test_events.py
```
