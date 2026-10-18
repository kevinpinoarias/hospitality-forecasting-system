# Events exploration (Series 36) — football and concerts, October 2026

> **Naming (from 18 October 2026).** *Log model* is the final model on the log-sales target (earlier text: "log final model", "final model"). *Post model* is the
> Track A package (earlier text: "Track A", "the package"): the log model plus the raw-£ Saturday model, the Friday-to-Sunday inputs and the Saturday and Sunday
> correction. The sections below keep the names they were written with, including the rules committed before each run, so that those texts are unchanged.

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

## 7. Scotland national team (added 18 October 2026, rules written before the run)

**List.** `config/events/scotland_matches.csv`: every Scotland senior men's match from 22 March 2024 to the scheduled November 2026 fixtures
(36 rows: 15 at Hampden, 11 away, 10 on neutral ground; 20 competitive, 10 friendly, 6 tournament). Source: Wikipedia's "Scotland national football
team results (2020-present)", read on 18 Oct 2026. Cross-checked against independent pages for the September to November 2025 qualifiers (FotMob, Bleacher
Report), the 2026 home fixtures (Scottish FA, ScotRail) and the World Cup times (a December 2025 schedule article). **Not independently confirmed:** the
March and June 2025 matches and the Euro 2024 dates (they agree with the Wikipedia list and are consistent with what is known). Dates are UK dates: the
World Cup opener against Haiti, 13 June in the United States, kicked off at 2am UK on 14 June and is stored as 14 June; the Morocco and Brazil games were
at 11pm UK. Kick-off times for the other matches are not stored.

**Flags** (`scotland_calendar` in `src/features/events.py`): `scotland_home` (a game at Hampden), `scotland_away` (away or neutral ground, not a
tournament game), `scotland_tournament` (Euro 2024 and the 2026 World Cup group games), `scotland_plays` (any of them). In the 585-day backtest span
(15 Jan 2025 to 6 Sep 2026) there are 17 matches: 7 at home, 7 away or neutral non-tournament, 3 World Cup games. The Euro 2024 games and the
2024 matches fall in the training period only.

**Rules, fixed before the run.**
1. *Residual check*: weekday-adjusted least squares of (actual − forecast) on `scotland_home`, `scotland_away` and `scotland_tournament` together, for the
   manager and the deployed model, 585 backtest days, 7-day block bootstrap. A flag is called **clear** only if its 95% interval excludes zero; with 3 to 7 days per
   flag, anything else is **not shown**.
2. *Model test*: the final model's recipe with `scotland_home`, `scotland_away` and `scotland_tournament` added to the 38 features, against the same recipe
   without them, 585-day rolling backtest, seeds 0-24, pooled MAE change with a 95% moving-block bootstrap. The features **help** only if that interval is
   entirely below zero **and** the residual check agrees in sign for at least one flag; a replicate on seeds 100-124 is run only in that case.
3. *No promotion*, whatever the result: the flags would enter the weekly log and be judged with the package's rule (26 weeks, interval excluding zero).
4. *Context only*: the 39 new days contain four Scotland games (26 Sep and 3 Oct away, 29 Sep and 6 Oct at Hampden). They are not used for any decision.

**Expectation, stated in advance.** With about a dozen matches in the model's learning data, the model test is unlikely to say anything either way; the
residual check on the three World Cup days is the only place a clear effect could appear.

### 7.1 Results

Rule 1, residual check (585 backtest days, weekday-adjusted, £/day, + = sales above the forecast; days flagged 7 / 7 / 3):

| Flag | vs manager | vs deployed model |
|---|---|---|
| `scotland_home` | −99 (−1,575 to +1,027) | −296 (−1,349 to +621) |
| `scotland_away` | −18 (−746 to +1,021) | −450 (−1,043 to +315) |
| `scotland_tournament` | −888 (−2,514 to +500) | −551 (−1,624 to +1,009) |

**Nothing is clear.** The three World Cup days were below both forecasts on average (Sunday 14 June £6,688 against forecasts of £8,030 and £7,902; Friday 19
June £8,736 against £11,400 and £10,473; Wednesday 24 June £7,740 against £7,700 and £6,615), but three days cannot separate that from chance.

Rule 2, model test (log1p final recipe, seeds 0-24): MAE 908.2 → 912.0, **change +3.8 (−2.0 to +9.8)**. Slightly worse, interval spanning zero.
On the 17 Scotland days alone: 904 → 953 (+49, −42 to +157). **The features do not help**, so no replicate was run (rule 2).

Context only (rule 4), the four Scotland games among the 39 new days: Saturday 26 Sep (away) +£988 against the manager and +£1,052 against the model;
Tuesday 29 Sep (home) +£221 and +£137; Saturday 3 Oct (away) +£2,744 and +£1,596; Tuesday 6 Oct (home) +£1,391 and +£985. All four were above both forecasts, but
26 Sep and 3 Oct are also Hydro and race weekend days, and 6 Oct had a Hydro show, so nothing here separates Scotland from the other events.

**Reading.** As expected in advance, the data hold too few Scotland days to say anything. The flags are not pursued as inputs; the weekly log can record
Scotland's matches as facts, and the World Cup days are a reminder that a rare, very large TV event may matter more than ordinary internationals but cannot be
learned from three days.

## 8. Note for later — where football features might still earn their place (added 18 October 2026)

Football, club and national team, is parked as a model input for this venue. The venue is a cafe-restaurant that does not show matches, so a game in Glasgow is
at best a crowd passing by and at worst a reason for regulars to stay away; the data lean towards slightly lower sales on match days (home-match days −£135 against
the manager, Old Firm and World Cup days lower still) but no interval excludes zero. Two reasons to come back to it:

1. **A different kind of venue.** For a pub or a bar that shows the games, football is the demand: kick-off times, the opponent, a derby or a tournament knockout would be
   core inputs, and the Celtic / Rangers / Scotland lists, flags and tests built here (`config/events/`, `src/features/events.py`) are the starting point. If this
   system is ever used for such a venue, test football first.
2. **Better features for this venue.** A feature that actually helps would probably need what was missing here: a complete fixture list with kick-off times (a 12:30
   Sunday derby and a 3pm Saturday league game are different days), weekend-only and kick-off-window versions, and a match-day-displacement flag (regulars staying away)
   judged separately from a crowd-passing-through flag. Any such test needs its rules written first, a replicate on fresh seeds, and a check on the weekly log.

The football and Scotland results stay in the Series 36 log entry as they are: not shown, not pursued for this venue.

## 9. The Hydro flag — checks on the list, and the test rules (written 18 October 2026, before any run)

### 9.1 Chasing the list's gaps

**Missing music shows.** The earlier note said the Hydro list was about 6–12% below Concert Archives' counts. That comparison is probably a counting difference (Concert Archives counts
listings, not show-days; its site blocks our reader, so it could not be reconciled line by line). An independent check on one month: every Hydro date found by searching November 2025
(2, 3, 6, 11, 16, 19, 22, 23, 25 and 30 November) is in `config/events/concert_days.csv`: 10 of 10. Spot checks of Westlife, Kevin Bridges, Hans Zimmer and Duran Duran also agree.
The setlist.fm and Hydro-site archives could not be used (setlist.fm gives upcoming shows and a handful of recent ones; the web archive is blocked). **No shows were added**; the list stands as
read from Songkick, with the caveat that one sampled month is not a proof of completeness.

**Non-music days — is the forecast off?** Hydro non-music events found: WWE Clash at the Castle (Sat 15 Jun 2024), PFL Europe 3 (Sat 28 Sep 2024), Gladiators Live (20 and 21 Dec 2025),
Cirque's Wizard of Oz (Sun 28 Dec 2025), WWE Live (Sat 10 Jan 2026). Actual minus forecast, £:

| Day | Manager | Model (where it exists) | Normal size of a miss |
|---|---|---|---|
| Sat 15 Jun 2024, WWE Clash | −5,099 | n/a (before the backtest) | manager MAE 1,163 on 2024 Saturdays |
| Sat 28 Sep 2024, PFL | +1,214 | n/a | same |
| Sat 20 Dec 2025, Gladiators Live | +349 | +9 | model MAE 908 |
| Sun 21 Dec 2025, Gladiators Live | −2,369 | −1,157 | |
| Sun 28 Dec 2025, Cirque | −1,175 | −403 | |
| Sat 10 Jan 2026, WWE Live | +1,393 | +3,716 | |

Mixed: some days far below the forecast, one far above, two ordinary. **There is no consistent pattern, so non-music events are not added** (six days, five events, and the busy
December and January periods carry their own swings). The flag therefore means what the list already holds: *a music or comedy show listed at the OVO Hydro that day.*

### 9.2 Disclosure

The Hydro was picked from a residual check on these same 585 backtest days (about five flags looked at: three venues, "any show" and a capacity sum), after the first idea came from five days
chosen for their errors. So this test is **not** a clean out-of-sample test of the idea; it tests whether a model *using* the flag, learning only from earlier days at every step, predicts later
days better. The clean evidence is the weekly log from 19 October.

### 9.3 Rules

**Feature.** `hydro_show` = 1 when `config/events/concert_days.csv` has a row for the OVO Hydro on that day (music or comedy), else 0. Same-day only: no flag for the day before or after, no
count of consecutive nights, no split between music and comedy, no other venue. 153 of the 585 backtest days are flagged.

**Models and protocol.** The final model's recipe (log1p target, 38 features, depth 4, 300 iterations, rate 0.05) with and without `hydro_show`; the 585-day rolling-origin backtest;
seeds 0-24; the same fits on the raw-£ target as a secondary result. Replicate on seeds 100-124 for both.

**Placebo.** The same test with the flags shuffled: the 153 flagged weekdays are reassigned at random among the days of the same weekday over the whole date range (fixed seed 20261018). A real
effect should not survive the shuffle.

**Metric.** Pooled MAE over the 585 days, change = with the flag minus without it, 95% interval from the 7-day moving-block bootstrap (`masked_block_bootstrap`); the same on flagged days, on
unflagged days, Monday-Thursday and Friday-Sunday.

**The flag passes only if all of these hold.**
1. *Primary*: on the log1p model, seeds 0-24, the pooled change is below zero and its 95% interval is entirely below zero.
2. *Replicate*: on seeds 100-124 the pooled interval is also entirely below zero.
3. *Where it comes from*: on seeds 0-24 the change on flagged days is below zero and more negative than the change on unflagged days, and the unflagged days' interval is not entirely above zero
   (the flag must not buy its gain by hurting the other days).
4. *Placebo*: the shuffled flag's pooled interval is not entirely below zero (if it is, the test is broken and the result is void).

Reported either way, not rules: the raw-£ results, bias on flagged days, the gain over the manager (manager MAE 1,074.5), and the same on the 39 new days (13 Hydro days; context only,
they have been seen).

**Decision.** *Pass*: the flag is recorded as supported on the backtest and enters the weekly log as a candidate input; it is **not** added to the product model on this evidence. Promotion needs
the log to reach 26 weeks with an interval excluding zero, the same bar as the Track A package, and a way to know the Hydro programme at forecast time (the Hydro and Ticketmaster publish weeks
ahead). *Fail*: the result is logged as Series 37, the flag is not pursued, and the residual check's +£324 to +£366 stays recorded as a fact about the venue, not an input.

**Expectation, stated in advance.** If the effect is as large as the residual check suggests and the model can learn it, the pooled gain is a few tens of pounds a day. If the model already
captures it through the manager's forecast and the weekday, the change will be near zero.

### 9.4 Result (run 18 October 2026, after the rules were committed in `96e3ed2`)

Average miss in £/day on the 585 backtest days (change = with the flag minus without; 95% interval); 153 Hydro days, 432 others. Manager MAE 1,074.5.

| Run | All days | Hydro days | Other days |
|---|---|---|---|
| log1p, seeds 0-24 | 908.2 → 904.9; **−3.3 (−17.9 to +10.0)** | 958.8 → 913.8; **−45.0 (−89.5 to −5.5)** | +11.4 (−1.0 to +23.7) |
| log1p, seeds 100-124 | 908.9 → 907.0; **−1.9 (−15.5 to +10.9)** | −34.4 (−75.4 to +2.3) | +9.7 (−3.8 to +22.7) |
| raw-£, seeds 0-24 (secondary) | 916.0 → 908.2; −7.8 (−23.6 to +5.1) | −16.8 (−58.8 to +18.7) | −4.7 |
| raw-£, seeds 100-124 (secondary) | −9.5 (−24.6 to +4.3) | −21.1 (−63.5 to +15.3) | −5.4 |
| **Placebo** (shuffled flags), log1p, seeds 0-24 | +0.4 (−4.1 to +4.4) | −4.5 (−12.7 to +2.5) | +2.2 |

Bias on Hydro days (forecast minus actual, log1p): −£352 without the flag, −£109 with it. On the other days: +£19 without, −£62 with.

**Verdict against the rules: FAIL.**

| Rule | Result |
|---|---|
| 1. Primary: pooled interval entirely below zero | **FAIL** (−3.3, interval −17.9 to +10.0) |
| 2. Replicate: pooled interval entirely below zero | **FAIL** (−1.9, interval −15.5 to +10.9) |
| 3. Gain comes from Hydro days and the other days are not clearly hurt | PASS |
| 4. Placebo does not show a clear gain | PASS (+0.4) |

The shuffled flag does nothing, so the test is sound, and the flag really does help on Hydro days: the average miss on those days falls by about £34 to £45, the bias there shrinks from −£352 to −£109, and the
gain is far larger than the placebo's. But the model pays for it on the other 432 days (+£10 to +£11 a day, nearly clear), so the pooled change is a few pounds and well inside the noise. The flag is
not added to the model and is not pursued as an input, under the pre-registered rule.

**Context only, the 39 new days** (seen before the rules were written): one fit on the 950 rows, with and without the flag. MAE 905 → 872 (−£33); Hydro days 1,232 → 1,072; other days 741 → 771. The
same pattern as the backtest, which says the pattern is real and not that the model has become better overall.

**Reading, and an idea that is NOT tested.** The flag carries real information about Hydro days (consistent in the residual check, the model fits, the replicate and the new days), but a tree model that
learns it as one more feature loses a little on ordinary days. A separate, additive correction on Hydro days only (the way Series 24 handles weekends), instead of a new model input, might keep the
Hydro-day gain without the ordinary-day cost. That is a new hypothesis, chosen after seeing this result, so it needs its own rules and a replicate before it is tried, and it would not be a clean test on
these 585 days either. It is recorded here and nowhere else.

## 10. The Hydro-day correction — test rules (written 18 October 2026, before any code or run)

**Why.** Section 9.4: the `hydro_show` flag helps on Hydro days (£34 to £45 off the average miss) but, as a model input, costs about £10 a day on the other 432 days. A correction applied after the
forecast, to Hydro days only, cannot touch the other days. **This hypothesis was chosen after seeing the section 9.4 result and uses the same 585 days**, so it is not a clean test; the weekly log is
the clean one.

**The correction** (a Hydro twin of `run_series24_weekend_correction.correct`, using its untuned package settings: half-life 8 weeks, strength 0.5, at least 4 days of history). For block *k* ≥ 1 of
the 60-day rolling-origin blocks, every day with a Hydro show gets its forecast **moved by −0.5 × the recency-weighted mean of (forecast − actual) over the Hydro days in blocks before *k***. Weights are
0.5 ^ (age in weeks ÷ 8). Block 0, and every day without a Hydro show, is unchanged. A flat £ amount: no split by weekday, show size, music or comedy. No grid and no settings are tried.

**Bases.** The correction is added to (1) **primary: the log final model**, bagged over 25 seeds, the same forecasts as the `hydro_show` test (seeds 0-24 and the replicate 100-124); (2) **secondary: the Track
A package** on the verified calendar, rebuilt by `run_package_validation.package_steps` with its weekend correction already applied, the error history for the Hydro correction taken from the package's own
forecasts so a weekend Hydro night is not corrected twice. No new model is fitted; both sets of bases already exist.

**Metric.** Mean change in absolute error against the same base without the correction, on all 585 days, on the 153 Hydro days, and on Monday-Thursday and Friday-Sunday Hydro days; 95% interval from
`masked_block_bootstrap` (7-day-equivalent blocks as before). Days without a Hydro show are unchanged by construction.

**It passes only if all of these hold, on the primary base.**
1. *Hydro days*: the change on Hydro days has a 95% interval entirely below zero on seeds 0-24 **and** on seeds 100-124.
2. *Pooled*: the change over all 585 days has a point estimate at or below zero on both seed sets. (The interval will probably span zero: Hydro days are a quarter of the days, so even a correction as good as
   the flag's £45 would move the pooled average by only about £11 against an interval of roughly ±£14. The pooled interval is therefore not a rule.)
3. *Placebo*: the same correction applied to the shuffled flags (the section 9.3 shuffle, seed 20261018) shows a change on those days whose 95% interval is **not** entirely below zero.
4. *Stability*: the correction amount is positive (sales above the forecast) in at least 8 of the 9 corrected blocks on seeds 0-24.

**Reported, not rules.** The secondary base (the package) with the same four numbers; the raw size of the correction in £ per block; the same correction applied to the manager's own forecast; the 39 new days
(context only, seen).

**Decision.** *Pass*: the correction is a candidate for the weekly log, **not** added to the product model; promotion needs the log to reach 26 weeks on Hydro nights with an interval excluding zero, and
a source for the Hydro programme at forecast time. It would also be a plain rule the rota planner can apply by hand. *Fail*: logged as Series 38 and not pursued; the Hydro effect stays a fact about the
venue.

**Expectation, stated in advance.** If the Hydro-day under-forecast is as steady as the residual check says (about £350 a day), the correction should remove most of it and the Hydro-day change should
be of the order of −£30 to −£45; if the effect drifts from block to block the correction will lag it and the gain will shrink.

### 10.1 Result (run 18 October 2026, after the rules were committed in `2b84830`)

Average miss in £/day, 585 backtest days; change = with the correction minus without (95% interval). Days without a Hydro show are unchanged by construction (checked in the code).

| Base | Seeds | All days (585) | Hydro days (153) |
|---|---|---|---|
| **Log final model (primary)** | 0-24 | 908.2 → 903.4; **−4.8 (−15.1 to +4.5)** | 958.8 → 940.4; **−18.4 (−59.1 to +16.4)** |
| | 100-124 | **−4.3 (−14.7 to +5.4)** | **−16.4 (−57.4 to +19.2)** |
| Track A package (secondary) | 0-24 | 864.6 → 861.6; −3.1 (−11.5 to +4.9) | −11.7 (−45.5 to +17.7) |
| | 100-124 | −3.5 (−12.0 to +4.5) | −13.3 (−47.4 to +16.5) |
| **Placebo** (shuffled flags, log final model) | 0-24 | −1.1 (−10.7 to +7.6) | −4.1 (−37.9 to +28.2) on its 161 days |
| Manager's forecast (descriptive) | | −0.5 (−7.3 to +6.5) | −1.9 (−28.3 to +24.3) |

Hydro Monday-Thursday −16 (−69 to +34) and Friday-Sunday −21 (−82 to +38) on the primary base: the same size in both. The correction added to Hydro days, by block (log final model, seeds 0-24): £349, 223, 327,
326, 192, 138, 225, 75, 125.

**Verdict against the rules: FAIL.**

| Rule | Result |
|---|---|
| 1. Hydro-day interval entirely below zero on both seed sets | **FAIL** (intervals −59 to +16 and −57 to +19) |
| 2. Pooled point estimate at or below zero on both seed sets | PASS (−4.8, −4.3) |
| 3. Placebo interval not entirely below zero | PASS |
| 4. Correction positive in at least 8 of 9 blocks | PASS (9 of 9) |

**Reading.** The correction moves the forecasts the right way every block and the pooled change is a few pounds below zero, as expected in advance, but on the Hydro days themselves the gain is about £16 to £18 (about
a third of the flag's £45) and its interval spans zero, so it cannot be told from nothing. Two things weaken it: the correction **shrinks over time** (about £330 early, about £100 in the last two blocks), so
either the Hydro effect is fading in 2026 or the model has partly learned it through the manager's forecast; and the shuffled-flag placebo also helps on Friday to Sunday (−27, −64 to +3), which means part of any
weekend gain is the base model's general weekend under-forecast and not the Hydro. The package base gains slightly less than the plain model (−12 to −13), as it should once its weekend correction is in place.

**Context only, the 39 new days:** a correction of +£92 per Hydro day (the amount the method would add from 7 September, from the 585 days' history), applied to the D forecasts: MAE 905 → 888; Hydro days 1,232 → 1,182;
other days unchanged.

**Decision.** Not pursued; logged as Series 38. The Hydro effect is recorded as a fact about the venue and a hand rule for the rota planner (a Hydro night has averaged about £350 above what the models and the manager expect;
the last two blocks suggest less), not as a model input or an automatic correction. Hydro nights are recorded in the weekly log so the effect can be re-estimated on clean days.

## 11. The Hydro flag inside the post model — test rules (written 18 October 2026, before any code or run)

**Where this stands.** Four combinations were asked for. (1) A plain raw-£ model, Monday to Sunday, with the Hydro flag and nothing else: done in section 9.4 (secondary run; pooled −7.8 and −9.5, Hydro days −16.8 and −21.1, intervals
spanning zero). (2) The log model plus the Hydro correction on every day: done in section 10.1 (primary base). (4) The post model plus the Hydro correction: done in section 10.1 (secondary base). **(3) the post model with the Hydro
flag as an input to its own models has not been run.** It is the only new test here.

**What is built.** The post model is the log model on Monday to Thursday; on Friday the log model with `A_all` (sales on the same weekday 1-4 weeks ago); on Saturday the raw-£ model with `A_lags` (sales 2 and 3 weeks ago); on
Sunday the log model with option `B` (weekend interactions); and the Saturday and Sunday correction (half-life 8 weeks, strength 0.5). The weekday options are those chosen in Series 27 / 31 (Friday A_all, Saturday A_lags, Sunday
B) and are **not re-chosen**. The flagged post model adds `hydro_show` to the inputs of every one of these models (Monday-Thursday: the log model with the flag, already fitted in section 9; Friday, Saturday, Sunday: the same
options fitted again with the flag) and leaves everything else, including the correction, unchanged. The correction is computed from the flagged post model's own errors. No Hydro correction is added on top.

**Comparison.** The same post model without the flag (the forecasts used as the secondary base in section 10.1). Seeds 0-24 and a replicate on 100-124. A sanity refit of the three weekend options without the flag (seeds 0-24)
must reproduce the stored post model to within £0.05 a day, or the comparison is not like for like and the result is void.

**Placebo.** The same fits with the shuffled flag of section 9.3 (seed 20261018), seeds 0-24.

**Metric.** Change in MAE (with the flag minus without), 95% moving-block bootstrap, on all 585 days, on the 153 Hydro days, on the 432 other days, and by Hydro day-type (Monday-Thursday, Friday, Saturday, Sunday; the last
three are descriptive).

**It passes only if all of these hold.**
1. *Hydro days*: the change on Hydro days has a 95% interval entirely below zero on seeds 0-24 **and** on seeds 100-124.
2. *Pooled*: the point estimate over all 585 days is at or below zero on both seed sets. (Pooled intervals are not a rule: Hydro days are a quarter of the days.)
3. *Other days*: the change on non-Hydro days has an interval that is not entirely above zero on both seed sets (the flag must not clearly hurt them, as in section 9).
4. *Placebo*: the shuffled flag's change on its own flagged days has an interval that is not entirely below zero.

**Reported, not rules.** The weekday table; the comparison with the Hydro correction (section 10.1) on the same base; the 39 new days (context only).

**Decision.** *Pass*: a candidate for the weekly log, **not** added to the product model (promotion needs 26 weeks on Hydro nights with an interval excluding zero and a source for the Hydro programme at forecast time). *Fail*: logged as
Series 39 and not pursued.

**Disclosure.** This is the third look at the same 585 days for the same idea, and the post-model version was proposed after the weekday table showed the effect concentrated on Saturday (26 Hydro Saturdays). A pass would
therefore be weaker evidence than the numbers suggest; the weekly log is the clean test.

**Expectation, stated in advance.** The post model already lifts Saturday forecasts (its Saturday bias on Hydro nights is −£530 against −£1,191 for the plain log model), so the room for the flag is smaller than in Series 37. Gains
of a few pounds on Hydro days, intervals spanning zero, are the likely outcome.

### 11.1 Result (run 18 October 2026, after the rules were committed in `29e23eb`)

**Sanity check passed:** the three weekend options refitted without the flag reproduce the stored post model exactly (MAE 864.630 against 864.630, largest single-day difference 0.000).

Average miss in £/day on the 585 backtest days; change = post model with the flag minus the post model without it (95% interval). 153 Hydro days, 432 others.

| | All days | Hydro days | Other days |
|---|---|---|---|
| Seeds 0-24 | 864.6 → 864.1; **−0.5 (−13.6 to +10.5)** | 907.1 → 881.0; **−26.1 (−67.8 to +10.0)** | +8.6 (−4.3 to +20.4) |
| Seeds 100-124 | **−7.6 (−19.9 to +2.7)** | **−33.6 (−71.6 to +0.1)** | +1.6 (−10.5 to +12.8) |
| **Placebo** (shuffled flags), seeds 0-24 | +4.6 (+1.2 to +8.2) | +4.0 (−3.3 to +12.1) on its 161 days | +4.8 |

By Hydro day-type (change; descriptive):

| Hydro days | n | Seeds 0-24 | Seeds 100-124 |
|---|---|---|---|
| Monday-Thursday | 79 | −32.7 (−89.1 to +21.7) | −29.7 (−80.7 to +20.4) |
| Friday | 29 | −32.9 (−124.9 to +51.6) | −52.4 (−157.1 to +38.5) |
| Saturday | 26 | −62.6 (−149.4 to +36.0) | −77.7 (−162.7 to +19.5) |
| Sunday | 19 | **+61.6** (−54.1 to +174.8) | **+39.4** (−74.3 to +149.9) |

Bias on Hydro days (forecast minus actual): −£278 without the flag, −£59 with it (seeds 0-24); Hydro Sundays go from +£371 to +£609.

**Verdict against the rules: FAIL.**

| Rule | Result |
|---|---|
| 1. Hydro-day interval entirely below zero on both seed sets | **FAIL** (−67.8 to +10.0 and −71.6 to +0.1: the replicate misses by 10p, the first set by £10) |
| 2. Pooled point estimate at or below zero on both seed sets | PASS (−0.5, −7.6) |
| 3. Other days: interval not entirely above zero | PASS |
| 4. Placebo not clearly helped | PASS (it is slightly worse) |

**Reading.** The flag inside the post model does about what the correction did and a little more on Hydro days (−£26 to −£34 against the correction's −£12 to −£13), with a similar pooled result (−£0.5 and −£7.6 against −£3.1 and
−£3.5). It is the closest of the Hydro attempts to clearing the bar, and still does not clear it. The gain is spread over Monday-Thursday, Friday and Saturday (−£30 to −£78, none individually clear) and **Sunday gets worse**, as in every
earlier run: the models over-forecast Hydro Sundays and the flag pushes them higher still. The post model's Saturday gain (−£63 to −£78) is smaller than the plain models' (−£111 to −£177) because the post model already forecasts
Hydro Saturdays higher.

**Not run:** the 39 new days, which the rules listed as context only; the post model's served path (as it would run on the new days) is not rebuilt with the flag.

**Decision.** Not pursued; logged as Series 39. Of the four tests asked for, the flag in a plain model (Series 37), the correction on the log model and on the post model (Series 38) and the flag in the post model (this one) all point the same
way: a small gain on Hydro days that never separates from zero, and a loss on Sundays.

## 12. A Thursday-only Hydro flag — test rules (written 18 October 2026, before any code or run)

**Why.** Sections 9 to 11: every Hydro input gains a little on Hydro days, never clearly, and loses on Sundays and Mondays and a little on ordinary days. The weekday tables show the gain concentrated on Thursday (17 Hydro Thursdays; the
plain log model under-forecast them by about £820; the flag cut the miss by £209, interval −298 to −100) and Saturday. **Thursday was chosen because it was the best of seven weekdays in results already seen**, so this is a post-hoc slice.
With seven weekdays, about three times in ten at least one would look clear by luck alone.

**Feature.** `hydro_thu` = 1 on a Thursday on which `config/events/concert_days.csv` has a row for the OVO Hydro, else 0. 17 of the 585 backtest days are flagged. No other weekday, no other venue.

**Models and protocol.** The log model's recipe (log1p target, 38 features, depth 4, 300 iterations, rate 0.05), with and without `hydro_thu`; the 585-day rolling-origin backtest; seeds 0-24 and a replicate on 100-124.
(Monday to Thursday are forecast by the log model in the post model, so a Thursday result applies to the post model unchanged.)

**Placebo.** The same fit with the 17 flags reassigned at random among all Thursdays (fixed seed 20261018), seeds 0-24.

**Metric.** Change in MAE (with the flag minus without it), 95% interval from `masked_block_bootstrap`, on the 17 Hydro Thursdays, on the other 568 days, and on all 585 days; the number of the 17 Hydro Thursdays that are better with
the flag.

**It passes only if all of these hold.**
1. *Hydro Thursdays*: the change has a 95% interval entirely below zero on seeds 0-24 **and** on seeds 100-124.
2. *Everything else*: the change on the other 568 days has an interval that is not entirely above zero on both seed sets (the flag must not clearly hurt anything else).
3. *Sign test*: on seeds 0-24 at least 13 of the 17 Hydro Thursdays are better with the flag (one-sided binomial p = 0.025).
4. *Placebo*: on the shuffled flags' own 17 days the interval is not entirely below zero.

**Reported, not rules.** The pooled change (it will span zero: 17 days cannot move a 585-day average by more than a few pounds); the Hydro Thursdays in the 39 new days (1 October, Westlife; 15 October, Duran Duran) as context only.

**Decision.** *Pass*: the flag is recorded as a candidate for the weekly log, **not** added to the product model; it needs 26 weeks and a Hydro-programme source at forecast time like every other Hydro input. *Fail*: logged as Series 40
and not pursued; Hydro Thursdays are recorded in the weekly log all the same.

**Disclosure.** A fourth look at the same 585 days for one idea, with the weekday picked after the fact. A pass would be a lead, not evidence; the weekly log is the clean test.

**Expectation, stated in advance.** If Thursday is a real effect, the Hydro-Thursday change should be of the order of −£100 to −£200 (about what the full flag showed) and the pooled change −£3 to −£6 with an interval spanning zero. If it was a
lucky slice, the replicate seeds and the placebo will show it.

### 12.1 Result (run 18 October 2026, after the rules were committed in `7d827e0`)

Average miss in £/day on the 585 backtest days; change = log model with `hydro_thu` minus without (95% interval). 17 Hydro Thursdays, 568 other days.

| | Hydro Thursdays (17) | All other days (568) | All 585 days |
|---|---|---|---|
| Seeds 0-24 | 925.6 → 650.9; **−274.8 (−436.4 to −92.3)** | +1.0 (−6.5 to +8.3) | 908.2 → 901.2; −7.1 (−17.1 to +1.9) |
| Seeds 100-124 | 930.6 → 650.1; **−280.5 (−431.3 to −113.0)** | +4.2 (−2.5 to +10.6) | −4.1 (−13.2 to +4.3) |
| **Placebo** (17 flags reassigned among Thursdays; it came out as 20 flagged days) | 507.7 → 478.7; **−29.0 (−57.8 to −4.8)** | +1.7 (−2.8 to +6.3) | +0.6 (−3.8 to +5.2) |

Bias on Hydro Thursdays (forecast minus actual): −£824 without the flag, −£357 with it. Other Thursdays: −£7.6 and −£6.0 (intervals span zero). **14 of the 17 Hydro Thursdays are better with the flag** (one-sided sign test p = 0.006). By block (seeds
0-24) the change is negative in seven blocks and positive in one (block 2, +£116 on three days).

**Verdict against the rules: FAIL, on rule 4 alone.**

| Rule | Result |
|---|---|
| 1. Hydro-Thursday interval entirely below zero on both seed sets | PASS (−436 to −92; −431 to −113) |
| 2. Everything else not clearly hurt | PASS (−6.5 to +8.3; −2.5 to +10.6) |
| 3. Sign test, at least 13 of 17 better | PASS (14, p = 0.006) |
| 4. Placebo interval not entirely below zero | **FAIL** (−57.8 to −4.8 on the shuffled flags' own days) |

**Reading.** The Thursday flag is by far the strongest result of the whole Hydro series: a gain of about £275 on the 17 Hydro Thursdays, more than six times the all-days flag's £45, in both seed sets, with nothing lost elsewhere. But the one
placebo draw (a random 20 Thursdays) also came out slightly better with the flag, by £29, interval below zero, and the pre-registered rule says a placebo that looks like a gain means the test cannot be trusted. The placebo's own days happened to be
unusually easy ones (average miss 508 against 926 for the real Hydro Thursdays), so a small gain there is the kind of thing one random draw can produce, but a single draw cannot say how often. **The formal result is a fail; whether it is also a
fluke of that draw is not known.** A permutation check (many random shuffles, to see how often a random set of 17 Thursdays gains £275) would answer it; it is not part of the rules and would need its own rules first.

**Context only, the 39 new days.** There were three Hydro Thursdays (10 Sep André Rieu, 1 Oct and 15 Oct; the rules said two, which was a miscount): their average miss 1,780 → 1,232, the other 36 days 832 → 829, all 39 days 905 → 860.

**Decision.** Logged as Series 40, formally a fail. No change to the model. The Thursday finding goes to the weekly log as the strongest lead of the series, with the placebo caveat attached.

## 13. The Thursday flag against many random placebos — rules (written 18 October 2026, before any code or run)

**Why.** Section 12.1: the Thursday flag gained £275 on its 17 days, but the one placebo draw (a random 20 Thursdays) also gained £29 with an interval below zero, which fails rule 4. One draw cannot say how often chance gives a gain. This repeats
the placebo many times. **It does not change the Series 40 verdict, which stays a formal fail;** it asks how unusual the real result is.

**Procedure.** For each of 200 draws *i* = 1 to 200: the same number of Thursday flags as the real `hydro_thu` are reassigned at random among all Thursdays in the model's rows (seed 20261018 + *i*), the log model is refitted with that flag
as an extra input (same recipe and rolling-origin protocol as before), and the change in average miss (with the flag minus without) is taken **on that draw's own flagged days among the 585 backtest days**. To keep 200 draws affordable each fit uses **5 seeds
(0-4)**; the real flag and the unflagged baseline are taken from the same 5 seeds (the first chunk of the stored 25-seed fits), so the comparison is like for like. No draw is discarded or repeated.

**Statistic.** The average change in absolute error on the flagged days (negative is a gain), as for the real flag.

**Rules.**
1. *Rank*: the real flag's 5-seed change (on its 17 Hydro Thursdays) is more negative than at least 95% of the 200 placebo changes (one-sided p ≤ 0.05).
2. *Size*: the real change is also more negative than the 99th-percentile-best placebo gain, i.e. beyond all but two of the 200 draws, **or** at least twice the best placebo gain; whichever is looser.
3. *Spread*: the 200 placebo changes are reported as a distribution (mean, 5th, 50th and 95th percentiles, the best); the first placebo (seed 20261018, −£29) is located within it.

**Reading, fixed in advance.** If rules 1 and 2 hold, the Thursday result is **unlikely to be chance on the backtest** and is recorded as the leading candidate for the weekly log (still not promoted; a fourth look at the same days, a weekday chosen after the fact).
If rule 1 fails, the Thursday result is **indistinguishable from what a random set of Thursdays can give** and is dropped to a note. If rule 1 holds and rule 2 fails, it is recorded as unclear.

**Limits, stated in advance.** Placebo days vary in difficulty (the first draw's days were easy ones: average miss 508 against 926 for the real days), so the statistic's spread comes partly from which days are drawn; that is what the draws are for. Five seeds
are noisier than 25; the real flag and the baseline use the same five, so the comparison is fair but the real change at 5 seeds will differ a little from the −£275 at 25 seeds.

**Expectation, stated in advance.** If the Thursday effect is real, the real change should sit far in the tail (a p of 0.01 or lower), with most placebo draws between about +£20 and −£40.
