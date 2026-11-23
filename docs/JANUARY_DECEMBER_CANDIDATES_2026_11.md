# The year-end candidates: a register and the reading for Series 90

Written Monday 9 November 2026, **before Series 90 was run** (the same practice as `docs/NO_MANAGER_TRAINING_2026_11.md`). Kevin's instruction: before deciding to adopt the stand-in and specialist-window approach, test the other
alternatives that have been popping up, see which behaves best, and register the candidates. Nothing in this document is registered in the forecast log: registering a model version needs an amendment to the log's
`PREREGISTRATION.md` and Kevin's decision, and a change to the stand-in needs an amendment too.

## The problem

Around the new year the forecast log is weak. The model's recent-sales inputs (last week's sales, the 7- and 14-day averages and the like) hold the busy Christmas fortnight, so they overstate the level in early January and
mislead in late December; and the live log has no real manager's number for the day (it arrives after the day), so it serves a stand-in, the weekday-and-season median of the manager's past numbers, which is contaminated around the
holidays by closed-day zeros. The unit below is the average daily miss (pounds) over the 585 backtest days, served the way the live log serves.

## The register

| # | Candidate | Where it acts | Status (9 November 2026) | Evidence |
|---|---|---|---|---|
| S0 | The current stand-in (median of the same weekday within 10 days of the same day of the year) | the input | in the log today | 946 a day; the real number gives 854 |
| S1 | Closed-day-aware median (closed days left out, 0 on 25 December and 1 January) | the input | tested, replicates | 903 (-43), Series 88-89 |
| S3 | The same date a year earlier in the holiday window 24 Dec-15 Jan, S0 elsewhere | the input | tested, replicates | 904 (-42) |
| **S6** | **Hybrid: the same date a year earlier for 24 Dec-2 Jan, S1 elsewhere** | the input | tested, replicates, leads | **892 (-54)**; designed after seeing S1 and S3 fail on the same days, so probably optimistic |
| M1 | Specialist windows: the 1-8 Jan masked model on 1-8 Jan, the companion (no history inputs) on 26-31 Dec, the unmasked model elsewhere | which model forecasts | tested, replicates | -5.9 to -11.0 with any stand-in; needs new fitted models |
| M2 | Whole-model mask on 1-8 Jan | training and forecast rows | tested: a wash with a stand-in (-0.9 to -4.3), -8 to -10 with the real number | Series 80-87 |
| M3 | Partial masking on 26-31 Dec (one input group) | training and forecast rows | tested, no group captures both | Series 84 |
| M4 | Closed days as missing or filled in the history inputs | the history inputs | tested, worse (+3.1, +8.9) | Series 81 |
| M5 | A January penalty in the intensity (Saturday pieces) | an input | tested, nothing | Series 79 |
| M6 | The closed-day intensity column | an input | in forecast-log version v1.12 | -1.6 with the real number |
| **B** | **Year-ago sales as an input**: `sales_364`, the sales on the same weekday 364 days earlier, in every piece | an input | **Series 90** | untested; only rows from January 2025 have it |
| **C** | **A days-since-Christmas input** `days_since_xmas` (days since 25 December, clipped at 60, 99 otherwise) in every piece, so the trees can learn to discount the history | an input | **Series 90** | untested; the model already has `days_since_bank_holiday` and `is_christmas_break` and has not learned to discount with two winters |
| **D** | **History inputs without the Christmas break**: the ten history inputs rebuilt from the sales series with 24 Dec-2 Jan left out (averages over the days that exist) | the history inputs | **Series 90** | untested; all of December is busy, so it may still overstate January |
| **E** | **A lean companion**: the companion for 26-31 Dec drops only the averages and lags (`rolling_7_sales`, `rolling_14_sales`, `lag_7_sales`, `lag_14_sales`, `lag_21_sales`) and keeps the forecast-error and same-weekday-summary inputs, which Series 84 found harmless to mask | which model forecasts | **Series 90** | untested |
| **F** | **The closed-day intensity in every piece**: `intensity_fixed` in the Monday-Thursday, Friday and Sunday pieces too (only Saturday has it) | an input | **Series 90** | untested; the Saturday-only choice came from the payday-heat result, not from a test of a closed-day or penalty column |
| **H** | **Random training masks** (option 2a): train with the ten history inputs hidden on a random 20% of rows, so the model learns to cope without them, then hide them at forecast time on 1-8 Jan and 26-31 Dec | training rows, then forecast rows | **Series 90** | untested |
| G | A correction rule after the model for 26 Dec-15 Jan (a share of the other year's error) | after the model | **not testable in the backtest** (one January: the backtest has no forecast for 1-14 January 2025) | sketched on early January 2026 vs 2025 only: -40 to -51 a day on 5-15 Jan, both Januaries over-forecast from the 5th |
| J | Year-ago lags in place of the recent ones, the wider skip-the-holiday lag and a longer-memory input | the history inputs | not run: B and D cover the idea | |

## Series 90: what is tested and how it is read (written before it ran)

Candidates B, C, D, E, F and H, each refitted in the four pieces of the current package (Monday-Thursday with the hydro flag, Friday, Saturday with `intensity_fixed`, Sunday), 25 seeds (200-224), 10 folds, 585 days, the deployed
weekend correction. Every candidate is **served with the S6 stand-in** (the best of the stand-ins, so the candidates are judged in the world the log will live in) and compared with the unmasked package served with S6 (Series
89's fits on the same seeds) and with M1, the specialist windows, served with S6 (the reference to beat).

* **measure**: the net change of the post package over the 585 days against the unmasked package served with S6 (pounds a day; negative = better), with the change over the late-December days (26-31 Dec), the early-January days
  (1-8 Jan) and all other days beside it; for E, the change against M1.
* **helps**: net change at most -2.0 (the 25-seed noise is about 2). **nothing**: within 2.0 of zero. **hurts**: above +2.0.
* **beats the specialist windows**: net change at least 2.0 below M1's on the same seeds.
* **E** is read against M1: it replaces M1's companion with the lean one, and is better if its net change is at least 2.0 below M1's.
* **ranking**: the candidates that help, in order of net change, with M1 as the bar; a candidate that helps without a new fitted model for particular days (B, C, D, F: new inputs, still one model) is noted as simpler than M1.
* **limits**: one December and one January in the backtest; 25 seeds; the same 585 days; the weather is the real weather; the stand-in is built from the fold's training rows; a candidate that helps is a candidate to repeat on
  100 and fresh seeds, not a result; adopting any of them is Kevin's decision and needs an amendment (and, for a model change, a new forecast-log version) first.

## The version plan (agreed with Kevin, 9 November 2026; built and registered 14 November 2026, see `docs/FORECAST_LOG_V1_13_V1_14_REGISTRATION.md`)

How the versions stack. Only the hydro arm is extended; the portfolio and post arms stay frozen as benchmarks. Each version is a copy of the previous hydro arm with one idea, runs beside its parent, and is scored against it.

    v1 (1 Nov): portfolio (frozen single model, benchmark) | post (weekend package, no hydro flag, benchmark) | hydro (weekend package + Thursday hydro flag)
      v1.11  hydro arm + the payday intensity in the Saturday pieces                       scored as Q4 against v1's hydro
        v1.12  hydro arm + the closed-day intensity instead; the holiday rule beside it     scored as Q6 against v1.11
          v1.13  hydro arm + the year-end specialist models (below)                         Q7 against v1.12
            v1.14  hydro arm + the year-ago sales input (below)                             Q8 against v1.13

**v1.13, the year-end fix.** The hydro arm of v1.12 with two specialist models used only on two stretches, everything else v1.12's own files copied unchanged: on 26-31 December a companion (each piece refitted without
the ten recent-sales inputs), and on 1-8 January a masked model (each piece refitted with the ten inputs missing on those rows, training rows and forecast days). It differs from v1.12 on 14 days a year, so Q7 answers one
question; it is descriptive until the first window has passed (after 8 January 2027). Backtest expectation (Series 89, 92; 100 and fresh seeds): about -6 pounds a day over the 585 days served with today's stand-in (about 300 a
day on the 14 days), about -11 with the better stand-in S6. The alternative of the same strength, the history inputs rebuilt without 24 Dec-2 Jan plus the companion (Series 92: tied), changes every day's forecast and is
not the first choice for that reason. Parent: v1.12; needs an amendment, a refit and Kevin's go-ahead.

**v1.14, the year-ago sales input.** The hydro arm of v1.13 with one new input in every piece, `sales_364` (the sales on the same weekday 364 days earlier; 655 of the 1,006 training rows have it, none before 2025). It
changes every day's forecast, so it is scored over all days. Backtest expectation (Series 92): the days outside the holiday windows improve by about 6 to 8 pounds a day on average (served with S6), unevenly (Sundays get worse by about 19 and the early-2025 folds, short of year-ago history, too), late December does not move, and with
the companion it was the best package (-12 to -14). "At some version": the exact form is settled by one small 25-seed test before registering (last year's same weekday against, for example, the average of the days around
last year's date, and the input with and without the Sunday piece, since Sundays get worse). Parent: v1.13; needs an amendment, a refit and Kevin's go-ahead.
Series 93 (9 November 2026, composed from stored forecasts, read against v1.13 as Q8 would): the input in Monday-Saturday gives -8.03 (100 seeds) and -9.73 (fresh) pounds a day, the best form; in all four pieces -5.95 and -8.18 (Sunday +14.45 and +10.79, 44 of 84 Sundays worse); leaving Sunday out is better by 2.07 and 1.55, short of the 2.0 bar on the fresh seeds, so it is the default, not a proven necessity. With Sunday left out the Sunday forecasts equal v1.13's exactly.

**Series 94, the exact form of the v1.14 input (planned 9 November 2026, before it ran).** Candidates, each added to the Monday-Thursday (hydro), Friday and Saturday pieces (the Sunday piece stays v1.13's): A `sales_364`
(the reference), B `sales_365` (the same date), C `ya_wk3` (the average of the same weekday 357, 364 and 371 days earlier), D `ya_c7` (the average of the 7 days centred on last year's date, Kevin's suggestion), E `ya_ratio`
(`sales_364` over `ya_c7`), F `sales_364` and `ya_c7` together. 25 seeds, served with S6, read against v1.13 with its specialist routing. A form is better than A if at least 2.0 pounds a day below it, tied within 2.0, worse
above; a form that is better than A, or tied with a simpler build, goes on to 100 and fresh seeds; nothing is decided on 25 seeds.
**Series 94 result (9 November 2026, 25 seeds):** nothing beat A `sales_364` (-5.16 against v1.13): B +0.66, C +1.40, D +1.36, E -1.46, F -2.66 (every other form 2.5 to 6.6 pounds a day worse than A; smoothing the year-ago value or lining up the date loses what the single same weekday carries; a second input dilutes it). v1.14 keeps `sales_364` in the Monday-Thursday (hydro), Friday and Saturday pieces. Untested: a growth-adjusted year-ago value. See `docs/EXPERIMENT_LOG.md`, Series 94.

**The stand-in** is not a model version: it changes an input for every arm at once and needs no refit. Proposed: log it as a parallel variant beside today's stand-in (the log's `variant` column) for a while, so both are
scored on the same days, and decide after Christmas 2026; an amendment first, stating the date from which it applies. The other holidays (Easter, summer) are tested later on their own terms and get their own version
after v1.14 or a bundle, Kevin's decision. Every registered version must be back-run from Monday 2 November 2026 (Kevin's rule of 9 November, `replay_run.py`).

## What is registered, and what is not

Registered here: the candidates and the reading above (this file, committed before the run). **Not** registered: any model version in the forecast log, or any change to its stand-in. Results are added below after Series 90.

## The other holidays: a plan for later (written 9 November 2026, nothing run)

Kevin (9 November 2026): the rest of the holidays will be tested later, on their own terms, because he does not think they relate much to Christmas. Nothing from the Christmas work is assumed to carry over; this section only
records a first look at the data and the test that would be run, so that its reading exists before the results do. Which fixes go into forecast-log version v1.13 (the Christmas fix alone, or later all the holidays together) is
undecided and is Kevin's.

**A first look, without fits** (the package with the real manager number, 100 seeds; the break flags `is_christmas_break`, `is_easter_break`, `is_summer_break` of the engineered features; the 14-day input on the first day of
the transition against the level the next 14 days sold, and the package's average error over the first 7 days; backtest events only for the error):

| Break | Start: input vs level | Start: package error | End: input vs level | End: package error | Events in the backtest |
|---|---|---|---|---|---|
| Christmas | +22% | +960 | +39% | +1,024 | 2 (one winter) |
| Easter | -6% | -681 | +13% | +345 | 4 (2025, 2026) |
| Summer | -7% | -1,101 | +10% | -34 | 4 (2025, 2026) |

The directions match the idea that the recent-sales inputs lag a change of level (too high after a break ends, too low when one starts) but the size is about a third of Christmas's, there are no closed days, Easter moves,
Series 33 left Christmas out of its first-week correction because its first week is flat, and the manager's number was not biased at the summer start (+109). The summer start (-1,101 a day, 2025 and 2026) stands out; the
summer end shows nothing.

**The test, when it is run** (to be fixed and committed before it runs, as the Christmas one was):
* windows fixed from the calendar flags, not from the results: the first 7 days from each break start and the first 7 days after each break end, Easter and summer, 8 events in the backtest;
* the candidates: the ten recent-sales inputs set missing on those windows (training rows and forecast days alike), pooled over the 8 events; and, separately, a stand-in for the manager's number aligned to the holiday
  (Easter-relative for Easter, the break's own start for summer) instead of the date;
* controls: placebo windows of the same length placed at random away from any break, as in Series 80;
* served with the live stand-in and with the best stand-in of Series 89, 25 seeds first, repeated on 100 and fresh seeds only if it helps;
* reading, written before the run: HELPS if the pooled net change over the 585 days is at most -2.0 and it beats at least 9 of 10 placebo draws; NOTHING within 2.0 of zero; HURTS above +2.0; the six per-break
  groups (Easter start, Easter end, summer start, summer end) are reported separately, not judged;
* limits: eight events, two Easters and two summers; 25 seeds; the same 585 days.

Not registered in the forecast log; nothing in this section changes any model or input.

## Results

Series 90 (9 November 2026, 25 seeds, served with S6; the reading above was written before the run). Net change against the unmasked package served with S6, pounds a day:

| Candidate (served with S6; 25 seeds) | Net, 585 days | 26-31 Dec | 1-8 Jan | All other days | Reading set beforehand |
|---|---|---|---|---|---|
| M1 specialist windows (the reference) | **-10.11** | -522 | -331 | -0.23 | |
| B year-ago sales input (`sales_364`) | -3.02 | +81 | -56 | -3.16 | HELPS |
| C days-since-Christmas input | +5.34 | +86 | +4 | +4.51 | HURTS |
| D history inputs without the Christmas break | -4.76 | -139 | -322 | +1.09 | HELPS |
| E specialist windows with the lean companion | -6.22 | -124 | -331 | -0.43 | WORSE than M1 |
| F closed-day column in every piece | -1.31 | +83 | +41 | -2.78 | NOTHING |
| H random training masks, masked at forecast time | +9.68 | -228 | -240 | +15.67 | HURTS |

Combinations built afterwards from the stored forecasts (not fitted, not in the reading): D + the companion on 26-31 December -8.62 (26-31 Dec -518, 1-8 Jan -322, other days +1.13); D + the masked model on 1-8 January + the companion -8.77; B + the companion -8.83 (other days -2.91).

Ranking: M1 (specialist windows) -10.11; then the combinations D + companion -8.62 and B + companion -8.83; then E -6.22 (worse than M1); D -4.76; B -3.02; F -1.31 (nothing); C +5.34 and H +9.68 hurt. The simplest candidate that gets most of M1's gain is D (one model, an input change, no second model). Status: no candidate is registered in the forecast log; D and B are candidates to repeat on 100 and fresh seeds. See `docs/EXPERIMENT_LOG.md`, Series 90.

Series 92 (9 November 2026, the same candidates on the other 75 seeds and on fresh seeds 300-399, served with S6): net change against the unmasked package, pounds a day:

| Package (served with S6) | 25 seeds (Series 90) | All 100 seeds | Fresh 300-399 | 26-31 Dec (100 / fresh) | 1-8 Jan (100 / fresh) | Other days (100 / fresh) |
|---|---|---|---|---|---|---|
| M1 specialist windows | -10.11 | **-10.95** | **-10.65** | -575 / -548 | -349 / -345 | -0.29 / -0.31 |
| D history inputs without the Christmas break | -4.76 | -6.36 | -5.57 | -147 / -152 | -316 / -322 | -0.53 / +0.39 |
| D + the companion on 26-31 Dec | -8.62 | -10.69 | -9.59 | -570 / -544 | -316 / -322 | -0.53 / +0.39 |
| D + the masked model on 1-8 Jan + the companion | -8.77 | -11.43 | -10.17 | -570 / -544 | -349 / -343 | -0.84 / +0.10 |
| B year-ago sales input | -3.02 | -6.42 | -8.33 | +7 / +43 | -52 / -52 | **-5.93 / -8.26** |
| B + the companion on 26-31 Dec | -8.83 | **-12.00** | **-14.00** | -560 / -534 | -52 / -52 | -5.68 / -8.01 |

M1 holds (-10.95, -10.65); D + the companion is TIED with it (-10.69, -9.59) and D + the masked model + the companion is TIED (-11.43, -10.17); D alone and B alone are behind M1; B improves the days outside the windows (-5.9, -8.3) and B + the companion is numerically best (-12.00, -14.00), but B's gain is not a holiday fix and is uneven (Monday to Thursday and Saturday better, Sunday +18.6 worse; December-February and June-August better, early 2025 and spring 2026 worse; 56% of days improve). Status: no candidate registered in the forecast log; the choice between M1 and D + the companion is simplicity against nothing measurable; B goes to the general feature question.
