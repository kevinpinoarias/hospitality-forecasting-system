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

## What is registered, and what is not

Registered here: the candidates and the reading above (this file, committed before the run). **Not** registered: any model version in the forecast log, or any change to its stand-in. Results are added below after Series 90.

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
