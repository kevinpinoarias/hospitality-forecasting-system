# The model picker (design record)

Status: **built** on 23 and 24 November 2026 (branch `model-picker`). Decisions are Kevin's; the reasoning is kept here.

## 1. What it is

Anyone using the chat, the API or the MCP tools can choose which model answers: the standard (live) model, or one of the eight versions the weekly forecast log tests (v1-post, v1-hydro, v1.11, v1.12, v1.13, v1.14, v1.131, v1.141). The point is the portfolio's: a visitor can compare the live model with the experimental ones. Every version is labelled **under test**, because the forecast log's formal readings come only after 26 to 52 weeks.

## 2. Decisions

| Question | Decision | Why |
|---|---|---|
| Which choices | All nine, each with a one-line legend and "under test since <date>" | Shows the whole line of work; the label stops anyone reading a version as a proven improvement |
| Rain scenarios for versions | Yes, worked out as for the standard model (the same rows with the rain swapped) | Answers look the same whichever model is chosen, and the assistant's rain rules keep working |
| Past dates | Each version's own backtest, the live model's protocol (10 folds, 585 days), with the manager's real number | The same footing as the live model's "made before the day" forecasts; the stand-in plays no part, so v1.131 = v1.13 and v1.141 = v1.14 here |
| How versions are refitted for the backtest | From each piece's registered definition, after proving all 23 distinct pieces reproduce the registered models' predictions exactly | The backtest is of the versions as registered, not of a re-implementation; CatBoost files differ byte-wise between two identical fits, so predictions are compared |
| Where the forecasting code lives | Moved unchanged into `src/forecast_core/` (option B), shared by the forecast log, the experiments and the API | The experiment modules pulled in xgboost, scikit-learn, wandb and matplotlib; adding those to the API image (option A) would have added 300 to 400 MB of unused libraries. One copy of the code means the log and the API cannot drift apart. Proven: every logged Monday run and a Christmas-to-January block replay identically (largest difference 0.0) |
| Where the model files live | Assets of the GitHub release `model-versions-1`; CI fetches and fingerprint-checks them | Too large for git and never changed; before this they existed only on one PC (no backup) |
| Memory | Versions load the first time they are picked; shared files load once; the API gets 1 GB | Measured: 223 MB for the standard model, about 645 MB with every version loaded, above Cloud Run's default 512 MB |
| How the chat's choice reaches the tools | A hidden per-message setting (option A): LangChain passes it to the forecast tools, the LLM never handles it; a model named in the chat wins | Option B (telling the LLM which model is selected) could be forgotten silently |
| Speed | Not optimised (Kevin, 24 November): a version answers in about 1.4 s per request; a weather cache and cached rows were considered and dropped as not worth it now | |

## 3. Results

Average miss over the 585 backtest days: standard £909.57, v1-post £864.05, v1-hydro £858.32, v1.11 £856.62, v1.12 £855.62, v1.13 £846.75, v1.14 £845.82. These match the experiment log's figures for the same recipes (the staircase of Section 35 and Series 82). v1.14's gain over v1.13 is about £1 a day here against about £8 in Series 93, because Series 89 to 94 were run without the manager's number (the stand-in); last year's sales help most when that number is missing. The versions were chosen by looking at these same days, so their backtest flatters them a little; the forecast log is the clean test.

## 4. Limits

- A version answers a past date only inside the backtest (15 January 2025 to 6 September 2026); later past dates were in its training data.
- A Thursday beyond the Hydro show list (it reaches 31 December 2026) gets no forecast from a model with the Hydro flag, as in the log.
- With `ASSISTANT_TOOLS=mcp` (off in the live app) only a model named in the chat is used, not the dropdown.
- The 19-case assistant evaluation has to be re-run with the new instructions; Gemini was unavailable (503) on 23 and 24 November.
