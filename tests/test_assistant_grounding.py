"""
The chat assistant quotes hardcoded figures (src/assistant/
model_comparison_data.py and sales_patterns_data.py). These tests check
every one against what src/evaluation/assistant_grounding.py computes from
the pipeline's and experiment log's result files, so a retrain or re-run
can't leave the assistant quoting stale numbers.

Needs reports/ (gitignored): run `python main.py` and
`python -m src.experiments.run_all_series` first, otherwise skipped.
"""

from __future__ import annotations

import pytest

from src.evaluation.assistant_grounding import EXPERIMENTS_DIR, RESULTS_DIR

REQUIRED = [
    RESULTS_DIR / "final_model_backtest_predictions.csv",
    RESULTS_DIR / "final_model_metrics.csv",
    EXPERIMENTS_DIR / "series17_summary.csv",
    EXPERIMENTS_DIR / "series17_manual_comparison_matched_days.csv",
    EXPERIMENTS_DIR / "series18_summary.csv",
    EXPERIMENTS_DIR / "business_impact_summary.csv",
]

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED),
    reason="needs the pipeline and experiment results in reports/",
)


def test_final_model_figures_match_the_results():
    from src.assistant import model_comparison_data as data
    from src.evaluation.assistant_grounding import final_model_results

    computed = final_model_results()
    assert data.VS_MANUAL_FORECAST == computed["VS_MANUAL_FORECAST"]
    assert data.PIPELINE_BACKTEST == computed["PIPELINE_BACKTEST"]
    assert data.ACCURACY_BY_DAYS_AHEAD == computed["ACCURACY_BY_DAYS_AHEAD"]
    assert data.BUSINESS_IMPACT_SIMULATION == computed["BUSINESS_IMPACT_SIMULATION"]
    assert data.FINAL_MODEL_MAPE_PCT == computed["VS_MANUAL_FORECAST"]["final_model"]["mape_pct"]
    assert data.FINAL_MODEL_PCT_BETTER_THAN_MANUAL == computed["VS_MANUAL_FORECAST"]["final_model_pct_better_than_manual"]


def test_sales_patterns_match_the_backtest():
    from src.assistant import sales_patterns_data as data
    from src.evaluation.assistant_grounding import sales_patterns

    computed = sales_patterns()
    for key in ["start_date", "end_date", "total_days"]:
        assert data.DATA_WINDOW[key] == computed["DATA_WINDOW"][key]

    def without(record: dict, *keys: str) -> dict:
        return {k: v for k, v in record.items() if k not in keys}

    for name in ["highest_sales_day", "lowest_sales_day", "best_forecast_day", "worst_forecast_day"]:
        assert without(data.BEST_WORST_SINGLE_DAY[name], "note") == computed["BEST_WORST_SINGLE_DAY"][name]
    assert without(data.BEST_WORST_SINGLE_WEEK, "definition") == computed["BEST_WORST_SINGLE_WEEK"]
    assert without(data.BEST_WORST_SINGLE_WEEKEND, "definition") == computed["BEST_WORST_SINGLE_WEEKEND"]
    assert data.BY_DAY_OF_WEEK == computed["BY_DAY_OF_WEEK"]
    assert data.BY_MONTH == computed["BY_MONTH"]
