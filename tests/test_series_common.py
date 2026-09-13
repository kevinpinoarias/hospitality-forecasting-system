import importlib.util

import pandas as pd
import pytest

from src.experiments.run_all_series import LATER_SERIES
from src.experiments.series_common import apply_december_ratio, fold_mean_metrics, pooled_mae, require_output


def _preds():
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-11-30", "2025-12-01", "2025-12-24", "2026-01-01"]),
        "actual": [100.0, 200.0, 300.0, 400.0],
        "pred": [110.0, 180.0, 330.0, 360.0],
        "fold_id": [0, 0, 1, 1],
    })


def test_december_ratio_multiplies_only_december_days():
    out = apply_december_ratio(_preds(), {1: 2.0, 24: 0.5}, out_col="corrected")

    assert out["corrected"].tolist() == [110.0, 360.0, 165.0, 360.0]


def test_december_ratio_leaves_original_predictions_untouched():
    preds = _preds()
    out = apply_december_ratio(preds, {1: 2.0, 24: 0.5}, out_col="corrected")

    assert out["pred"].tolist() == preds["pred"].tolist()


def test_pooled_mae_overall_and_december_only():
    preds = _preds()

    assert pooled_mae(preds, "pred") == pytest.approx((10 + 20 + 30 + 40) / 4)
    assert pooled_mae(preds, "pred", december_only=True) == pytest.approx((20 + 30) / 2)


def test_fold_mean_metrics_averages_per_fold_not_per_day():
    preds = pd.DataFrame({
        "actual": [100.0, 100.0, 100.0, 100.0],
        "pred": [110.0, 110.0, 110.0, 190.0],
        "fold_id": [0, 0, 0, 1],
    })
    # Fold 0 MAE = 10 (three days), fold 1 MAE = 90 (one day): the fold
    # average is 50, whereas pooling all four days would give 30.
    assert fold_mean_metrics(preds, "pred")["MAE_mean"] == pytest.approx(50.0)


def test_require_output_names_the_script_to_run():
    with pytest.raises(FileNotFoundError, match="run_series9_multiplicative_correction"):
        require_output("definitely_not_a_real_output.csv", "run_series9_multiplicative_correction")


def test_every_orchestrated_series_module_exists():
    for _, module_name in LATER_SERIES:
        assert importlib.util.find_spec(f"src.experiments.{module_name}") is not None, module_name
