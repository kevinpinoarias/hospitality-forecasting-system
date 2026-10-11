"""
Tests for the Series 19-21 experiment harness: that the tasks are what the docs say,
the random column orders are valid and reproducible, and - the one that matters most -
that the experiment's CatBoost loop produces exactly the predictions the real
BaggedCatBoost does, so every figure is comparable with the final model's.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.experiments import run_series19_column_order as s19
from src.experiments import run_series20_neural_v2 as s20
from src.experiments import run_series21_catboost_depth_iterations as s21
from src.experiments.report_series19_21 import block_bootstrap
from src.experiments.series19_21_common import bagged_log1p_predictions, read_done, score
from src.experiments.splits import generate_rolling_origin_folds
from src.models.final_model import DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL, BaggedCatBoost


# --- Series 19: the orders ---------------------------------------------------------


def test_random_orders_are_distinct_permutations_of_the_38_columns_and_not_the_current_order():
    orders = s19.random_orders(100)
    assert len(orders) == 100 and len({tuple(o) for o in orders}) == 100
    for order in orders:
        assert sorted(order) == sorted(FINAL_FEATURES) and order != FINAL_FEATURES


def test_random_orders_are_reproducible_and_depend_on_the_seed():
    assert s19.random_orders(10) == s19.random_orders(10)
    assert s19.random_orders(10) != s19.random_orders(10, rng_seed=20)


def test_structured_orders_are_valid_permutations():
    for name, order in s19.structured_orders().items():
        assert sorted(order) == sorted(FINAL_FEATURES), name
    assert s19.structured_orders()["candidates_first"][:23] == FINAL_FEATURES[15:]


def test_series19_stage1_has_the_documented_tasks():
    tasks = s19.stage1_tasks()
    kinds = [t["kind"] for t in tasks]
    assert len(tasks) == 1 + 100 + 3 + 30
    assert kinds.count("random_order") == 100 and kinds.count("seed_control") == 30
    assert len({t["id"] for t in tasks}) == len(tasks)
    base = tasks[0]
    assert base["id"] == "baseline_order" and base["features"] == FINAL_FEATURES and base["seeds"] == list(range(25))
    assert base["params"] == FINAL_PARAMS


def test_seed_controls_use_the_current_order_and_distinct_seed_sets():
    controls = [t for t in s19.stage1_tasks() if t["kind"] == "seed_control"]
    assert all(t["features"] == FINAL_FEATURES and len(t["seeds"]) == 25 for t in controls)
    all_seeds = [s for t in controls for s in t["seeds"]]
    assert len(all_seeds) == len(set(all_seeds))  # no seed is reused between sets


def test_confirmation_picks_the_best_random_orders_and_pairs_them_with_the_current_order():
    stage1 = {f"random_{i:03d}": {"id": f"random_{i:03d}", "kind": "random_order", "bag": {"MAE": 900 + i}} for i in range(1, 11)}
    orders = {k: list(reversed(FINAL_FEATURES)) for k in stage1}
    tasks = s19.confirm_tasks(stage1, orders, top=3)
    assert len(tasks) == 3 * 5 + 5
    assert {t["id"].split("_")[2] + "_" + t["id"].split("_")[3] for t in tasks if t["kind"] == "confirm_order"} == {"random_001", "random_002", "random_003"}
    assert sum(t["kind"] == "confirm_baseline" for t in tasks) == 5


# --- Series 20 / 21: the grids -------------------------------------------------------


def test_series20_feature_sets():
    assert len(s20.FEATURE_SETS["v1"]) == 15 and len(s20.FEATURE_SETS["v2"]) == 38
    assert s20.FEATURE_SETS["v2"][:15] == s20.FEATURE_SETS["v1"]
    assert s20.FEATURE_SETS["v2_no_year"] == [f for f in FINAL_FEATURES if f != "year"]


def test_series20_stages_have_the_documented_size():
    assert len(s20.stage_a_tasks()) == 4 * 3 * 5
    b = s20.stage_b_tasks()
    assert len(b) == 72 * 2 and len({t["id"] for t in b}) == len(b)
    assert all(t["seed"] == 42 for t in b)
    assert {t["feature_set"] for t in b} == {"v1", "v2"}


def test_series20_current_configs_are_the_production_defaults_and_the_series1_winners():
    from src.models import train_lstm, train_transformer

    assert s20.CURRENT_CONFIGS["lstm_production"][1] == {
        "hidden_size": train_lstm.HIDDEN_SIZE, "num_layers": train_lstm.NUM_LAYERS, "window": 30, "epochs": train_lstm.EPOCHS}
    trf = s20.CURRENT_CONFIGS["transformer_production"][1]
    assert trf["d_model"] == train_transformer.D_MODEL and trf["nhead"] == train_transformer.N_HEAD and trf["epochs"] == train_transformer.EPOCHS


def test_series20_transformer_grid_never_asks_for_heads_that_do_not_divide_the_model_width():
    for arch, hp in s20.grid_points():
        if arch == "transformer":
            assert hp["d_model"] % hp["nhead"] == 0


def test_series21_grid_is_60_unique_configurations_including_the_current_model():
    tasks = s21.grid_tasks()
    assert len(tasks) == 3 * 5 * 4 and len({t["id"] for t in tasks}) == 60
    current = [t for t in tasks if t["params"] == FINAL_PARAMS]
    assert len(current) == 1 and current[0]["features"] == FINAL_FEATURES and current[0]["seeds"] == list(range(25))


def test_series21_confirmation_uses_fresh_seeds_and_includes_the_current_model():
    grid = {t["id"]: {"id": t["id"], "params": t["params"], "bag": {"MAE": 900 + i}} for i, t in enumerate(s21.grid_tasks())}
    tasks = s21.confirm_tasks(grid, top=5)
    assert len(tasks) == 6 and all(t["seeds"] == s21.FRESH_SEEDS for t in tasks)
    assert set(s21.FRESH_SEEDS).isdisjoint(range(25))
    assert any(t["params"] == FINAL_PARAMS for t in tasks)


# --- the harness reproduces the real model -------------------------------------------


def synthetic_frame(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data = {DATE_COL: pd.date_range("2024-01-01", periods=rows)}
    for name in FINAL_FEATURES:
        data[name] = rng.normal(size=rows)
    df = pd.DataFrame(data)
    df[TARGET_COL] = np.exp(8 + 0.3 * df["forecast_sales"] + 0.1 * rng.normal(size=rows))
    return df


def test_the_experiment_loop_matches_bagged_catboost_exactly():
    df = synthetic_frame()
    folds = generate_rolling_origin_folds(df[DATE_COL], initial_train_days=200, test_days=50, step_days=50)
    params, seeds = {"iterations": 15, "learning_rate": 0.1, "depth": 3}, [0, 1, 2]
    mine = bagged_log1p_predictions(df, folds, FINAL_FEATURES, params, seeds).mean(axis=0)
    expected = []
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        expected.append(BaggedCatBoost.fit(train, FINAL_FEATURES, params, seeds).predict(test))
    np.testing.assert_array_equal(mine, np.concatenate(expected))


def test_column_order_can_change_the_predictions():
    """The premise of Series 19: with a single seed, a different column order gives different predictions."""
    df = synthetic_frame()
    folds = generate_rolling_origin_folds(df[DATE_COL], initial_train_days=200, test_days=50, step_days=50)[:1]
    params = {"iterations": 15, "learning_rate": 0.1, "depth": 3}
    a = bagged_log1p_predictions(df, folds, FINAL_FEATURES, params, [0])
    b = bagged_log1p_predictions(df, folds, list(reversed(FINAL_FEATURES)), params, [0])
    assert not np.array_equal(a, b)


# --- scoring, bootstrap, resumability ------------------------------------------------


def test_score_reports_pooled_and_fold_mean_mae():
    actual, pred = np.array([10.0, 20.0, 30.0, 40.0]), np.array([12.0, 18.0, 33.0, 36.0])
    out = score(actual, pred, np.array([0, 0, 1, 1]))
    assert out["MAE"] == pytest.approx(2.75)
    assert out["per_fold_MAE"] == pytest.approx([2.0, 3.5]) and out["MAE_fold_mean"] == pytest.approx(2.75)


def test_block_bootstrap_ci_covers_a_zero_mean_difference_and_collapses_for_a_constant_one():
    rng = np.random.default_rng(0)
    mean, lo, hi = block_bootstrap(rng.normal(0, 50, size=300))
    assert lo < 0 < hi and lo <= mean <= hi
    assert block_bootstrap(np.full(100, 7.0)) == (7.0, 7.0, 7.0)


def test_read_done_returns_finished_task_ids(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({"id": "a", "x": 1}) + "\n" + json.dumps({"id": "b", "x": 2}) + "\n", encoding="utf-8")
    assert set(read_done(path)) == {"a", "b"}
    assert read_done(tmp_path / "missing.jsonl") == {}


# --- Series 22: letting the network see the day it predicts ---------------------------


def window_frame(n: int = 12, gap_after: int | None = None) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n)
    if gap_after is not None:  # drop one calendar day, as the real data does for Christmas/New Year
        dates = dates.delete(gap_after)
    return pd.DataFrame({DATE_COL: dates, TARGET_COL: np.arange(len(dates), dtype=float) * 100,
                         "f": np.arange(len(dates), dtype=float)})


def test_default_windows_are_the_days_before_the_target_and_exclude_it():
    from src.models.neural_common import build_windows

    df = window_frame()
    X, y, dates = build_windows(df, ["f"], window=3)
    assert X.shape == (9, 3, 1) and y[0] == 300.0           # first target is row 3 (needs 3 days before it)
    assert X[0, :, 0].tolist() == [0.0, 1.0, 2.0]           # the three days BEFORE row 3 - not row 3 itself
    assert dates.iloc[0] == df[DATE_COL].iloc[3]


def test_include_target_day_ends_the_window_at_the_predicted_day_and_never_uses_the_target_value():
    from src.models.neural_common import build_windows

    df = window_frame()
    X, y, dates = build_windows(df, ["f"], window=3, include_target_day=True)
    assert X.shape == (10, 3, 1)                            # one more usable day: the first target is row 2
    assert X[0, :, 0].tolist() == [0.0, 1.0, 2.0] and y[0] == 200.0 and dates.iloc[0] == df[DATE_COL].iloc[2]
    # the target (sales) is in y only - the feature column f is not derived from it
    X1, y1, _ = build_windows(df, ["f"], window=1, include_target_day=True)
    assert X1.shape == (12, 1, 1) and X1[:, 0, 0].tolist() == list(range(12))   # window 1 = the day's own features only


def test_windows_spanning_a_missing_calendar_day_are_skipped_in_both_modes():
    from src.models.neural_common import build_windows

    df = window_frame(gap_after=5)                          # 2025-01-06 has no row
    for include in (False, True):
        X, _, dates = build_windows(df, ["f"], window=3, include_target_day=include)
        for d in dates:
            span = pd.date_range(end=d, periods=3 if include else 4)
            assert pd.Timestamp("2025-01-06") not in span


def test_series22_selects_the_best_confirmed_settings_per_network_and_feature_set(monkeypatch):
    from src.experiments import run_series22_neural_target_day as s22

    best = {("lstm", "v1"): [{"hp": {"hidden_size": 32, "num_layers": 1, "window": 14, "epochs": 40}, "mae": 1100.0}] * 3,
            ("transformer", "v2"): [{"hp": {"d_model": 16, "nhead": 2, "window": 7, "epochs": 40}, "mae": 1200.0}] * 3}
    d, e = s22.stage_d_tasks(best), s22.stage_e_tasks(best)
    assert len(d) == 2 * 3 * 5 and all(t["include_target_day"] and t["seed"] in range(100, 105) for t in d)
    assert len(e) == 2 * 5 * 3 and {t["hp"]["window"] for t in e} == {1, 3, 7, 14, 30}
    assert len({t["id"] for t in d + e}) == len(d) + len(e)


# --- the error autopsy ------------------------------------------------------------------


def test_holiday_context_names_the_day_or_its_neighbour():
    from src.experiments.run_error_autopsy import holiday_context

    cal = {pd.Timestamp("2025-05-26").date(): "Spring Bank Holiday"}
    assert holiday_context(pd.Timestamp("2025-05-26"), cal) == "Spring Bank Holiday"
    assert holiday_context(pd.Timestamp("2025-05-25"), cal) == "1 day before Spring Bank Holiday"
    assert holiday_context(pd.Timestamp("2025-05-28"), cal) == "2 days after Spring Bank Holiday"
    assert holiday_context(pd.Timestamp("2025-06-10"), cal) == ""


def test_same_weekday_average_uses_the_previous_weeks_and_needs_at_least_two():
    from src.experiments.run_error_autopsy import same_weekday_average

    sales = pd.Series({pd.Timestamp("2025-01-01") + pd.Timedelta(days=7 * k): float(k) for k in range(6)})
    assert same_weekday_average(pd.Timestamp("2025-02-12"), sales) == pytest.approx(np.mean([5.0, 4.0, 3.0, 2.0]))
    assert np.isnan(same_weekday_average(pd.Timestamp("2025-01-08"), sales))  # only one earlier week exists


def test_school_phase_labels_each_part_of_a_break():
    from src.experiments.run_error_autopsy import school_phase

    breaks = pd.DataFrame([{"break_type": "summer", "start": "2025-06-30", "end": "2025-08-15"}])
    assert school_phase(pd.Timestamp("2025-06-25"), breaks) == "last week of term"
    assert school_phase(pd.Timestamp("2025-07-02"), breaks) == "first week of a break"
    assert school_phase(pd.Timestamp("2025-07-20"), breaks) == "middle of a break"
    assert school_phase(pd.Timestamp("2025-08-12"), breaks) == "last week of a break"
    assert school_phase(pd.Timestamp("2025-05-01"), breaks) == "other term time"


def test_category_table_error_index_compares_share_of_error_with_share_of_days():
    from src.experiments.run_error_autopsy import category_table

    df = pd.DataFrame({"kind": ["a"] * 6 + ["b"] * 2, "abs_err": [10.0] * 6 + [60.0] * 2, "gap": [1.0] * 8})
    t = category_table(df, "kind").set_index("group")
    assert t.loc["a", "days"] == 6 and t.loc["b", "MAE"] == 60.0
    assert t.loc["b", "share_of_days_%"] == pytest.approx(25.0) and t.loc["b", "share_of_error_%"] == pytest.approx(66.667, abs=0.01)
    assert t.loc["b", "error_index"] > 2 > t.loc["a", "error_index"]


def test_forward_chained_corrections_help_a_steady_bias_and_only_use_earlier_blocks():
    from src.experiments.run_error_autopsy import forward_chained_corrections

    rng = np.random.default_rng(0)
    n = 280
    ranked = pd.DataFrame({"fold_id": np.repeat(np.arange(4), 70),
                           "weekday": np.tile(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 40),
                           "actual": rng.uniform(5000, 15000, n)})
    ranked["model"] = ranked["actual"] * 0.9 + rng.normal(0, 200, n)   # the model is always 10% too low
    out = forward_chained_corrections(ranked).set_index("correction (learned from earlier blocks only)")
    assert (out["uncorrected MAE"] > 0).all()
    assert out.loc["Each weekday shifted by its past average bias (full strength)", "change"] < 0   # a steady bias is correctable
