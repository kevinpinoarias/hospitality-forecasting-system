"""
Series 14 - Log1p target transform.

Hypothesis: daily sales are right-skewed with occasional large spikes that
dominate a squared-error objective. Training on log1p(sales) and inverting
predictions with expm1 compresses those spikes and should reduce MAE.
log1p rather than log because the four known closure days have zero sales.

Method:
1. Log1p at the reference hyperparameters, against the raw-scale reference.
2. A dedicated 18-combination grid in log space, since the transform
   changes the loss surface the hyperparameters were tuned on.
3. The best log-space configuration re-checked on the sliding-365 window.
4. Full per-fold metrics and an actual-vs-predicted figure for the winner.

Result reproduced:

    | Configuration                                    | MAE    | MAE std |
    | Raw scale, reference hyperparameters             | 907.77 | 182.08  |
    | Log1p, same hyperparameters                      | 919.55 | 148.93  |
    | Log1p, best of grid (300 / 0.05 / 4)             | 892.38 | 153.52  |
    | Log1p, best config, sliding-365 window           | 933.56 | 159.63  |

Confirmed at the time as the first genuine improvement since Series 2 -
but see Series 16, which shows most of this £15.39 gap was a favourable
seed rather than the transform. See docs/EXPERIMENT_LOG.md, Series 14.

Outputs (reports/experiments/):
- series14_log_transform_grid_results.csv
- series14_best_config_per_fold.csv
- series14_plot_data.csv            (reused by Series 15 and 16)
- series14_full_metrics_per_fold.csv
Figure: reports/figures/series14_log_transform_vs_reference_stacked.png
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.experiments.series_common import (
    ADD_ALL_CANDIDATES, DATE_COL, FIGURES_DIR, LOG1P_PARAMS, REFERENCE_PARAMS, TARGET_COL,
    build_folds, fit_predict_catboost_seeded, load_data, results_path, slice_test_window, slice_train_window,
)

LOG1P_GRID = [
    {"iterations": it, "learning_rate": lr, "depth": d}
    for it, lr, d in product([100, 300, 500], [0.03, 0.05, 0.1], [4, 6])
]


def _score(params: dict, window: int | None, log1p_target: bool) -> pd.DataFrame:
    df = load_data()
    rows = []
    for fold in build_folds(df):
        train_df = slice_train_window(df, DATE_COL, fold, window)
        test_df = slice_test_window(df, DATE_COL, fold)
        pred = fit_predict_catboost_seeded(train_df, test_df, params, ADD_ALL_CANDIDATES, log1p_target=log1p_target)
        rows.append({"fold_id": fold.fold_id, **regression_metrics(test_df[TARGET_COL], pred)})
    return pd.DataFrame(rows)


def _summary(per_fold: pd.DataFrame) -> tuple[float, float]:
    return per_fold["MAE"].mean(), per_fold["MAE"].std()


def build_plot_data() -> pd.DataFrame:
    df = load_data()
    records = []
    for fold in build_folds(df):
        train_df = slice_train_window(df, DATE_COL, fold, None)
        test_df = slice_test_window(df, DATE_COL, fold)
        ref_pred = fit_predict_catboost_seeded(train_df, test_df, REFERENCE_PARAMS, ADD_ALL_CANDIDATES)
        log_pred = fit_predict_catboost_seeded(train_df, test_df, LOG1P_PARAMS, ADD_ALL_CANDIDATES, log1p_target=True)
        for d, actual, rp, lp in zip(test_df[DATE_COL], test_df[TARGET_COL], ref_pred, log_pred):
            records.append({"date": d, "fold_id": fold.fold_id, "actual": actual, "ref_pred": rp, "log_pred": lp})
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def plot_stacked(plot_data: pd.DataFrame, ref_mae: float, log_mae: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # Blank row at each fold boundary so lines don't bridge the gap between
    # two independent test windows. Plotted directly rather than via the
    # shared plotting helper, whose internal dropna() would erase these.
    rows, prev = [], None
    for _, r in plot_data.iterrows():
        if prev is not None and r["fold_id"] != prev:
            rows.append({"date": r["date"], "fold_id": r["fold_id"], "actual": np.nan, "ref_pred": np.nan, "log_pred": np.nan})
        rows.append(r.to_dict())
        prev = r["fold_id"]
    gapped = pd.DataFrame(rows)
    bounds = plot_data.groupby("fold_id")["date"].agg(["min", "max"])

    fig, axes = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    panels = [
        (axes[0], "ref_pred", "#d62728", f"Reference (raw-scale) - MAE £{ref_mae:.2f}", "Actual vs. Reference Model (raw-scale)"),
        (axes[1], "log_pred", "#1f77b4", f"Log1p transform - MAE £{log_mae:.2f}", "Actual vs. Log1p-Transformed Model (Series 14 winner)"),
    ]
    for ax, col, colour, label, title in panels:
        ax.plot(gapped["date"], gapped["actual"], color="#222222", linewidth=1.8, label="Actual", zorder=3)
        ax.plot(gapped["date"], gapped[col], color=colour, linewidth=1.5, label=label, zorder=2)
        for i, (_, b) in enumerate(bounds.iterrows()):
            if i % 2 == 0:
                ax.axvspan(b["min"], b["max"], color="#000000", alpha=0.03, zorder=0)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Total Sales (£)")
        ax.legend(loc="upper left", frameon=True, fontsize=10)
        ax.grid(True, alpha=0.25)
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.suptitle("10 independent rolling-origin test folds", fontsize=11, y=0.995)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "series14_log_transform_vs_reference_stacked.png", dpi=150)
    plt.close(fig)


def run() -> dict:
    ref = _summary(_score(REFERENCE_PARAMS, None, log1p_target=False))
    log_same = _summary(_score(REFERENCE_PARAMS, None, log1p_target=True))

    grid_rows = []
    for params in LOG1P_GRID:
        per_fold = _score(params, None, log1p_target=True)
        per_fold.insert(0, "params", str(params))
        grid_rows.append(per_fold)
    grid = pd.concat(grid_rows, ignore_index=True)
    grid.to_csv(results_path("series14_log_transform_grid_results.csv"), index=False)
    grid_board = (
        grid.groupby("params").agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std")).reset_index().sort_values("MAE_mean")
    )

    best_expanding = _score(LOG1P_PARAMS, None, log1p_target=True)
    best_expanding.to_csv(results_path("series14_best_config_per_fold.csv"), index=False)
    best_sliding = _summary(_score(LOG1P_PARAMS, 365, log1p_target=True))

    plot_data = build_plot_data()
    plot_data.to_csv(results_path("series14_plot_data.csv"), index=False)

    per_fold_rows = []
    for fold_id, g in plot_data.groupby("fold_id"):
        per_fold_rows.append({"fold_id": fold_id, "model": "reference", **regression_metrics(g["actual"], g["ref_pred"])})
        per_fold_rows.append({"fold_id": fold_id, "model": "log1p", **regression_metrics(g["actual"], g["log_pred"])})
    full_metrics = pd.DataFrame(per_fold_rows)
    full_metrics.to_csv(results_path("series14_full_metrics_per_fold.csv"), index=False)

    ref_plot_mae = full_metrics.query("model == 'reference'")["MAE"].mean()
    log_plot_mae = full_metrics.query("model == 'log1p'")["MAE"].mean()
    plot_stacked(plot_data, ref_plot_mae, log_plot_mae)

    return {
        "raw_scale_reference": ref,
        "log1p_same_hyperparameters": log_same,
        "log1p_grid_leaderboard": grid_board,
        "log1p_best_expanding": _summary(best_expanding),
        "log1p_best_sliding_365": best_sliding,
        "aggregate_metrics": full_metrics.groupby("model")[["MAE", "RMSE", "MAPE", "Bias"]].agg(["mean", "std"]),
    }


if __name__ == "__main__":
    out = run()
    for key in ["raw_scale_reference", "log1p_same_hyperparameters", "log1p_best_expanding", "log1p_best_sliding_365"]:
        mae, std = out[key]
        print(f"{key}: MAE={mae:.2f}  std={std:.2f}")
    print()
    print(out["log1p_grid_leaderboard"].head(5).to_string(index=False))
    print()
    print(out["aggregate_metrics"].round(2).to_string())
