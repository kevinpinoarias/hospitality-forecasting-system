"""
Series 24 - an adaptive Friday-to-Sunday bias correction on the "raw Saturday" hybrid.

Series 23 found the raw-£ target beats the log target on Saturdays only (£1,177 against
£1,290) while the log target is as good or better on every other day. This series starts
from that HYBRID (raw model on Saturdays, the final log model on every other day) and asks
whether a correction learned from past misses improves it further.

THE CORRECTION. Before each 60-day test block, take each of Friday, Saturday and Sunday
separately and compute the hybrid's average (forecast minus actual) over that weekday's days
in EARLIER blocks only, giving recent days more weight (weight halves every `half_life`
weeks of age, measured from the start of the block being corrected). Subtract `strength` times
that bias from the block's forecasts for that weekday. Monday to Thursday are left alone. Each
weekday gets its own correction. The first block has no history and is left as it is, so scoring
is on blocks 1-9 (the autopsy's convention).

PRE-REGISTERED PLAN (written before any result was looked at)
  grid       half-life {4, 8, 16} weeks x strength {0.25, 0.5, 0.75} = 9 variants, scope Friday to
             Sunday (the Saturday-only and all-days scopes of the roadmap's 27-variant grid were
             dropped by decision, so 9 variants are tried on one base)
  base       raw_same_params on Saturdays, log1p_final elsewhere, seeds 0-24 (Series 23 predictions)
  step 1     the variant with the lowest pooled MAE on blocks 1-9 is the winner
  step 2     refit the log and raw models on 25 FRESH seeds (100-124), build the hybrid, apply the
             winner's exact settings (no re-choosing) and measure the change against the
             uncorrected hybrid on the same days
  bar        the winner is kept only if on the fresh seeds (a) the 95% moving-block bootstrap interval
             of the pooled MAE change excludes zero and is negative, and (b) neither Friday nor Sunday
             is significantly worse (change interval entirely above zero)
  reporting  the number of variants tried (9) is stated; the step-1 gain is optimistic, the step-2
             figure is the one that counts. Anything else (the other 8 variants on the fresh seeds, the
             correction on the log-everywhere base) is context, not a selection.

Run:  python -m src.experiments.run_series24_weekend_correction [--workers 10]
"""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments.series19_21_common import SUFFIX
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments.series19_21_common import OUT_DIR, init_worker, load_protocol, read_done, run_parallel, test_days

RESULTS_FRESH = OUT_DIR / "series24_fresh_seeds.jsonl"
REPORT_DIR = CONFIG.path("experiments_dir") / f"series24_weekend_correction{SUFFIX}"
FRESH_SEEDS = list(range(100, 125))
HALF_LIVES = [4, 8, 16]
STRENGTHS = [0.25, 0.5, 0.75]
SCOPE = ("Friday", "Saturday", "Sunday")
MIN_HISTORY = 4
BASE_VARIANTS = ("log1p_final", "raw_same_params")


def hybrid(weekday: np.ndarray, log_pred: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
    """Raw-target forecast on Saturdays, log-target forecast on every other day."""
    return np.where(weekday == "Saturday", raw_pred, log_pred)


def correct(days: pd.DataFrame, pred: np.ndarray, half_life_weeks: float, strength: float, scope: tuple[str, ...] = SCOPE) -> np.ndarray:
    """The adaptive weekday correction. Block k's forecasts for each weekday in `scope` are shifted by
    -strength x (the recency-weighted mean of forecast minus actual for that weekday in blocks BEFORE k).
    Block 0, and weekdays outside `scope`, are unchanged."""
    out = pred.astype(float).copy()
    fold = days["fold_id"].to_numpy()
    weekday = days["weekday"].to_numpy()
    dates = pd.to_datetime(days["date"]).to_numpy()
    error = pred - days["actual"].to_numpy()
    for k in sorted(set(fold.tolist()))[1:]:
        block = fold == k
        start = dates[block].min()
        for day in scope:
            history = (fold < k) & (weekday == day)
            if history.sum() < MIN_HISTORY:
                continue
            age_weeks = (start - dates[history]) / np.timedelta64(7, "D")
            weights = 0.5 ** (age_weeks / half_life_weeks)
            bias = float(np.sum(weights * error[history]) / np.sum(weights))
            out[block & (weekday == day)] = pred[block & (weekday == day)] - strength * bias
    return out


def mae(actual: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(pred[mask] - actual[mask]).mean())


def grid_table(days: pd.DataFrame, base: np.ndarray) -> pd.DataFrame:
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    scored = days["fold_id"].to_numpy() >= 1
    base_mae = mae(a, base, scored)
    rows = []
    for half_life, strength in product(HALF_LIVES, STRENGTHS):
        corrected = correct(days, base, half_life, strength)
        row = {"half_life_weeks": half_life, "strength": strength, "MAE": mae(a, corrected, scored), "uncorrected_MAE": base_mae}
        row["change"] = row["MAE"] - base_mae
        for day in SCOPE:
            m = scored & (wd == day)
            row[f"{day}_change"] = mae(a, corrected, m) - mae(a, base, m)
        rows.append(row)
    return pd.DataFrame(rows)


def change_report(days: pd.DataFrame, base: np.ndarray, corrected: np.ndarray) -> pd.DataFrame:
    """Change in MAE (corrected minus uncorrected) overall and by weekday, with 95% block-bootstrap intervals,
    on the scored blocks only."""
    scored = days["fold_id"].to_numpy() >= 1
    a, wd = days["actual"].to_numpy()[scored], days["weekday"].to_numpy()[scored]
    diff = np.abs(corrected[scored] - a) - np.abs(base[scored] - a)
    rows = []
    for name, mask in [("All scored days", np.ones(len(a), bool))] + [(d, wd == d) for d in s23.WEEKDAYS]:
        mean, lo, hi = s23.masked_block_bootstrap(diff, mask.astype(float))
        rows.append({"segment": name, "days": int(mask.sum()), "uncorrected_MAE": float(np.abs(base[scored] - a)[mask].mean()),
                     "corrected_MAE": float(np.abs(corrected[scored] - a)[mask].mean()), "change": mean, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def fresh_tasks() -> list[dict]:
    out = []
    for variant in BASE_VARIANTS:
        for i in range(0, len(FRESH_SEEDS), s23.SEED_CHUNK):
            seeds = FRESH_SEEDS[i:i + s23.SEED_CHUNK]
            out.append({"id": f"{variant}_s{seeds[0]}-{seeds[-1]}", "variant": variant, "seeds": seeds})
    return out


def load_bags(path, expected_seeds: int) -> dict[str, np.ndarray]:
    done = read_done(path)
    bags = {}
    for variant in BASE_VARIANTS:
        per_seed = np.concatenate([np.array(r["per_seed"]) for r in done.values() if r["variant"] == variant], axis=0)
        assert per_seed.shape[0] == expected_seeds, (variant, per_seed.shape)
        bags[variant] = per_seed.mean(axis=0)
    return bags


def make_figures(grid: pd.DataFrame, days: pd.DataFrame, base: np.ndarray, corrected: np.ndarray, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    pivot = grid.pivot(index="half_life_weeks", columns="strength", values="change")
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(3), [f"{s}" for s in pivot.columns]); ax.set_yticks(range(3), [f"{h} wk" for h in pivot.index])
    ax.set_xlabel("strength"); ax.set_ylabel("memory half-life")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{pivot.to_numpy()[i, j]:+.1f}", ha="center", va="center", fontsize=9)
    ax.set_title("Change in MAE vs the uncorrected hybrid (£, seeds 0-24)", fontsize=9)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(REPORT_DIR / "grid.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    a, wd, fold = days["actual"].to_numpy(), days["weekday"].to_numpy(), days["fold_id"].to_numpy()
    for ax, day in zip(axes, SCOPE):
        m = wd == day
        for pred, name, colour in ((base, "uncorrected hybrid", "#c64600"), (corrected, f"corrected ({label})", "#1a5fb4")):
            ax.plot(sorted(set(fold[m].tolist())), [(pred[m & (fold == k)] - a[m & (fold == k)]).mean() for k in sorted(set(fold[m].tolist()))],
                    marker="o", label=name, color=colour)
        ax.axhline(0, color="#888", lw=1); ax.set_title(day); ax.set_xlabel("60-day block")
    axes[0].set_ylabel("average forecast minus actual (£)"); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(REPORT_DIR / "bias_by_block.png", dpi=130); plt.close(fig)


def main(workers: int) -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, wd, scored = days["actual"].to_numpy(), days["weekday"].to_numpy(), days["fold_id"].to_numpy() >= 1
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

    # ---- step 1: choose the winner on the seeds 0-24 predictions
    bags = s23.load_bags()
    base = hybrid(wd, bags["log1p_final"], bags["raw_same_params"])
    print(f"\nScored days (blocks 1-9): {int(scored.sum())}.  MAE on them:  log everywhere {mae(a, bags['log1p_final'], scored):.2f}   "
          f"raw everywhere {mae(a, bags['raw_same_params'], scored):.2f}   hybrid (raw Saturdays) {mae(a, base, scored):.2f}")
    grid = grid_table(days, base)
    grid.to_csv(REPORT_DIR / "grid.csv", index=False)
    print("\nStep 1 - the 9 variants on seeds 0-24 (change vs the uncorrected hybrid, £; negative = better):\n", grid.round(2).to_string(index=False))
    best = grid.loc[grid["MAE"].idxmin()]
    hl, st = int(best["half_life_weeks"]), float(best["strength"])
    print(f"\nWinner: half-life {hl} weeks, strength {st}: MAE {best['MAE']:.2f}, change {best['change']:+.2f} (optimistic: it picked itself)")
    change_report(days, base, correct(days, base, hl, st)).to_csv(REPORT_DIR / "winner_seeds0_24.csv", index=False)

    # ---- step 2: fresh seeds, same settings
    run_parallel(fresh_tasks(), s23.fit_task, RESULTS_FRESH, workers, init_worker, "series24 fresh seeds")
    fresh = load_bags(RESULTS_FRESH, len(FRESH_SEEDS))
    base_f = hybrid(wd, fresh["log1p_final"], fresh["raw_same_params"])
    corrected_f = correct(days, base_f, hl, st)
    print(f"\nFresh seeds 100-124, scored days: log everywhere {mae(a, fresh['log1p_final'], scored):.2f}   hybrid {mae(a, base_f, scored):.2f}   "
          f"hybrid + winner {mae(a, corrected_f, scored):.2f}")
    report = change_report(days, base_f, corrected_f)
    report.to_csv(REPORT_DIR / "winner_fresh_seeds.csv", index=False)
    print("\nStep 2 - the winner on FRESH seeds (change = corrected minus uncorrected MAE, 95% interval):\n", report.round(1).to_string(index=False))
    context = grid_table(days, base_f)
    context.to_csv(REPORT_DIR / "grid_fresh_seeds_context.csv", index=False)
    print("\nContext (not a selection): all 9 variants on the fresh seeds:\n", context.round(2).to_string(index=False))

    overall = report.iloc[0]
    fri, sun = report[report["segment"] == "Friday"].iloc[0], report[report["segment"] == "Sunday"].iloc[0]
    ok_overall = overall["hi"] < 0
    ok_days = (fri["lo"] <= 0) and (sun["lo"] <= 0)
    print(f"\nBar: (a) overall interval entirely below zero: {ok_overall}  (b) Friday and Sunday not significantly worse: {ok_days}  ->  "
          f"{'PASSES' if ok_overall and ok_days else 'DOES NOT PASS'}")
    make_figures(grid, days, base_f, corrected_f, f"{hl} wk, {st}")
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    main(parser.parse_args().workers)
