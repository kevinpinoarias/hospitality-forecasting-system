"""
Series 36 - do football and concerts explain what the forecasts miss?

EXPLORATORY. Three stages, none of which changes the model or the pipeline (the event lists are in config/events/, the flags
in src/features/events.py):

  residuals   Are the manager's and the deployed model's errors related to event days, after allowing for the weekday? A weekday-adjusted
              least-squares fit of (actual - forecast) on the event flags, with a 7-day moving-block bootstrap interval. Football, then concerts.
  football    The 585-day rolling backtest (365-day start, 60-day blocks, seeds 0-24, the final model's recipe) with Celtic / Rangers features
              added to FINAL_FEATURES: "plays" (celtic_plays, rangers_plays, old_firm) and "home_away" (home and away flags per club + old_firm).
  saturday    The pre-registered Saturday test: one flag, sat_big_home (a Saturday with Celtic or Rangers at home), on the raw-pound and log1p models,
              seeds 0-24 and a replicate on 100-124. Rules in docs/EVENTS_EXPLORATION_2026_10.md, section 4 (written before the run).

Run:  python -m src.experiments.run_series36_event_checks residuals
      python -m src.experiments.run_series36_event_checks football [--workers 11]
      python -m src.experiments.run_series36_event_checks saturday [--workers 11]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.experiments.run_series23_log_vs_raw_peaks import WEEKDAYS, masked_block_bootstrap, raw_predictions
from src.experiments.series19_21_common import (
    _STATE, FINAL_PARAMS, OUT_DIR, bagged_log1p_predictions, load_protocol, run_parallel, test_days,
)
from src.features.events import event_calendar, football_calendar
from src.models.final_model import FINAL_FEATURES

BACKTEST_PREDICTIONS = PROJECT_ROOT / "reports" / "results" / "final_model_backtest_predictions.csv"
HOLDOUT_PREDICTIONS = PROJECT_ROOT / "data" / "holdout" / "reports" / "results" / "holdout_predictions.csv"
RESULTS_FOOTBALL = OUT_DIR / "series36_football.jsonl"
RESULTS_SATURDAY = OUT_DIR / "series36_saturday.jsonl"
SEED_CHUNK = 5
SEEDS = {"s0_24": list(range(25)), "s100_124": list(range(100, 125))}

FOOTBALL_VARIANTS = {
    "baseline": [],
    "plays": ["celtic_plays", "rangers_plays", "old_firm"],
    "home_away": ["celtic_home", "celtic_away", "rangers_home", "rangers_away", "old_firm"],
}
SATURDAY_VARIANTS = {  # name -> (target, extra features)
    "raw_base": ("raw", []), "raw_flag": ("raw", ["sat_big_home"]),
    "log_base": ("log1p", []), "log_flag": ("log1p", ["sat_big_home"]),
}


# --------------------------------------------------------------------------- residuals


def _with_event_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(event_calendar(), on="date", how="left")
    out["dow"] = out["date"].dt.dayofweek
    out["miss_manager"] = out["actual"] - out["manager"]  # positive: sales came in above the forecast
    out["miss_model"] = out["actual"] - out["model"]
    return out


def weekday_adjusted_effect(frame: pd.DataFrame, flags: list[str], y: str, reps: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares effect of each flag on `y` with weekday dummies, and a 95% interval from a 7-day moving-block bootstrap."""
    def fit(d: pd.DataFrame) -> np.ndarray:
        X = pd.get_dummies(d["dow"], prefix="d", dtype=float)
        for f in flags:
            X[f] = d[f].astype(float)
        return np.linalg.lstsq(X.to_numpy(), d[y].to_numpy(), rcond=None)[0][-len(flags):]

    rng = np.random.default_rng(0)
    weeks = (frame["date"] - frame["date"].min()).dt.days // 7
    groups = [g.index.to_numpy() for _, g in frame.groupby(weeks)]
    draws = []
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(fit(frame.loc[idx].reset_index(drop=True)))
    return fit(frame), np.percentile(draws, [2.5, 97.5], axis=0)


def _print_effects(frame: pd.DataFrame, flags: list[str], title: str) -> None:
    print(f"\n== {title}: {len(frame)} days; days flagged {({f: int((frame[f] > 0).sum()) for f in flags})} ==")
    for y, name in (("miss_manager", "manager"), ("miss_model", "deployed model")):
        beta, ci = weekday_adjusted_effect(frame, flags, y)
        print(f"  actual minus {name} forecast, GBP/day, weekday-adjusted (+ = sales above the forecast)")
        for i, f in enumerate(flags):
            print(f"    {f:20s} {beta[i]:+8.0f}   95% [{ci[0][i]:+.0f}, {ci[1][i]:+.0f}]")


def run_residuals() -> None:
    bt = pd.read_csv(BACKTEST_PREDICTIONS, parse_dates=["date"]).rename(
        columns={"total_sales": "actual", "forecast_sales": "manager", "final_model_prediction": "model"})[["date", "actual", "manager", "model"]]
    backtest = _with_event_flags(bt)
    _print_effects(backtest, ["any_home_big"], "backtest: any Celtic or Rangers home match")
    _print_effects(backtest, ["celtic_home", "rangers_home", "old_firm"], "backtest: separate football flags")
    _print_effects(backtest, ["celtic_away", "rangers_away"], "backtest: away matches")
    _print_effects(backtest, ["any_show"], "backtest: any show at the Hydro, Armadillo or O2 Academy")
    _print_effects(backtest, ["hydro_show", "o2_show", "armadillo_show"], "backtest: by venue")
    _print_effects(backtest, ["concert_capacity_k"], "backtest: summed capacity, per 1,000 seats")
    if HOLDOUT_PREDICTIONS.exists():  # context only: the 39 new days are not clean evidence for features chosen after seeing them
        ho = pd.read_csv(HOLDOUT_PREDICTIONS, parse_dates=["date"])
        ho = ho[(ho["mode"] == "A") & (ho["variant"] == "with_manager") & (ho["arm"] == "P")]
        ho = _with_event_flags(ho.rename(columns={"forecast": "model"})[["date", "actual", "manager", "model"]])
        hydro = ho["hydro_show"] == 1
        print(f"\nThe 39 new days (context only): Hydro days {int(hydro.sum())}; mean actual minus model {ho[hydro].miss_model.mean():+.0f} on Hydro days, "
              f"{ho[~hydro].miss_model.mean():+.0f} on the others; minus manager {ho[hydro].miss_manager.mean():+.0f} / {ho[~hydro].miss_manager.mean():+.0f}")


# --------------------------------------------------------------------------- backtests


def init_worker36() -> None:
    df, folds = load_protocol()
    _STATE["df"], _STATE["folds"] = df.merge(football_calendar(), on="date", how="left"), folds
    assert _STATE["df"]["sat_big_home"].notna().all()


def fit_task(task: dict) -> dict:
    started = time.time()
    target, extra = task["target"], task["extra"]
    features = list(FINAL_FEATURES) + extra
    predict = raw_predictions if target == "raw" else bagged_log1p_predictions
    per_seed = predict(_STATE["df"], _STATE["folds"], features, dict(FINAL_PARAMS), task["seeds"])
    return {"id": task["id"], "variant": task["variant"], "set": task["set"], "per_seed": [[round(float(x), 2) for x in r] for r in per_seed],
            "seconds": round(time.time() - started, 1)}


def _tasks(variants: dict[str, tuple[str, list[str]]], sets: dict[str, list[int]]) -> list[dict]:
    return [{"id": f"{v}_{s}_{i}", "variant": v, "set": s, "target": target, "extra": extra, "seeds": seeds[i:i + SEED_CHUNK]}
            for v, (target, extra) in variants.items() for s, seeds in sets.items() for i in range(0, len(seeds), SEED_CHUNK)]


def _bags(path) -> dict[tuple[str, str], np.ndarray]:
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        grouped.setdefault((r["variant"], r["set"]), []).append(np.array(r["per_seed"]))
    out = {}
    for key, parts in grouped.items():
        stacked = np.concatenate(parts, axis=0)
        assert stacked.shape[0] == 25, (key, stacked.shape)
        out[key] = stacked.mean(axis=0)
    return out


def _protocol_days() -> pd.DataFrame:
    df, folds = load_protocol()
    d = test_days(df, folds)
    d["date"] = pd.to_datetime(d["date"])
    return d.merge(football_calendar(), on="date", how="left")


def run_football(workers: int) -> None:
    variants = {name: ("log1p", extra) for name, extra in FOOTBALL_VARIANTS.items()}
    run_parallel(_tasks(variants, {"s0_24": SEEDS["s0_24"]}), fit_task, RESULTS_FOOTBALL, workers, init_worker36, "series36-football")
    bags = _bags(RESULTS_FOOTBALL)
    days = _protocol_days()
    a, m = days["actual"].to_numpy(), days["manual"].to_numpy()
    wd = days["date"].dt.day_name().to_numpy()
    base = bags[("baseline", "s0_24")]
    pd.set_option("display.width", 220)
    print(f"585 test days; manager MAE {np.abs(m - a).mean():.1f}")
    rows = []
    for v in FOOTBALL_VARIANTS:
        p = bags[(v, "s0_24")]
        mean, lo, hi = masked_block_bootstrap(np.abs(p - a) - np.abs(base - a), np.ones(len(a)))
        rows.append({"variant": v, "features added": len(FOOTBALL_VARIANTS[v]), "MAE": np.abs(p - a).mean(), "change vs baseline": mean, "lo": lo, "hi": hi,
                     "bias": (p - a).mean(), "gain over manager": np.abs(m - a).mean() - np.abs(p - a).mean()})
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    home = ((days["celtic_home"] + days["rangers_home"]) > 0).to_numpy()
    segments = {"home-match days": home, "Old Firm days": (days["old_firm"] == 1).to_numpy(),
                "no football at all": ((days["celtic_plays"] + days["rangers_plays"]) == 0).to_numpy(), **{d: wd == d for d in WEEKDAYS}}
    seg_rows = []
    for name, mask in segments.items():
        for v in ("plays", "home_away"):
            mean, lo, hi = masked_block_bootstrap(np.abs(bags[(v, "s0_24")] - a) - np.abs(base - a), mask.astype(float))
            seg_rows.append({"segment": name, "variant": v, "days": int(mask.sum()), "baseline MAE": np.abs(base - a)[mask].mean(),
                             "new MAE": np.abs(bags[(v, 's0_24')] - a)[mask].mean(), "change": mean, "lo": lo, "hi": hi})
    print(pd.DataFrame(seg_rows).round(0).to_string(index=False))


def run_saturday(workers: int) -> None:
    run_parallel(_tasks(SATURDAY_VARIANTS, SEEDS), fit_task, RESULTS_SATURDAY, workers, init_worker36, "series36-saturday")
    bags = _bags(RESULTS_SATURDAY)
    days = _protocol_days()
    a, m = days["actual"].to_numpy(), days["manual"].to_numpy()
    sat = (days["date"].dt.dayofweek == 5).to_numpy()
    flagged = sat & (days["sat_big_home"] == 1).to_numpy()
    segments = {"all Saturdays": sat, "flagged Saturdays": flagged, "unflagged Saturdays": sat & ~flagged}
    print(f"Saturdays {sat.sum()}  flagged {flagged.sum()}  unflagged {(sat & ~flagged).sum()}  | manager Saturday MAE {np.abs(m - a)[sat].mean():.0f}")
    rows = []
    for tgt in ("raw", "log"):
        for s in SEEDS:
            base, flag = bags[(f"{tgt}_base", s)], bags[(f"{tgt}_flag", s)]
            for name, mask in segments.items():
                mean, lo, hi = masked_block_bootstrap(np.abs(flag - a) - np.abs(base - a), mask.astype(float))
                rows.append({"model": tgt, "seeds": s, "segment": name, "days": int(mask.sum()), "base MAE": np.abs(base - a)[mask].mean(),
                             "flag MAE": np.abs(flag - a)[mask].mean(), "change": mean, "lo": lo, "hi": hi,
                             "base bias": (base - a)[mask].mean(), "flag bias": (flag - a)[mask].mean()})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(0).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["residuals", "football", "saturday"])
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    {"residuals": run_residuals, "football": lambda: run_football(args.workers), "saturday": lambda: run_saturday(args.workers)}[args.stage]()
