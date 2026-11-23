"""
Series 36 - do football and concerts explain what the forecasts miss?

EXPLORATORY. Three stages, none of which changes the model or the pipeline (the event lists are in config/events/, the flags
in src/features/events.py):

  residuals   Are the manager's and the deployed model's errors related to event days, after allowing for the weekday? A weekday-adjusted
              least-squares fit of (actual - forecast) on the event flags, with a 7-day moving-block bootstrap interval. Football, then concerts.
  football    The 585-day rolling backtest (365-day start, 60-day blocks, seeds 0-24, the final model's recipe) with Celtic / Rangers features
              added to FINAL_FEATURES: "plays" (celtic_plays, rangers_plays, old_firm) and "home_away" (home and away flags per club + old_firm).
  scotland    The Scotland national team flags (scotland_home, scotland_away, scotland_tournament) added to the final model, rules in section 7 of the write-up.
  hydro       The pre-registered Hydro test (section 9 of the write-up, committed before the run): hydro_show added to the final model, log1p and raw-pound,
              seeds 0-24 and 100-124, plus a shuffled-flag placebo on seeds 0-24 and a verdict against the four rules.
  hydrocorr   The pre-registered Hydro-day correction (section 10 of the write-up, committed before this code): after the forecast, Hydro-show days are moved
              by -0.5 x the recency-weighted (half-life 8 weeks) mean forecast error on earlier Hydro days. No model is refitted: the bases are the log final model's
              bags from the `hydro` stage and the Track A package rebuilt from the verified-calendar results. Verdict against the four rules.
  postflag    The pre-registered test of hydro_show inside the post model (section 11 of the write-up, committed before this code): the Friday (A_all), Saturday (A_lags)
              and Sunday (B) models of the post model refitted with the flag, Monday-Thursday from the log model with the flag, the Saturday and Sunday correction recomputed,
              against the stored post model; seeds 0-24 and 100-124, a shuffled-flag placebo and a sanity refit without the flag. Verdict against four rules.
  hydrothu    The pre-registered Thursday-only Hydro flag (section 12 of the write-up, committed before this code): hydro_thu = a Hydro show on a Thursday, added to the
              log model, seeds 0-24 and 100-124, a placebo shuffled among Thursdays, a sign test on the 17 Hydro Thursdays and a verdict against four rules.
  thuperm     The repeated placebo for the Thursday flag (section 13 of the write-up, committed before this code): 200 random reassignments of the Thursday flags
              among all Thursdays, 5 seeds each, each scored on its own flagged days, against the real flag's change on the same 5 seeds.
  saturday    The pre-registered Saturday test: one flag, sat_big_home (a Saturday with Celtic or Rangers at home), on the raw-pound and log1p models,
              seeds 0-24 and a replicate on 100-124. Rules in docs/EVENTS_EXPLORATION_2026_10.md, section 4 (written before the run).

Run:  python -m src.experiments.run_series36_event_checks residuals
      python -m src.experiments.run_series36_event_checks football [--workers 11]
      python -m src.experiments.run_series36_event_checks saturday [--workers 11]
      python -m src.experiments.run_series36_event_checks scotland [--workers 11]
      python -m src.experiments.run_series36_event_checks hydro [--workers 11]
      python -m src.experiments.run_series36_event_checks hydrocorr
      python -m src.experiments.run_series36_event_checks postflag [--workers 11]
      python -m src.experiments.run_series36_event_checks hydrothu [--workers 11]
      python -m src.experiments.run_series36_event_checks thuperm [--workers 11]
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
from src.features.events import event_calendar
from src.models.final_model import FINAL_FEATURES

BACKTEST_PREDICTIONS = PROJECT_ROOT / "reports" / "results" / "final_model_backtest_predictions.csv"
HOLDOUT_PREDICTIONS = PROJECT_ROOT / "data" / "holdout" / "reports" / "results" / "holdout_predictions.csv"
RESULTS_FOOTBALL = OUT_DIR / "series36_football.jsonl"
RESULTS_SATURDAY = OUT_DIR / "series36_saturday.jsonl"
RESULTS_SCOTLAND = OUT_DIR / "series36_scotland.jsonl"
RESULTS_HYDRO = OUT_DIR / "series36_hydro.jsonl"
RESULTS_POSTFLAG = OUT_DIR / "series36_postflag.jsonl"
RESULTS_THU = OUT_DIR / "series36_hydrothu.jsonl"
RESULTS_PERM = OUT_DIR / "series36_thuperm.jsonl"
PERM_DRAWS = 200
PLACEBO_SEED = 20261018
SEED_CHUNK = 5
SEEDS = {"s0_24": list(range(25)), "s100_124": list(range(100, 125))}

FOOTBALL_VARIANTS = {
    "baseline": [],
    "plays": ["celtic_plays", "rangers_plays", "old_firm"],
    "home_away": ["celtic_home", "celtic_away", "rangers_home", "rangers_away", "old_firm"],
}
SCOTLAND_FLAGS = ["scotland_home", "scotland_away", "scotland_tournament"]
SCOTLAND_VARIANTS = {"baseline": ("log1p", []), "scotland": ("log1p", SCOTLAND_FLAGS)}
HYDRO_VARIANTS = {  # name -> (target, extra features, seed sets)
    "log_base": ("log1p", [], ("s0_24", "s100_124")), "log_flag": ("log1p", ["hydro_show"], ("s0_24", "s100_124")),
    "raw_base": ("raw", [], ("s0_24", "s100_124")), "raw_flag": ("raw", ["hydro_show"], ("s0_24", "s100_124")),
    "log_placebo": ("log1p", ["hydro_placebo"], ("s0_24",)),
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
    _print_effects(backtest, SCOTLAND_FLAGS, "backtest: Scotland national team (home, away or neutral, tournament)")
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


def placebo_flag(dates: pd.Series, flag: pd.Series, seed: int) -> np.ndarray:
    """The flags shuffled among the days of the same weekday (the count of flagged days per weekday is unchanged)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(flag), dtype=int)
    weekday = pd.to_datetime(dates).dt.dayofweek.to_numpy()
    values = flag.to_numpy()
    for d in range(7):
        idx = np.where(weekday == d)[0]
        out[idx] = rng.permutation(values[idx])
    return out


def init_worker36() -> None:
    df, folds = load_protocol()
    _STATE["df"], _STATE["folds"] = df.merge(event_calendar(), on="date", how="left"), folds
    assert _STATE["df"]["sat_big_home"].notna().all()
    _STATE["df"]["hydro_placebo"] = placebo_flag(_STATE["df"]["date"], _STATE["df"]["hydro_show"], PLACEBO_SEED)
    _STATE["df"]["hydro_thu"] = _STATE["df"]["hydro_show"] * (pd.to_datetime(_STATE["df"]["date"]).dt.dayofweek == 3).astype(int)
    _STATE["df"]["thu_placebo"] = placebo_flag(_STATE["df"]["date"], _STATE["df"]["hydro_thu"], PLACEBO_SEED)


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
    return d.merge(event_calendar(), on="date", how="left")


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


def run_scotland(workers: int) -> None:
    run_parallel(_tasks(SCOTLAND_VARIANTS, {"s0_24": SEEDS["s0_24"]}), fit_task, RESULTS_SCOTLAND, workers, init_worker36, "series36-scotland")
    bags = _bags(RESULTS_SCOTLAND)
    days = _protocol_days()
    a, m = days["actual"].to_numpy(), days["manual"].to_numpy()
    base, new = bags[("baseline", "s0_24")], bags[("scotland", "s0_24")]
    diff = np.abs(new - a) - np.abs(base - a)
    mean, lo, hi = masked_block_bootstrap(diff, np.ones(len(a)))
    print(f"585 test days: baseline MAE {np.abs(base - a).mean():.1f}, with Scotland flags {np.abs(new - a).mean():.1f}; change {mean:+.1f} (95% interval {lo:+.1f} to {hi:+.1f}); "
          f"manager {np.abs(m - a).mean():.1f}")
    rows = []
    for flag in SCOTLAND_FLAGS + ["scotland_plays"]:
        mask = (days[flag] == 1).to_numpy()
        if mask.sum():
            mean, lo, hi = masked_block_bootstrap(diff, mask.astype(float))
            rows.append({"segment": flag, "days": int(mask.sum()), "baseline MAE": np.abs(base - a)[mask].mean(), "new MAE": np.abs(new - a)[mask].mean(),
                         "change": mean, "lo": lo, "hi": hi, "baseline bias": (base - a)[mask].mean(), "new bias": (new - a)[mask].mean()})
    print(pd.DataFrame(rows).round(0).to_string(index=False))


def run_hydro(workers: int) -> None:
    tasks = [{"id": f"{v}_{st}_{i}", "variant": v, "set": st, "target": target, "extra": extra, "seeds": SEEDS[st][i:i + SEED_CHUNK]}
             for v, (target, extra, sets) in HYDRO_VARIANTS.items() for st in sets for i in range(0, 25, SEED_CHUNK)]
    run_parallel(tasks, fit_task, RESULTS_HYDRO, workers, init_worker36, "series36-hydro")
    bags = _bags(RESULTS_HYDRO)
    days = _protocol_days()
    a, m = days["actual"].to_numpy(), days["manual"].to_numpy()
    hydro = (days["hydro_show"] == 1).to_numpy()
    wd = days["date"].dt.dayofweek.to_numpy()
    segments = {"all days": np.ones(len(a), bool), "Hydro days": hydro, "non-Hydro days": ~hydro, "Mon-Thu": wd <= 3, "Fri-Sun": wd >= 4,
                "Hydro Mon-Thu": hydro & (wd <= 3), "Hydro Fri-Sun": hydro & (wd >= 4)}
    print(f"585 test days; Hydro days {int(hydro.sum())}; manager MAE {np.abs(m - a).mean():.1f}")
    rows, result = [], {}
    for label, base_key, new_key, st in (("log1p seeds 0-24", "log_base", "log_flag", "s0_24"), ("log1p seeds 100-124", "log_base", "log_flag", "s100_124"),
                                         ("raw-GBP seeds 0-24", "raw_base", "raw_flag", "s0_24"), ("raw-GBP seeds 100-124", "raw_base", "raw_flag", "s100_124"),
                                         ("PLACEBO log1p seeds 0-24", "log_base", "log_placebo", "s0_24")):
        base, new = bags[(base_key, st)], bags[(new_key, st)]
        diff = np.abs(new - a) - np.abs(base - a)
        for name, mask in segments.items():
            mean, lo, hi = masked_block_bootstrap(diff, mask.astype(float))
            result[(label, name)] = (mean, lo, hi)
            rows.append({"run": label, "segment": name, "days": int(mask.sum()), "base MAE": np.abs(base - a)[mask].mean(), "new MAE": np.abs(new - a)[mask].mean(),
                         "change": mean, "lo": lo, "hi": hi, "base bias": (base - a)[mask].mean(), "new bias": (new - a)[mask].mean()})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    log0, log1, plc = result[("log1p seeds 0-24", "all days")], result[("log1p seeds 100-124", "all days")], result[("PLACEBO log1p seeds 0-24", "all days")]
    h, u = result[("log1p seeds 0-24", "Hydro days")], result[("log1p seeds 0-24", "non-Hydro days")]
    rules = {"1 primary: pooled interval entirely below zero (seeds 0-24)": log0[2] < 0,
             "2 replicate: pooled interval entirely below zero (seeds 100-124)": log1[2] < 0,
             "3 gain comes from Hydro days, non-Hydro days not clearly hurt": h[0] < 0 and h[0] < u[0] and not u[1] > 0,
             "4 placebo: pooled interval NOT entirely below zero": not (plc[2] < 0)}
    print("\nVERDICT against the pre-registered rules:")
    for k, v in rules.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("  => " + ("PASS: supported on the backtest; enters the weekly log as a candidate, not promoted" if all(rules.values()) else "FAIL: not pursued; record as Series 37"))


CORRECTION_HALF_LIFE_WEEKS, CORRECTION_STRENGTH, CORRECTION_MIN_HISTORY = 8, 0.5, 4


def hydro_correct(days: pd.DataFrame, pred: np.ndarray, flag: np.ndarray, half_life_weeks: float = CORRECTION_HALF_LIFE_WEEKS,
                  strength: float = CORRECTION_STRENGTH, min_history: int = CORRECTION_MIN_HISTORY) -> tuple[np.ndarray, dict[int, float]]:
    """Block k's flagged days move by -strength x the recency-weighted mean of (forecast - actual) over flagged days in blocks BEFORE k.
    Block 0, unflagged days and blocks with too little history are unchanged. Returns the new forecasts and the amount added per block."""
    out = pred.astype(float).copy()
    fold = days["fold_id"].to_numpy()
    dates = pd.to_datetime(days["date"]).to_numpy()
    error = pred - days["actual"].to_numpy()
    amounts: dict[int, float] = {}
    for k in sorted(set(fold.tolist()))[1:]:
        block = fold == k
        history = (fold < k) & flag
        if history.sum() < min_history:
            continue
        age_weeks = (dates[block].min() - dates[history]) / np.timedelta64(7, "D")
        weights = 0.5 ** (age_weeks / half_life_weeks)
        bias = float(np.sum(weights * error[history]) / np.sum(weights))
        out[block & flag] = pred[block & flag] - strength * bias
        amounts[int(k)] = -strength * bias
    return out, amounts


def run_hydrocorr() -> None:
    from src.experiments import run_package_validation as pv

    df, folds = load_protocol()
    frame = df.merge(event_calendar(), on="date", how="left")
    frame["hydro_placebo"] = placebo_flag(frame["date"], frame["hydro_show"], PLACEBO_SEED)
    days = _protocol_days()
    days = days.merge(frame[["date", "hydro_placebo"]], on="date", how="left")
    days["weekday"] = days["date"].dt.day_name()
    days["date"] = days["date"].dt.strftime("%Y-%m-%d")  # test_days' own form, which the package code expects
    a, m = days["actual"].to_numpy(), days["manual"].to_numpy()
    hydro, placebo = (days["hydro_show"] == 1).to_numpy(), (days["hydro_placebo"] == 1).to_numpy()
    wd = pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    segments = {"all 585 days": np.ones(len(a), bool), "Hydro days": hydro, "non-Hydro days": ~hydro,
                "Hydro Mon-Thu": hydro & (wd <= 3), "Hydro Fri-Sun": hydro & (wd >= 4)}

    hydro_bags = _bags(RESULTS_HYDRO)
    data = pv.load_set("_cal2")
    chosen = pv.choose_on_first_blocks(days, data["log_raw_0_24"], data["features_0_24"])
    bases = {
        ("log final model", "s0_24"): hydro_bags[("log_base", "s0_24")], ("log final model", "s100_124"): hydro_bags[("log_base", "s100_124")],
        ("Track A package", "s0_24"): pv.package_steps(days, data["log_raw_0_24"], data["features_0_24"], chosen)[pv.STEPS[-1]],
        ("Track A package", "s100_124"): pv.package_steps(days, data["log_raw_fresh"], data["features_fresh"], chosen)[pv.STEPS[-1]],
    }
    print(f"585 test days; Hydro days {int(hydro.sum())}; weekday options chosen for the package: {chosen}")
    print(f"log base MAE (hydro stage) {np.abs(bases[('log final model', 's0_24')] - a).mean():.1f} | package MAE "
          f"{np.abs(bases[('Track A package', 's0_24')] - a).mean():.1f} | manager {np.abs(m - a).mean():.1f}")
    rows, result, amounts_primary = [], {}, {}
    runs = [(f"{name} {st}", base, hydro, name, st) for (name, st), base in bases.items()]
    runs += [("PLACEBO log final model s0_24", bases[("log final model", "s0_24")], placebo, "placebo", "s0_24"),
             ("manager's forecast (descriptive)", m.astype(float), hydro, "manager", "-")]
    for label, base, flag, name, st in runs:
        new, amounts = hydro_correct(days, base, flag)
        if name == "log final model" and st == "s0_24":
            amounts_primary = amounts
        assert np.array_equal(new[~flag], base[~flag]), "unflagged days must be unchanged"
        diff = np.abs(new - a) - np.abs(base - a)
        for seg, mask in segments.items():
            if name == "placebo":  # the placebo is judged on its own (shuffled) flagged days
                mask = flag if seg == "Hydro days" else mask
            mean, lo, hi = masked_block_bootstrap(diff, mask.astype(float))
            result[(label, seg)] = (mean, lo, hi)
            rows.append({"run": label, "segment": seg, "days": int(mask.sum()), "base MAE": np.abs(base - a)[mask].mean(),
                         "new MAE": np.abs(new - a)[mask].mean(), "change": mean, "lo": lo, "hi": hi})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print("\nCorrection added to Hydro days per block (log final model, seeds 0-24), GBP:", {k: round(v) for k, v in amounts_primary.items()})

    p0, p1 = result[("log final model s0_24", "Hydro days")], result[("log final model s100_124", "Hydro days")]
    q0, q1 = result[("log final model s0_24", "all 585 days")], result[("log final model s100_124", "all 585 days")]
    pl = result[("PLACEBO log final model s0_24", "Hydro days")]
    positive = sum(v > 0 for v in amounts_primary.values())
    rules = {"1 Hydro days: interval entirely below zero on both seed sets": p0[2] < 0 and p1[2] < 0,
             "2 pooled point estimate at or below zero on both seed sets": q0[0] <= 0 and q1[0] <= 0,
             "3 placebo: interval NOT entirely below zero": not (pl[2] < 0),
             f"4 stability: correction positive in at least 8 of 9 blocks ({positive} of {len(amounts_primary)})": positive >= 8 and len(amounts_primary) == 9}
    print("\nVERDICT against the pre-registered rules (primary base):")
    for k, v in rules.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("  => " + ("PASS: a candidate for the weekly log, not promoted" if all(rules.values()) else "FAIL: not pursued; record as Series 38"))


POST_OPTIONS = {"Friday": ("fri", "log1p", "A_all"), "Saturday": ("sat", "raw", "A_lags"), "Sunday": ("sun", "log1p", "B")}


def init_worker_post() -> None:
    from src.experiments import run_series26_weekend_features as s26

    df, folds = s26.load_frame()
    df = df.merge(event_calendar(), on="date", how="left")
    df["hydro_placebo"] = placebo_flag(df["date"], df["hydro_show"], PLACEBO_SEED)
    _STATE["df"], _STATE["folds"] = df, folds


def fit_task_post(task: dict) -> dict:
    started = time.time()
    features = list(FINAL_FEATURES) + task["extra"]
    predict = raw_predictions if task["target"] == "raw" else bagged_log1p_predictions
    per_seed = predict(_STATE["df"], _STATE["folds"], features, dict(FINAL_PARAMS), task["seeds"])
    return {"id": task["id"], "variant": task["variant"], "set": task["set"], "per_seed": [[round(float(x), 2) for x in r] for r in per_seed],
            "seconds": round(time.time() - started, 1)}


def run_postflag(workers: int) -> None:
    from src.experiments import run_package_validation as pv
    from src.experiments import run_series24_weekend_correction as s24
    from src.experiments import run_series26_weekend_features as s26

    tasks = []
    for tag, flag_col, sets in (("flag", "hydro_show", ("s0_24", "s100_124")), ("placebo", "hydro_placebo", ("s0_24",)), ("base", None, ("s0_24",))):
        for day, (short, target, option) in POST_OPTIONS.items():
            extra = list(s26.VARIANTS[option]["extra"]) + ([flag_col] if flag_col else [])
            for st in sets:
                for i in range(0, 25, SEED_CHUNK):
                    tasks.append({"id": f"{short}_{tag}_{st}_{i}", "variant": f"{short}_{tag}", "set": st, "target": target, "extra": extra, "seeds": SEEDS[st][i:i + SEED_CHUNK]})
    run_parallel(tasks, fit_task_post, RESULTS_POSTFLAG, workers, init_worker_post, "series36-postflag")
    bags = _bags(RESULTS_POSTFLAG)
    hydro_bags = _bags(RESULTS_HYDRO)

    days = _protocol_days()
    df, folds = load_protocol()
    frame = df.merge(event_calendar(), on="date", how="left")
    frame["hydro_placebo"] = placebo_flag(frame["date"], frame["hydro_show"], PLACEBO_SEED)
    days = days.merge(frame[["date", "hydro_placebo"]], on="date", how="left")
    days["weekday"] = days["date"].dt.day_name()
    wd = days["weekday"].to_numpy()
    days["date"] = days["date"].dt.strftime("%Y-%m-%d")
    a, m = days["actual"].to_numpy(), days["manual"].to_numpy()
    hydro, placebo = (days["hydro_show"] == 1).to_numpy(), (days["hydro_placebo"] == 1).to_numpy()
    data = pv.load_set("_cal2")
    chosen = pv.choose_on_first_blocks(days, data["log_raw_0_24"], data["features_0_24"])
    expected = {d: o for d, (_, _, o) in POST_OPTIONS.items()}
    assert chosen == expected, (chosen, expected)
    stored = {"s0_24": pv.package_steps(days, data["log_raw_0_24"], data["features_0_24"], chosen)[pv.STEPS[-1]],
              "s100_124": pv.package_steps(days, data["log_raw_fresh"], data["features_fresh"], chosen)[pv.STEPS[-1]]}
    raw_stored = {"s0_24": data["log_raw_0_24"]["raw_same_params"], "s100_124": data["log_raw_fresh"]["raw_same_params"]}

    def assemble(log_pred: np.ndarray, tag: str, st: str, raw: np.ndarray) -> np.ndarray:
        weekend = s24.hybrid(wd, log_pred, raw)
        for day, (short, _, _) in POST_OPTIONS.items():
            weekend = np.where(wd == day, bags[(f"{short}_{tag}", st)], weekend)
        return s24.correct(days, weekend, 8, 0.5, scope=("Saturday", "Sunday"))

    rebuilt_base = assemble(hydro_bags[("log_base", "s0_24")], "base", "s0_24", raw_stored["s0_24"])
    sanity = abs(np.abs(rebuilt_base - a).mean() - np.abs(stored["s0_24"] - a).mean())
    print(f"SANITY: refit post model without the flag MAE {np.abs(rebuilt_base - a).mean():.3f} vs stored {np.abs(stored['s0_24'] - a).mean():.3f} "
          f"(difference {sanity:.3f}; largest single-day difference {np.abs(rebuilt_base - stored['s0_24']).max():.3f}); must be below 0.05 -> {'OK' if sanity < 0.05 else 'VOID'}")

    runs = {"post model + flag, seeds 0-24": (assemble(hydro_bags[("log_flag", "s0_24")], "flag", "s0_24", raw_stored["s0_24"]), stored["s0_24"], hydro),
            "post model + flag, seeds 100-124": (assemble(hydro_bags[("log_flag", "s100_124")], "flag", "s100_124", raw_stored["s100_124"]), stored["s100_124"], hydro),
            "PLACEBO post model, seeds 0-24": (assemble(hydro_bags[("log_placebo", "s0_24")], "placebo", "s0_24", raw_stored["s0_24"]), stored["s0_24"], placebo)}
    segments = {"all 585 days": lambda f: np.ones(len(a), bool), "Hydro days": lambda f: f, "non-Hydro days": lambda f: ~f}
    day_types = {"Hydro Mon-Thu": np.isin(wd, ["Monday", "Tuesday", "Wednesday", "Thursday"]), "Hydro Friday": wd == "Friday",
                 "Hydro Saturday": wd == "Saturday", "Hydro Sunday": wd == "Sunday"}
    rows, result = [], {}
    for label, (new, base, flag) in runs.items():
        diff = np.abs(new - a) - np.abs(base - a)
        masks = {name: fn(flag) for name, fn in segments.items()}
        masks.update({k: flag & v for k, v in day_types.items()})
        for seg, mask in masks.items():
            mean, lo, hi = masked_block_bootstrap(diff, mask.astype(float))
            result[(label, seg)] = (mean, lo, hi)
            rows.append({"run": label, "segment": seg, "days": int(mask.sum()), "base MAE": np.abs(base - a)[mask].mean(), "new MAE": np.abs(new - a)[mask].mean(),
                         "change": mean, "lo": lo, "hi": hi, "base bias": (base - a)[mask].mean(), "new bias": (new - a)[mask].mean()})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(f"585 test days; Hydro days {int(hydro.sum())}; stored post model MAE {np.abs(stored['s0_24'] - a).mean():.1f}; manager {np.abs(m - a).mean():.1f}")
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    r0, r1 = "post model + flag, seeds 0-24", "post model + flag, seeds 100-124"
    pl = result[("PLACEBO post model, seeds 0-24", "Hydro days")]
    rules = {"1 Hydro days: interval entirely below zero on both seed sets": result[(r0, "Hydro days")][2] < 0 and result[(r1, "Hydro days")][2] < 0,
             "2 pooled point estimate at or below zero on both seed sets": result[(r0, "all 585 days")][0] <= 0 and result[(r1, "all 585 days")][0] <= 0,
             "3 other days: interval not entirely above zero on both seed sets": not (result[(r0, "non-Hydro days")][1] > 0) and not (result[(r1, "non-Hydro days")][1] > 0),
             "4 placebo: interval NOT entirely below zero": not (pl[2] < 0)}
    print("\nVERDICT against the pre-registered rules:")
    for k, v in rules.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("  => " + ("PASS: a candidate for the weekly log, not promoted" if all(rules.values()) and sanity < 0.05 else "FAIL (or void): not pursued; record as Series 39"))


def run_hydrothu(workers: int) -> None:
    tasks = [{"id": f"{v}_{st}_{i}", "variant": v, "set": st, "target": "log1p", "extra": [col], "seeds": SEEDS[st][i:i + SEED_CHUNK]}
             for v, col, sets in (("log_thu", "hydro_thu", ("s0_24", "s100_124")), ("log_thu_placebo", "thu_placebo", ("s0_24",)))
             for st in sets for i in range(0, 25, SEED_CHUNK)]
    run_parallel(tasks, fit_task, RESULTS_THU, workers, init_worker36, "series36-hydrothu")
    bags, hydro_bags = _bags(RESULTS_THU), _bags(RESULTS_HYDRO)
    df, folds = load_protocol()
    frame = df.merge(event_calendar(), on="date", how="left")
    frame["hydro_thu"] = frame["hydro_show"] * (frame["date"].dt.dayofweek == 3).astype(int)
    frame["thu_placebo"] = placebo_flag(frame["date"], frame["hydro_thu"], PLACEBO_SEED)
    days = _protocol_days().merge(frame[["date", "hydro_thu", "thu_placebo"]], on="date", how="left")
    a, m = days["actual"].to_numpy(), days["manual"].to_numpy()
    flag, placebo = (days["hydro_thu"] == 1).to_numpy(), (days["thu_placebo"] == 1).to_numpy()
    thursday = (days["date"].dt.dayofweek == 3).to_numpy()
    print(f"585 test days; Hydro Thursdays {int(flag.sum())} (placebo {int(placebo.sum())}); all Thursdays {int(thursday.sum())}; manager MAE {np.abs(m - a).mean():.1f}")
    rows, result = [], {}
    for label, base_key, new_key, st, f in (("seeds 0-24", "log_base", "log_thu", "s0_24", flag), ("seeds 100-124", "log_base", "log_thu", "s100_124", flag),
                                            ("PLACEBO (Thursdays), seeds 0-24", "log_base", "log_thu_placebo", "s0_24", placebo)):
        base, new = hydro_bags[(base_key, st)], bags[(new_key, st)]
        diff = np.abs(new - a) - np.abs(base - a)
        masks = {"flagged days": f, "all other days": ~f, "all 585 days": np.ones(len(a), bool), "other Thursdays": thursday & ~f}
        for seg, mask in masks.items():
            mean, lo, hi = masked_block_bootstrap(diff, mask.astype(float))
            result[(label, seg)] = (mean, lo, hi)
            rows.append({"run": label, "segment": seg, "days": int(mask.sum()), "base MAE": np.abs(base - a)[mask].mean(), "new MAE": np.abs(new - a)[mask].mean(),
                         "change": mean, "lo": lo, "hi": hi, "base bias": (base - a)[mask].mean(), "new bias": (new - a)[mask].mean()})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    base, new = hydro_bags[("log_base", "s0_24")], bags[("log_thu", "s0_24")]
    better = int((np.abs(new - a)[flag] < np.abs(base - a)[flag]).sum())
    from math import comb
    p_sign = sum(comb(int(flag.sum()), k) for k in range(better, int(flag.sum()) + 1)) / 2 ** int(flag.sum())
    print(f"\nSign test, seeds 0-24: {better} of {int(flag.sum())} Hydro Thursdays are better with the flag (one-sided p = {p_sign:.3f})")
    per_block = pd.DataFrame({"fold": days["fold_id"], "change": (np.abs(new - a) - np.abs(base - a)), "flag": flag})
    pb = per_block[per_block["flag"]].groupby("fold")["change"].agg(["count", "mean"]).round(0)
    print("Hydro Thursdays by block (count, mean change in miss, seeds 0-24):\n" + pb.to_string())

    r0, r1, pl = "seeds 0-24", "seeds 100-124", "PLACEBO (Thursdays), seeds 0-24"
    rules = {"1 Hydro Thursdays: interval entirely below zero on both seed sets": result[(r0, "flagged days")][2] < 0 and result[(r1, "flagged days")][2] < 0,
             "2 everything else: interval not entirely above zero on both seed sets": not (result[(r0, "all other days")][1] > 0) and not (result[(r1, "all other days")][1] > 0),
             f"3 sign test: at least 13 of {int(flag.sum())} Hydro Thursdays better ({better})": better >= 13,
             "4 placebo: interval NOT entirely below zero": not (result[(pl, "flagged days")][2] < 0)}
    print("\nVERDICT against the pre-registered rules:")
    for k, v in rules.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("  => " + ("PASS: a candidate for the weekly log, not promoted" if all(rules.values()) else "FAIL: not pursued; record as Series 40"))


def fit_task_perm(task: dict) -> dict:
    started = time.time()
    df = _STATE["df"].copy()
    df["perm_flag"] = placebo_flag(df["date"], df["hydro_thu"], task["perm_seed"])
    per_seed = bagged_log1p_predictions(df, _STATE["folds"], list(FINAL_FEATURES) + ["perm_flag"], dict(FINAL_PARAMS), task["seeds"])
    return {"id": task["id"], "variant": "perm", "set": "perm", "draw": task["draw"], "per_seed": [[round(float(x), 2) for x in r] for r in per_seed],
            "seconds": round(time.time() - started, 1)}


def run_thuperm(workers: int) -> None:
    from src.experiments.series19_21_common import read_done

    tasks = [{"id": f"perm_{i}", "draw": i, "perm_seed": PLACEBO_SEED + i, "seeds": list(range(5))} for i in range(1, PERM_DRAWS + 1)]
    run_parallel(tasks, fit_task_perm, RESULTS_PERM, workers, init_worker36, "series36-thuperm")
    done = read_done(RESULTS_PERM)

    df, folds = load_protocol()
    frame = df.merge(event_calendar(), on="date", how="left")
    frame["hydro_thu"] = frame["hydro_show"] * (frame["date"].dt.dayofweek == 3).astype(int)
    days = _protocol_days().merge(frame[["date", "hydro_thu"]], on="date", how="left")
    a = days["actual"].to_numpy()
    real_flag = (days["hydro_thu"] == 1).to_numpy()

    def first_chunk(path, record_id: str) -> np.ndarray:
        rec = read_done(path)[record_id]
        arr = np.array(rec["per_seed"])
        assert arr.shape[0] == 5
        return arr.mean(axis=0)

    base = first_chunk(RESULTS_HYDRO, "log_base_s0_24_0")
    real = first_chunk(RESULTS_THU, "log_thu_s0_24_0")
    real_change = float((np.abs(real - a) - np.abs(base - a))[real_flag].mean())
    changes, counts = [], []
    for i in range(1, PERM_DRAWS + 1):
        pred = np.array(done[f"perm_{i}"]["per_seed"]).mean(axis=0)
        flag_all = placebo_flag(frame["date"], frame["hydro_thu"], PLACEBO_SEED + i)
        flag_days = days[["date"]].merge(pd.DataFrame({"date": frame["date"], "f": flag_all}), on="date", how="left")["f"].to_numpy() == 1
        changes.append(float((np.abs(pred - a) - np.abs(base - a))[flag_days].mean()))
        counts.append(int(flag_days.sum()))
    ch = np.array(changes)
    rank = int((ch <= real_change).sum())
    p = (rank + 1) / (len(ch) + 1)
    print(f"real flag, 5 seeds: change on its {int(real_flag.sum())} Hydro Thursdays = {real_change:+.1f} (25-seed figure was -274.8)")
    print(f"{len(ch)} placebo draws, flagged scored days per draw: min {min(counts)}, median {int(np.median(counts))}, max {max(counts)}")
    print(f"placebo change on own flagged days: mean {ch.mean():+.1f}, 5th pct {np.percentile(ch, 5):+.1f}, median {np.median(ch):+.1f}, 95th pct {np.percentile(ch, 95):+.1f}, "
          f"best (most negative) {ch.min():+.1f}, worst {ch.max():+.1f}")
    print(f"draws at least as good as the real flag: {rank} of {len(ch)} -> one-sided p = {p:.4f}")
    first = changes[0]
    print(f"the first placebo of Series 40 (seed {PLACEBO_SEED}) gave -29.0 at 25 seeds; draw 1 here (seed {PLACEBO_SEED + 1}) gave {first:+.1f}; share of draws more negative than -29.0: {(ch <= -29.0).mean():.3f}")
    rule1 = p <= 0.05
    best = ch.min()
    rule2 = (rank <= 2) or (real_change <= 2 * best)
    print("\nVERDICT against the pre-registered rules:")
    print(f"  [{'PASS' if rule1 else 'FAIL'}] 1 rank: real change beats at least 95% of placebo draws (p = {p:.4f})")
    print(f"  [{'PASS' if rule2 else 'FAIL'}] 2 size: beyond all but two draws (rank {rank}) or at least twice the best placebo ({best:+.1f}) -> {real_change:+.1f}")
    if rule1 and rule2:
        print("  => unlikely to be chance on the backtest: the leading candidate for the weekly log (not promoted)")
    elif not rule1:
        print("  => indistinguishable from a random set of Thursdays: dropped to a note")
    else:
        print("  => unclear")
    pd.Series(changes, name="placebo_change").to_csv(OUT_DIR / "series36_thuperm_changes.csv", index=False)


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
    parser.add_argument("stage", choices=["residuals", "football", "saturday", "scotland", "hydro", "hydrocorr", "postflag", "hydrothu", "thuperm"])
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    {"residuals": run_residuals, "football": lambda: run_football(args.workers), "saturday": lambda: run_saturday(args.workers),
     "scotland": lambda: run_scotland(args.workers), "hydro": lambda: run_hydro(args.workers), "hydrocorr": run_hydrocorr, "postflag": lambda: run_postflag(args.workers), "hydrothu": lambda: run_hydrothu(args.workers), "thuperm": lambda: run_thuperm(args.workers)}[args.stage]()
