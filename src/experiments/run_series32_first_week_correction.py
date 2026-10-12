"""
Series 32 - a correction for the first week of the Easter and summer breaks.

Series 30 found no feature that helps the transition weeks, and the first week of a break is still the hardest stretch of the
year. The first week of Easter and summer breaks sells above its usual level in all six breaks in the data (+10% to +30% over
the previous four same weekdays) and the model under-forecasts it in all four breaks where it has an out-of-sample forecast.
Christmas is left out: its first week is flat. This series asks whether a correction learned from earlier breaks helps, the way
the Saturday and Sunday correction of Series 24 does.

THE CORRECTION. For the days in scope of a break, learn the gap from the days in scope of OTHER breaks (forecast minus actual),
and move the forecasts by a share of it:
  additive   forecast - strength x (mean of forecast minus actual over the history days)
  ratio      forecast x (1 + strength x (r - 1)), r = sum of actual / sum of forecast over the history days

Two ways to build the history:
  forward-chained (primary)   only EARLIER breaks, on the package's out-of-sample forecasts (the Track A package on the verified
                              calendar: log model Monday to Thursday, raw model on Saturday, per-weekday weekend features, the
                              Saturday and Sunday correction). Only Easter 2025 has no earlier break; summer 2025, Easter 2026
                              and summer 2026 are scored: 3 breaks, 21 days (42 with the extra scope).
  leave one break out         for stability: the six Easter and summer breaks of 2024-2026, each forecast by the final model
                              trained WITHOUT a 42-day window around it (21 days either side of the break's start; nothing from the
                              break is in the training rows), and corrected from the other five breaks. Not realistic for
                              deployment (it learns from later breaks) and it covers 2024, which the backtest does not.

PRE-REGISTERED PLAN (written before any result was seen)
  variants   8: form {additive, ratio} x strength {0.5, 1.0} x scope {first week, first week + the last week of term}
  step 1     the variant with the lowest MAE on the scored days, forward-chained, on seeds 0-24 is the winner
  step 2     apply the same settings to the package built from 25 fresh seeds (100-124); not re-chosen
  bar        kept only if, on the fresh seeds, forward-chained, (a) the pooled MAE on the scored days falls, with a 95%
             moving-block bootstrap interval (7-day blocks) entirely below zero, and (b) at least two of the three scored breaks
             improve; AND on the leave-one-break-out version the same settings improve at least four of the six breaks
  otherwise  the result is parked as a lead to revisit with the Easter and summer 2027 breaks
  note       the step 1 rule ("lowest MAE on the scored days") compares MAE across scopes that score different days (21 against
             42), so it favours the scope with the easier days; it was run as written, and the first-week-only variants are
             reported separately below as a POST-HOC follow-up (not pre-registered)
  context    the effect on all 585 days, and the final model's first-week bias in each of the six breaks

Run:  python -m src.experiments.run_series32_first_week_correction [--workers 10] [--suffix _cal2]
"""

from __future__ import annotations

import argparse
import time
from itertools import product

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_package_validation as pv
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series29_school_calendar as s29
from src.experiments.series19_21_common import _STATE, OUT_DIR, SUFFIX, load_protocol, read_done, run_parallel, test_days
from src.features.build_features import SCHOOL_BREAKS
from src.models.final_model import DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL

RESULTS = OUT_DIR / "series32_leave_one_break_out.jsonl"
REPORT_DIR = CONFIG.path("experiments_dir") / f"series32_first_week_correction{SUFFIX}"
SEEDS = list(range(25))
FORMS = ("additive", "ratio")
STRENGTHS = (0.5, 1.0)
SCOPES = ("first week", "first week + last week of term")
PURGE_BEFORE, PURGE_AFTER = 21, 21  # days either side of a break's start left out of the leave-one-out training rows
PREDICT_BEFORE, PREDICT_AFTER = 7, 7  # the 14 days forecast: the last week of term and the first week of the break


# --------------------------------------------------------------------------- the breaks and the correction


def breaks_in_data(dates: pd.Series) -> pd.DataFrame:
    """The Easter and summer breaks whose last-week-of-term and first-week days are all in `dates`, in order."""
    have = set(pd.to_datetime(dates).dt.normalize())
    b = SCHOOL_BREAKS[SCHOOL_BREAKS["break_type"].isin(("easter", "summer"))].copy()
    b["start"] = pd.to_datetime(b["start"])
    b["name"] = b["break_type"].str.capitalize() + " " + b["start"].dt.year.astype(str)
    keep = [all(s + pd.Timedelta(days=k) in have for k in range(-PREDICT_BEFORE, PREDICT_AFTER)) for s in b["start"]]
    return b[keep].sort_values("start").reset_index(drop=True)[["name", "break_type", "start"]]


def scope_dates(start: pd.Timestamp, scope: str) -> list[pd.Timestamp]:
    first = [start + pd.Timedelta(days=k) for k in range(7)]
    term = [start - pd.Timedelta(days=k) for k in range(PREDICT_BEFORE, 0, -1)]
    return first if scope == "first week" else term + first


def shifted(pred: np.ndarray, hist_pred: np.ndarray, hist_actual: np.ndarray, form: str, strength: float) -> np.ndarray:
    """`pred` corrected by `strength` of the gap learned from the history days."""
    if form == "additive":
        return pred - strength * float(np.mean(hist_pred - hist_actual))
    ratio = float(np.sum(hist_actual) / np.sum(hist_pred))
    return pred * (1.0 + strength * (ratio - 1.0))


def forward_chained(days: pd.DataFrame, pred: np.ndarray, breaks: pd.DataFrame, form: str, strength: float, scope: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Correct each break's scope days from the scope days of EARLIER breaks only. Returns the corrected forecasts and, per break
    that could be corrected, the indices of the days that were."""
    a = days["actual"].to_numpy()
    position = {pd.Timestamp(d): i for i, d in enumerate(pd.to_datetime(days["date"]))}
    out = pred.astype(float).copy()
    history: list[int] = []
    corrected: dict[str, np.ndarray] = {}
    for b in breaks.itertuples():
        idx = np.array([position[d] for d in scope_dates(b.start, scope) if d in position], dtype=int)
        if len(idx) == 0:
            continue
        if history:
            h = np.array(history)
            out[idx] = shifted(pred[idx], pred[h], a[h], form, strength)
            corrected[b.name] = idx
        history += idx.tolist()
    return out, corrected


def leave_one_out(frame: pd.DataFrame, form: str, strength: float, scope: str) -> pd.DataFrame:
    """`frame`: one row per break and date with columns break, start, date, actual, pred. Each break's scope days are corrected from the
    scope days of every OTHER break. Returns the frame's scope rows with a `corrected` column."""
    own_scope = frame.apply(lambda r: r["date"] in set(scope_dates(r["start"], scope)), axis=1)
    rows = []
    for name in frame["break"].unique():
        mine = frame[(frame["break"] == name) & own_scope]
        others = frame[(frame["break"] != name) & own_scope]
        corrected = shifted(mine["pred"].to_numpy(), others["pred"].to_numpy(), others["actual"].to_numpy(), form, strength)
        rows.append(mine.assign(corrected=corrected))
    return pd.concat(rows, ignore_index=True)


def break_change(a: np.ndarray, pred: np.ndarray, corrected: np.ndarray, idx: np.ndarray) -> float:
    return float(np.abs(corrected[idx] - a[idx]).mean() - np.abs(pred[idx] - a[idx]).mean())


def cluster_bootstrap(per_break_change: np.ndarray, weights: np.ndarray, reps: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Day-weighted mean change with a 95% interval from resampling whole breaks."""
    rng = np.random.default_rng(seed)
    n = len(per_break_change)
    idx = rng.integers(0, n, size=(reps, n))
    means = (per_break_change[idx] * weights[idx]).sum(axis=1) / weights[idx].sum(axis=1)
    return float((per_break_change * weights).sum() / weights.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------- leave-one-break-out fits


def lobo_windows(start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    return (start - pd.Timedelta(days=PURGE_BEFORE), start + pd.Timedelta(days=PURGE_AFTER),
            start - pd.Timedelta(days=PREDICT_BEFORE), start + pd.Timedelta(days=PREDICT_AFTER))


def lobo_predictions(df: pd.DataFrame, start: pd.Timestamp, seeds: list[int]) -> np.ndarray:
    """The final model's 14 forecasts around a break from a model trained without the 42 days around it; one row per seed."""
    from catboost import CatBoostRegressor

    purge_lo, purge_hi, test_lo, test_hi = lobo_windows(start)
    train = df[~((df[DATE_COL] >= purge_lo) & (df[DATE_COL] < purge_hi))]
    test = df[(df[DATE_COL] >= test_lo) & (df[DATE_COL] < test_hi)]
    features = list(FINAL_FEATURES)
    target = np.log1p(train[TARGET_COL])
    out = []
    for seed in seeds:
        model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
        model.fit(train[features], target)
        out.append(np.expm1(model.predict(test[features])))
    return np.array(out)


def init_worker32() -> None:
    df, folds = load_protocol()
    _STATE["df"], _STATE["folds"] = s29.with_calendar(df), folds


def lobo_task(task: dict) -> dict:
    started = time.time()
    per_seed = lobo_predictions(_STATE["df"], pd.Timestamp(task["start"]), task["seeds"])
    return {"id": task["id"], "break": task["break"], "start": task["start"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def lobo_tasks(breaks: pd.DataFrame) -> list[dict]:
    out = []
    for b in breaks.itertuples():
        for i in range(0, len(SEEDS), s23.SEED_CHUNK):
            chunk = SEEDS[i:i + s23.SEED_CHUNK]
            out.append({"id": f"{b.name.replace(' ', '_')}_s{chunk[0]}-{chunk[-1]}", "break": b.name, "start": str(b.start.date()), "seeds": chunk})
    return out


def lobo_frame(df: pd.DataFrame, breaks: pd.DataFrame) -> pd.DataFrame:
    done = read_done(RESULTS)
    actual = df.set_index(DATE_COL)[TARGET_COL]
    rows = []
    for b in breaks.itertuples():
        bag = np.concatenate([np.array(r["per_seed"]) for r in done.values() if r["break"] == b.name], axis=0).mean(axis=0)
        lo = b.start - pd.Timedelta(days=PREDICT_BEFORE)
        for k, pred in enumerate(bag):
            d = lo + pd.Timedelta(days=k)
            rows.append({"break": b.name, "start": b.start, "date": d, "actual": float(actual[d]), "pred": float(pred)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- the package forecasts


def package_forecasts(days: pd.DataFrame, suffix: str) -> dict[str, np.ndarray]:
    """The Track A package (the last step of the package table) on seeds 0-24 and on the fresh seeds, from the results set `suffix`."""
    data = pv.load_set(suffix)
    chosen = pv.choose_on_first_blocks(days, data["log_raw_0_24"], data["features_0_24"])
    return {"seeds0_24": pv.package_steps(days, data["log_raw_0_24"], data["features_0_24"], chosen)[pv.STEPS[-1]],
            "fresh": pv.package_steps(days, data["log_raw_fresh"], data["features_fresh"], chosen)[pv.STEPS[-1]],
            "final_model": data["log_raw_0_24"]["log1p_final"]}


def per_break_table(days: pd.DataFrame, pred: np.ndarray, corrected: np.ndarray, hit: dict[str, np.ndarray]) -> pd.DataFrame:
    a = days["actual"].to_numpy()
    return pd.DataFrame([{"break": name, "days": len(idx), "before_MAE": float(np.abs(pred[idx] - a[idx]).mean()),
                          "after_MAE": float(np.abs(corrected[idx] - a[idx]).mean()), "change": break_change(a, pred, corrected, idx),
                          "before_bias": float((pred[idx] - a[idx]).mean())} for name, idx in hit.items()])


def forward_table(days: pd.DataFrame, pred: np.ndarray, breaks: pd.DataFrame) -> pd.DataFrame:
    a = days["actual"].to_numpy()
    rows = []
    for form, strength, scope in product(FORMS, STRENGTHS, SCOPES):
        corrected, hit = forward_chained(days, pred, breaks, form, strength, scope)
        idx = np.concatenate(list(hit.values()))
        per = per_break_table(days, pred, corrected, hit)
        rows.append({"form": form, "strength": strength, "scope": scope, "days": len(idx), "before_MAE": float(np.abs(pred[idx] - a[idx]).mean()),
                     "after_MAE": float(np.abs(corrected[idx] - a[idx]).mean()), "change": float(np.abs(corrected[idx] - a[idx]).mean() - np.abs(pred[idx] - a[idx]).mean()),
                     "breaks_improved": int((per["change"] < 0).sum()), "breaks_scored": len(per), "all_585_days_change": float(np.abs(corrected - a).mean() - np.abs(pred - a).mean())})
    return pd.DataFrame(rows)


def followup_first_week_only(days: pd.DataFrame, pkg: dict[str, np.ndarray], breaks: pd.DataFrame) -> pd.DataFrame:
    """POST-HOC (not pre-registered): the additive first-week-only corrections, forward-chained, on both seed sets, break by break."""
    a = days["actual"].to_numpy()
    rows = []
    for strength, seeds in product((0.5, 1.0), ("seeds0_24", "fresh")):
        corrected, hit = forward_chained(days, pkg[seeds], breaks, "additive", strength, "first week")
        idx = np.concatenate(list(hit.values()))
        diff = np.abs(corrected - a) - np.abs(pkg[seeds] - a)
        mask = np.zeros(len(a)); mask[idx] = 1.0
        mean, lo, hi = s23.masked_block_bootstrap(diff, mask, block=7)
        for r in per_break_table(days, pkg[seeds], corrected, hit).to_dict("records"):
            rows.append({"strength": strength, "seeds": seeds, "break": r["break"], "before_bias": r["before_bias"],
                         "before_MAE": r["before_MAE"], "after_MAE": r["after_MAE"], "change": r["change"]})
        rows.append({"strength": strength, "seeds": seeds, "break": f"POOLED {len(idx)} days", "before_bias": np.nan,
                     "before_MAE": float(np.abs(pkg[seeds] - a)[idx].mean()), "after_MAE": float(np.abs(corrected - a)[idx].mean()),
                     "change": mean, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def main(workers: int, suffix: str) -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    breaks = breaks_in_data(df[DATE_COL])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"Easter and summer breaks with data: {breaks['name'].tolist()}")

    pkg = package_forecasts(days, suffix)
    a = days["actual"].to_numpy()

    # ---- step 1: the 8 variants, forward-chained, on the package, seeds 0-24
    grid = forward_table(days, pkg["seeds0_24"], breaks)
    grid.to_csv(REPORT_DIR / "forward_chained_seeds0_24.csv", index=False)
    print("\nStep 1 - forward-chained on the package, seeds 0-24 (change = after minus before, £; negative = better):\n", grid.round(2).to_string(index=False))
    best = grid.loc[grid["after_MAE"].idxmin()]
    form, strength, scope = best["form"], float(best["strength"]), best["scope"]
    print(f"\nWinner: {form}, strength {strength}, scope '{scope}': change {best['change']:+.1f} on {int(best['days'])} days "
          f"({int(best['breaks_improved'])} of {int(best['breaks_scored'])} breaks improve; optimistic: it picked itself)")

    # ---- step 2: fresh seeds, same settings
    corrected_f, hit_f = forward_chained(days, pkg["fresh"], breaks, form, strength, scope)
    idx_f = np.concatenate(list(hit_f.values()))
    diff = np.abs(corrected_f - a) - np.abs(pkg["fresh"] - a)
    mask = np.zeros(len(a)); mask[idx_f] = 1.0
    mean, lo, hi = s23.masked_block_bootstrap(diff, mask, block=7)
    per = per_break_table(days, pkg["fresh"], corrected_f, hit_f)
    per.to_csv(REPORT_DIR / "winner_fresh_seeds_by_break.csv", index=False)
    print(f"\nStep 2 - the winner on FRESH seeds: {len(idx_f)} days, MAE {np.abs(pkg['fresh'] - a)[idx_f].mean():.1f} -> {np.abs(corrected_f - a)[idx_f].mean():.1f}, "
          f"change {mean:+.1f} (95% interval {lo:+.1f}, {hi:+.1f}); all 585 days {np.abs(corrected_f - a).mean() - np.abs(pkg['fresh'] - a).mean():+.2f}\n", per.round(1).to_string(index=False))
    ok_a, ok_b = bool(hi < 0), int((per["change"] < 0).sum()) >= 2
    print(f"Forward-chained bar: (a) interval below zero {ok_a}, (b) at least two of three breaks improve {ok_b}")

    # ---- leave one break out (six breaks, the final model)
    run_parallel(lobo_tasks(breaks), lobo_task, RESULTS, workers, init_worker32, "series32 leave-one-break-out")
    frame = lobo_frame(s29.with_calendar(df), breaks)
    first = frame[frame.apply(lambda r: r["date"] >= r["start"], axis=1)]
    bias = first.groupby("break", sort=False).apply(lambda g: pd.Series({"MAE": (g["pred"] - g["actual"]).abs().mean(), "bias": (g["pred"] - g["actual"]).mean(),
                                                                         "actual_vs_forecast_%": ((g["actual"] / g["pred"]).mean() - 1) * 100}), include_groups=False)
    print("\nLeave one break out: the final model's forecast of each first week (a model that never saw the break):\n", bias.round(1).to_string())
    lo_rows = []
    for f, s, sc in product(FORMS, STRENGTHS, SCOPES):
        out = leave_one_out(frame, f, s, sc)
        by = out.groupby("break", sort=False).apply(lambda g: pd.Series({"days": len(g), "change": (g["corrected"] - g["actual"]).abs().mean() - (g["pred"] - g["actual"]).abs().mean()}), include_groups=False)
        m, l, h = cluster_bootstrap(by["change"].to_numpy(), by["days"].to_numpy())
        lo_rows.append({"form": f, "strength": s, "scope": sc, "days": int(by["days"].sum()), "change": m, "lo": l, "hi": h,
                        "breaks_improved": int((by["change"] < 0).sum()), "breaks": len(by)})
    loo = pd.DataFrame(lo_rows)
    loo.to_csv(REPORT_DIR / "leave_one_break_out.csv", index=False)
    print("\nLeave one break out, the final model, six breaks (change in MAE on the scope days; interval from resampling whole breaks):\n", loo.round(2).to_string(index=False))
    same = loo[(loo["form"] == form) & (loo["strength"] == strength) & (loo["scope"] == scope)].iloc[0]
    ok_c = int(same["breaks_improved"]) >= 4
    print(f"\nThe winner's settings on leave-one-break-out: change {same['change']:+.1f} ({same['lo']:+.1f}, {same['hi']:+.1f}), {int(same['breaks_improved'])} of {int(same['breaks'])} breaks improve")
    print(f"Overall bar: {'PASSES' if ok_a and ok_b and ok_c else 'DOES NOT PASS (park as a lead for the 2027 breaks)'}")
    follow = followup_first_week_only(days, pkg, breaks)
    follow.to_csv(REPORT_DIR / "posthoc_first_week_only.csv", index=False)
    print("\nPOST-HOC follow-up (not pre-registered): additive, first week only, forward-chained, break by break:\n", follow.round(1).to_string(index=False))
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--suffix", default="_cal2", help="the results set holding the package re-run on the verified calendar")
    args = parser.parse_args()
    main(args.workers, args.suffix)
