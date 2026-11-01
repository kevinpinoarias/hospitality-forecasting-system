"""
Series 42 - training the log model for the day the manager's number is missing.

The rules are in docs/NO_MANAGER_TRAINING_2026_11.md, committed before this script existed. Three arms, all the log model (CatBoost, log1p,
the final model's settings, 25 seeds), refitted on the days before each of the protocol's 10 folds:

  N0  trained with the manager's real number, served with the stand-in (the model as deployed)
  N1  trained with the stand-in in place of the number (computed from the OTHER training rows), served with the stand-in
  N2  trained and served without the `forecast_sales` column

Each fold's test block is scored the way the weekly log serves a week (`served_rows`): for every week that lies wholly inside the block, next-week
days (NW) are forecast from the sales known up to the Sunday eight days before the week's Monday, this-week days (TW) from the Sunday before it.

Run:  python -m src.experiments.run_series42_no_manager run [--workers 10] [--smoke]     (resumable: finished tasks are kept)
      python -m src.experiments.run_series42_no_manager report [--smoke]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series35_holdout as h
from src.experiments.run_series23_log_vs_raw_peaks import masked_block_bootstrap
from src.experiments.series19_21_common import load_protocol, read_done, run_parallel, test_days
from src.features.serving_features import seasonal_medians
from src.models.final_model import DATE_COL, FINAL_FEATURES, TARGET_COL

MANAGER_COL = "forecast_sales"
ARMS = ("N0", "N1", "N2")
SEED_SETS = {"A": list(range(25)), "B": list(range(100, 125))}
KINDS = ("NW", "TW")
LEAD_DAYS = {"NW": 8, "TW": 1}  # days from the last known Sunday to the week's Monday
MIN_GAIN, MAX_BUSY_LOSS, MIN_FOLDS = 15.0, 25.0, 7  # rules 2, 4 and 3 of the pre-registration
SANITY_TOLERANCE = 0.05
ENGINEERED = CONFIG.path("features_dir") / "engineered_features.csv"
_S: dict = {}


def out_dir(smoke: bool = False) -> Path:
    return CONFIG.path("experiments_dir") / ("series42_smoke" if smoke else "series42")


# --------------------------------------------------------------------------- arms and weeks


def stand_in_leave_one_out(train: pd.DataFrame) -> np.ndarray:
    """The stand-in manager number for each training row, computed from the OTHER training rows only (the row itself is left out)."""
    out = np.empty(len(train))
    for i in range(len(train)):
        others = train.drop(train.index[i])
        out[i] = seasonal_medians(others, [train[DATE_COL].iloc[i]], MANAGER_COL)[0]
    return out


def arm_frame(arm: str, train: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The training rows and feature list of an arm."""
    if arm == "N0":
        return train, list(FINAL_FEATURES)
    if arm == "N1":
        replaced = train.copy()
        replaced[MANAGER_COL] = stand_in_leave_one_out(train)
        return replaced, list(FINAL_FEATURES)
    if arm == "N2":
        return train, [f for f in FINAL_FEATURES if f != MANAGER_COL]
    raise ValueError(f"unknown arm {arm!r}")


def whole_weeks(block_start: pd.Timestamp, block_end: pd.Timestamp) -> list[pd.Timestamp]:
    """Mondays of the weeks that lie wholly inside [block_start, block_end)."""
    monday = block_start + pd.Timedelta(days=(0 - block_start.weekday()) % 7)
    out = []
    while monday + pd.Timedelta(days=7) <= block_end:
        out.append(monday)
        monday += pd.Timedelta(days=7)
    return out


# --------------------------------------------------------------------------- the worker


def init_worker() -> None:
    df, folds = load_protocol()
    _S["df"], _S["folds"], _S["full"] = df, folds, h.load_engineered(ENGINEERED)


def fit_task(task: dict) -> dict:
    """One fold, one arm, one seed set: fit the bag, forecast every whole week of the test block as the log would."""
    started = time.time()
    df, folds, full = _S["df"], _S["folds"], _S["full"]
    fold = folds[task["fold"]]
    train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)].reset_index(drop=True)
    test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)].reset_index(drop=True)
    frame, features = arm_frame(task["arm"], train)
    bag = h.Bag.fit(frame, features, "log1p", seeds=task["seeds"], thread_count=1)
    scored = set(test[DATE_COL])
    served = []
    for monday in whole_weeks(fold.test_start, fold.test_end):
        for kind in KINDS:
            last_known = monday - pd.Timedelta(days=LEAD_DAYS[kind])
            targets = [monday + pd.Timedelta(days=i) for i in range(7)]
            rows = h.served_rows(full, last_known, targets, "v2", None)
            for date, pred in zip(rows[DATE_COL], bag.predict(rows)):
                if date in scored:
                    served.append({"date": date.strftime("%Y-%m-%d"), "kind": kind, "pred": round(float(pred), 4)})
    record = {"id": task["id"], "arm": task["arm"], "seedset": task["seedset"], "fold": task["fold"], "served": served,
              "seconds": round(time.time() - started, 1)}
    if task["arm"] == "N0" and task["seedset"] == "A":  # sanity check: real inputs and real history reproduce the stored backtest
        record["real_inputs"] = {d.strftime("%Y-%m-%d"): round(float(p), 4) for d, p in zip(test[DATE_COL], bag.predict(test))}
    return record


def build_tasks(folds: list[int], seeds_per_set: int | None = None) -> list[dict]:
    tasks = []
    for fold in folds:
        for arm in ARMS:
            for name, seeds in SEED_SETS.items():
                seeds = seeds[:seeds_per_set] if seeds_per_set else seeds
                tasks.append({"id": f"s42_{arm}_{name}_f{fold}", "arm": arm, "seedset": name, "fold": fold, "seeds": seeds})
    return tasks


# --------------------------------------------------------------------------- analysis


def served_frame(records: dict[str, dict], days: pd.DataFrame) -> pd.DataFrame:
    """One row per arm, seed set, kind and day: the forecast, the actual and the absolute error."""
    rows = [{"arm": r["arm"], "seedset": r["seedset"], "fold": r["fold"], **s} for r in records.values() for s in r["served"]]
    frame = pd.DataFrame(rows).merge(days[["date", "actual"]], on="date", how="left")
    frame["abs_err"] = (frame["pred"] - frame["actual"]).abs()
    return frame.sort_values(["arm", "seedset", "kind", "date"]).reset_index(drop=True)


def change_vs_n0(frame: pd.DataFrame, arm: str, kind: str) -> pd.DataFrame:
    """Per day: the arm's absolute error minus N0's, for each seed set and averaged over the two (negative is a gain)."""
    wide = frame[frame["kind"] == kind].pivot_table(index=["date", "fold", "actual"], columns=["arm", "seedset"], values="abs_err").reset_index()
    out = wide[["date", "fold", "actual"]].copy()
    for s in SEED_SETS:
        out[f"diff_{s}"] = (wide[(arm, s)] - wide[("N0", s)]).to_numpy()
    out["diff"] = out[[f"diff_{s}" for s in SEED_SETS]].mean(axis=1)
    return out.dropna().sort_values("date").reset_index(drop=True)


def interval(diff: np.ndarray) -> tuple[float, float, float]:
    return masked_block_bootstrap(np.asarray(diff, dtype=float), np.ones(len(diff)))


def judge(frame: pd.DataFrame, arm: str) -> dict:
    """The four pre-registered rules for one arm against N0 (see section 9 of the rules document for how 'pooled' is read)."""
    nw, tw = change_vs_n0(frame, arm, "NW"), change_vs_n0(frame, arm, "TW")
    per_set = {s: interval(nw[f"diff_{s}"]) for s in SEED_SETS}
    pooled = interval(nw["diff"])
    tw_sets = {s: interval(tw[f"diff_{s}"]) for s in SEED_SETS}
    fold_means = nw.groupby("fold")["diff"].mean()
    busy = nw[nw["actual"] >= nw["actual"].quantile(0.75)]
    busy_change = float(busy["diff"].mean())
    rules = {
        "1 gain, upper end below zero on both seed sets": all(hi < 0 for _, _, hi in per_set.values()),
        f"2 pooled gain of at least {MIN_GAIN:.0f}": pooled[0] <= -MIN_GAIN,
        f"3 better in at least {MIN_FOLDS} folds": int((fold_means < 0).sum()) >= MIN_FOLDS,
        "4 no harm: TW interval not wholly above zero, busiest quarter within limit": all(lo <= 0 for _, lo, _ in tw_sets.values()) and busy_change <= MAX_BUSY_LOSS,
    }
    return {"arm": arm, "nw_days": len(nw), "tw_days": len(tw), "nw_per_set": per_set, "nw_pooled": pooled, "tw_per_set": tw_sets,
            "folds_better": int((fold_means < 0).sum()), "folds": int(len(fold_means)), "busy_change": busy_change, "rules": rules,
            "candidate": all(rules.values())}


def sanity_check(records: dict[str, dict], days: pd.DataFrame) -> dict:
    """N0 with real inputs and real history against the stored log-model backtest on the verified calendar (the `_cal2` results set).
    models/final_model/backtest_predictions.csv is NOT the reference: it is the deployed model on the original school calendar."""
    from src.experiments import run_package_validation as pv

    stored = pv.load_set("_cal2")["log_raw_0_24"]["log1p_final"]
    ours = {}
    for r in records.values():
        ours.update(r.get("real_inputs", {}))
    mine = np.array([ours.get(d, np.nan) for d in days["date"]])
    diff = np.abs(mine - stored)
    return {"days": int(np.isfinite(diff).sum()), "max_abs_diff": float(np.nanmax(diff)) if np.isfinite(diff).any() else float("nan"),
            "passes": bool(np.isfinite(diff).all() and np.nanmax(diff) <= SANITY_TOLERANCE)}


def mae_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["kind", "arm", "seedset"])["abs_err"].agg(days="size", MAE="mean").round(1).reset_index()


def report(records: dict[str, dict], days: pd.DataFrame, smoke: bool = False, check_sanity: bool = True) -> str:
    frame = served_frame(records, days)
    lines = ["# Series 42: scores", ""]
    if check_sanity:
        s = sanity_check(records, days)
        lines += [f"Sanity check (N0, real inputs, against the stored verified-calendar log-model backtest): {s['days']} days, largest difference {s['max_abs_diff']:.4f}, "
                  f"{'PASS' if s['passes'] else 'FAIL'} (tolerance {SANITY_TOLERANCE}).", ""]
    lines += ["## Mean absolute error (£ a day)", "", mae_table(frame).to_markdown(index=False), ""]
    verdicts = [judge(frame, arm) for arm in ("N1", "N2")]
    for v in verdicts:
        lines += [f"## {v['arm']} against N0", "",
                  f"NW days {v['nw_days']}, TW days {v['tw_days']}. Change in absolute error, £ a day (negative is a gain), 95% block-bootstrap interval:", "",
                  "| Scoring | Change | Low | High |", "|---|---|---|---|"]
        for s, (m, lo, hi) in v["nw_per_set"].items():
            lines.append(f"| NW, seed set {s} | {m:+.1f} | {lo:+.1f} | {hi:+.1f} |")
        m, lo, hi = v["nw_pooled"]
        lines.append(f"| NW, pooled | {m:+.1f} | {lo:+.1f} | {hi:+.1f} |")
        for s, (m, lo, hi) in v["tw_per_set"].items():
            lines.append(f"| TW, seed set {s} | {m:+.1f} | {lo:+.1f} | {hi:+.1f} |")
        lines += ["", f"Better in {v['folds_better']} of {v['folds']} folds on NW days. Busiest quarter of NW days: {v['busy_change']:+.1f}.", "", "| Rule | Holds |", "|---|---|"]
        lines += [f"| {name} | {'yes' if ok else 'no'} |" for name, ok in v["rules"].items()]
        lines += ["", f"**{v['arm']}: {'candidate' if v['candidate'] else 'not a candidate'}**", ""]
    n12 = change_vs_n0(frame, "N1", "NW")["diff"].mean() - change_vs_n0(frame, "N2", "NW")["diff"].mean()
    lines += [f"N1 minus N2 (pooled NW change): {n12:+.1f} £ a day (within £5 means one finding).", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- command line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="rehearsal: last fold only, 2 seeds per set, written to its own folder")
    args = parser.parse_args()
    out = out_dir(args.smoke)
    df, folds = load_protocol()
    days = test_days(df, folds)
    path = out / "tasks.jsonl"
    fold_ids = [len(folds) - 1] if args.smoke else list(range(len(folds)))
    tasks = build_tasks(fold_ids, seeds_per_set=2 if args.smoke else None)
    if args.command == "run":
        run_parallel(tasks, fit_task, path, args.workers, initializer=init_worker, label="s42")
    done = read_done(path)
    records = {t["id"]: done[t["id"]] for t in tasks if t["id"] in done}
    if len(records) < len(tasks):
        print(f"[s42] {len(tasks) - len(records)} tasks not finished; run again to resume")
        return 1
    text = report(records, days, smoke=args.smoke, check_sanity=not args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(text, encoding="utf-8")
    served_frame(records, days).to_csv(out / "served_predictions.csv", index=False)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
