"""
Series 48 - why did the single-day flags stop helping once Friday, Saturday and Sunday got their own pieces?

Series 46 found the flags lowered the plain log model's error on Friday (-29 pounds) and Saturday (-37); Series 47 found almost nothing (-5, -2)
inside the real pieces. Two explanations are on the table, and this series tests both on the plain log model, where the flags did help:

  E1, information   the pieces' extra inputs already tell the model what the flags told it. Friday's inputs are sales on the same weekday over the
                    last 4 weeks, Saturday's the sales 2 and 3 weeks ago, Sunday's four weekend interactions (each is 0 on Monday-Thursday). Any of
                    these separates the weekend days from the rest by itself, so a flag adds little.
  E2, tree depth    the model's trees are 4 splits deep. Telling Saturday from every other day with `day_of_week` takes two splits (above 4, below 5);
                    `is_saturday` takes one. With deeper trees the flags would matter less.

Design: the plain log model (log1p target, 300 iterations, learning rate 0.05, 10 folds, 585 days, 100 seeds 200-299, as Series 46), with `is_weekend`
as today or replaced by the three single-day flags, under five settings:

  none               the final model's 38 columns, depth 4 (this is Series 46's comparison, repeated as the control and checked against it)
  friday_inputs      + the Friday piece's inputs (lag_14, lag_21, mean and max of the last 4 same weekdays)
  saturday_inputs    + the Saturday piece's inputs (lag_14, lag_21)
  sunday_inputs      + the Sunday piece's inputs (the four weekend interactions)
  depth6             the 38 columns, trees 6 splits deep

PRE-REGISTERED (written before any result was seen)
  quantity   the flags' effect: MAE with the flags minus MAE with `is_weekend`, per weekday group, in each setting (negative = the flags help)
  E1 holds   if the flags help in `none` on Friday and Saturday (interval excluding zero) and their effect is less than half as large, on the
             weekday the inputs are meant for, with friday_inputs (Friday), saturday_inputs (Saturday) and sunday_inputs (Sunday)
  E2 holds   if the same is true for depth6 on Friday and Saturday
  also read  the MAE of each setting on each weekday, to see whether the extra inputs do the flags' job better than the flags do (the question is
             whether `none` + flags, `friday_inputs` + `is_weekend` and `friday_inputs` + flags rank as the explanation predicts)
  limits     this is the log target everywhere; the real Saturday piece is a raw-pound model. A result here explains the Series 46 to 47 drop; it
             adopts nothing and changes no model.

Run:  python -m src.experiments.run_series48_why_flags_faded list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series26_weekend_features as s26
from src.experiments.run_series23_log_vs_raw_peaks import masked_block_bootstrap
from src.experiments.run_series45_year_and_weekend_flag import add_weekday_flags, variant_features as flag_features
from src.experiments.run_series46_calendar_confirmation import SEEDS, load_bags as series46_bags
from src.experiments.series19_21_common import (
    _STATE, FINAL_PARAMS, bagged_log1p_predictions, load_protocol, read_done, run_parallel, test_days,
)

CHUNK = 5
OUT_DIR = CONFIG.path("experiments_dir") / "series48"
RESULTS = OUT_DIR / "series48_why_flags_faded.jsonl"
EXTRAS = {
    "none": [], "friday_inputs": s26.VARIANTS["A_all"]["extra"], "saturday_inputs": s26.VARIANTS["A_lags"]["extra"],
    "sunday_inputs": s26.VARIANTS["B"]["extra"], "depth6": [],
}
PARAMS = {name: (dict(FINAL_PARAMS, depth=6) if name == "depth6" else dict(FINAL_PARAMS)) for name in EXTRAS}
VERSIONS = ("as_today", "single_day_flags")
GROUPS = {"all": [0, 1, 2, 3, 4, 5, 6], "Mon-Thu": [0, 1, 2, 3], "Fri": [4], "Sat": [5], "Sun": [6]}
OWN_DAY = {"friday_inputs": "Fri", "saturday_inputs": "Sat", "sunday_inputs": "Sun"}


def setting_features(setting: str, version: str) -> list[str]:
    return flag_features("kept", "as_today" if version == "as_today" else "single_days") + EXTRAS[setting]


def init_worker48() -> None:
    df, folds = s26.load_frame()
    _STATE["df"], _STATE["folds"] = add_weekday_flags(df), folds


def fit_task(task: dict) -> dict:
    started = time.time()
    per_seed = bagged_log1p_predictions(_STATE["df"], _STATE["folds"], task["features"], task["params"], task["seeds"])
    return {"id": task["id"], "setting": task["setting"], "version": task["version"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for setting in EXTRAS:
        for version in VERSIONS:
            for i in range(0, len(SEEDS), CHUNK):
                chunk = SEEDS[i:i + CHUNK]
                out.append({"id": f"{setting}_{version}_s{chunk[0]}-{chunk[-1]}", "setting": setting, "version": version, "seeds": chunk,
                            "features": setting_features(setting, version), "params": PARAMS[setting]})
    return out


def load_bags() -> dict[tuple[str, str], np.ndarray]:
    done = read_done(RESULTS)
    out = {}
    for setting in EXTRAS:
        for version in VERSIONS:
            rows = [np.array(r["per_seed"]) for r in done.values() if r["setting"] == setting and r["version"] == version]
            per_seed = np.concatenate(rows, axis=0)
            assert per_seed.shape[0] == len(SEEDS), (setting, version, per_seed.shape)
            out[(setting, version)] = per_seed.mean(axis=0)
    return out


def report() -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    a, dow = days["actual"].to_numpy(), pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    bags = load_bags()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    old = series46_bags("backtest")  # the control must reproduce Series 46's two log-model fits
    for version, name in (("as_today", "final_model"), ("single_day_flags", "single_day_flags")):
        gap = float(np.abs(bags[("none", version)] - old[name]).max())
        assert gap < 0.05, f"the control ({version}) differs from Series 46 by up to {gap}"
    print("Control checked: setting 'none' reproduces Series 46's log-model fits.\n")

    effect, interval, mae = {}, {}, {}
    for setting in EXTRAS:
        diff = np.abs(bags[(setting, "single_day_flags")] - a) - np.abs(bags[(setting, "as_today")] - a)
        for group, days_in in GROUPS.items():
            mask = np.isin(dow, days_in)
            change, lo, hi = masked_block_bootstrap(diff, mask.astype(float))
            effect[(setting, group)], interval[(setting, group)] = change, (lo, hi)
            for version in VERSIONS:
                mae[(setting, version, group)] = float(np.abs(bags[(setting, version)] - a)[mask].mean())

    eff = pd.DataFrame({g: [effect[(s, g)] for s in EXTRAS] for g in GROUPS}, index=list(EXTRAS))
    print("The flags' effect (MAE with flags minus MAE with is_weekend, pounds; negative = the flags help)")
    print(eff.round(2).to_string())
    print("\n95% interval on the flags' effect, Friday and Saturday")
    print(pd.DataFrame({g: [f"{interval[(s, g)][0]:+.1f} to {interval[(s, g)][1]:+.1f}" for s in EXTRAS] for g in ("Fri", "Sat")},
                       index=list(EXTRAS)).to_string())
    rows = [{"setting": s, "version": v, **{g: mae[(s, v, g)] for g in GROUPS}} for s in EXTRAS for v in VERSIONS]
    mae_table = pd.DataFrame(rows)
    mae_table.to_csv(OUT_DIR / "mae_by_weekday.csv", index=False)
    print("\nMAE by weekday for every setting and version")
    print(mae_table.round(1).to_string(index=False))

    def holds(setting: str, days_: tuple[str, ...]) -> bool:
        """The flags helped in the control on these weekdays (interval below zero) and help less than half as much in `setting`."""
        return all(interval[("none", g)][1] < 0 and abs(effect[(setting, g)]) < 0.5 * abs(effect[("none", g)]) for g in days_)

    e1 = {s: holds(s, (OWN_DAY[s],)) for s in OWN_DAY}
    e2 = holds("depth6", ("Fri", "Sat"))
    print("\nE1 (the pieces' inputs make the flags unnecessary), per piece:", e1)
    print("E2 (deeper trees make the flags unnecessary):", e2)
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(EXTRAS)} settings x {len(VERSIONS)} versions, {len(tasks)} tasks")
        for s in EXTRAS:
            print(f"  {s:16s} {len(setting_features(s, 'as_today'))} / {len(setting_features(s, 'single_day_flags'))} features, depth {PARAMS[s]['depth']}")
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, init_worker48, "series48")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
