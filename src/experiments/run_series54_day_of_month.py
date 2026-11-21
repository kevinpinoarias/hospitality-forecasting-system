"""
Series 54 - does where a day sits in the month help the weekly log's models?

Day of month has not been tried. Months have 28 to 31 days, so "day 31" is the last day in some months and does not exist in others; the same
fact is described four ways here, appended to the model's columns (after the piece's own extra inputs, so the baseline is a prefix of every variant):

  none                  today's piece, nothing added
  day_of_month          1 to 31
  day_and_days_to_end   day_of_month and days_to_month_end (days in the month minus the day, 0 on the last day): the distance from both edges
  circle                day_of_month_sin and day_of_month_cos, angle 2 pi (day - 1) / days in the month, so the last day of a month sits next to
                        the first day of the next whatever the length (Series 44 found the sine/cosine pair, not the plain number, carried
                        day of year)
  all                   the four columns above together

Every variant is added to every piece of the weekly log (src/forecast_log/v1.py), each as deployed:

  Monday-Thursday          the plain log model                                    (post model)
  Monday-Thursday (hydro)  the log model with the Thursday-Hydro flag             (hydro model)
  Friday                   log1p + the Friday inputs (A_all)
  Saturday                 raw pounds + the Saturday inputs (A_lags)
  Sunday                   log1p + the Sunday inputs (B)
  Saturday fallback (raw)  the plain raw-pound model, which the log uses on a Saturday whose A_lags inputs are not yet known

The post model is the first piece with the three weekend pieces; the hydro model swaps the Monday-Thursday piece for the hydro one; each goes through
the Series 24 correction (half life 8 weeks, strength 0.5, Saturday and Sunday). The fallback piece is not in either package (the backtest's
Saturdays have their inputs); it is scored on Saturdays on its own. Baselines for the four post-model pieces are the 100-seed fits already made in
Series 46-47 (identical frames, settings and seeds); the hydro and fallback baselines are fitted here.

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  protocol   the final model's: 10 folds, 585 backtest days, 100 seeds (200-299), and the 56 days from 7 September 2026
  look at    for each variant, the change in the post model's and the hydro model's MAE after the correction with a 95% moving-block bootstrap
             interval; where it comes from by weekday and piece; in how many folds it is better; the new days
  expected   little or nothing: a screen of the post model's errors (src/experiments/run_error_calendar_screen.py) found no pattern by day of
             month. This is the test of the features themselves, which a one-way screen cannot replace (interactions)
  limits     no pass/fail gate; nothing in production changes; the deployed v1 pieces are never refitted, so a gain would be a new version

Run:  python -m src.experiments.run_series54_day_of_month list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series50_drop_year_in_pieces as s50
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import CUTOFF, SEEDS
from src.experiments.series19_21_common import (
    _STATE, DATE_COL, FINAL_FEATURES, FINAL_PARAMS, TARGET_COL, bagged_log1p_predictions, load_protocol, read_done, run_parallel, test_days,
)
from src.forecast_log.v1 import add_hydro_thu

BACKTEST_CHUNK, NEW_DAYS_CHUNK = 5, 10
OUT_DIR = CONFIG.path("experiments_dir") / "series54"
RESULTS = OUT_DIR / "series54_day_of_month.jsonl"
VARIANT_COLUMNS = {
    "none": [],
    "day_of_month": ["day_of_month"],
    "day_and_days_to_end": ["day_of_month", "days_to_month_end"],
    "circle": ["day_of_month_sin", "day_of_month_cos"],
    "all": ["day_of_month", "days_to_month_end", "day_of_month_sin", "day_of_month_cos"],
}
# piece -> (extra inputs, target, weekdays scored, is the "none" baseline reused from Series 46-47)
PIECES: dict[str, tuple[list[str], str, list[int], bool]] = {
    "Monday-Thursday": ([], "log1p", [0, 1, 2, 3], True),
    "Friday": (s47.PIECES["Friday"][0], "log1p", [4], True),
    "Saturday": (s47.PIECES["Saturday"][0], "raw", [5], True),
    "Sunday": (s47.PIECES["Sunday"][0], "log1p", [6], True),
    "Monday-Thursday (hydro)": (["hydro_thu"], "log1p", [0, 1, 2, 3], False),
    "Saturday fallback (raw)": ([], "raw", [5], False),
}


def piece_features(piece: str, variant: str) -> list[str]:
    return list(FINAL_FEATURES) + PIECES[piece][0] + VARIANT_COLUMNS[variant]


def add_day_of_month_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    day, length = out[DATE_COL].dt.day, out[DATE_COL].dt.days_in_month
    out["day_of_month"] = day
    out["days_to_month_end"] = length - day
    angle = 2 * np.pi * (day - 1) / length
    out["day_of_month_sin"], out["day_of_month_cos"] = np.sin(angle), np.cos(angle)
    return out


def init_worker54() -> None:
    s47.init_worker47()
    clash = {"day_of_month", "days_to_month_end", "day_of_month_sin", "day_of_month_cos"} & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_day_of_month_columns(add_hydro_thu(_STATE["df"]))


def fit_task(task: dict) -> dict:
    df, folds = _STATE["df"], _STATE["folds"]
    started, features, target = time.time(), task["features"], PIECES[task["piece"]][1]
    if task["part"] == "backtest":
        if target == "log1p":
            per_seed = bagged_log1p_predictions(df, folds, features, FINAL_PARAMS, task["seeds"])
        else:
            per_seed = s23.raw_predictions(df, folds, features, FINAL_PARAMS, task["seeds"])
    else:
        from catboost import CatBoostRegressor

        train, test = df[df[DATE_COL] < CUTOFF], df[df[DATE_COL] >= CUTOFF]
        y = np.log1p(train[TARGET_COL]) if target == "log1p" else train[TARGET_COL]
        rows = []
        for seed in task["seeds"]:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
            model.fit(train[features], y)
            p = model.predict(test[features])
            rows.append(np.expm1(p) if target == "log1p" else p)
        per_seed = np.array(rows)
    return {"id": task["id"], "part": task["part"], "piece": task["piece"], "variant": task["variant"], "seeds": task["seeds"],
            "per_seed": [[round(float(x), 2) for x in row] for row in per_seed], "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for piece, (_, _, _, reused) in PIECES.items():
        for variant in VARIANT_COLUMNS:
            if variant == "none" and reused:
                continue
            features = piece_features(piece, variant)
            for part, size in (("backtest", BACKTEST_CHUNK), ("new_days", NEW_DAYS_CHUNK)):
                for i in range(0, len(SEEDS), size):
                    chunk = SEEDS[i:i + size]
                    out.append({"id": f"{part}_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece, "variant": variant,
                                "seeds": chunk, "features": features})
    return out


def load_bags(part: str, results=RESULTS, variant_columns=VARIANT_COLUMNS, none_results=None) -> dict[tuple[str, str], np.ndarray]:
    """(piece, variant) -> the 100-seed bag, with the reused Series 46-47 baselines for the four post-model pieces. Later series reuse this
    with their own `results` file and `variant_columns`, and `none_results` (this series' file) for the baselines of the other two pieces."""
    done = read_done(results)
    none_done = read_done(none_results) if none_results else done
    reused = s50.load_kept(part)
    out = {}
    for piece, (_, _, _, is_reused) in PIECES.items():
        for variant in variant_columns:
            if variant == "none" and is_reused:
                out[(piece, variant)] = reused[piece]
                continue
            source = none_done if variant == "none" else done
            rows = [np.array(r["per_seed"]) for r in source.values() if (r["part"], r["piece"], r["variant"]) == (part, piece, variant)]
            per_seed = np.concatenate(rows, axis=0)
            assert per_seed.shape[0] == len(SEEDS), (part, piece, variant, per_seed.shape)
            out[(piece, variant)] = per_seed.mean(axis=0)
    return out


def report(results=RESULTS, out_dir=OUT_DIR, variant_columns=VARIANT_COLUMNS, none_results=None) -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, fold = days["actual"].to_numpy(), days["fold_id"].to_numpy()
    dow = pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    new = df[df[DATE_COL] >= CUTOFF]
    a_new, dow_new = new[TARGET_COL].to_numpy(), new[DATE_COL].dt.dayofweek.to_numpy()
    bt, nd = load_bags("backtest", results, variant_columns, none_results), load_bags("new_days", results, variant_columns, none_results)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v for v in variant_columns if v != "none"]

    def interval(diff: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        return s23.masked_block_bootstrap(diff, mask.astype(float))

    rows = []
    for piece, (_, _, weekdays, _) in PIECES.items():
        own, own_new = np.isin(dow, weekdays), np.isin(dow_new, weekdays)
        for v in variants:
            d = np.abs(bt[(piece, v)] - a) - np.abs(bt[(piece, "none")] - a)
            change, lo, hi = interval(d, own)
            d_new = np.abs(nd[(piece, v)] - a_new) - np.abs(nd[(piece, "none")] - a_new)
            rows.append({"piece": piece, "variant": v, "change": change, "lo": lo, "hi": hi,
                         "folds_better": int(sum(d[own & (fold == k)].mean() < 0 for k in range(10))),
                         "new_days_change": float(d_new[own_new].mean())})
    pieces = pd.DataFrame(rows)
    pieces.to_csv(out_dir / "pieces.csv", index=False)
    print("EACH PIECE ON ITS OWN WEEKDAYS (change = with the variant minus as deployed; negative = better; 585 days, 100 seeds)")
    for v in variants:
        print(f"\n{v}")
        print(pieces[pieces["variant"] == v].drop(columns="variant").round(2).to_string(index=False))

    def package(variant: str, mon_thu: str, source: dict) -> tuple[np.ndarray, np.ndarray]:
        pred = np.select([dow == 4, dow == 5, dow == 6],
                         [source[("Friday", variant)], source[("Saturday", variant)], source[("Sunday", variant)]], default=source[(mon_thu, variant)])
        return pred, s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    out = []
    for name, mon_thu in (("post model", "Monday-Thursday"), ("hydro model", "Monday-Thursday (hydro)")):
        base_bt = package("none", mon_thu, bt)
        for v in variants:
            cand = package(v, mon_thu, bt)
            for step, idx in (("before the correction", 0), ("after the correction", 1)):
                diff = np.abs(cand[idx] - a) - np.abs(base_bt[idx] - a)
                change, lo, hi = interval(diff, np.ones(len(a), bool))
                out.append({"model": name, "variant": v, "step": step, "MAE_as_deployed": float(np.abs(base_bt[idx] - a).mean()),
                            "MAE_variant": float(np.abs(cand[idx] - a).mean()), "change": change, "lo": lo, "hi": hi,
                            "Mon-Thu": float(diff[dow < 4].mean()), "Fri": float(diff[dow == 4].mean()), "Sat": float(diff[dow == 5].mean()),
                            "Sun": float(diff[dow == 6].mean()), "folds_better": int(sum(diff[fold == k].mean() < 0 for k in range(10)))})
    packages = pd.DataFrame(out)
    packages.to_csv(out_dir / "packages.csv", index=False)
    print("\n\nTHE WHOLE MODELS (all 585 days)")
    for name in ("post model", "hydro model"):
        print(f"\n{name}")
        print(packages[packages["model"] == name].drop(columns="model").round(2).to_string(index=False))

    print("\n\nNEW DAYS (56 days from 7 September 2026), the models before the correction, change in pounds a day, day-weighted over the pieces")
    for name, mon_thu in (("post model", "Monday-Thursday"), ("hydro model", "Monday-Thursday (hydro)")):
        base = np.select([dow_new == 4, dow_new == 5, dow_new == 6], [nd[("Friday", "none")], nd[("Saturday", "none")], nd[("Sunday", "none")]], default=nd[(mon_thu, "none")])
        line = {}
        for v in variants:
            cand = np.select([dow_new == 4, dow_new == 5, dow_new == 6], [nd[("Friday", v)], nd[("Saturday", v)], nd[("Sunday", v)]], default=nd[(mon_thu, v)])
            line[v] = round(float((np.abs(cand - a_new) - np.abs(base - a_new)).mean()), 2)
        print(f"  {name:12s} {line}")
    print(f"\nSaved to {out_dir}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(PIECES)} pieces x {len(VARIANT_COLUMNS)} variants (4 baselines reused), {len(tasks)} tasks")
        for piece in PIECES:
            print(f"  {piece:26s} baseline {len(piece_features(piece, 'none'))} columns, 'all' {len(piece_features(piece, 'all'))}, target {PIECES[piece][1]}")
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, init_worker54, "series54")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
