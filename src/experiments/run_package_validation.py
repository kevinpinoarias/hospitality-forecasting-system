"""
Package validation: does the Track A package (Series 23-27) hold up when its inputs change?

Re-running every series after every change is expensive, so each change is tested on its own against the baseline and the
package is re-validated at milestones. This script is the milestone check. It rebuilds the package from two complete sets of
results and prints one table:

  results set  ""      the original runs (original school calendar)
  results set  "_cal2" the same series re-run on the verified calendar (SERIES_RESULTS_SUFFIX=_cal2), including every
                       selection step: the grid winner of Series 24, the weekday options of Series 27

The package is: the log model on Monday to Thursday, the raw-£ model on Saturday, each weekend day's own feature option
(chosen on test blocks 0-4, as in Series 27) and the Saturday and Sunday correction (8 weeks, strength 0.5; the grid of
Series 24 is re-run and reported to see whether it still picks them). Scored on all 585 days, on the scored blocks 1-9
(the correction needs history) and on the judging blocks 5-9 that the weekday choices never saw, on the original seeds
(0-24) and on 25 fresh seeds (100-124).

Run:  python -m src.experiments.run_package_validation [--new _cal2] [--old ""]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.forecast_core.correction import CORRECTION  # noqa: F401  (moved there unchanged)
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series25_weekend_training as s25
from src.experiments import run_series27_weekend_per_weekday as s27
from src.experiments.series19_21_common import load_protocol, read_done, test_days

STEPS = ["final model (log everywhere)", "+ raw Saturday (hybrid)", "+ weekend features per weekday", "+ Saturday and Sunday correction"]


def results_dir(suffix: str):
    return CONFIG.path("experiments_dir") / f"series19_21{suffix}"


def bag_dict(path, key_fields: tuple[str, ...], n_seeds: int) -> dict:
    """Mean over seeds of each variant's per-seed predictions in a results file, keyed by the given record fields."""
    groups: dict = {}
    for rec in read_done(path).values():
        groups.setdefault(tuple(rec[f] for f in key_fields), []).append(np.array(rec["per_seed"]))
    out = {}
    for key, rows in groups.items():
        per_seed = np.concatenate(rows, axis=0)
        assert per_seed.shape[0] == n_seeds, (key, per_seed.shape)
        out[key if len(key) > 1 else key[0]] = per_seed.mean(axis=0)
    return out


def load_set(suffix: str) -> dict:
    d = results_dir(suffix)
    return {
        "log_raw_0_24": bag_dict(d / "series23_log_vs_raw.jsonl", ("variant",), 25),
        "log_raw_fresh": bag_dict(d / "series24_fresh_seeds.jsonl", ("variant",), 25),
        "features_0_24": bag_dict(d / "series26_weekend_features.jsonl", ("variant", "target"), 25),
        "features_fresh": bag_dict(d / "series26_fresh_seeds.jsonl", ("variant", "target"), 25),
    }


def package_steps(days: pd.DataFrame, log_raw: dict, features: dict, chosen: dict[str, str]) -> dict[str, np.ndarray]:
    wd = days["weekday"].to_numpy()
    log, raw = log_raw["log1p_final"], log_raw["raw_same_params"]
    hybrid = s24.hybrid(wd, log, raw)
    options = {}
    for day, option in chosen.items():
        target = s27.DAY_TARGET[day]
        base = log if target == "log1p" else raw
        options[day] = {"none": base, option: features[(option, target)]} if option != "none" else {"none": base}
    weekend = s27.assemble_weekend(days, hybrid, options, chosen)
    corrected = s24.correct(days, weekend, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])
    return dict(zip(STEPS, [log, hybrid, weekend, corrected]))


def choose_on_first_blocks(days: pd.DataFrame, log_raw: dict, features_0_24: dict) -> dict[str, str]:
    baseline = {"log1p_final": log_raw["log1p_final"], "raw_same_params": log_raw["raw_same_params"]}
    options = {d: s27.option_predictions(d, features_0_24, baseline) for d in s25.FRI_SUN}
    return s27.choose_options(days, options, days["fold_id"].to_numpy() < s27.SELECTION_BLOCKS)


def grid_winner(days: pd.DataFrame, log_raw: dict) -> tuple[int, float, float]:
    hybrid = s24.hybrid(days["weekday"].to_numpy(), log_raw["log1p_final"], log_raw["raw_same_params"])
    grid = s24.grid_table(days, hybrid)
    best = grid.loc[grid["MAE"].idxmin()]
    return int(best["half_life_weeks"]), float(best["strength"]), float(best["change"])


def scores(days: pd.DataFrame, steps: dict[str, np.ndarray]) -> pd.DataFrame:
    a, fold = days["actual"].to_numpy(), days["fold_id"].to_numpy()
    masks = {"all 585 days": np.ones(len(a), bool), "blocks 1-9 (540)": fold >= 1, "blocks 5-9 (300)": fold >= s27.SELECTION_BLOCKS}
    rows = []
    for name, pred in steps.items():
        rows.append({"step": name, **{label: s24.mae(a, pred, m) for label, m in masks.items()}})
    return pd.DataFrame(rows)


def main(old: str, new: str) -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    out_dir = CONFIG.path("experiments_dir") / "package_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for label, suffix in (("original calendar", old), ("verified calendar", new)):
        data = load_set(suffix)
        chosen = choose_on_first_blocks(days, data["log_raw_0_24"], data["features_0_24"])
        hl, st, change = grid_winner(days, data["log_raw_0_24"])
        print(f"\n===== {label} (results set '{suffix}')\nweekday options chosen on blocks 0-4: {chosen}\n"
              f"Series 24 grid winner on this set's hybrid: {hl} weeks, strength {st} (change {change:+.2f})")
        for seeds, key_lr, key_ft in (("seeds 0-24", "log_raw_0_24", "features_0_24"), ("fresh seeds 100-124", "log_raw_fresh", "features_fresh")):
            if seeds.startswith("fresh") and any((o, s27.DAY_TARGET[d]) not in data["features_fresh"] for d, o in chosen.items() if o != "none"):
                print(f"\n[{seeds}] not all chosen options have fresh-seed fits in this set; skipped")
                continue
            table = scores(days, package_steps(days, data[key_lr], data[key_ft], chosen))
            table.insert(0, "calendar / seeds", f"{label}, {seeds}")
            frames.append(table)
            print(f"\n[{seeds}]\n", table.drop(columns="calendar / seeds").round(2).to_string(index=False))
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(out_dir / "package_validation.csv", index=False)

    print("\n===== The package against the final model, both calendars (change in MAE £ and %):")
    for tag in sorted(set(result["calendar / seeds"])):
        t = result[result["calendar / seeds"] == tag].set_index("step")
        for col in ("all 585 days", "blocks 1-9 (540)", "blocks 5-9 (300)"):
            base, last = t.loc[STEPS[0], col], t.loc[STEPS[-1], col]
            print(f"{tag:42s} {col:18s} {base:7.2f} -> {last:7.2f}  ({last - base:+6.2f}, {100 * (last - base) / base:+.1f}%)")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default="")
    parser.add_argument("--new", default="_cal2")
    args = parser.parse_args()
    main(args.old, args.new)
