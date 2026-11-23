"""
Series 29 - what does the corrected school calendar do to the model?

The school-holiday flags came from a table that started at Christmas 2024 and had several wrong dates (see
docs/SCHOOL_CALENDAR.md): 2024's Easter and summer holidays were term time, the first weekend of every holiday was
term time, and some starts and ends were days or a week off. This series refits the final model and the raw-£
Saturday model with ONLY the four school columns replaced by the verified calendar, and compares them with the
Series 23 fits (original calendar) on the same 585 test days.

  log1p_final       the final model (log1p target, 38 features, depth 4, 300 iterations, learning rate 0.05)
  raw_same_params   the same settings on the raw-£ target (the Saturday model of the hybrid)

PRE-REGISTERED (written before any result was seen)
  H29        the verified calendar lowers the final model's MAE
  seeds      25 seeds (0-24), repeated on 25 different seeds (100-124) as a check
  reading    pooled MAE over the 585 days, the change against the original calendar with a 95% moving-block
             bootstrap interval, and the same by weekday and by where a day sits in a school break
  decision   the corrected calendar is kept whatever the result, because it is the correct input; the result tells us how
             much of the model's accuracy was riding on the wrong one, and restates the baseline for later series
  tag        `--tag` names the calendar version and the results file; stage 1 is Christmas, Easter and summer, a later
             stage adds the October week and the February mid-term

Run:  python -m src.experiments.run_series29_school_calendar [--workers 10] [--tag stage1]
"""

from __future__ import annotations

import argparse
import functools

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.forecast_core.rows import SCHOOL_COLUMNS, with_calendar  # noqa: F401  (moved there unchanged)
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments.run_error_autopsy import school_phase
from src.experiments.series19_21_common import _STATE, OUT_DIR, load_protocol, run_parallel, test_days
from src.features.build_features import (
    LONG_BREAK_TYPES, SCHOOL_BREAK_TYPES, SCHOOL_BREAKS, SCHOOL_BREAKS_V1,
)

SEEDS = {"seeds0_24": list(range(25)), "seeds100_124": list(range(100, 125))}
VARIANT_TARGETS = ("log1p_final", "raw_same_params")
STAGE_TYPES = {"stage1": LONG_BREAK_TYPES, "stage2": SCHOOL_BREAK_TYPES}  # stage 2 adds the October week and February mid-term
LONG_BREAKS = SCHOOL_BREAKS[SCHOOL_BREAKS["break_type"].isin(LONG_BREAK_TYPES)]  # the phase table is about the long holidays
PHASES = ["last week of term", "first week of a break", "middle of a break", "last week of a break", "other term time"]


def init_worker29(tag: str = "stage2") -> None:
    df, folds = load_protocol()
    _STATE["df"], _STATE["folds"] = with_calendar(df, break_types=STAGE_TYPES[tag]), folds


def results_path(tag: str):
    return OUT_DIR / f"series29_school_calendar_{tag}.jsonl"


def make_tasks() -> list[dict]:
    out = []
    for set_name, seeds in SEEDS.items():
        for variant in VARIANT_TARGETS:
            for i in range(0, len(seeds), s23.SEED_CHUNK):
                chunk = seeds[i:i + s23.SEED_CHUNK]
                out.append({"id": f"{variant}_{set_name}_s{chunk[0]}-{chunk[-1]}", "variant": variant, "seeds": chunk, "set": set_name})
    return out


def load_new_bags(path) -> dict[str, dict[str, np.ndarray]]:
    from src.experiments.series19_21_common import read_done

    done = read_done(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for set_name in SEEDS:
        out[set_name] = {}
        for variant in VARIANT_TARGETS:
            rows = [np.array(r["per_seed"]) for r in done.values() if r["variant"] == variant and f"_{set_name}_s" in r["id"]]
            per_seed = np.concatenate(rows, axis=0)
            assert per_seed.shape[0] == len(SEEDS[set_name]), (set_name, variant, per_seed.shape)
            out[set_name][variant] = per_seed.mean(axis=0)
    return out


def old_bags() -> dict[str, dict[str, np.ndarray]]:
    b0 = s23.load_bags()
    f = s24.load_bags(s24.RESULTS_FRESH, 25)
    return {"seeds0_24": {"log1p_final": b0["log1p_final"], "raw_same_params": b0["raw_same_params"]},
            "seeds100_124": {"log1p_final": f["log1p_final"], "raw_same_params": f["raw_same_params"]}}


def change_row(name: str, a: np.ndarray, old: np.ndarray, new: np.ndarray, mask: np.ndarray) -> dict:
    diff = np.abs(new - a) - np.abs(old - a)
    mean, lo, hi = s23.masked_block_bootstrap(diff, mask.astype(float))
    return {"segment": name, "days": int(mask.sum()), "old_MAE": float(np.abs(old - a)[mask].mean()),
            "new_MAE": float(np.abs(new - a)[mask].mean()), "change": mean, "lo": lo, "hi": hi}


def main(workers: int, tag: str) -> None:
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, wd = days["actual"].to_numpy(), days["weekday"].to_numpy()
    phase = np.array([school_phase(pd.Timestamp(d), LONG_BREAKS) for d in days["date"]])
    path = results_path(tag)
    out_dir = CONFIG.path("experiments_dir") / f"series29_school_calendar_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    run_parallel(make_tasks(), s23.fit_task, path, workers, functools.partial(init_worker29, tag), "series29")
    new, old = load_new_bags(path), old_bags()

    flagged_old = with_calendar(df, SCHOOL_BREAKS_V1, LONG_BREAK_TYPES)["is_school_holiday"].sum()
    patched = with_calendar(df, break_types=STAGE_TYPES[tag])
    print(f"\nSchool-holiday days in the 950 model rows: original calendar {int(flagged_old)}, verified calendar ({tag}) {int(patched['is_school_holiday'].sum())}")
    for set_name in SEEDS:
        print(f"\n===== {set_name} =====")
        rows = []
        for variant, label in (("log1p_final", "final model (log1p)"), ("raw_same_params", "raw-£ model")):
            rows.append(change_row(label, a, old[set_name][variant], new[set_name][variant], np.ones(len(a), bool)))
        hyb_old = s24.hybrid(wd, old[set_name]["log1p_final"], old[set_name]["raw_same_params"])
        hyb_new = s24.hybrid(wd, new[set_name]["log1p_final"], new[set_name]["raw_same_params"])
        rows.append(change_row("hybrid (raw Saturdays)", a, hyb_old, hyb_new, np.ones(len(a), bool)))
        print(pd.DataFrame(rows).round(2).to_string(index=False))
        seg = []
        for d in s23.WEEKDAYS:
            seg.append(change_row(d, a, old[set_name]["log1p_final"], new[set_name]["log1p_final"], wd == d))
        for p in PHASES:
            seg.append(change_row(p, a, old[set_name]["log1p_final"], new[set_name]["log1p_final"], phase == p))
        table = pd.DataFrame(seg)
        table.to_csv(out_dir / f"final_model_by_segment_{set_name}.csv", index=False)
        print("\nFinal model by weekday and by place in a school break (verified calendar's phases):\n", table.round(1).to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--tag", default="stage2", choices=sorted(STAGE_TYPES))
    args = parser.parse_args()
    main(args.workers, args.tag)
