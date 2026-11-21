"""
Series 52 - does putting the three single-day flags in the base change the post model's weekend choices?

The post model's Friday, Saturday and Sunday pieces were chosen in Series 26-27 with `is_weekend` in the base: each weekday picked its own extra
inputs (Friday the same-weekday memory A_all, Saturday the sales of 2 and 3 weeks ago A_lags, Sunday the weekend interactions B) from six options,
on the first five test blocks. Series 47 put the flags into those pieces as chosen and found nothing, but a different option, or a different
target, could win once the flags are in the base. This series repeats the selection from the start with the flags there, and compares the whole
assembled post model with the one built on `is_weekend`.

Both bases are refitted on today's data with identical seeds, so the comparison is paired and not confounded by data revisions since the stored
Series 23-27 results (`is_weekend` base = what the post model has today):

  bases      is_weekend (the final model's column)  |  single_day_flags (is_friday, is_saturday, is_sunday in its place and order)
  options    none, A_all, A_summary, A_lags, B, AB  (the Series 26 variants), each on the log1p and the raw-pound target
  seeds      0-24 (selection and first look) and 100-124 (fresh), 10 folds, 585 test days

PRE-REGISTERED (written before any result was seen)
  procedure  exactly Series 27's: each of Friday, Saturday, Sunday takes the option with the lowest MAE on its own days in blocks 0-4 (seeds 0-24)
             on its fixed target (log1p for Friday and Sunday, raw pounds for Saturday), ties to "none"; the weekend is assembled on the Series 23
             hybrid (raw model on Saturdays) and put through the Series 24 correction (half life 8 weeks, strength 0.5, Saturday and Sunday),
             as `run_package_validation.package_steps` does
  expectation the options chosen stay the same and the package moves by a few pounds at most (Series 47 and 48 point that way)
  reading    by judgment on the net effect, as agreed with Kevin: the package's change after the correction, flags minus `is_weekend`, with a 95%
             moving-block bootstrap interval, on both seed sets; which options each base chooses; the same by weekday and by fold. Also reported,
             not used to choose: each weekday allowed both targets (12 options), chosen on blocks 0-4 and judged on blocks 5-9 with the fresh seeds
  closes     the question of whether the single-day flags belong anywhere in the post model: whichever way this comes out, it is the last test of
             them. Nothing in production changes; the deployed forecast-log v1 pieces are never refitted, so a gain would be a new version.

Run:  python -m src.experiments.run_series52_weekend_selection_with_flags list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_package_validation as pv
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series25_weekend_training as s25
from src.experiments import run_series26_weekend_features as s26
from src.experiments import run_series27_weekend_per_weekday as s27
from src.experiments.run_series45_year_and_weekend_flag import variant_features as flag_features
from src.experiments.run_series47_weekend_flags_in_pieces import init_worker47
from src.experiments.series19_21_common import _STATE, FINAL_PARAMS, bagged_log1p_predictions, load_protocol, read_done, run_parallel, test_days

CHUNK = 5
SEED_SETS = {"seeds0_24": list(range(25)), "seeds100_124": list(range(100, 125))}
BASES = ("is_weekend", "single_day_flags")
OPTIONS = ["none"] + list(s26.VARIANTS)
TARGETS = ("log1p", "raw")
OUT_DIR = CONFIG.path("experiments_dir") / "series52"
RESULTS = OUT_DIR / "series52_weekend_selection_with_flags.jsonl"


def option_features(base: str, option: str) -> list[str]:
    columns = flag_features("kept", "as_today" if base == "is_weekend" else "single_days")
    return columns + ([] if option == "none" else s26.VARIANTS[option]["extra"])


def fit_task(task: dict) -> dict:
    df, folds = _STATE["df"], _STATE["folds"]
    started = time.time()
    if task["target"] == "log1p":
        per_seed = bagged_log1p_predictions(df, folds, task["features"], FINAL_PARAMS, task["seeds"])
    else:
        per_seed = s23.raw_predictions(df, folds, task["features"], FINAL_PARAMS, task["seeds"])
    return {"id": task["id"], "base": task["base"], "option": task["option"], "target": task["target"], "set": task["set"],
            "seeds": task["seeds"], "per_seed": [[round(float(x), 2) for x in row] for row in per_seed],
            "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for set_name, seeds in SEED_SETS.items():
        for base in BASES:
            for option in OPTIONS:
                for target in TARGETS:
                    for i in range(0, len(seeds), CHUNK):
                        chunk = seeds[i:i + CHUNK]
                        out.append({"id": f"{base}_{option}_{target}_{set_name}_s{chunk[0]}-{chunk[-1]}", "base": base, "option": option,
                                    "target": target, "set": set_name, "seeds": chunk, "features": option_features(base, option)})
    return out


def load_bags() -> dict[tuple[str, str], dict[tuple[str, str], np.ndarray]]:
    """(base, seed set) -> (option, target) -> the 25-seed bag in pounds, shape (585,)."""
    done = read_done(RESULTS)
    out: dict = {}
    for base in BASES:
        for set_name, seeds in SEED_SETS.items():
            bags = {}
            for option in OPTIONS:
                for target in TARGETS:
                    rows = [np.array(r["per_seed"]) for r in done.values()
                            if (r["base"], r["option"], r["target"], r["set"]) == (base, option, target, set_name)]
                    per_seed = np.concatenate(rows, axis=0)
                    assert per_seed.shape[0] == len(seeds), (base, option, target, set_name, per_seed.shape)
                    bags[(option, target)] = per_seed.mean(axis=0)
            out[(base, set_name)] = bags
    return out


def pv_inputs(bags: dict[tuple[str, str], np.ndarray]) -> tuple[dict, dict]:
    """The two dicts run_package_validation expects: the plain log and raw models, and every option keyed (variant, target)."""
    log_raw = {"log1p_final": bags[("none", "log1p")], "raw_same_params": bags[("none", "raw")]}
    features = {(v, t): bags[(v, t)] for v in s26.VARIANTS for t in TARGETS}
    return log_raw, features


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, wd, fold = days["actual"].to_numpy(), days["weekday"].to_numpy(), days["fold_id"].to_numpy()
    sel, ev = fold < s27.SELECTION_BLOCKS, fold >= s27.SELECTION_BLOCKS
    bags = load_bags()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chosen, steps = {}, {}
    for base in BASES:
        log_raw0, features0 = pv_inputs(bags[(base, "seeds0_24")])
        chosen[base] = pv.choose_on_first_blocks(days, log_raw0, features0)
        sel_table = pd.DataFrame({d: {o: s24.mae(a, s27.option_predictions(d, features0, log_raw0)[o], sel & (wd == d))
                                      for o in OPTIONS} for d in s25.FRI_SUN})
        print(f"\n=== base: {base} ===\nMAE by option on the selection blocks 0-4 (seeds 0-24, each weekday on its own target):")
        print(sel_table.round(1).to_string())
        print("Chosen:", chosen[base])
        for set_name in SEED_SETS:
            log_raw, features = pv_inputs(bags[(base, set_name)])
            steps[(base, set_name)] = pv.package_steps(days, log_raw, features, chosen[base])

    print("\n\nTHE POST MODEL BUILT ON EACH BASE (MAE in pounds; the choices come from seeds 0-24 blocks 0-4 and are applied unchanged to both seed sets)")
    for set_name in SEED_SETS:
        for base in BASES:
            table = pv.scores(days, steps[(base, set_name)])
            table.insert(0, "base", base)
            print(f"\n{set_name}")
            print(table.round(2).to_string(index=False))

    print("\n\nTHE CHANGE FROM PUTTING THE FLAGS IN THE BASE (flags minus is_weekend; negative = the flags help)")
    rows = []
    for set_name in SEED_SETS:
        for step in pv.STEPS:
            old, new = steps[("is_weekend", set_name)][step], steps[("single_day_flags", set_name)][step]
            diff = np.abs(new - a) - np.abs(old - a)
            change, lo, hi = s23.masked_block_bootstrap(diff, np.ones(len(a)))
            rows.append({"seeds": set_name, "step": step, "change": change, "lo": lo, "hi": hi,
                         "Mon-Thu": float(diff[np.isin(wd, ["Monday", "Tuesday", "Wednesday", "Thursday"])].mean()),
                         "Fri": float(diff[wd == "Friday"].mean()), "Sat": float(diff[wd == "Saturday"].mean()),
                         "Sun": float(diff[wd == "Sunday"].mean()), "folds_better": int(sum(diff[fold == k].mean() < 0 for k in range(10)))})
    changes = pd.DataFrame(rows)
    changes.to_csv(OUT_DIR / "flags_minus_is_weekend.csv", index=False)
    print(changes.round(2).to_string(index=False))

    print("\n\nEXPLORATORY, not used to choose: each weekday allowed both targets (12 options), chosen on blocks 0-4 (seeds 0-24), judged on blocks 5-9")
    for base in BASES:
        b0, bf = bags[(base, "seeds0_24")], bags[(base, "seeds100_124")]
        picks = {}
        for day in s25.FRI_SUN:
            mask = sel & (wd == day)
            scores = {f"{o}/{t}": s24.mae(a, b0[(o, t)], mask) for o in OPTIONS for t in TARGETS}
            picks[day] = min(scores, key=scores.get)
        judged = {day: s24.mae(a, bf[tuple(picks[day].split("/"))], ev & (wd == day)) for day in s25.FRI_SUN}
        fixed = {day: s24.mae(a, bf[(chosen[base][day], s27.DAY_TARGET[day])], ev & (wd == day)) for day in s25.FRI_SUN}
        print(f"{base:18s} picks {picks}")
        print(f"{'':18s} blocks 5-9, fresh seeds, MAE: both-targets choice {({k: round(v, 1) for k, v in judged.items()})}  fixed-target choice {({k: round(v, 1) for k, v in fixed.items()})}")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(BASES)} bases x {len(OPTIONS)} options x {len(TARGETS)} targets x {len(SEED_SETS)} seed sets, {len(tasks)} tasks")
        for base in BASES:
            print(f"  {base:18s} none: {len(option_features(base, 'none'))} columns; A_all: {len(option_features(base, 'A_all'))}; B: {len(option_features(base, 'B'))}")
    elif step == "run":
        run_parallel(make_tasks(), fit_task, RESULTS, workers, init_worker47, "series52")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
