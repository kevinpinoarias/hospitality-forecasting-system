"""
Series 74 - the 3-week forecast ratio in the Monday-to-Thursday and Friday pieces only, on a third seed set.

Series 66 and 69 (seeds 200-299 and 300-399): `fc_ratio_3wk` (the manager's forecast divided by the mean of the sales 14, 21 and 28 days earlier) was 4.49 and 4.39 pounds a day better for the post model after the
correction (intervals -11.9 to +3.1 and -11.9 to +2.5), in 7 of 10 folds, better on Monday to Thursday (-6.2 and -8.6 on those days) and Friday (-25.3 and -24.5), flat on Saturday and WORSE on Sunday (+19.9 and +26.2).
By calendar year its gain sat in 2025 (-7.7 and -8.0) and was about zero in 2026; the 56 new days were worse (+4.6 and +8.3), mostly Friday. This tests the obvious package: the ratio only where it helped.

THE CATCH, stated first: the weekdays were chosen after seeing Series 66 and 69, on the same 585 days, so even a good result here carries that bias. What this run can add is: a third seed set (seed luck), the package
assembled across all three seed sets (300 seeds), and a read of the second half of the days, 2026 and the 56 new days.

The package: the ratio column in the Monday-to-Thursday piece, the hydro Monday-to-Thursday piece and the Friday piece; Saturday, Sunday and everything else exactly as deployed. Its baseline is the deployed pieces on the same seeds.
Seeds 200-299 and 300-399 are Series 66 and 69's own fits (no refit); seeds 400-499, which no earlier series used, are fitted here for those three pieces, with and without the ratio (180 fits). Saturday and Sunday are the
same deployed bags in the package and its baseline, so they cancel; the correction (Saturday and Sunday only) is therefore identical in both.

READING (set before any new number was seen; by judgment on the net effect, as agreed with Kevin)
  expected     from Series 66 and 69's per-piece figures: about -7 pounds a day for the post model before the correction (Monday to Thursday -3.5, Friday -3.6 of the daily average), weaker in 2026, and worse on the new days
               because of Friday (+11.8 and +19.2 there)
  replicates   on the third seed set (400-499) the post and hydro models are each at least 3.5 pounds a day better after the correction over all 585 days
  supported    on the pooled 300 seeds, in BOTH models: at least 2.0 pounds better on folds 5-9, better in at least 3 of those 5 folds, no worse than 0 in 2026, and no worse than +1.0 on the 56 new days (before the correction)
  not supported   any of those fails; reported as it is, with the numbers
  limits       same days and folds throughout and weekdays chosen on them (above); the gain is about 0.8% of the daily error at most; no pass/fail gate; nothing in production changes; a new column would change the serving code too

Run:  python -m src.experiments.run_series74_ratio_weekdays list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series66_forecast_ratio as s66
from src.experiments import run_series69_forecast_ratio_fresh_seeds as s69
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import CUTOFF
from src.experiments.series19_21_common import DATE_COL, TARGET_COL, load_protocol, read_done, run_parallel, test_days

OUT_DIR = CONFIG.path("experiments_dir") / "series74"
RESULTS = OUT_DIR / "series74_ratio_weekdays.jsonl"
SEEDS_C = list(range(400, 500))
PIECES = ["Monday-Thursday", "Friday", "Monday-Thursday (hydro)"]
RATIO = "fc_ratio_3wk"
SET_NAMES = ["seeds 200-299 (Series 66)", "seeds 300-399 (Series 69)", "seeds 400-499 (new)"]


def make_tasks() -> list[dict]:
    out = []
    for piece in PIECES:
        for variant, column in (("none_c", None), ("ratio_c", RATIO)):
            features = s66.piece_features(piece, column) if column else s66.piece_features(piece, "none")
            for part, size in (("backtest", s54.BACKTEST_CHUNK), ("new_days", s54.NEW_DAYS_CHUNK)):
                for i in range(0, len(SEEDS_C), size):
                    chunk = SEEDS_C[i:i + size]
                    out.append({"id": f"{part}_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece, "variant": variant, "seeds": chunk, "features": features})
    return out


def set_bags(part: str) -> list[dict[tuple[str, str], np.ndarray]]:
    """For each of the three seed sets, (piece, 'none' | 'ratio') -> the 100-seed bag, for the three pieces the ratio goes into plus Saturday and Sunday (deployed)."""
    a = s54.load_bags(part, s66.RESULTS, s66.VARIANT_COLUMNS, s54.RESULTS)
    b = s69.load_fresh_bags(part)
    done = read_done(RESULTS)
    c = {}
    for piece in PIECES:
        for variant, key in (("none_c", "none"), ("ratio_c", "ratio")):
            rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == (part, piece, variant)]
            per_seed = np.concatenate(rows, axis=0)
            assert per_seed.shape[0] == len(SEEDS_C), (part, piece, variant, per_seed.shape)
            c[(piece, key)] = per_seed.mean(axis=0)
    out = []
    for source, ratio_name in ((a, RATIO), (b, RATIO)):
        out.append({**{(p, "none"): source[(p, "none")] for p in PIECES + ["Saturday", "Sunday"]}, **{(p, "ratio"): source[(p, ratio_name)] for p in PIECES}})
    out.append({**c, ("Saturday", "none"): a[("Saturday", "none")], ("Sunday", "none"): a[("Sunday", "none")]})  # Saturday and Sunday cancel in the comparison: the same deployed bags in both packages
    return out


def pooled(sets: list[dict]) -> dict:
    keys = [k for k in sets[0] if k[0] in PIECES]
    out = {k: np.mean([s[k] for s in sets], axis=0) for k in keys}
    out.update({("Saturday", "none"): sets[0][("Saturday", "none")], ("Sunday", "none"): sets[0][("Sunday", "none")]})
    return out


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow, fold = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy(), days["fold_id"].to_numpy()
    first, second = fold < 5, fold >= 5
    y26 = (days["date"] >= "2026-01-01").to_numpy()
    new = df[df[DATE_COL] >= CUTOFF]
    a_new, dow_new = new[TARGET_COL].to_numpy(), new[DATE_COL].dt.dayofweek.to_numpy()
    bt_sets, nd_sets = set_bags("backtest"), set_bags("new_days")
    bt_sets.append(pooled(bt_sets))
    nd_sets.append(pooled(nd_sets[:3]))
    names = SET_NAMES + ["pooled, 300 seeds"]
    pd.set_option("display.width", 250)

    def package(bags: dict, variant: str, mon_thu: str, dows: np.ndarray) -> np.ndarray:
        return np.select([dows == 4, dows == 5, dows == 6], [bags[("Friday", variant)], bags[("Saturday", "none")], bags[("Sunday", "none")]], default=bags[(mon_thu, variant)])

    rows = []
    for name, bt, nd in zip(names, bt_sets, nd_sets):
        for model, mon_thu in (("post model", "Monday-Thursday"), ("hydro model", "Monday-Thursday (hydro)")):
            base = s24.correct(days, package(bt, "none", mon_thu, dow), CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])
            cand = s24.correct(days, package(bt, "ratio", mon_thu, dow), CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])
            diff = np.abs(cand - a) - np.abs(base - a)
            change, lo, hi = s23.masked_block_bootstrap(diff, np.ones(len(a)))
            new_change = float((np.abs(package(nd, "ratio", mon_thu, dow_new) - a_new) - np.abs(package(nd, "none", mon_thu, dow_new) - a_new)).mean())
            rows.append({"seeds": name, "model": model, "all 585": change, "lo": lo, "hi": hi, "folds 0-4": float(diff[first].mean()), "folds 5-9": float(diff[second].mean()),
                         "2026": float(diff[y26].mean()), "folds better 5-9": int(sum(diff[fold == k].mean() < 0 for k in range(5, 10))),
                         "folds better": int(sum(diff[fold == k].mean() < 0 for k in range(10))), "new days": new_change})
    table = pd.DataFrame(rows)
    print("THE RATIO IN THE MONDAY-TO-THURSDAY AND FRIDAY PIECES ONLY, against the deployed pieces on the same seeds (pounds a day; negative = better)\n")
    print("after the correction on the 585 days; 'new days' = the 56 days from 7 September, before the correction\n")
    print(table.round(2).to_string(index=False))
    print("\nREADING SET BEFOREHAND")
    third = table[table["seeds"] == SET_NAMES[2]]
    replicates = bool((third["all 585"] <= -3.5).all())
    print(f"  replicates on the third seed set (each model at least 3.5 better over all 585 days): {'YES' if replicates else 'NO'}  ({third.set_index('model')['all 585'].round(2).to_dict()})")
    pool = table[table["seeds"] == names[3]].set_index("model")
    checks = {m: (pool.loc[m, "folds 5-9"] <= -2.0, pool.loc[m, "folds better 5-9"] >= 3, pool.loc[m, "2026"] <= 0.0, pool.loc[m, "new days"] <= 1.0) for m in pool.index}
    for m, (c1, c2, c3, c4) in checks.items():
        print(f"  pooled {m:11s} folds 5-9 at least 2.0 better: {c1} ({pool.loc[m, 'folds 5-9']:+.2f}); 3 of 5 folds better: {c2} ({int(pool.loc[m, 'folds better 5-9'])}); 2026 no worse than 0: {c3} ({pool.loc[m, '2026']:+.2f}); "
              f"new days no worse than +1.0: {c4} ({pool.loc[m, 'new days']:+.2f})")
    print("  SUPPORTED" if all(all(v) for v in checks.values()) else "  NOT SUPPORTED", "(the pooled 300 seeds, both models, all four conditions)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "packages.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(PIECES)} pieces x 2 variants on seeds {SEEDS_C[0]}-{SEEDS_C[-1]}: {len(tasks)} tasks")
        for piece in PIECES:
            print(f"  {piece:26s} deployed {len(s66.piece_features(piece, 'none'))} columns, with the ratio {len(s66.piece_features(piece, RATIO))}")
        assert not set(SEEDS_C) & (set(range(200, 400))), "the third seed set overlaps Series 66 or 69"
        d, _ = load_protocol()
        print("ratio column missing values:", int(s66.add_ratio_columns(d, s66.load_daily_sales())[RATIO].isna().sum()), "of", len(d))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, s66.init_worker66, "series74")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
