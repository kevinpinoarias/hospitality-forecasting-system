"""
Series 69 - the 3-week forecast ratio again, on fresh seeds.

Series 66 added three comparisons of the manager's forecast with the recent level of sales, one at a time. The best, `fc_ratio_3wk` (the forecast divided by the mean
of the sales 14, 21 and 28 days earlier), was better by 4.49 pounds a day for the post model and 4.96 for the hydro model after the correction (intervals −11.9 to +3.1
and −12.5 to +2.5, better in 7 and 6 folds of 10) but 4.6 and 3.7 pounds worse on the 56 new days. It was the best of three variants, so part of that gain may be seed luck.

What this series can and cannot tell. It re-fits the deployed pieces and the ratio variant on seeds 300-399, which no earlier series used, and compares them with each
other, as Series 46 did for the calendar columns. A gain that is seed luck will shrink or vanish. It cannot fix a gain that is luck of the DAYS: the 585 backtest days and
the 56 new days are the same, so the day-level noise (the bootstrap interval) stays what it was. A confirmation here means "not seed luck", not "adopt".

Both the deployed pieces (variant `none`) and the ratio variant are fitted on the fresh seeds, for all six pieces of the weekly log; the Series 46-47 and 54 baselines use seeds 200-299
and are not reused. Series 54's design otherwise (10 folds, 585 days, the 56 new days, the two models assembled and corrected, the moving-block bootstrap).

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin)
  replicates   the change after the correction is at least 2.3 pounds better (half of Series 66's gain) in BOTH models, and better in at least 6 of 10 folds in both
  seed luck    the change is under 1 pound in either model, or the sign flips
  in between   reported as it is; no verdict
  adopt?       not decided here: it would also need an interval below zero and the new days not worse, and the new days did not support it in Series 66
  limits       no pass/fail gate; nothing in production changes; a new column would change the serving code too

Run:  python -m src.experiments.run_series69_forecast_ratio_fresh_seeds list | run [--workers 10] | report
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
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import CUTOFF
from src.experiments.series19_21_common import DATE_COL, TARGET_COL, load_protocol, read_done, run_parallel, test_days

OUT_DIR = CONFIG.path("experiments_dir") / "series69"
RESULTS = OUT_DIR / "series69_forecast_ratio_fresh_seeds.jsonl"
FRESH_SEEDS = list(range(300, 400))
VARIANTS = ["none", "fc_ratio_3wk"]
SERIES_66 = {"post model": -4.49, "hydro model": -4.96}  # after the correction, Series 66, seeds 200-299


def make_tasks() -> list[dict]:
    out = []
    for piece in s54.PIECES:
        for variant in VARIANTS:
            features = s66.piece_features(piece, variant)
            for part, size in (("backtest", s54.BACKTEST_CHUNK), ("new_days", s54.NEW_DAYS_CHUNK)):
                for i in range(0, len(FRESH_SEEDS), size):
                    chunk = FRESH_SEEDS[i:i + size]
                    out.append({"id": f"{part}_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece, "variant": variant,
                                "seeds": chunk, "features": features})
    return out


def load_fresh_bags(part: str) -> dict[tuple[str, str], np.ndarray]:
    done = read_done(RESULTS)
    out = {}
    for piece in s54.PIECES:
        for variant in VARIANTS:
            rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == (part, piece, variant)]
            per_seed = np.concatenate(rows, axis=0)
            assert per_seed.shape[0] == len(FRESH_SEEDS), (part, piece, variant, per_seed.shape)
            out[(piece, variant)] = per_seed.mean(axis=0)
    return out


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    a, fold = days["actual"].to_numpy(), days["fold_id"].to_numpy()
    dow = pd.to_datetime(days["date"]).dt.dayofweek.to_numpy()
    new = df[df[DATE_COL] >= CUTOFF]
    a_new, dow_new = new[TARGET_COL].to_numpy(), new[DATE_COL].dt.dayofweek.to_numpy()
    bt, nd = load_fresh_bags("backtest"), load_fresh_bags("new_days")
    pd.set_option("display.width", 250)

    def package(variant: str, mon_thu: str) -> tuple[np.ndarray, np.ndarray]:
        pred = np.select([dow == 4, dow == 5, dow == 6], [bt[("Friday", variant)], bt[("Saturday", variant)], bt[("Sunday", variant)]], default=bt[(mon_thu, variant)])
        return pred, s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    print("FRESH SEEDS 300-399: fc_ratio_3wk against the deployed pieces fitted on the same seeds (negative = better; 585 days)\n")
    rows = []
    for piece, (_, _, weekdays, _) in s54.PIECES.items():
        own = np.isin(dow, weekdays)
        d = np.abs(bt[(piece, "fc_ratio_3wk")] - a) - np.abs(bt[(piece, "none")] - a)
        change, lo, hi = s23.masked_block_bootstrap(d, own.astype(float))
        own_new = np.isin(dow_new, weekdays)
        d_new = np.abs(nd[(piece, "fc_ratio_3wk")] - a_new) - np.abs(nd[(piece, "none")] - a_new)
        rows.append({"piece": piece, "change": change, "lo": lo, "hi": hi, "folds_better": int(sum(d[own & (fold == k)].mean() < 0 for k in range(10))),
                     "new_days_change": float(d_new[own_new].mean())})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nTHE WHOLE MODELS (all 585 days)")
    out = []
    for name, mon_thu in (("post model", "Monday-Thursday"), ("hydro model", "Monday-Thursday (hydro)")):
        base, cand = package("none", mon_thu), package("fc_ratio_3wk", mon_thu)
        for step, idx in (("before the correction", 0), ("after the correction", 1)):
            diff = np.abs(cand[idx] - a) - np.abs(base[idx] - a)
            change, lo, hi = s23.masked_block_bootstrap(diff, np.ones(len(a)))
            out.append({"model": name, "step": step, "MAE_deployed": float(np.abs(base[idx] - a).mean()), "MAE_ratio": float(np.abs(cand[idx] - a).mean()),
                        "change": change, "lo": lo, "hi": hi, "Mon-Thu": float(diff[dow < 4].mean()), "Fri": float(diff[dow == 4].mean()),
                        "Sat": float(diff[dow == 5].mean()), "Sun": float(diff[dow == 6].mean()),
                        "folds_better": int(sum(diff[fold == k].mean() < 0 for k in range(10)))})
    table = pd.DataFrame(out)
    print(table.round(2).to_string(index=False))

    print("\nAGAINST SERIES 66 (after the correction; seeds 200-299 then, 300-399 now)")
    after = table[table["step"] == "after the correction"].set_index("model")
    for name in SERIES_66:
        now = float(after.loc[name, "change"])
        folds = int(after.loc[name, "folds_better"])
        print(f"  {name:12s} Series 66 {SERIES_66[name]:+.2f}   fresh seeds {now:+.2f}   folds better {folds} of 10")
    verdict = all(float(after.loc[n, "change"]) <= -2.3 and int(after.loc[n, "folds_better"]) >= 6 for n in SERIES_66)
    luck = any(float(after.loc[n, "change"]) > -1.0 for n in SERIES_66)
    print("  reading set beforehand:", "REPLICATES" if verdict else ("SEED LUCK" if luck else "in between"))

    print("\nNEW DAYS (56 days), before the correction, day-weighted over the pieces")
    for name, mon_thu in (("post model", "Monday-Thursday"), ("hydro model", "Monday-Thursday (hydro)")):
        def assembled(variant: str) -> np.ndarray:
            return np.select([dow_new == 4, dow_new == 5, dow_new == 6], [nd[("Friday", variant)], nd[("Saturday", variant)], nd[("Sunday", variant)]], default=nd[(mon_thu, variant)])
        print(f"  {name:12s} {float((np.abs(assembled('fc_ratio_3wk') - a_new) - np.abs(assembled('none') - a_new)).mean()):+.2f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "packages.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(s54.PIECES)} pieces x {len(VARIANTS)} variants (baselines refitted on the fresh seeds), {len(tasks)} tasks, seeds {FRESH_SEEDS[0]}-{FRESH_SEEDS[-1]}")
        assert not set(FRESH_SEEDS) & set(range(200, 300)), "the fresh seeds overlap the Series 54-66 seeds"
        for piece in s54.PIECES:
            print(f"  {piece:26s} none {len(s66.piece_features(piece, 'none'))} columns, fc_ratio_3wk {len(s66.piece_features(piece, 'fc_ratio_3wk'))}")
            assert len(s66.piece_features(piece, "fc_ratio_3wk")) == len(s66.piece_features(piece, "none")) + 1
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, s66.init_worker66, "series69")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
