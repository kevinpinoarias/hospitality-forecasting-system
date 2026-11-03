"""
Series 51 - the combined calendar candidate inside the real post-model pieces.

Series 46 found that, on the plain log model, replacing `is_weekend` with `is_friday` / `is_saturday` / `is_sunday`, dropping the plain `month`
number and dropping `year` all at once scored -15.30 pounds on the 585 backtest days (interval -36.6 to +1.2) and -4.58 on the 56 days from
7 September 2026. Series 47 tested the flags alone inside the real pieces (package -1.0), Series 50 tested dropping `year` alone there
(package +1.8); the three changes together were never run inside the pieces. This series does that. Each piece of the post model, as deployed
except for the three changes:

  Monday-Thursday   the plain log model                                   (log1p target)
  Friday            + the Friday inputs (A_all)                           (log1p target)
  Saturday          + the Saturday inputs (A_lags)                        (raw-pound target)
  Sunday            + the Sunday inputs (B)                               (log1p target)

The "as deployed" side is not refitted: it is the 100-seed bags already produced with identical settings and seeds (Series 47's as-today weekend
pieces and Series 46's `final_model` for Monday-Thursday).

READING (set before any result was seen; by judgment on the net effect, as agreed with Kevin after Series 49)
  protocol   as Series 47 and 50: 585 backtest days, 10 folds, 100 seeds (200-299), the 56 days from 7 September 2026, the package put through
             the Series 24 correction (half life 8 weeks, strength 0.5, Saturday and Sunday)
  look at    (1) the package's net change in pounds after the correction, with its 95% moving-block bootstrap interval; (2) whether it is spread
             over many folds or sits in one; (3) each piece and weekday, to see where it comes from; (4) the new days. No pass/fail gate: the
             result is read as a whole. Nothing in production or the served models changes; the deployed v1 models are never refitted.

Run:  python -m src.experiments.run_series51_combined_candidate_in_pieces list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series50_drop_year_in_pieces as s50
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series45_year_and_weekend_flag import variant_features as flag_features
from src.experiments.run_series46_calendar_confirmation import CUTOFF, SEEDS
from src.experiments.series19_21_common import DATE_COL, TARGET_COL, load_protocol, read_done, run_parallel, test_days

OUT_DIR = CONFIG.path("experiments_dir") / "series51"
RESULTS = OUT_DIR / "series51_combined_candidate_in_pieces.jsonl"
FOLD_6 = 6  # 10 January to 10 March 2026
PIECES = s50.PIECES


def piece_features(piece: str) -> list[str]:
    """The piece's columns with the three changes: the single-day flags in is_weekend's place, no `month`, no `year`."""
    return [f for f in flag_features("kept", "single_days") if f not in ("month", "year")] + PIECES[piece][0]


def make_tasks() -> list[dict]:
    out = []
    for piece in PIECES:
        features = piece_features(piece)
        for part, size in (("backtest", s50.BACKTEST_CHUNK), ("new_days", s50.NEW_DAYS_CHUNK)):
            for i in range(0, len(SEEDS), size):
                chunk = SEEDS[i:i + size]
                out.append({"id": f"{part}_{piece}_candidate_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece,
                            "seeds": chunk, "features": features})
    return out


def load_candidate(part: str) -> dict[str, np.ndarray]:
    done = read_done(RESULTS)
    out = {}
    for piece in PIECES:
        per_seed = np.concatenate([np.array(r["per_seed"]) for r in done.values() if r["part"] == part and r["piece"] == piece], axis=0)
        assert per_seed.shape[0] == len(SEEDS), (part, piece, per_seed.shape)
        out[piece] = per_seed.mean(axis=0)
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
    kept, cand = s50.load_kept("backtest"), load_candidate("backtest")
    kept_new, cand_new = s50.load_kept("new_days"), load_candidate("new_days")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def interval(diff: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        return s23.masked_block_bootstrap(diff, mask.astype(float))

    def package(version: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pred = np.select([dow == 4, dow == 5, dow == 6], [version["Friday"], version["Saturday"], version["Sunday"]], default=version["Monday-Thursday"])
        return pred, s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    pk_kept, pk_cand = package(kept), package(cand)
    out = []
    for step, idx in (("before the correction", 0), ("after the Saturday and Sunday correction", 1)):
        diff = np.abs(pk_cand[idx] - a) - np.abs(pk_kept[idx] - a)
        change, lo, hi = interval(diff, np.ones(len(a), bool))
        change6, lo6, hi6 = interval(diff, fold != FOLD_6)
        out.append({"step": step, "MAE_as_deployed": float(np.abs(pk_kept[idx] - a).mean()), "MAE_candidate": float(np.abs(pk_cand[idx] - a).mean()),
                    "change": change, "lo": lo, "hi": hi, "change_without_fold6": change6, "lo_no6": lo6, "hi_no6": hi6,
                    "Mon-Thu": float(diff[dow < 4].mean()), "Fri": float(diff[dow == 4].mean()),
                    "Sat": float(diff[dow == 5].mean()), "Sun": float(diff[dow == 6].mean())})
    pkg = pd.DataFrame(out)
    pkg.to_csv(OUT_DIR / "package.csv", index=False)
    print("THE PACKAGE, all 585 days (change = candidate minus as deployed; negative = the candidate is better)")
    print(pkg.round(2).to_string(index=False))

    diff = np.abs(pk_cand[1] - a) - np.abs(pk_kept[1] - a)
    by_fold = pd.DataFrame({"fold": range(10), "from": [days.loc[fold == k, "date"].iloc[0] for k in range(10)],
                            "change_after_correction": [diff[fold == k].mean() for k in range(10)]})
    print("\nPackage after the correction, change by fold  (better in", int((by_fold["change_after_correction"] < 0).sum()), "of 10)")
    print(by_fold.round(1).to_string(index=False))

    rows = []
    for piece, (_, _, weekdays) in PIECES.items():
        own, own_new = np.isin(dow, weekdays), np.isin(dow_new, weekdays)
        d = np.abs(cand[piece] - a) - np.abs(kept[piece] - a)
        change, lo, hi = interval(d, own)
        d_new = np.abs(cand_new[piece] - a_new) - np.abs(kept_new[piece] - a_new)
        rows.append({"piece": piece, "days": int(own.sum()), "MAE_as_deployed": float(np.abs(kept[piece] - a)[own].mean()),
                     "MAE_candidate": float(np.abs(cand[piece] - a)[own].mean()), "change": change, "lo": lo, "hi": hi,
                     "folds_better": int(sum(d[own & (fold == k)].mean() < 0 for k in range(10))),
                     "new_days_n": int(own_new.sum()), "new_days_change": float(d_new[own_new].mean())})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "pieces.csv", index=False)
    print("\nEach piece on its own weekdays")
    print(table.round(2).to_string(index=False))

    weights = {p: int(np.isin(dow_new, w).sum()) for p, (_, _, w) in PIECES.items()}
    net_new = sum(table.set_index("piece").loc[p, "new_days_change"] * n for p, n in weights.items()) / sum(weights.values())
    print(f"\nNet change over the {sum(weights.values())} new days, before the correction (day-weighted over the four pieces): {net_new:+.2f}")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(PIECES)} pieces, candidate columns, {len(tasks)} tasks")
        for p in PIECES:
            print(f"  {p:16s} {len(piece_features(p))} features, target {PIECES[p][1]}")
    elif step == "run":
        run_parallel(make_tasks(), s50.fit_task, RESULTS, workers, s47.init_worker47, "series51")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
