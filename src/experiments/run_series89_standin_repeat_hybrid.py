"""
Series 89 - the better stand-ins on the rest of the seeds, and a hybrid.

Series 88 (25 seeds, 200-224) served the unmasked package with six stand-ins for the manager's number: the closed-day-aware median (S1: 902 pounds a day average miss against 945 for the live log's current stand-in S0, the real
number's floor being 854) and the same date a year earlier inside the holiday window (S3: 903, far better on 26-31 December, worse in early January because the date, not the weekday, is lined up) were the best, and the specialist
windows (the masked model on 1 to 8 January, the companion on 26 to 31 December, the unmasked model elsewhere) kept their gain under every stand-in. The result rests on one winter (24 holiday-window days). Kevin (9 November
2026): repeat it on the other seeds and try a hybrid.

Stand-ins, each served on the fold's test days exactly as in Series 87/88 (models fitted on the real training rows, only the test rows' `forecast_sales` differs, history from the training rows only):
  S0  the live log's current stand-in (reproduces Series 87/88)
  S1  the closed-day-aware median (closed days left out of the history, 0 on 25 December and 1 January)
  S3  the manager's number 365 days earlier (the same date a year ago) inside the holiday window 24 December to 15 January, S0 elsewhere
  S6  the HYBRID: the same date a year ago (365 days) for the Christmas week 24 December to 2 January, and S1 everywhere else (so from 3 January it is the closed-day-aware median); a missing year-ago number falls back to S1
  real  the manager's real number, the floor
Three model variants as Series 87/88 (unmasked, the 1 to 8 January mask, the companion without the ten history inputs). Seeds: all 100 of the old set (200-299) and the fresh seeds 300-399; 25-seed figures from Series 88
are not reused because S6 is new. 3 variants x 4 pieces x 40 tasks of 5 seeds = 480 tasks; each task predicts every stand-in. A sanity step checks S0, S1, S3 and the real number against Series 88's stored forecasts.

READING (set before any result was seen)
  question 1   the unmasked package's average daily miss over the 585 days with each stand-in, on (a) the 100 seeds and (b) the fresh seeds; also over the 24 holiday-window days, 26-31 December and 1-8 January
  replicates   S1 is better than S0 by at least 5 pounds a day overall on BOTH seed sets, and S3 likewise (Series 88: 43 and 42)
  tied         S1 and S3 are TIED if their overall misses are within 5 pounds a day on both seed sets
  hybrid       S6 BEATS BOTH if its overall miss is lower than S1's and S3's by at least 5 pounds a day on BOTH seed sets; MATCHES the better one if within 5; otherwise WORSE
  question 2   the specialist windows with S1, S3 and S6, each against the unmasked package served with the SAME stand-in: STILL HELP if the net change is at most -2.0 on both seed sets
  also         the live-like picture: today (S0, no tricks) against each stand-in with and without the specialist windows, and the real-number floor
  limits       one December and one January in the backtest (24 holiday-window days); the stand-in is built from the fold's training rows (staler than the live log's); the weather is the real weather; the same 585 days,
               so this checks the seeds and the hybrid, not the days; a different stand-in changes an input of the pre-registered log, so it needs an amendment first, but no refit

Run:  python -m src.experiments.run_series89_standin_repeat_hybrid list | sanity | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series82_mask_windows as s82
from src.experiments import run_series88_better_standin as s88
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features.serving_features import seasonal_medians
from src.models.final_model import FINAL_PARAMS, TARGET_COL

OUT_DIR = CONFIG.path("experiments_dir") / "series89"
RESULTS = OUT_DIR / "series89_standin_repeat_hybrid.jsonl"
MANAGER_COL = s88.MANAGER_COL
PIECES, VARIANTS = s88.PIECES, s88.VARIANTS
FRESH = list(range(300, 400))
STANDINS = ["S0", "S1", "S3", "S6", "real"]
CHRISTMAS_WEEK = ((12, 24), (1, 2))


def standins(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    base = s88.standins(train, test)  # S0, S1, S2, S3, S4, S5, real, exactly as Series 88
    dates = test[DATE_COL]
    week = s82.in_window(dates, *CHRISTMAS_WEEK)
    ya365 = s88.year_ago(dates, 365)
    s6 = np.where(week & ~np.isnan(ya365), ya365, base["S1"])
    return {"S0": base["S0"], "S1": base["S1"], "S3": base["S3"], "S6": s6, "real": base["real"]}


def fit_task89(task: dict) -> dict:
    from catboost import CatBoostRegressor

    df, folds = _STATE["df"], _STATE["folds"]
    started, features, target = time.time(), task["features"], s54.PIECES[task["piece"]][1]
    out = {name: [] for name in STANDINS}
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)].copy()
        served = standins(train, test)
        y = np.log1p(train[TARGET_COL]) if target == "log1p" else train[TARGET_COL]
        per = {name: [] for name in STANDINS}
        for seed in task["seeds"]:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
            model.fit(train[features], y)
            for name in STANDINS:
                frame = test.copy()
                frame[MANAGER_COL] = served[name]
                p = model.predict(frame[features])
                per[name].append(np.expm1(p) if target == "log1p" else p)
        for name in STANDINS:
            out[name].append(np.array(per[name]))
    per_standin = {name: [[round(float(x), 2) for x in row] for row in np.concatenate(out[name], axis=1)] for name in STANDINS}
    return {"id": task["id"], "part": "backtest", "piece": task["piece"], "variant": task["variant"], "seeds": task["seeds"], "per_seed": per_standin["S0"], "per_standin": per_standin,
            "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for variant, features_of in VARIANTS.items():
        for piece in PIECES:
            features = features_of(piece)
            for name, seeds in ((variant, SEEDS), (f"{variant}_fresh", FRESH)):
                for i in range(0, len(seeds), s54.BACKTEST_CHUNK):
                    chunk = seeds[i:i + s54.BACKTEST_CHUNK]
                    out.append({"id": f"backtest_{piece}_{name}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": name, "seeds": chunk, "features": features})
    return out


def mine(done: dict, piece: str, variant: str, standin: str) -> np.ndarray:
    rows = [np.array(r["per_standin"][standin]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == 100, (piece, variant, standin, per_seed.shape)
    return per_seed.mean(axis=0)


def sanity() -> None:
    """S0, S1, S3 and the real number of one task must reproduce Series 88's stored forecasts (same piece, variant and seeds)."""
    from src.features import payday_intensity as pi
    from src.forecast_log.v1 import add_hydro_thu

    piece = "Monday-Thursday (hydro)"
    s47.init_worker47()
    _STATE["df"] = pi.add_payday_intensity_fixed(add_hydro_thu(_STATE["df"]), DATE_COL)
    got = fit_task89({"id": "sanity", "piece": piece, "variant": "sd_base", "seeds": SEEDS[:5], "features": VARIANTS["sd_base"](piece)})["per_standin"]
    stored = [r["per_standin"] for r in read_done(s88.RESULTS).values() if (r["piece"], r["variant"]) == (piece, "sd_base") and r["seeds"] == SEEDS[:5]][0]
    gaps = {k: float(np.max(np.abs(np.array(got[k]) - np.array(stored[k])))) for k in ("S0", "S1", "S3", "real")}
    print("sanity: largest difference from Series 88's stored forecasts:", {k: round(v, 4) for k, v in gaps.items()})
    assert max(gaps.values()) < 0.01, "the loop does not reproduce Series 88"


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    six = s82.in_window(days["date"], (12, 26), (12, 31))
    jan8 = s82.in_window(days["date"], (1, 1), (1, 8))
    hol = s82.in_window(days["date"], *s88.HOLIDAY)
    done = read_done(RESULTS)
    pd.set_option("display.width", 250)

    def package(suffix: str, standin: str, kind: str = "base") -> np.ndarray:
        def piece_preds(variant: str) -> dict:
            return {pc: mine(done, pc, variant + suffix, standin) for pc in PIECES}

        main = piece_preds("sd_base")
        if kind == "spec":
            msk, comp = piece_preds("sd_j1_8"), piece_preds("sd_companion")
            p = {pc: np.where(six, comp[pc], np.where(jan8, msk[pc], main[pc])) for pc in PIECES}
        else:
            p = main if kind == "base" else piece_preds(kind)
        pred = np.select([dow == 4, dow == 5, dow == 6], [p["Friday"], p["Saturday"], p["Sunday"]], default=p["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    tables, verdicts = {}, {}
    for label, suffix in (("all 100 seeds", ""), ("fresh seeds 300-399", "_fresh")):
        rows = {}
        for s in STANDINS:
            err = np.abs(package(suffix, s) - a)
            rows[s] = {"all 585 days": float(err.mean()), f"{int(hol.sum())} holiday-window days": float(err[hol].mean()), "26-31 Dec": float(err[six].mean()), "1-8 Jan": float(err[jan8].mean())}
        tables[label] = pd.DataFrame(rows).T
        print(f"QUESTION 1, {label}: THE UNMASKED PACKAGE SERVED WITH EACH STAND-IN (average miss, pounds; lower is better)\n")
        print(tables[label].round(0).to_string())
        print()
    allm = {k: t["all 585 days"] for k, t in tables.items()}
    for s in ("S1", "S3"):
        ok = all(allm[k]["S0"] - allm[k][s] >= 5.0 for k in allm)
        print(f"READING SET BEFOREHAND, {s}: {'REPLICATES (at least 5 better than S0 on both seed sets)' if ok else 'DOES NOT REPLICATE'}: " + ", ".join(f"{k}: {allm[k][s] - allm[k]['S0']:+.1f}" for k in allm))
    tied = all(abs(allm[k]["S1"] - allm[k]["S3"]) <= 5.0 for k in allm)
    print(f"S1 and S3: {'TIED (within 5 on both seed sets)' if tied else 'NOT TIED'}: " + ", ".join(f"{k}: S3 - S1 = {allm[k]['S3'] - allm[k]['S1']:+.1f}" for k in allm))
    beats = all(allm[k]["S6"] <= min(allm[k]["S1"], allm[k]["S3"]) - 5.0 for k in allm)
    matches = all(allm[k]["S6"] <= min(allm[k]["S1"], allm[k]["S3"]) + 5.0 for k in allm)
    print(f"READING SET BEFOREHAND, hybrid S6: {'BEATS BOTH' if beats else 'MATCHES the better one' if matches else 'WORSE'}: " + ", ".join(f"{k}: {allm[k]['S6'] - min(allm[k]['S1'], allm[k]['S3']):+.1f} against the better of S1 and S3" for k in allm))
    # question 2: do the tricks still help
    rows2 = {}
    for label, suffix in (("all 100 seeds", ""), ("fresh seeds 300-399", "_fresh")):
        for s in ("S0", "S1", "S3", "S6"):
            base = package(suffix, s)
            for name, kind in (("whole-model mask", "sd_j1_8"), ("specialist windows", "spec")):
                d = np.abs(package(suffix, s, kind) - a) - np.abs(base - a)
                rows2[(label, s, name)] = {"net, 585 days": float(d.mean()), "26-31 Dec": float(d[six].mean()), "1-8 Jan": float(d[jan8].mean()), "all other days": float(d[~(six | jan8)].mean())}
    t2 = pd.DataFrame(rows2).T
    print("\nQUESTION 2: THE TRICKS AGAINST THE UNMASKED PACKAGE SERVED WITH THE SAME STAND-IN (pounds a day; negative = better)\n")
    print(t2.round(2).to_string())
    for s in ("S0", "S1", "S3", "S6"):
        nets = [float(t2.loc[(k, s, "specialist windows"), "net, 585 days"]) for k in allm]
        print(f"READING SET BEFOREHAND, specialist windows with {s}: {nets[0]:+.2f} and {nets[1]:+.2f}: {'STILL HELP' if all(n <= -2.0 for n in nets) else 'DO NOT STILL HELP'}")
    # the live-like picture
    print("\nTHE LIVE-LIKE PICTURE (average miss over the 585 days; change against today = S0, no tricks)\n")
    pic = {}
    for label, suffix in (("all 100 seeds", ""), ("fresh seeds 300-399", "_fresh")):
        today = float(np.abs(package(suffix, "S0") - a).mean())
        for name, s, kind in (("today (S0, no tricks)", "S0", "base"), ("specialist windows only (S0)", "S0", "spec"), ("S1 only", "S1", "base"), ("S1 + specialist windows", "S1", "spec"),
                              ("S3 only", "S3", "base"), ("S3 + specialist windows", "S3", "spec"), ("S6 hybrid only", "S6", "base"), ("S6 hybrid + specialist windows", "S6", "spec"), ("the floor (real number)", "real", "base")):
            m = float(np.abs(package(suffix, s, kind) - a).mean())
            pic.setdefault(name, {})[label + ": miss"] = m
            pic[name][label + ": vs today"] = m - today
    print(pd.DataFrame(pic).T.round(1).to_string())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(OUT_DIR / f"standins_{'fresh' if 'fresh' in k else 'seeds100'}.csv")
    t2.to_csv(OUT_DIR / "tricks_by_standin.csv")
    pd.DataFrame(pic).T.to_csv(OUT_DIR / "live_like_picture.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(VARIANTS)} variants x {len(PIECES)} pieces x 40 tasks of {s54.BACKTEST_CHUNK} seeds (20 on seeds 200-299, 20 on the fresh seeds) = {len(tasks)} tasks (backtest only; each predicts {len(STANDINS)} stand-ins)")
        df, folds = load_protocol()
        fold = next(f for f in folds if f.test_start <= pd.Timestamp("2025-12-30") < f.test_end)
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[df[DATE_COL].between("2025-12-24", "2026-01-08") & (df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        s = standins(train, test)
        print(pd.DataFrame({"date": test[DATE_COL].dt.strftime("%a %d %b %Y").to_numpy(), **{k: np.round(v, 0) for k, v in s.items()}}).to_string(index=False))
    elif step == "sanity":
        sanity()
    elif step == "run":
        run_parallel(make_tasks(), fit_task89, RESULTS, workers, s82.init_worker82, "series89")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "sanity", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
