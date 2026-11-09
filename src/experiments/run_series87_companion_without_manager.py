"""
Series 87 - the companion model and the January mask when the manager's number is not available.

Every backtest so far gave the models the manager's REAL forecast for the day (`forecast_sales`), as an input. The live forecast log cannot: the manager's number reaches us only after the day (Series 42-43), so each
log forecast is served with a STAND-IN, the weekday-and-season median of the manager's past forecasts (`seasonal_medians`, the same weekday within 10 days of the same day of the year). In late December the manager's real number
was almost exact (29 to 31 December 2025: 12,500, 12,500 and 9,700 against 13,035, 12,369 and 9,849 sold), and the companion (Series 85) leans on it, so some of its gain may be the manager's knowledge, not the model's.
Kevin's question (9 November 2026): how does the companion behave without that lucky, pin-point forecast?

The test serves every piece with the stand-in, the way the log does ("N0" of Series 42: trained with the manager's real number, served with the stand-in): each fold's models are fitted on the real training rows, and on the fold's
test days `forecast_sales` is replaced by the stand-in computed from the training rows only (the manager's numbers before the fold's test start, so nothing from the future). Everything else is untouched (the weather is still the
actual weather, which is optimistic in the same way for every variant, and the history inputs are the real history).
Three variants, 25 seeds (200-224), 10 folds, 585 days, the deployed weekend correction; 3 x 4 pieces x 5 tasks of 5 seeds = 60 tasks:
  sd_base        the unmasked package
  sd_j1_8        the 1 to 8 January mask (the ten history inputs missing on those rows, training and forecast days alike)
  sd_companion   each piece refitted without the ten history inputs (Series 85's companion)
and the combination sd_comp_dec = sd_j1_8 with the companion's forecast on 26 to 31 December. A sanity step (`sanity`) re-runs one task with the real manager's number and checks that it reproduces the stored backtest.

READING (set before any result was seen)
  measure      the NET change of the post package over the 585 days (pounds a day; negative = better), the six days 26-31 December and the other days; all served with the stand-in
  mask holds   sd_j1_8 against sd_base is at least 4.5 better (half of Series 82's -9.07 with the real number)
  companion    sd_comp_dec against sd_j1_8: SURVIVES if it is at least 1.1 better (half of Series 85's 2.22), WEAKENS if better but by less, FAILS if not better; the six days are shown (Series 85: -301 against -64)
  also         how far the stand-in was from the real number on 26-31 December 2025, and what serving the stand-in costs the unmasked package (Series 43: about +99 overall)
  limits       one December (six days); 25 seeds; the stand-in is computed from the fold's training rows (slightly staler than the live log's, which uses the history up to the Sunday before the week); the weather is
               still the real weather; the same 585 days; a refit would be a new forecast-log version and Kevin's decision

Run:  python -m src.experiments.run_series87_companion_without_manager list | sanity | run [--workers 10] | report
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series81_closed_days_in_lags as s81
from src.experiments import run_series82_mask_windows as s82
from src.experiments import run_series85_companion_model as s85
from src.experiments.run_package_validation import CORRECTION
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features.serving_features import seasonal_medians
from src.models.final_model import FINAL_PARAMS, TARGET_COL

OUT_DIR = CONFIG.path("experiments_dir") / "series87"
RESULTS = OUT_DIR / "series87_companion_without_manager.jsonl"
MANAGER_COL = "forecast_sales"
SEEDS25, PIECES = s81.SEEDS25, s81.PIECES
VARIANTS = {"sd_base": lambda piece: s81.piece_features(piece, "base"), "sd_j1_8": lambda piece: s82.piece_features(piece, "j1_8"), "sd_companion": s85.piece_features}


def fit_task87(task: dict) -> dict:
    """Like `s54.fit_task` for the backtest, but each fold's test rows carry the stand-in for the manager's number (unless the task says `real`)."""
    from catboost import CatBoostRegressor

    df, folds = _STATE["df"], _STATE["folds"]
    started, features, target = time.time(), task["features"], s54.PIECES[task["piece"]][1]
    out = []
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)].copy()
        if not task.get("real"):
            test[MANAGER_COL] = seasonal_medians(train, test[DATE_COL], MANAGER_COL)
        y = np.log1p(train[TARGET_COL]) if target == "log1p" else train[TARGET_COL]
        per_seed = []
        for seed in task["seeds"]:
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
            model.fit(train[features], y)
            p = model.predict(test[features])
            per_seed.append(np.expm1(p) if target == "log1p" else p)
        out.append(np.array(per_seed))
    per_seed = np.concatenate(out, axis=1)
    return {"id": task["id"], "part": "backtest", "piece": task["piece"], "variant": task["variant"], "seeds": task["seeds"], "per_seed": [[round(float(x), 2) for x in row] for row in per_seed],
            "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for variant, features_of in VARIANTS.items():
        for piece in PIECES:
            features = features_of(piece)
            for i in range(0, len(SEEDS25), s54.BACKTEST_CHUNK):
                chunk = SEEDS25[i:i + s54.BACKTEST_CHUNK]
                out.append({"id": f"backtest_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": variant, "seeds": chunk, "features": features})
    return out


def mine(done: dict, piece: str, variant: str) -> np.ndarray:
    rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(SEEDS25), (piece, variant, per_seed.shape)
    return per_seed.mean(axis=0)


def sanity() -> None:
    """One task with the REAL manager's number must reproduce Series 81's stored `base` fit for the same piece and seeds (the same code path, so to the penny)."""
    piece = "Monday-Thursday (hydro)"
    task = {"id": "sanity", "part": "backtest", "piece": piece, "variant": "sd_base", "seeds": SEEDS25[:5], "features": VARIANTS["sd_base"](piece), "real": True}
    from src.experiments import run_series47_weekend_flags_in_pieces as s47
    from src.forecast_log.v1 import add_hydro_thu

    s47.init_worker47()
    _STATE["df"] = add_hydro_thu(_STATE["df"])
    from src.features import payday_intensity as pi

    _STATE["df"] = pi.add_payday_intensity_fixed(_STATE["df"], DATE_COL)
    got = np.array(fit_task87(task)["per_seed"])
    stored = [np.array(r["per_seed"]) for r in read_done(s81.RESULTS).values() if (r["piece"], r["variant"]) == (piece, "base") and r["seeds"] == SEEDS25[:5]]
    assert stored, "Series 81's base fit for these seeds is missing"
    gap = float(np.max(np.abs(got - stored[0])))
    print(f"sanity: the largest difference from Series 81's stored base fit, real manager number: {gap:.4f} pounds")
    assert gap < 0.01, "the stand-in loop does not reproduce the stored fit"


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    six = s82.in_window(days["date"], (12, 26), (12, 31))
    assert six.sum() == 6
    done, done81, done82 = read_done(RESULTS), read_done(s81.RESULTS), read_done(s82.RESULTS)
    pd.set_option("display.width", 250)

    def per_piece(source: dict, variant: str) -> dict:
        return {pc: mine(source, pc, variant) for pc in PIECES}

    def package(pieces: dict) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [pieces["Friday"], pieces["Saturday"], pieces["Sunday"]], default=pieces["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    real_base = package(per_piece(done81, "base"))
    sd_base, sd_jan, sd_comp = per_piece(done, "sd_base"), per_piece(done, "sd_j1_8"), per_piece(done, "sd_companion")
    variants = {"sd_base (unmasked, stand-in)": package(sd_base), "sd_j1_8 (1-8 Jan mask)": package(sd_jan), "sd_comp_dec (mask + companion 26-31 Dec)": package({pc: np.where(six, sd_comp[pc], sd_jan[pc]) for pc in PIECES}),
                "sd_companion on every day": package(sd_comp)}
    ref = variants["sd_base (unmasked, stand-in)"]
    rows = {}
    for name, p in variants.items():
        diff = np.abs(p - a) - np.abs(ref - a)
        rows[name] = {"net, 585 days": float(diff.mean()), "the 6 days 26-31 Dec": float(diff[six].mean()), "all other days": float(diff[~six].mean())}
    table = pd.DataFrame(rows).T
    print("EACH VARIANT MINUS THE UNMASKED PACKAGE, all served with the stand-in for the manager's number (pounds a day; negative = better; seeds 200-224)\n")
    print(table.round(2).to_string())
    print(f"\nwhat serving the stand-in costs the unmasked package, against the same package with the real number: average miss {np.abs(ref - a).mean():,.0f} against {np.abs(real_base - a).mean():,.0f} ({np.abs(ref - a).mean() - np.abs(real_base - a).mean():+.0f} a day)")
    mask_gain = float(table.loc["sd_j1_8 (1-8 Jan mask)", "net, 585 days"])
    comp_gain = float(table.loc["sd_comp_dec (mask + companion 26-31 Dec)", "net, 585 days"]) - mask_gain
    print(f"\nthe mask against the unmasked package: {mask_gain:+.2f} ({'HOLDS (at least 4.5 better)' if mask_gain <= -4.5 else 'does not hold'}; with the real number: -9.07)")
    print(f"the companion on 26-31 December against the mask alone: {comp_gain:+.2f} ({'SURVIVES' if comp_gain <= -1.1 else 'WEAKENS' if comp_gain < 0 else 'FAILS'}; with the real number: -2.22)")
    # how far the stand-in was from the real number on the six days
    full = df.set_index(pd.to_datetime(df[DATE_COL]))
    rows6 = []
    for fold in folds:
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)]
        if not test[DATE_COL].between("2025-12-26", "2025-12-31").any():
            continue
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        t6 = test[test[DATE_COL].between("2025-12-26", "2025-12-31")]
        standin = seasonal_medians(train, t6[DATE_COL], MANAGER_COL)
        rows6 += [{"date": d.strftime("%a %d %b"), "actual": float(s_), "manager (real)": float(m), "stand-in": float(si)} for d, s_, m, si in zip(t6[DATE_COL], t6[TARGET_COL], t6[MANAGER_COL], standin)]
    t = pd.DataFrame(rows6)
    pick = np.where(six)[0]
    t["package, stand-in, unmasked"], t["package, stand-in, mask"], t["package, stand-in, mask + companion"] = ref[pick], variants["sd_j1_8 (1-8 Jan mask)"][pick], variants["sd_comp_dec (mask + companion 26-31 Dec)"][pick]
    print("\n26-31 DECEMBER 2025: THE MANAGER'S REAL NUMBER, THE STAND-IN, AND THE PACKAGE'S FORECASTS WITH THE STAND-IN\n")
    print(t.round(0).to_string(index=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "variants.csv")
    t.round(0).to_csv(OUT_DIR / "late_december_days.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(VARIANTS)} variants x {len(PIECES)} pieces x {len(SEEDS25) // s54.BACKTEST_CHUNK} tasks of {s54.BACKTEST_CHUNK} seeds = {len(tasks)} tasks (backtest only, 25 seeds)")
        for variant, features_of in VARIANTS.items():
            for piece in PIECES:
                f = features_of(piece)
                assert MANAGER_COL in f, f"{variant}/{piece}: the manager's number is not an input"
        print("the manager's number is an input of every piece of every variant")
    elif step == "sanity":
        sanity()
    elif step == "run":
        run_parallel(make_tasks(), fit_task87, RESULTS, workers, s82.init_worker82, "series87")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "sanity", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
