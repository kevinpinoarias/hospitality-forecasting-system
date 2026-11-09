"""
Series 90 - the untested year-end candidates, served the way the forecast log serves.

The register and the reading are in docs/JANUARY_DECEMBER_CANDIDATES_2026_11.md, committed before this script ran. Six candidates, each refitted in the four pieces of the current package (Monday-Thursday with the hydro
flag, Friday, Saturday with `intensity_fixed`, Sunday), 25 seeds (200-224), 10 folds, 585 days, the deployed weekend correction, every one SERVED WITH THE S6 STAND-IN (the hybrid of Series 89: the same date a year earlier
for 24 December to 2 January, the closed-day-aware median elsewhere; the models are fitted on the real training rows and only the test rows' `forecast_sales` differs, as in Series 87-89):
  B  `sales_364` (the sales on the same weekday 364 days earlier) added to every piece; rows before 16 January 2025 have none
  C  `days_since_xmas` (days since 25 December, 99 beyond 60 days) added to every piece
  D  the ten history inputs rebuilt from the sales series with 24 December to 2 January left out (averages over the days that exist, from half the window)
  E  the lean companion: like Series 85's companion but dropping only the averages and lags (`rolling_7_sales`, `rolling_14_sales`, `lag_7_sales`, `lag_14_sales`, `lag_21_sales`), keeping the forecast-error and
     same-weekday-summary inputs; used on 26-31 December inside the specialist windows (the 1 to 8 January masked model on 1-8 January, the unmasked model elsewhere)
  F  `intensity_fixed` added to the Monday-Thursday, Friday and Sunday pieces (the Saturday piece has it)
  H  random training masks: the ten history inputs hidden (set missing) on a random 20% of the training rows of every seed, then hidden at forecast time on 1-8 January and 26-31 December
References: the unmasked package served with S6 and the specialist windows (M1) served with S6, both from Series 89's fits on the same seeds (200-224). Tasks: 6 x 4 pieces x 5 = 120. A sanity step reproduces Series 89's
stored S6 forecast for the unmasked piece to the penny.

READING (written in the register before the run)
  measure     the net change of the post package over the 585 days against the unmasked package served with S6 (pounds a day; negative = better), with 26-31 Dec, 1-8 Jan and all other days beside it; E against M1
  helps       net change at most -2.0 (25-seed noise is about 2); nothing: within 2.0 of zero; hurts: above +2.0
  beats M1    net change at least 2.0 below the specialist windows' on the same seeds
  E           better than M1 if its net change is at least 2.0 below M1's
  limits      one December and one January; 25 seeds; the same 585 days; a candidate that helps is a candidate to repeat on 100 and fresh seeds, not a result; any adoption is Kevin's decision and needs an amendment
  (and, for a model change, a new forecast-log version)

Run:  python -m src.experiments.run_series90_year_end_candidates list | sanity | run [--workers 10] | report
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
from src.experiments import run_series81_closed_days_in_lags as s81
from src.experiments import run_series82_mask_windows as s82
from src.experiments import run_series87_companion_without_manager as s87
from src.experiments import run_series89_standin_repeat_hybrid as s89
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu
from src.models.final_model import FINAL_PARAMS, TARGET_COL

OUT_DIR = CONFIG.path("experiments_dir") / "series90"
RESULTS = OUT_DIR / "series90_year_end_candidates.jsonl"
MANAGER_COL = s89.MANAGER_COL
SEEDS25, PIECES, INPUTS = s81.SEEDS25, s81.PIECES, s81.INPUTS
LEAN_DROP = ["rolling_7_sales", "rolling_14_sales", "lag_7_sales", "lag_14_sales", "lag_21_sales"]
DROPOUT = 0.2
CANDIDATES = ["B", "C", "D", "E", "F", "H"]


def base_features(piece: str) -> list[str]:
    return s81.piece_features(piece, "base")


def features_for(candidate: str, piece: str) -> list[str]:
    f = base_features(piece)
    if candidate == "B":
        return f + ["sales_364"]
    if candidate == "C":
        return f + ["days_since_xmas"]
    if candidate == "D":
        return ["bk_" + x if x in INPUTS else x for x in f]
    if candidate == "E":
        return [x for x in f if x not in LEAN_DROP]
    if candidate == "F":
        return f if pi.COLUMN_FIXED in f else f + [pi.COLUMN_FIXED]
    if candidate == "H":
        return f  # the original names; the test-time masks are applied inside the fit
    raise ValueError(candidate)


def init_worker90() -> None:
    s47.init_worker47()
    df = pi.add_payday_intensity_fixed(add_hydro_thu(_STATE["df"]), DATE_COL)
    sales, fe = s81.load_series()
    dates = pd.DatetimeIndex(df[DATE_COL])
    d = pd.to_datetime(df[DATE_COL])
    new = {"sales_364": sales.shift(364).reindex(dates).to_numpy()}
    xmas_this = pd.to_datetime(d.dt.year.astype(str) + "-12-25")
    xmas_prev = pd.to_datetime((d.dt.year - 1).astype(str) + "-12-25")
    since = np.where(d >= xmas_this, (d - xmas_this).dt.days, (d - xmas_prev).dt.days)
    new["days_since_xmas"] = np.where(since > 60, 99, since).astype(float)
    brk = s82.in_window(pd.Series(sales.index), (12, 24), (1, 2))
    s_nb, fe_nb = sales.copy(), fe.copy()
    s_nb[brk], fe_nb[brk] = np.nan, np.nan
    ins = s81.inputs_from(s_nb, fe_nb, loose=True).reindex(dates)
    window = s82.in_window(df[DATE_COL], (1, 1), (1, 8)) | s82.in_window(df[DATE_COL], (12, 26), (12, 31))
    for c in INPUTS:
        new["bk_" + c] = ins[c].to_numpy()
        col = df[c].to_numpy(dtype=float).copy()
        col[window] = np.nan
        new["hm_" + c] = col
    _STATE["df"] = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def fit_task90(task: dict) -> dict:
    """Fit on the real training rows and serve the S6 stand-in on the fold's test days; with `dropout` the training rows get random masks and the test rows the window masks."""
    from catboost import CatBoostRegressor

    df, folds = _STATE["df"], _STATE["folds"]
    started, features, target = time.time(), task["features"], s54.PIECES[task["piece"]][1]
    dropout = task.get("dropout", 0.0)
    cols = [c for c in features if c in INPUTS]
    out = []
    for fold in folds:
        train = df[(df[DATE_COL] >= fold.train_start) & (df[DATE_COL] < fold.train_end)]
        test = df[(df[DATE_COL] >= fold.test_start) & (df[DATE_COL] < fold.test_end)].copy()
        test[MANAGER_COL] = s89.standins(train, test)["S6"]
        y = np.log1p(train[TARGET_COL]) if target == "log1p" else train[TARGET_COL]
        te = test[features].copy()
        if dropout:
            for c in cols:
                te[c] = test["hm_" + c].to_numpy()
        per_seed = []
        for seed in task["seeds"]:
            tr = train[features].copy()
            if dropout:
                mask = np.random.default_rng(seed).random(len(tr)) < dropout
                tr.loc[mask, cols] = np.nan
            model = CatBoostRegressor(random_state=seed, verbose=False, allow_writing_files=False, thread_count=1, **FINAL_PARAMS)
            model.fit(tr, y)
            p = model.predict(te)
            per_seed.append(np.expm1(p) if target == "log1p" else p)
        out.append(np.array(per_seed))
    per_seed = np.concatenate(out, axis=1)
    return {"id": task["id"], "part": "backtest", "piece": task["piece"], "variant": task["variant"], "seeds": task["seeds"], "per_seed": [[round(float(x), 2) for x in row] for row in per_seed],
            "seconds": round(time.time() - started, 1)}


def make_tasks() -> list[dict]:
    out = []
    for candidate in CANDIDATES:
        for piece in PIECES:
            features = features_for(candidate, piece)
            for i in range(0, len(SEEDS25), s54.BACKTEST_CHUNK):
                chunk = SEEDS25[i:i + s54.BACKTEST_CHUNK]
                task = {"id": f"backtest_{piece}_{candidate}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": candidate, "seeds": chunk, "features": features}
                if candidate == "H":
                    task["dropout"] = DROPOUT
                out.append(task)
    return out


def mine(done: dict, piece: str, variant: str) -> np.ndarray:
    rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(SEEDS25), (piece, variant, per_seed.shape)
    return per_seed.mean(axis=0)


def s89_first25(done89: dict, piece: str, variant: str) -> np.ndarray:
    """Series 89's S6-served forecast of a variant on seeds 200-224 (its first five chunks)."""
    rows = [np.array(r["per_standin"]["S6"]) for r in done89.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant) and r["seeds"][0] in SEEDS25]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(SEEDS25), (piece, variant, per_seed.shape)
    return per_seed.mean(axis=0)


def sanity() -> None:
    piece = "Monday-Thursday (hydro)"
    init_worker90()
    got = np.array(fit_task90({"id": "sanity", "piece": piece, "variant": "sanity", "seeds": SEEDS25[:5], "features": base_features(piece)})["per_seed"])
    stored = [np.array(r["per_standin"]["S6"]) for r in read_done(s89.RESULTS).values() if (r["piece"], r["variant"]) == (piece, "sd_base") and r["seeds"] == SEEDS25[:5]][0]
    gap = float(np.max(np.abs(got - stored)))
    print(f"sanity: the largest difference from Series 89's stored S6 forecast: {gap:.4f} pounds")
    assert gap < 0.01, "the loop does not reproduce Series 89"


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    six = s82.in_window(days["date"], (12, 26), (12, 31))
    jan8 = s82.in_window(days["date"], (1, 1), (1, 8))
    done, done89 = read_done(RESULTS), read_done(s89.RESULTS)
    pd.set_option("display.width", 250)

    def package(p: dict) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [p["Friday"], p["Saturday"], p["Sunday"]], default=p["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    base_p = {pc: s89_first25(done89, pc, "sd_base") for pc in PIECES}
    msk_p = {pc: s89_first25(done89, pc, "sd_j1_8") for pc in PIECES}
    comp_p = {pc: s89_first25(done89, pc, "sd_companion") for pc in PIECES}
    lean_p = {pc: mine(done, pc, "E") for pc in PIECES}
    spec = {pc: np.where(six, comp_p[pc], np.where(jan8, msk_p[pc], base_p[pc])) for pc in PIECES}
    spec_lean = {pc: np.where(six, lean_p[pc], np.where(jan8, msk_p[pc], base_p[pc])) for pc in PIECES}
    base = package(base_p)
    names = {"M1 specialist windows (the reference)": package(spec), "B year-ago sales input": package({pc: mine(done, pc, "B") for pc in PIECES}), "C days-since-Christmas input": package({pc: mine(done, pc, "C") for pc in PIECES}),
             "D history without the Christmas break": package({pc: mine(done, pc, "D") for pc in PIECES}), "E specialist windows with the lean companion": package(spec_lean),
             "F closed-day column in every piece": package({pc: mine(done, pc, "F") for pc in PIECES}), "H random training masks, masked at forecast time": package({pc: mine(done, pc, "H") for pc in PIECES})}
    rows = {}
    for name, p in names.items():
        d = np.abs(p - a) - np.abs(base - a)
        rows[name] = {"net, 585 days": float(d.mean()), "26-31 Dec": float(d[six].mean()), "1-8 Jan": float(d[jan8].mean()), "all other days": float(d[~(six | jan8)].mean())}
    table = pd.DataFrame(rows).T
    print("EACH CANDIDATE MINUS THE UNMASKED PACKAGE, all served with the S6 stand-in (pounds a day; negative = better; seeds 200-224)\n")
    print(table.round(2).to_string())
    m1 = float(table.iloc[0]["net, 585 days"])
    print(f"\nthe specialist windows (the reference): net {m1:+.2f}\n")
    for name in list(names)[1:]:
        net = float(table.loc[name, "net, 585 days"])
        if name.startswith("E"):
            verdict = "BETTER than M1 (at least 2.0 below)" if net <= m1 - 2.0 else "NO BETTER than M1 (within 2.0)" if net <= m1 + 2.0 else "WORSE than M1"
        else:
            verdict = ("HELPS" if net <= -2.0 else "HURTS" if net > 2.0 else "NOTHING") + (", and BEATS the specialist windows (at least 2.0 below M1)" if net <= m1 - 2.0 else "")
        print(f"READING SET BEFOREHAND, {name}: net {net:+.2f}: {verdict}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "candidates.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(CANDIDATES)} candidates x {len(PIECES)} pieces x {len(SEEDS25) // s54.BACKTEST_CHUNK} tasks of {s54.BACKTEST_CHUNK} seeds = {len(tasks)} tasks (backtest only, 25 seeds, all served with S6)")
        init_worker90()
        frame = _STATE["df"]
        for c in CANDIDATES:
            for piece in PIECES:
                missing = [f for f in features_for(c, piece) if f not in frame.columns]
                assert not missing, f"{c}/{piece}: missing columns {missing}"
        print("every candidate's inputs exist in the frame")
        for c in CANDIDATES:
            added = [f for f in features_for(c, "Friday") if f not in base_features("Friday")]
            dropped = [f for f in base_features("Friday") if f not in features_for(c, "Friday")]
            print(f"  {c}  Friday piece: added {added}, dropped {dropped}" + (f"; training masks on {int(DROPOUT * 100)}% of rows, window masks at forecast time" if c == "H" else ""))
        print(f"sales_364: {int(frame['sales_364'].notna().sum())} of {len(frame)} rows have it; days_since_xmas range {frame['days_since_xmas'].min():.0f} to {frame['days_since_xmas'].max():.0f}")
    elif step == "sanity":
        sanity()
    elif step == "run":
        run_parallel(make_tasks(), fit_task90, RESULTS, workers, init_worker90, "series90")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "sanity", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
