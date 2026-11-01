"""
Series 43 - what the missing manager number costs the post model, by distance (Series 18's test, repeated with the post model).

The rules are in docs/SERVING_HORIZON_POST_MODEL_2026_11.md, committed before this script existed. For each of Series 18's 8 origins (every fold
whose origin has 180 days of data after it), every piece of the post model (the log model, the raw-pound model and the Friday, Saturday and Sunday
pieces; seeds 0 to 24) is refitted on the days before the origin and asked for the 180 days after it in three ways:

  real      every real input (real history, real weather, the manager's real number): the ceiling, and the sanity checks
  with      as the API sees those days: history filled with weekday-and-season medians, real weather for 16 days then medians, the real number
  without   the same, with the stand-in (weekday-and-season median of the manager's past forecasts) in place of the number

The `report` step assembles the post model from the pieces (weekday routing as the weekly log does it at any distance, then the Saturday and Sunday
correction learnt from the post model's own errors before the origin), runs the two sanity checks and stops if either fails; only then does it
write the distance table.

Run:  python -m src.experiments.run_series43_post_model_horizon run [--workers 10]     (resumable: finished tasks are kept)
      python -m src.experiments.run_series43_post_model_horizon report
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_package_validation as pv
from src.experiments import run_series35_holdout as h
from src.experiments.run_series23_log_vs_raw_peaks import masked_block_bootstrap
from src.experiments.series19_21_common import load_protocol, read_done, run_parallel, test_days
from src.features.serving_features import (
    WEATHER_COLS, build_feature_frame, extend_daily_frame, fill_future_sales_seasonally, with_history_features,
)
from src.models.final_model import DATE_COL, FINAL_FEATURES, TARGET_COL

MANAGER_COL = "forecast_sales"
HORIZON_DAYS = 180
LIVE_WEATHER_DAYS = 16  # Series 18's stand-in for a live weather forecast
HORIZON_BUCKETS = [(1, 7), (8, 16), (17, 30), (31, 60), (61, 90), (91, 180), (1, 180)]
SEEDS = list(range(25))
CONDITIONS = ("real", "with", "without")
# name -> (extra inputs on top of the 38 final-model features, target): the post model's pieces without the Hydro flag (src/forecast_log/v1.py)
PIECES: dict[str, tuple[list[str], str]] = {
    "log": ([], "log1p"),
    "raw": ([], "raw"),
    "fri_A_all": (h.EXTRA["Friday"], "log1p"),
    "sat_A_lags": (h.EXTRA["Saturday"], "raw"),
    "sun_B": (h.EXTRA["Sunday"], "log1p"),
}
SANITY_TOLERANCE = 0.05
ENGINEERED = CONFIG.path("features_dir") / "engineered_features.csv"
OUT_DIR = CONFIG.path("experiments_dir") / "series43"
_S: dict = {}


# --------------------------------------------------------------------------- origins and inputs


def origins(folds: list, last_date: pd.Timestamp) -> list:
    """Series 18's origins: the folds whose test start has HORIZON_DAYS days of data after it."""
    return [f for f in folds if f.test_start + pd.Timedelta(days=HORIZON_DAYS) <= last_date + pd.Timedelta(days=1)]


def served_rows(full: pd.DataFrame, origin: pd.Timestamp, dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Feature rows for `dates` as the API sees them on the origin date (Series 18's layout): real history before the origin, weekday-and-season
    medians after it (sales, the manager's number, weather), with the weather that happened on the first LIVE_WEATHER_DAYS days."""
    history = full[full[DATE_COL] < origin]
    end = origin + pd.Timedelta(days=HORIZON_DAYS - 1)
    daily = extend_daily_frame(history, end.date())
    real = full.set_index(DATE_COL)
    live = (daily[DATE_COL] >= origin) & (daily[DATE_COL] < origin + pd.Timedelta(days=LIVE_WEATHER_DAYS)) & daily[DATE_COL].isin(real.index)
    for col in WEATHER_COLS:
        daily.loc[live, col] = real.loc[daily.loc[live, DATE_COL], col].to_numpy()
    frame = build_feature_frame(daily)
    first_future = len(history)
    position = {d: i for i, d in enumerate(frame[DATE_COL])}
    sales = fill_future_sales_seasonally(frame, first_future, history)
    return with_history_features(frame, sales, [position[d] for d in dates]).reset_index(drop=True)


def sales_known_before(full: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """Daily sales known on the origin date (real before it, missing from it on), over the whole window."""
    return h.sales_known_series(full, origin - pd.Timedelta(days=1), origin + pd.Timedelta(days=HORIZON_DAYS - 1))


# --------------------------------------------------------------------------- the worker


def init_worker() -> None:
    df, folds = load_protocol()
    full = h.load_engineered(ENGINEERED)
    _S.update(df=df, folds=folds, full=full, sales=full.set_index(DATE_COL)[TARGET_COL])


def fit_task(task: dict) -> dict:
    """One origin, one piece: fit the bag on the days before the origin and forecast the scored days after it in the three conditions."""
    started = time.time()
    df, folds, full, sales = _S["df"], _S["folds"], _S["full"], _S["sales"]
    fold = next(f for f in folds if f.fold_id == task["fold"])
    origin = fold.test_start
    extra, target = PIECES[task["piece"]]
    train = df[df[DATE_COL] < origin].reset_index(drop=True)
    window = df[(df[DATE_COL] >= origin) & (df[DATE_COL] < origin + pd.Timedelta(days=HORIZON_DAYS))].reset_index(drop=True)
    dates = list(window[DATE_COL])
    bag = h.Bag.fit(h.with_weekend(train, sales), list(FINAL_FEATURES) + extra, target, seeds=SEEDS, thread_count=1)
    known = sales_known_before(full, origin)
    without = served_rows(full, origin, dates)
    with_number = without.copy()
    with_number[MANAGER_COL] = window[MANAGER_COL].to_numpy()
    preds = {
        "real": bag.predict(h.with_weekend(window, sales)),
        "with": bag.predict(h.with_weekend(with_number, known)),
        "without": bag.predict(h.with_weekend(without, known)),
    }
    return {"id": task["id"], "fold": task["fold"], "origin": origin.strftime("%Y-%m-%d"), "piece": task["piece"],
            "dates": [d.strftime("%Y-%m-%d") for d in dates], **{c: [round(float(x), 4) for x in preds[c]] for c in CONDITIONS},
            "seconds": round(time.time() - started, 1)}


def build_tasks(fold_ids: list[int]) -> list[dict]:
    return [{"id": f"s43_f{f}_{p}", "fold": f, "piece": p} for f in fold_ids for p in PIECES]


# --------------------------------------------------------------------------- assembling the post model


def route(dates: pd.Series, preds: dict[str, np.ndarray], known: pd.Series | None) -> np.ndarray:
    """The post model's weekday routing at any distance (as src/forecast_log/predict.py does it, without the Hydro flag): Friday uses A_all only if
    the sales 7, 14, 21 and 28 days earlier are known, Saturday uses A_lags if the sales 14 and 21 days earlier are known (else the raw model),
    Sunday always B, Monday to Thursday the log model. `known=None` routes by weekday alone (the validated backtest's routing)."""
    wd = pd.to_datetime(pd.Series(list(dates))).dt.day_name().to_numpy()
    everything = np.ones(len(wd), dtype=bool)
    fri_ok = everything if known is None else h_lags_known(known, dates, (7, 14, 21, 28))
    sat_ok = everything if known is None else h_lags_known(known, dates, (14, 21))
    out = np.where(wd == "Friday", np.where(fri_ok, preds["fri_A_all"], preds["log"]), preds["log"])
    out = np.where(wd == "Saturday", np.where(sat_ok, preds["sat_A_lags"], preds["raw"]), out)
    return np.where(wd == "Sunday", preds["sun_B"], out)


def h_lags_known(known: pd.Series, dates, lags: tuple[int, ...]) -> np.ndarray:
    ok = pd.Series(True, index=known.index)
    for k in lags:
        ok &= known.shift(k).notna()
    return ok.reindex(pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates))))).fillna(False).to_numpy()


def assemble(records: dict[str, dict], df: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """One row per origin and scored day: actual, the manager's number, and the log and post models in the three conditions."""
    by_origin: dict[str, dict[str, dict]] = {}
    for r in records.values():
        by_origin.setdefault(r["origin"], {})[r["piece"]] = r
    real_sales = full.set_index(DATE_COL)[TARGET_COL]
    real_sales = real_sales.reindex(pd.date_range(real_sales.index.min(), real_sales.index.max(), freq="D"))  # one row per calendar day, for the lag checks
    frames = []
    for origin_text, pieces in sorted(by_origin.items()):
        origin = pd.Timestamp(origin_text)
        dates = pd.to_datetime(pd.Series(pieces["log"]["dates"]))
        assert all(p["dates"] == pieces["log"]["dates"] for p in pieces.values()), "pieces disagree on the scored days"
        hist_days, hist_pred = h.package_history(origin)
        out = pd.DataFrame({"origin": origin_text, "date": dates.dt.strftime("%Y-%m-%d"), "horizon_day": (dates - origin).dt.days + 1})
        window = df.set_index(DATE_COL).loc[dates]
        out["actual"], out["manager"] = window[TARGET_COL].to_numpy(), window[MANAGER_COL].to_numpy()
        for c in CONDITIONS:
            preds = {name: np.array(p[c]) for name, p in pieces.items()}
            out[f"log_{c}"] = preds["log"]
            # every real input: routed by weekday alone, as the validated backtest is (a missing sales day in the data, e.g. 24 January 2025,
            # does not switch a Friday to the plain model there); served: the log's availability rule, which does
            routed = route(dates, preds, None if c == "real" else sales_known_before(full, origin))
            # the first backtest block has no earlier errors to learn the correction from, so it is left as routed (as in the stored backtest)
            out[f"post_{c}"] = h.correct_new_block(hist_days, hist_pred, list(dates), routed) if len(hist_days) else routed
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- sanity checks and the table


def sanity_checks(table: pd.DataFrame) -> list[dict]:
    """Both models with every real input against their stored verified-calendar backtests, on each origin's 60-day test block."""
    df, folds = load_protocol()
    days = test_days(df, folds)
    days["weekday"] = pd.to_datetime(days["date"]).dt.day_name()
    data = pv.load_set("_cal2")
    stored = {
        "log model (series19_21_cal2, log1p_final)": data["log_raw_0_24"]["log1p_final"],
        "post model (package final step)": pv.package_steps(days, data["log_raw_0_24"], data["features_0_24"], h.CHOSEN)[pv.STEPS[3]],
    }
    stored_by_date = {name: pd.Series(values, index=days["date"]) for name, values in stored.items()}
    checks = []
    for name, column in (("log model (series19_21_cal2, log1p_final)", "log_real"), ("post model (package final step)", "post_real")):
        diffs = []
        for fold in folds:
            block = table[(table["origin"] == fold.test_start.strftime("%Y-%m-%d")) & (table["horizon_day"] <= (fold.test_end - fold.test_start).days)]
            if len(block):
                diffs.append(np.abs(block[column].to_numpy() - stored_by_date[name].reindex(block["date"]).to_numpy()))
        diff = np.concatenate(diffs) if diffs else np.array([np.nan])
        checks.append({"check": name, "days": int(np.isfinite(diff).sum()), "max_abs_diff": float(np.nanmax(diff)),
                       "passes": bool(np.isfinite(diff).all() and np.nanmax(diff) <= SANITY_TOLERANCE)})
    return checks


def mae(a: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(p - a)))


COLUMNS = {"manager": "Manager alone", "log_real": "Log model, every real input", "log_with": "Log model, with the number",
           "log_without": "Log model, without it", "post_real": "Post model, every real input", "post_with": "Post model, with the number",
           "post_without": "Post model, without it"}


def distance_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for low, high in HORIZON_BUCKETS:
        sub = table[table["horizon_day"].between(low, high)]
        rows.append({"days ahead": f"{low}-{high}", "scores": len(sub), **{label: round(mae(sub["actual"].to_numpy(), sub[c].to_numpy()), 1) for c, label in COLUMNS.items()}})
    return pd.DataFrame(rows)


def cost_table(table: pd.DataFrame) -> pd.DataFrame:
    """Cost of losing the number (MAE without minus with) for both models per bucket, and post cost minus log cost, with 95% block-bootstrap intervals
    (days ordered by origin and date; the bucket is a mask on the whole series)."""
    ordered = table.sort_values(["origin", "date"]).reset_index(drop=True)
    a = ordered["actual"].to_numpy()
    err = {c: np.abs(ordered[c].to_numpy() - a) for c in ("log_with", "log_without", "post_with", "post_without")}
    diffs = {"log model": err["log_without"] - err["log_with"], "post model": err["post_without"] - err["post_with"]}
    diffs["post cost minus log cost"] = diffs["post model"] - diffs["log model"]
    rows = []
    for low, high in HORIZON_BUCKETS:
        mask = ordered["horizon_day"].between(low, high).to_numpy().astype(float)
        row = {"days ahead": f"{low}-{high}"}
        for name, d in diffs.items():
            m, lo, hi = masked_block_bootstrap(d, mask)
            row[name] = f"{m:+.1f} ({lo:+.1f}, {hi:+.1f})"
        rows.append(row)
    return pd.DataFrame(rows)


def reading(table: pd.DataFrame) -> str:
    ordered = table.sort_values(["origin", "date"]).reset_index(drop=True)
    a = ordered["actual"].to_numpy()
    d = (np.abs(ordered["post_without"] - a) - np.abs(ordered["post_with"] - a)) - (np.abs(ordered["log_without"] - a) - np.abs(ordered["log_with"] - a))
    m, lo, hi = masked_block_bootstrap(d.to_numpy(), np.ones(len(d)))
    verdict = "loses less without the number" if hi < 0 else "loses more without the number" if lo > 0 else "the same cost"
    return f"Post cost minus log cost over all 180 days: {m:+.1f} ({lo:+.1f}, {hi:+.1f}) -> by the rule in section 5: **{verdict}**."


# --------------------------------------------------------------------------- command line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    df, folds = load_protocol()
    full = h.load_engineered(ENGINEERED)
    chosen = origins(folds, full[DATE_COL].max())
    print(f"[s43] {len(chosen)} origins: {[f.test_start.date().isoformat() for f in chosen]}")
    tasks = build_tasks([f.fold_id for f in chosen])
    path = OUT_DIR / "tasks.jsonl"
    if args.command == "run":
        run_parallel(tasks, fit_task, path, args.workers, initializer=init_worker, label="s43")
    done = read_done(path)
    records = {t["id"]: done[t["id"]] for t in tasks if t["id"] in done}
    if len(records) < len(tasks):
        print(f"[s43] {len(tasks) - len(records)} tasks not finished; run again to resume")
        return 1
    table = assemble(records, df, full)
    checks = sanity_checks(table)
    for c in checks:
        print(f"[s43] sanity: {c['check']}: {c['days']} days, largest difference {c['max_abs_diff']:.4f} -> {'PASS' if c['passes'] else 'FAIL'}")
    if not all(c["passes"] for c in checks):
        print("[s43] a sanity check failed: no table is written (rules, section 4)")
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "predictions.csv", index=False)
    lines = ["# Series 43: scores", "", *[f"Sanity: {c['check']}: {c['days']} days, largest difference {c['max_abs_diff']:.4f}, PASS." for c in checks], "",
             "## MAE by distance (£ a day)", "", distance_table(table).to_markdown(index=False), "",
             "## Cost of losing the number (without minus with, £ a day, 95% interval)", "", cost_table(table).to_markdown(index=False), "", reading(table), ""]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("[s43] table written to", OUT_DIR / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
