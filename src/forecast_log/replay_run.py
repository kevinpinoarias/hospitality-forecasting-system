"""
Replay a logged run of the forecast log with another registered version, from the inputs the run used.

A run is made from the engineered-features file (sales through the last known day) and the live weather it fetched, which the run saves in `log/weather_used.csv`. This rebuilds exactly that situation
for a past run: the same issue time, the same last known day, the weather the run recorded (nothing is fetched), the same targets, the same correction history; it forecasts with the main version (v1) and
with any further versions asked for, and checks that its v1 numbers equal the ones the run logged, to the penny. Nothing is written to the log repository.

  python -m src.forecast_log.replay_run --run-id 20261102T1649-monday --also v1.11 [--csv out.csv]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.experiments import run_series35_holdout as h
from src.forecast_log import predict, run_forecasts, schedule, v1
from src.models.final_model import DATE_COL, TARGET_COL


def run_row(log_repo: Path, run_id: str) -> pd.Series:
    runs = pd.read_csv(log_repo / "log" / "runs.csv", dtype=str)
    row = runs[runs["run_id"] == run_id]
    if row.empty:
        raise RuntimeError(f"run {run_id} is not in {log_repo / 'log' / 'runs.csv'}")
    return row.iloc[0]


def run_weather(log_repo: Path, run_id: str) -> pd.DataFrame:
    """The weather the run used, as it saved it (max temperature, rain, sun hours for each day it covered)."""
    saved = pd.read_csv(log_repo / "log" / "weather_used.csv")
    weather = saved[saved["run_id"] == run_id][["date", "max_temp", "rain_mm", "sun_hours"]].rename(columns={"date": DATE_COL}).reset_index(drop=True)
    weather[DATE_COL] = pd.to_datetime(weather[DATE_COL]).dt.normalize()
    return weather


def replay(log_repo: Path, run_id: str, also: tuple[str, ...], version: str = v1.VERSION) -> pd.DataFrame:
    """One row per forecast type and target day: the main version's portfolio, post and hydro forecasts, the further versions' hydro arms, the stand-in manager number, and what the run logged."""
    row = run_row(log_repo, run_id)
    issue = datetime.fromisoformat(row["issue_time"])
    full = h.load_engineered(run_forecasts.FEATURES)
    known = full.dropna(subset=[TARGET_COL])
    last_known = known[DATE_COL].max()
    if str(last_known.date()) != row["data_through"]:
        raise RuntimeError(f"the features file runs through {last_known.date()}, the run used data through {row['data_through']}: the run cannot be replayed from this file")
    weather = run_weather(log_repo, run_id)
    models = predict.load_models(log_repo, version)
    history = v1.load_correction_history(v1.correction_history_path(version))
    extras = [(m, v1.load_correction_history(v1.correction_history_path(m.version))) for m in (predict.load_models(log_repo, v) for v in also)]
    logged = pd.read_csv(log_repo / "log" / "forecasts.csv", dtype={"variant": str})
    logged = logged[logged["run_id"] == run_id]
    frames = []
    for ftype in schedule.forecast_types_for(row["trigger"]):
        targets = [d for d in schedule.target_dates(ftype, issue) if pd.Timestamp(d) > last_known]
        result = predict.forecast_days(models, full, last_known, [pd.Timestamp(d) for d in targets], history, weather, extras)
        result.insert(0, "forecast_type", ftype)
        frames.append(result)
    out = pd.concat(frames, ignore_index=True)
    out["target_date"] = out["target_date"].astype(str)
    for model in predict.MODELS:
        key = logged[(logged["model"] == model)].set_index(["forecast_type", "target_date"])["forecast"]
        out[f"{model}_logged"] = [key.get((t, d), float("nan")) for t, d in zip(out["forecast_type"], out["target_date"])]
    return out


def reproduces_the_log(out: pd.DataFrame, tolerance: float = 0.01) -> bool:
    """True if every portfolio, post and hydro forecast replayed equals the logged one (rounded to the pence, as the log writes it)."""
    ok = True
    for model in predict.MODELS:
        have = out[f"{model}_logged"].notna()
        ok &= bool(((out.loc[have, model].round(2) - out.loc[have, f"{model}_logged"]).abs() <= tolerance).all())
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--also", nargs="*", default=[], help="further registered versions whose hydro arm is replayed (v1.11)")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    settings = run_forecasts.load_settings()
    out = replay(Path(settings["log_repo"]), args.run_id, tuple(args.also))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out.round(0).to_string(index=False))
    print(f"\nreplayed v1 forecasts equal the logged ones: {reproduces_the_log(out)}")
    if args.csv:
        out.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
