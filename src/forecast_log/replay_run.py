"""
Replay a logged run of the forecast log with another registered version, from the inputs the run used.

A run is made from the engineered-features file (sales through the last known day) and the live weather it fetched, which the run saves in `log/weather_used.csv`. This rebuilds exactly that situation
for a past run: the same issue time, the same last known day, the weather the run recorded (nothing is fetched), the same targets, the same correction history; it forecasts with the main version (v1) and
with any further versions asked for, and checks that its v1 numbers equal the ones the run logged, to the penny. Newer sales in the features file are cut off at the day the run's data ended.

`--backfill` gives registered versions the forecasts of every past Monday run that stands (not a withdrawn one), each with only what was known that Monday. A version trained on days after a run's data has
seen the days that run forecasts, so it is left out of that run. Nothing is written unless `--write` is given; then the rows go to the log labelled `replay` with the time they were really made, so they are
scored apart from the live forecasts. If any replay's v1 numbers differ from the log, nothing is written.

  python -m src.forecast_log.replay_run --run-id 20261102T1649-monday --also v1.11 [--csv out.csv]
  python -m src.forecast_log.replay_run --backfill v1.11 v1.12 [--write]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.experiments import run_series35_holdout as h
from src.forecast_log import logio, predict, run_forecasts, schedule, v1
from src.models.final_model import DATE_COL, TARGET_COL

REPLAY_LABEL = "replay"


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


def features_as_of(full: pd.DataFrame, data_through: pd.Timestamp) -> pd.DataFrame:
    """The engineered features as a run saw them: nothing after the day its sales data ran through. A file that stops earlier cannot stand in for that run."""
    last_known = full.dropna(subset=[TARGET_COL])[DATE_COL].max()
    if last_known < data_through:
        raise RuntimeError(f"the features file runs through {last_known.date()}, the run used data through {data_through.date()}: the run cannot be replayed from this file")
    return full[full[DATE_COL] <= data_through].reset_index(drop=True)


def issued_for(logged: pd.DataFrame, issue: datetime) -> datetime:
    """The time a run's days and leads count from: its issue time, or for a reissue of a withdrawn run (made later, with the earlier run's days), the day its leads point back to."""
    if logged.empty:
        return issue
    day = (pd.to_datetime(logged["target_date"]) - pd.to_timedelta(logged["lead_days"].astype(int), unit="D")).min().date()
    return datetime.combine(day, issue.timetz()) if day < issue.date() else issue


def replay(log_repo: Path, run_id: str, also: tuple[str, ...], version: str = v1.VERSION) -> pd.DataFrame:
    """One row per forecast type and target day: the main version's portfolio, post and hydro forecasts, the further versions' hydro arms, the stand-in manager number, and what the run logged."""
    row = run_row(log_repo, run_id)
    logged = pd.read_csv(log_repo / "log" / "forecasts.csv", dtype={"variant": str})
    logged = logged[logged["run_id"] == run_id]
    issue = issued_for(logged, datetime.fromisoformat(row["issue_time"]))
    last_known = pd.Timestamp(row["data_through"])
    full = features_as_of(h.load_engineered(run_forecasts.FEATURES), last_known)
    weather = run_weather(log_repo, run_id)
    models = predict.load_models(log_repo, version)
    history = v1.load_correction_history(v1.correction_history_path(version))
    extras = [(m, v1.load_correction_history(v1.correction_history_path(m.version))) for m in (predict.load_models(log_repo, v) for v in also)]
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


def trained_through(log_repo: Path, version: str) -> pd.Timestamp:
    """The last day of sales a registered version was trained on."""
    registry = json.loads((log_repo / "models" / "versions.json").read_text(encoding="utf-8"))
    entry = next((v for v in registry["versions"] if v["version"] == version), None)
    if entry is None:
        raise RuntimeError(f"version {version} is not registered in {log_repo / 'models' / 'versions.json'}")
    return pd.Timestamp(entry["last_training_day"])


def standing_runs(log_repo: Path) -> pd.DataFrame:
    """The past Monday runs that stand. A withdrawn, skipped or aborted run has no forecasts for a replay to be checked against."""
    runs = pd.read_csv(log_repo / "log" / "runs.csv", dtype=str)
    return runs[(runs["trigger"] == "monday") & (runs["status"] == "ok")]


def replay_rows(out: pd.DataFrame, log_repo: Path, run_id: str, fingerprints: dict[str, str], replay_id: str, issued: str, commit: str) -> list[dict]:
    """Log rows for further versions' hydro arms in a replayed run. Each copies the run's own hydro row for that day (data through, lead, weather source, input hash, stand-in), so only what the replay
    adds differs. A day the run has no hydro row for (a Hydro Thursday beyond the list) gets none."""
    logged = pd.read_csv(log_repo / "log" / "forecasts.csv", dtype=str).fillna("")
    logged = logged[(logged["run_id"] == run_id) & (logged["model"] == "hydro")].set_index(["forecast_type", "target_date"], drop=False)
    rows = []
    for _, day in out.iterrows():
        key = (day["forecast_type"], day["target_date"])
        if key not in logged.index:
            continue
        for version, fingerprint in fingerprints.items():
            arm = predict.extra_arm_name(version)
            rows.append({**logged.loc[key].to_dict(), "run_id": replay_id, "issue_time": issued, "model": arm, "model_version": version, "forecast": round(float(day[arm]), 2),
                         "model_fingerprint": fingerprint, "code_commit": commit, "on_time": REPLAY_LABEL,
                         "manager_forecast_used": round(float(day[arm + predict.STANDIN_SUFFIX]), 2)})  # the stand-in this version was given
    return rows


def backfill(log_repo: Path, versions: tuple[str, ...], now: datetime, commit: str) -> list[tuple[list[dict], dict]]:
    """For every standing run, the rows these versions are missing and that run's line for runs.csv. Raises, so nothing is written, if a replay does not reproduce the log."""
    trained = {v: trained_through(log_repo, v) for v in versions}
    prints = {v: predict.load_models(log_repo, v).combined_fingerprint for v in versions}
    issued = now.isoformat(timespec="seconds")
    batches = []
    for _, run in standing_runs(log_repo).iterrows():
        clean = tuple(v for v in versions if trained[v] <= pd.Timestamp(run["data_through"]))
        if not clean:
            continue
        out = replay(log_repo, run["run_id"], clean)
        if not reproduces_the_log(out):
            raise RuntimeError(f"the replay of {run['run_id']} does not equal the logged forecasts: nothing is written")
        replay_id = f"{now:%Y%m%dT%H%M}-replay-{run['run_id']}"
        rows = logio.new_rows_only(replay_rows(out, log_repo, run["run_id"], {v: prints[v] for v in clean}, replay_id, issued, commit), log_repo / "log" / "forecasts.csv")
        if rows:
            batches.append((rows, {"run_id": replay_id, "issue_time": issued, "trigger": REPLAY_LABEL, "status": "ok", "reason": f"replay of {run['run_id']} with {' '.join(clean)}",
                                   "data_through": run["data_through"], "forecast_types_made": run["forecast_types_made"], "code_commit": commit}))
    return batches


def run_backfill(log_repo: Path, versions: tuple[str, ...], write: bool) -> int:
    with run_forecasts.RunLock():
        batches = backfill(log_repo, versions, datetime.now().astimezone(), v1.code_commit() if write else "")
        for rows, run in batches:
            print(f"[replay] {run['reason']}: {len(rows)} new rows")
        if not batches:
            print("[replay] nothing is missing")
        elif write:
            for rows, run in batches:
                logio.write_run(log_repo, rows, run)
            total = sum(len(rows) for rows, _ in batches)
            print(f"[replay] {total} rows, {logio.commit_and_push(log_repo, f'replay: {total} rows for ' + ' '.join(versions))}")
        else:
            print("[replay] not written (add --write)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--also", nargs="*", default=[], help="further registered versions whose hydro arm is replayed (v1.11)")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--backfill", nargs="+", metavar="VERSION", help="registered versions to give every past Monday run they were trained before")
    parser.add_argument("--write", action="store_true", help="with --backfill: append the rows to the log, commit and push")
    args = parser.parse_args()
    if bool(args.run_id) == bool(args.backfill):
        parser.error("give either --run-id or --backfill")
    settings = run_forecasts.load_settings()
    if args.backfill:
        return run_backfill(Path(settings["log_repo"]), tuple(args.backfill), args.write)
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
