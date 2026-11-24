"""
Backtests of the forecast log's registered versions: what each version would have forecast for every past day,
made the way the live model's past forecasts were made (the rolling-origin protocol: 10 folds, 585 days from
15 January 2025 to 6 September 2026, each fold's models trained only on the days before it, the manager's real
number as the input). The model picker serves these for past dates (src/api/model_versions.py).

How a version is refitted: every piece is fitted again, per fold, from its REGISTERED definition (its inputs
and target, read from the piece's own meta.json) on the same frozen features file the versions were fitted on,
with the same extra columns (the Hydro flag, both payday intensities, last year's sales) and the same rule for
v1.13's masked pieces. `verify` first refits every distinct piece at the registered cut-off and requires its
predictions to equal the registered model's (CatBoost's files are not byte-identical between two fits of the
same model, so predictions, not bytes, are compared). Pieces shared by several versions are fitted once.

Each day is then forecast with the version's own routing (src/forecast_core.routing), v1.13's specialist
windows for the versions that have them, and the Saturday and Sunday correction learned from the earlier folds
of the same backtest (fold 1 has no history and stays uncorrected, as in every experiment). With the manager's
real number the stand-in plays no part, so v1.131 equals v1.13 and v1.141 equals v1.14 here.

  python -m src.forecast_log.version_backtest verify
  python -m src.forecast_log.version_backtest run [--out models/version_backtests.csv]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.experiments import run_series35_holdout as h
from src.experiments.series19_21_common import load_protocol, test_days
from src.features.payday_intensity import add_payday_intensity, add_payday_intensity_fixed
from src.features.year_ago import add_year_ago_sales
from src.forecast_core.bags import Bag, load_engineered
from src.forecast_core.correction import CORRECTION, correct
from src.forecast_core.routing import MASK_PREFIX, Models, route_post, with_year_end_specialists
from src.forecast_core.rows import add_hydro_thu, weekday_names, with_weekend
from src.forecast_log import v1, v1_13

DATE_COL, TARGET_COL = "date", "total_sales"
CHOICES_PATH = PROJECT_ROOT / "config" / "model_versions.json"
MODELS_ROOT = PROJECT_ROOT / "models"
DEFAULT_OUT = MODELS_ROOT / "version_backtests.csv"
FIRST_DAY, LAST_DAY, DAYS = "2025-01-15", "2026-09-06", 585


def choices() -> list[dict]:
    return [c for c in json.loads(CHOICES_PATH.read_text(encoding="utf-8"))["choices"] if c["id"] != "standard"]


def distinct_pieces() -> dict[str, tuple[str, str]]:
    """fingerprint -> (version folder, piece name) for every distinct registered piece."""
    out: dict[str, tuple[str, str]] = {}
    for c in choices():
        for name, fp in c["pieces"].items():
            out.setdefault(fp, (c["version"], name))
    return out


def base_name(piece: str) -> str:
    for prefix in (v1_13.COMPANION_PREFIX, MASK_PREFIX):
        if piece.startswith(prefix):
            return piece[len(prefix):]
    return piece


def feature_frames(full: pd.DataFrame, train_end: pd.Timestamp, train_start: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training rows before `train_end` (plain, and with the weekend inputs), with every extra column any registered piece uses, built as the versions' fits built them."""
    sales = full.set_index(DATE_COL)[TARGET_COL]
    rows = h.complete_rows(full)
    rows = rows[(rows[DATE_COL] < train_end) & ((rows[DATE_COL] >= train_start) if train_start is not None else True)].reset_index(drop=True)
    rows_w = with_weekend(rows, sales)

    def extras(frame: pd.DataFrame) -> pd.DataFrame:
        return add_year_ago_sales(add_payday_intensity_fixed(add_payday_intensity(frame, DATE_COL), DATE_COL), sales, DATE_COL)

    return extras(rows), extras(rows_w)


def fit_piece(version: str, piece: str, plain: pd.DataFrame, weekend: pd.DataFrame) -> Bag:
    meta = json.loads((MODELS_ROOT / version / piece / "meta.json").read_text(encoding="utf-8"))
    frame = weekend if base_name(piece) in v1.WEEKEND_PIECES else plain
    if piece.startswith(MASK_PREFIX):
        frame = v1_13.masked_copy(frame)
    return Bag.fit(frame, meta["features"], meta["target"])


def frozen_full() -> pd.DataFrame:
    if h.file_sha256(v1.TRAINING_FEATURES) != v1.TRAINING_FEATURES_SHA:
        raise RuntimeError(f"{v1.TRAINING_FEATURES} is not the frozen features file the versions were fitted on")
    return add_hydro_thu(load_engineered(v1.TRAINING_FEATURES))


def verify() -> bool:
    """Refit every distinct piece at the registered cut-off; its predictions must equal the registered model's."""
    full = frozen_full()
    plain, weekend = feature_frames(full, v1.FIRST_EXCLUDED_DAY)
    ok = True
    for fp, (version, piece) in distinct_pieces().items():
        started = time.time()
        refit = fit_piece(version, piece, plain, weekend)
        registered = Bag.load(MODELS_ROOT / version / piece)
        frame = weekend if base_name(piece) in v1.WEEKEND_PIECES else plain
        if piece.startswith(MASK_PREFIX):
            frame = v1_13.masked_copy(frame)
        diff = float(np.max(np.abs(refit.predict(frame) - registered.predict(frame))))
        ok &= diff < 1e-6
        print(f"[verify] {version}/{piece}: largest difference {diff:.2e} ({time.time() - started:.0f}s)", flush=True)
    print(f"[verify] {'ALL' if ok else 'NOT ALL'} {len(distinct_pieces())} pieces reproduce the registered models", flush=True)
    return ok


def run(out: Path) -> pd.DataFrame:
    full = frozen_full()
    df, folds = load_protocol()
    days = test_days(df, folds)
    if (len(days), days["date"].min(), days["date"].max()) != (DAYS, FIRST_DAY, LAST_DAY):
        raise RuntimeError(f"the protocol's test days are {len(days)} from {days['date'].min()} to {days['date'].max()}, not the live model's {DAYS} ({FIRST_DAY} to {LAST_DAY})")
    pieces = distinct_pieces()
    sales_known = full.set_index(DATE_COL)[TARGET_COL].asfreq("D")
    all_rows = h.complete_rows(full)
    raw = {c["id"]: [] for c in choices()}
    for fold in folds:
        started = time.time()
        plain, weekend = feature_frames(full, fold.train_end, fold.train_start)
        fitted = {fp: fit_piece(version, piece, plain, weekend) for fp, (version, piece) in pieces.items()}
        test = all_rows[(all_rows[DATE_COL] >= fold.test_start) & (all_rows[DATE_COL] < fold.test_end)].reset_index(drop=True)
        test = add_year_ago_sales(add_payday_intensity_fixed(add_payday_intensity(test, DATE_COL), DATE_COL), full.set_index(DATE_COL)[TARGET_COL], DATE_COL)
        for c in choices():
            models = Models(c["version"], None, {name: fitted[fp] for name, fp in c["pieces"].items()}, None, dict(c["pieces"]), c.get("standin", "S0"))
            forecast = route_post(models, test, sales_known, hydro=c["arm"] == "hydro")
            if c["version"] != "v1":
                forecast = with_year_end_specialists(models, test, sales_known, forecast)
            raw[c["id"]].append(forecast)
        print(f"[run] fold {fold.fold_id}: {len(test)} days, {len(fitted)} pieces refitted ({time.time() - started:.0f}s)", flush=True)

    result = days[["date", "fold_id", "actual"]].copy()
    result["weekday"] = weekday_names(result["date"])
    for choice_id, parts in raw.items():
        forecast = np.concatenate(parts)
        result[choice_id] = correct(result, forecast, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, float_format="%.6f")
    print(f"[run] wrote {out}: {len(result)} days; sha256 {h.file_sha256(out)}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "verify":
        raise SystemExit(0 if verify() else 1)
    run(args.out)


if __name__ == "__main__":
    main()
