"""
The model picker's back end: forecasts from the forecast log's registered versions, beside the standard model.

The choices, their legends and the fingerprints of their files are in config/model_versions.json (made by
scripts/export_model_versions.py from the log's registry). A version is loaded the first time someone picks
it, never at start-up, and every file is checked against its registered fingerprint first: a version whose
files are missing or differ is listed as unavailable and never served. Files shared by two versions (v1.131
uses v1.13's pieces, v1.141 v1.14's) are loaded once.

The forecast itself is the forecast log's own code (src/forecast_core), so a version answers here exactly as
it does in the log: the same rows, the same routing, the same Saturday and Sunday correction, the same
stand-in for the manager's number. Two things are added for the chat, worked out the way the live API does
them for the standard model: the dry and heavy-rain scenarios (the same rows with the rain swapped), and a
manager's estimate given by the person asking takes the stand-in's place.

Days the versions do not cover yet:
- dates up to the end of the data: the versions have no forecast made before the day (their backtests are
  not run yet), so the standard model answers and says so;
- Thursdays beyond the Hydro list: whether there is a show is unknown, so a model with the Hydro flag does
  not forecast them (the forecast log's rule), and the standard model answers and says so.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.weather import DRY_MM, HEAVY_RAIN_MM, fetch_live_weather, live_weather_end
from src.config import PROJECT_ROOT
from src.features import year_ago
from src.features.payday_intensity import add_payday_intensity, add_payday_intensity_fixed
from src.features.serving_features import typical_heavy_rain_mm
from src.forecast_core.bags import Bag, file_sha256, fingerprint
from src.forecast_core.correction import correct_new_block, load_correction_history
from src.forecast_core.routing import Models, route_post, with_year_end_specialists
from src.forecast_core.rows import add_hydro_thu, sales_known_series, served_rows
from src.forecast_log import standin as standin_module
from src.forecast_log.inputs import hydro_list_through, unknown_hydro_thursdays

DATE_COL = "date"
STANDARD = "standard"
CHOICES_PATH = PROJECT_ROOT / "config" / "model_versions.json"
# Where the registered files are: models/<version>/<piece>/ and data/training/<version>_correction_history.csv,
# as in the forecast log's runner folder. The image build puts them there; both can be pointed elsewhere.
MODELS_ROOT = Path(os.environ.get("MODEL_VERSIONS_DIR", PROJECT_ROOT / "models"))
CORRECTION_DIR = Path(os.environ.get("CORRECTION_HISTORY_DIR", PROJECT_ROOT / "data" / "training"))


class VersionUnavailable(RuntimeError):
    """A version whose registered files are missing here or do not match their fingerprints."""


@dataclass
class VersionForecast:
    best: float | None
    dry: float | None
    heavy: float | None
    note: str | None = None


def load_choices(path: Path = CHOICES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["choices"]


def _with_rain(rows: pd.DataFrame, rain_mm) -> pd.DataFrame:
    """The same rows with another amount of rain: only the rain-derived inputs depend on it (as in src/api/inference.py)."""
    out = rows.copy()
    out["rain_mm"] = rain_mm
    out["is_heavy_rain"] = (out["rain_mm"] > HEAVY_RAIN_MM).astype(int)
    out["is_dry_day"] = (out["rain_mm"] < DRY_MM).astype(int)
    return out


class VersionService:
    def __init__(self, history, models_root: Path = MODELS_ROOT, correction_dir: Path = CORRECTION_DIR, choices: list[dict] | None = None) -> None:
        self.history = history
        self.models_root, self.correction_dir = Path(models_root), Path(correction_dir)
        self.choices = {c["id"]: c for c in (choices if choices is not None else load_choices())}
        self._lock = threading.Lock()
        self._bags: dict[str, Bag] = {}  # by fingerprint, so two versions sharing a file share the loaded model
        self._loaded: dict[str, tuple[Models, tuple[pd.DataFrame, np.ndarray]]] = {}

    # ----------------------------------------------------------------- the menu

    def is_known(self, choice_id: str) -> bool:
        return choice_id in self.choices

    def is_available(self, choice_id: str) -> bool:
        """The standard model always is; a version only if all its registered files are here (their contents are checked when it loads)."""
        if choice_id == STANDARD:
            return True
        c = self.choices.get(choice_id)
        if c is None:
            return False
        pieces_here = all((self.models_root / c["version"] / name / "meta.json").exists() for name in c["pieces"])
        return pieces_here and (self.correction_dir / f"{c['version']}_correction_history.csv").exists()

    def menu(self) -> list[dict]:
        keys = ("id", "label", "legend", "status", "registered")
        return [{**{k: c.get(k) for k in keys}, "available": self.is_available(c["id"])} for c in self.choices.values()]

    # ----------------------------------------------------------------- loading

    def _load(self, choice_id: str) -> tuple[Models, tuple[pd.DataFrame, np.ndarray]]:
        with self._lock:
            if choice_id in self._loaded:
                return self._loaded[choice_id]
            c = self.choices[choice_id]
            if not self.is_available(choice_id):
                raise VersionUnavailable(f"the files of {choice_id} are not installed on this server")
            pieces, prints = {}, {}
            for name, expected in c["pieces"].items():
                folder = self.models_root / c["version"] / name
                got = fingerprint(folder)
                if got != expected:
                    raise VersionUnavailable(f"{choice_id}: model file {name} does not match its registered fingerprint")
                if got not in self._bags:
                    self._bags[got] = Bag.load(folder)
                pieces[name], prints[name] = self._bags[got], got
            history_file = self.correction_dir / f"{c['version']}_correction_history.csv"
            if file_sha256(history_file) != c["correction_history_sha256"]:
                raise VersionUnavailable(f"{choice_id}: the correction history does not match its registered fingerprint")
            loaded = (Models(c["version"], None, pieces, None, prints, c.get("standin", "S0")), load_correction_history(history_file))
            self._loaded[choice_id] = loaded
            return loaded

    # ----------------------------------------------------------------- forecasting

    def _weather(self, last_known: pd.Timestamp, today: dt.date) -> pd.DataFrame:
        """The live weather for the days after the data, as the forecast log fetches it (an empty frame on failure: historical averages)."""
        weather, _ = fetch_live_weather((last_known + pd.Timedelta(days=1)).date(), live_weather_end(today))
        weather = weather.copy()
        weather[DATE_COL] = pd.to_datetime(weather[DATE_COL]).dt.normalize()
        return weather.reset_index(drop=True)

    def forecast(self, choice_id: str, dates: list[dt.date], today: dt.date | None = None,
                 manager_estimates: dict[dt.date, float] | None = None) -> dict[dt.date, VersionForecast]:
        """The version's forecasts (best estimate, dry, heavy rain) for each date, or a note saying why the standard model must answer instead."""
        today = today or dt.datetime.now(dt.timezone.utc).date()
        c = self.choices[choice_id]
        last_known = pd.Timestamp(self.history.max_date)
        out: dict[dt.date, VersionForecast] = {}
        for d in dates:
            if pd.Timestamp(d) <= last_known:
                out[d] = VersionForecast(None, None, None, f"{choice_id} was not running on this date, so the standard model answers (the versions have no forecasts made before past days yet).")
        hydro = c["arm"] == "hydro"
        if hydro:
            for d in unknown_hydro_thursdays([d for d in dates if d not in out], hydro_list_through()):
                out[d] = VersionForecast(None, None, None, f"{choice_id} uses the Hydro show list, which does not reach this Thursday yet, so the standard model answers.")
        targets = sorted(d for d in dates if d not in out)
        if not targets:
            return out

        models, (hist_days, hist_pred) = self._load(choice_id)
        full = self.history.df
        stamps = [pd.Timestamp(d) for d in targets]
        sales_known = sales_known_series(full, last_known, max(stamps))
        rows = add_hydro_thu(served_rows(full, last_known, stamps, "v2", self._weather(last_known, today)))
        extra_version = c["version"] != "v1"
        if extra_version:  # as forecast_days builds the further versions' rows
            rows = add_payday_intensity_fixed(add_payday_intensity(rows, DATE_COL), DATE_COL)
            rows[year_ago.COLUMN] = year_ago.year_ago_sales(rows[DATE_COL], sales_known).to_numpy()
            if models.standin == standin_module.NAME:
                rows["forecast_sales"] = standin_module.hybrid_standin(full, last_known, rows[DATE_COL])
        if manager_estimates:
            rows["forecast_sales"] = [manager_estimates.get(d, v) for d, v in zip(targets, rows["forecast_sales"].to_numpy())]

        def run(frame: pd.DataFrame) -> np.ndarray:
            raw = route_post(models, frame, sales_known, hydro=hydro)
            if extra_version:
                raw = with_year_end_specialists(models, frame, sales_known, raw)
            return correct_new_block(hist_days, hist_pred, stamps, raw)

        rain = rows["rain_mm"].to_numpy(dtype=float)
        heavy_mm = np.where(rain > HEAVY_RAIN_MM, rain, typical_heavy_rain_mm(full))
        best, dry, heavy = run(rows), run(_with_rain(rows, 0.0)), run(_with_rain(rows, heavy_mm))
        for i, d in enumerate(targets):
            out[d] = VersionForecast(float(best[i]), float(dry[i]), float(heavy[i]))
        return out
