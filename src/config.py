"""
Loads and validates config/config.yaml - the one place for the settings that
used to be repeated across the code (paths, split dates, weather thresholds,
model settings, which feature set each model uses).

Validation is strict on purpose: an unknown key, a missing path name or a
feature-set name that does not exist stops the program at start-up with a
message that says what is wrong, instead of the value being silently ignored.

    from src.config import CONFIG
    CONFIG.path("features_dir")      # absolute Path under the project root
    CONFIG.weather.heavy_rain_mm     # 5.0
    CONFIG.features.use["catboost"]  # "v2"

The feature sets themselves (the lists of column names) live in
src/features/build_features.py; this file only holds their names.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
CONFIG_PATH_ENV = "FORECAST_CONFIG"  # point at another file, e.g. in tests

FEATURE_SET_NAMES = ("v1", "v2")
# Every model that picks a feature set must appear under features.use.
FEATURE_USERS = ("xgboost", "neural", "experiments_baseline", "catboost")
REQUIRED_PATHS = (
    "raw_data_dir", "raw_private_dir", "interim_dir", "interim_sales_dir", "interim_labour_dir",
    "processed_sales_dir", "processed_labour_dir", "daily_sales_master", "daily_labour_master",
    "features_dir", "engineered_features", "cache_dir", "results_dir", "figures_dir",
    "experiments_dir", "models_dir", "final_model_dir",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Project(_Strict):
    name: str
    timezone: str
    council: str
    uk_subdivision: str


class Splits(_Strict):
    baseline_split_date: str
    sarimax_split_date: str
    xgboost_split_date: str
    neural_split_date: str

    @field_validator("*")
    @classmethod
    def _is_a_date(cls, value: str) -> str:
        import datetime as dt

        try:
            dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{value!r} is not a YYYY-MM-DD date") from exc
        return value


class Features(_Strict):
    use: dict[str, str]

    @model_validator(mode="after")
    def _known_sets_and_users(self) -> "Features":
        missing = [user for user in FEATURE_USERS if user not in self.use]
        extra = [user for user in self.use if user not in FEATURE_USERS]
        if missing or extra:
            raise ValueError(f"features.use needs exactly {list(FEATURE_USERS)}; missing {missing}, unexpected {extra}")
        for user, name in self.use.items():
            if name not in FEATURE_SET_NAMES:
                raise ValueError(f"features.use.{user} is {name!r}; it must be one of {list(FEATURE_SET_NAMES)}")
        return self


class Weather(_Strict):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    warm_temp_c: float
    hot_scotland_c: float
    heavy_rain_mm: float = Field(gt=0)
    anomaly_window: int = Field(gt=0)


class Sarimax(_Strict):
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    exog_features: list[str]


class XGBoost(_Strict):
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    max_depth: int = Field(gt=0)
    subsample: float = Field(gt=0, le=1)
    colsample_bytree: float = Field(gt=0, le=1)
    objective: str
    random_state: int

    def params(self) -> dict:
        return self.model_dump()


class CatBoost(_Strict):
    iterations: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    depth: int = Field(gt=0)
    n_seeds: int = Field(gt=0)

    def params(self) -> dict:
        """The CatBoost constructor settings (n_seeds is not one of them)."""
        return {"iterations": self.iterations, "learning_rate": self.learning_rate, "depth": self.depth}

    def seeds(self) -> list[int]:
        return list(range(self.n_seeds))


class Analysis(_Strict):
    best_window_days: int = Field(gt=0)
    top_n_error_days: int = Field(gt=0)


class Config(_Strict):
    project: Project
    paths: dict[str, str]
    splits: Splits
    features: Features
    weather: Weather
    sarimax: Sarimax
    xgboost: XGBoost
    catboost: CatBoost
    analysis: Analysis

    @field_validator("paths")
    @classmethod
    def _all_paths_present(cls, paths: dict[str, str]) -> dict[str, str]:
        missing = [name for name in REQUIRED_PATHS if name not in paths]
        extra = [name for name in paths if name not in REQUIRED_PATHS]
        if missing or extra:
            raise ValueError(f"paths is missing {missing} and has unknown {extra}")
        for name, value in paths.items():
            # Judged by both Windows' and Linux's rules, so "C:/data" is refused on a
            # Linux machine (CI) as well as on the Windows one it was written on.
            pure = (PurePosixPath(value), PureWindowsPath(value))
            if any(p.is_absolute() or p.drive or p.root or ".." in p.parts for p in pure):
                raise ValueError(f"paths.{name} must be relative to the project root, not {value!r}")
        return paths

    def path(self, name: str) -> Path:
        """Absolute path for a name under `paths`."""
        return PROJECT_ROOT / self.paths[name]


def load_config(path: Path | str | None = None) -> Config:
    """Read and validate a config file (the project's by default)."""
    path = Path(path or os.environ.get(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


CONFIG = get_config()
