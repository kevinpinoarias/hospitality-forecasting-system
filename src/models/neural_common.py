"""
Shared data preparation, training loop, and prediction utilities for the
neural forecasters (LSTM and small Transformer). Both architectures reuse
this exact windowing, scaling, and training loop and differ only in the
model class itself, so any difference in results reflects the architecture,
not inconsistent data handling.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "model_features.csv"

DATE_COL = "date"
TARGET_COL = "total_sales"
DEFAULT_SPLIT_DATE = "2024-12-31"
DEFAULT_WINDOW = 30

# Uses the GPU automatically when one is available. Note this mostly matters
# for larger models/datasets than this project's - on a model this small,
# GPU and CPU training take roughly the same wall-clock time, since the
# overhead of moving data to the GPU can outweigh the tiny amount of actual
# computation per batch.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same final feature set used by the XGBoost model, for a fair comparison.
FEATURES = [
    "forecast_sales",
    "month_sin",
    "day_of_week",
    "day_of_year_cos",
    "day_of_year_sin",
    "day_of_year",
    "month_cos",
    "month",
    "day_of_week_sin",
    "is_bank_holiday",
    "days_to_bank_holiday",
    "days_since_payday",
    "is_payday_window_pm3",
    "is_long_weekend",
    "is_heavy_rain",
]


def set_seed(seed: int) -> None:
    """Fixes every source of randomness used during training (weight
    initialisation, DataLoader shuffling) so a run can be reproduced exactly
    from its logged seed - without this, two runs of identical code and
    hyperparameters can land on meaningfully different results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_data(path: Path = INPUT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return df


class Scaler:
    """
    Mean/std scaler fit on the training split only, applied to both the
    feature matrix and the target. Neural networks train far more reliably
    on normalised inputs than on raw sales figures in the thousands -
    unlike XGBoost, which is scale-invariant and needed none of this.
    """

    def __init__(self) -> None:
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Scaler":
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0)
        self.feature_std[self.feature_std == 0] = 1.0
        self.target_mean = float(y.mean())
        self.target_std = float(y.std()) or 1.0
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feature_mean) / self.feature_std

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.target_std + self.target_mean


def build_windows(
    df: pd.DataFrame,
    features: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Build (window, n_features) sequences and next-day targets.

    Skips any window where the `window` input days plus the target day are
    not fully consecutive calendar days - a handful of dates in this
    dataset (Christmas Day, New Year's Day) have no row at all rather than
    a zero, and a naive row-based window would silently treat those gaps
    as adjacent days.
    """
    feature_matrix = df[features].to_numpy(dtype=np.float32)
    target = df[TARGET_COL].to_numpy(dtype=np.float32)

    X_list, y_list, date_list = [], [], []

    for end_idx in range(window, len(df)):
        span_dates = df[DATE_COL].iloc[end_idx - window : end_idx + 1]
        day_gaps = span_dates.diff().dt.days.iloc[1:]
        if (day_gaps != 1).any():
            continue

        X_list.append(feature_matrix[end_idx - window : end_idx])
        y_list.append(target[end_idx])
        date_list.append(df[DATE_COL].iloc[end_idx])

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    target_dates = pd.Series(date_list)

    return X, y, target_dates


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def prepare_datasets(
    split_date: str = DEFAULT_SPLIT_DATE,
    window: int = DEFAULT_WINDOW,
    features: list[str] = FEATURES,
) -> dict:
    """Load data, build gap-aware windows, split by date, and scale using
    statistics fit on the training split only (no leakage from validation)."""
    df = load_model_data()
    X, y, target_dates = build_windows(df, features, window)

    split_ts = pd.to_datetime(split_date)
    train_mask = (target_dates < split_ts).to_numpy()
    valid_mask = ~train_mask

    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError("Train/validation split produced an empty set.")

    n_features = X.shape[-1]
    scaler = Scaler().fit(X[train_mask].reshape(-1, n_features), y[train_mask])

    def scale(X_subset: np.ndarray, y_subset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, w, f = X_subset.shape
        X_scaled = scaler.transform_X(X_subset.reshape(-1, f)).reshape(n, w, f)
        y_scaled = scaler.transform_y(y_subset)
        return X_scaled.astype(np.float32), y_scaled.astype(np.float32)

    X_train, y_train = scale(X[train_mask], y[train_mask])
    X_valid, y_valid = scale(X[valid_mask], y[valid_mask])

    return {
        "train_dataset": SequenceDataset(X_train, y_train),
        "valid_dataset": SequenceDataset(X_valid, y_valid),
        "valid_dates": target_dates[valid_mask].reset_index(drop=True),
        "y_valid_actual": y[valid_mask],
        "scaler": scaler,
        "n_features": n_features,
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
    }


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> list[float]:
    """Standard PyTorch training loop, shared by both architectures."""
    model.to(DEVICE)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch).squeeze(-1)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        history.append(epoch_loss / len(train_dataset))

    return history


def predict(model: nn.Module, dataset: Dataset, scaler: Scaler) -> np.ndarray:
    """Run the model over a dataset and return predictions in original units."""
    model.to(DEVICE)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            preds_scaled.append(model(X_batch).squeeze(-1).cpu())
    preds_scaled = torch.cat(preds_scaled).numpy()
    return scaler.inverse_transform_y(preds_scaled)
