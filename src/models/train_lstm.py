"""
Trains and evaluates an LSTM neural forecaster, as a comparison model
against the existing XGBoost/SARIMAX baselines.

Unlike XGBoost (one row of features per day), this model reads a window of
recent days directly and learns temporal patterns itself, instead of
relying on hand-engineered lag/rolling features.

Outputs
-------
- models/lstm_model.pt
- reports/results/lstm_metrics.csv
- reports/results/lstm_predictions.csv
- reports/figures/lstm_forecast_vs_actual.png
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.evaluation.metrics import regression_metrics
from src.evaluation.plots import plot_forecast_vs_actual
from src.evaluation.experiment_tracking import log_run
from src.models.neural_common import (
    DATE_COL,
    DEFAULT_SPLIT_DATE,
    DEFAULT_WINDOW,
    FEATURES,
    TARGET_COL,
    predict,
    prepare_datasets,
    set_seed,
    train_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


class LSTMForecaster(nn.Module):
    """Reads a window of days, predicts the next day's total sales."""

    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # the hidden state after the final day in the window
        return self.fc(last_step)


def run_lstm(
    split_date: str = DEFAULT_SPLIT_DATE,
    window: int = DEFAULT_WINDOW,
    seed: int = 42,
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    data = prepare_datasets(split_date=split_date, window=window, features=FEATURES)

    model = LSTMForecaster(n_features=data["n_features"])
    history = train_model(
        model,
        data["train_dataset"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
    )

    preds = predict(model, data["valid_dataset"], data["scaler"])
    metrics = regression_metrics(data["y_valid_actual"], preds)

    metrics_df = pd.DataFrame([{"model": "lstm", **metrics}])
    predictions_df = pd.DataFrame({
        DATE_COL: data["valid_dates"],
        TARGET_COL: data["y_valid_actual"],
        "lstm_pred": preds,
    })

    figure_path = None
    if save_outputs:
        torch.save(model.state_dict(), MODEL_DIR / "lstm_model.pt")

        metrics_path = RESULTS_DIR / "lstm_metrics.csv"
        predictions_path = RESULTS_DIR / "lstm_predictions.csv"
        metrics_df.to_csv(metrics_path, index=False)
        predictions_df.to_csv(predictions_path, index=False)

        figure_path = FIGURES_DIR / "lstm_forecast_vs_actual.png"
        plot_forecast_vs_actual(
            df=predictions_df,
            date_col=DATE_COL,
            actual_col=TARGET_COL,
            pred_col="lstm_pred",
            title="LSTM Forecast vs Actual Sales",
            output_path=figure_path,
            pred_label="LSTM Forecast",
        )

    # Canonical run (save_outputs=True) is named "lstm" and feeds the model
    # comparison script. Extra seed runs are named separately so they don't
    # overwrite it in W&B, and exist purely to show the true run-to-run
    # spread rather than presenting one seed's result as definitive.
    log_run(
        model_name="lstm" if save_outputs else f"lstm_seed{seed}",
        config={
            "split_date": split_date,
            "window": window,
            "seed": seed,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "n_train": data["n_train"],
            "n_valid": data["n_valid"],
            "final_train_loss": history[-1],
        },
        metrics=metrics,
        figure_path=figure_path,
        job_type="train" if save_outputs else "seed_check",
    )

    print(f"LSTM (seed={seed}) MAE:  {metrics['MAE']:.2f}")
    print(f"LSTM (seed={seed}) RMSE: {metrics['RMSE']:.2f}")
    print(f"LSTM (seed={seed}) MAPE: {metrics['MAPE']:.2f}%")

    return predictions_df, metrics_df


if __name__ == "__main__":
    run_lstm()
