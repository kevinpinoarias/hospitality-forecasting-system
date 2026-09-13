"""
Trains and evaluates a small Transformer neural forecaster, as a second
comparison model alongside the LSTM, XGBoost, and SARIMAX.

Shares the exact same data windowing/scaling/training loop as train_lstm.py
(see src/models/neural_common.py) - only the model architecture differs.

Outputs
-------
- models/transformer_model.pt
- reports/results/transformer_metrics.csv
- reports/results/transformer_predictions.csv
- reports/figures/transformer_forecast_vs_actual.png
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

D_MODEL = 32
N_HEAD = 4
NUM_LAYERS = 1
DIM_FEEDFORWARD = 64
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


class TransformerForecaster(nn.Module):
    """
    Reads a window of days all at once (via self-attention, not step by
    step) and predicts the next day's total sales.

    A Transformer has no inherent sense of sequence order - attention looks
    at every position simultaneously - so a learned positional embedding is
    added to each day's representation to tell the model "this is day 1 of
    the window, this is day 2," etc.
    """

    def __init__(
        self,
        n_features: int,
        window: int = DEFAULT_WINDOW,
        d_model: int = D_MODEL,
        nhead: int = N_HEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, n_features)
        x = self.input_proj(x) + self.pos_embedding
        encoded = self.encoder(x)
        last_step = encoded[:, -1, :]  # the representation of the most recent day
        return self.fc(last_step)


def run_transformer(
    split_date: str = DEFAULT_SPLIT_DATE,
    window: int = DEFAULT_WINDOW,
    seed: int = 42,
    save_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    data = prepare_datasets(split_date=split_date, window=window, features=FEATURES)

    model = TransformerForecaster(n_features=data["n_features"], window=window)
    history = train_model(
        model,
        data["train_dataset"],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
    )

    preds = predict(model, data["valid_dataset"], data["scaler"])
    metrics = regression_metrics(data["y_valid_actual"], preds)

    metrics_df = pd.DataFrame([{"model": "transformer", **metrics}])
    predictions_df = pd.DataFrame({
        DATE_COL: data["valid_dates"],
        TARGET_COL: data["y_valid_actual"],
        "transformer_pred": preds,
    })

    figure_path = None
    if save_outputs:
        torch.save(model.state_dict(), MODEL_DIR / "transformer_model.pt")

        metrics_path = RESULTS_DIR / "transformer_metrics.csv"
        predictions_path = RESULTS_DIR / "transformer_predictions.csv"
        metrics_df.to_csv(metrics_path, index=False)
        predictions_df.to_csv(predictions_path, index=False)

        figure_path = FIGURES_DIR / "transformer_forecast_vs_actual.png"
        plot_forecast_vs_actual(
            df=predictions_df,
            date_col=DATE_COL,
            actual_col=TARGET_COL,
            pred_col="transformer_pred",
            title="Transformer Forecast vs Actual Sales",
            output_path=figure_path,
            pred_label="Transformer Forecast",
        )

    log_run(
        model_name="transformer" if save_outputs else f"transformer_seed{seed}",
        config={
            "split_date": split_date,
            "window": window,
            "seed": seed,
            "d_model": D_MODEL,
            "nhead": N_HEAD,
            "num_layers": NUM_LAYERS,
            "dim_feedforward": DIM_FEEDFORWARD,
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

    print(f"Transformer (seed={seed}) MAE:  {metrics['MAE']:.2f}")
    print(f"Transformer (seed={seed}) RMSE: {metrics['RMSE']:.2f}")
    print(f"Transformer (seed={seed}) MAPE: {metrics['MAPE']:.2f}%")

    return predictions_df, metrics_df


if __name__ == "__main__":
    run_transformer()
