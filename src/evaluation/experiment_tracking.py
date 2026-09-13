"""
Thin wrapper around Weights & Biases so every training/baseline script logs
runs the same way - one comparable row per model in the same W&B project,
rather than each script inventing its own logging shape.
"""

from __future__ import annotations

from pathlib import Path

import wandb

PROJECT_NAME = "hospitality-forecasting"


def log_run(
    model_name: str,
    config: dict,
    metrics: dict,
    figure_path: Path | None = None,
    job_type: str = "train",
) -> None:
    """Log one model's config and metrics as a single W&B run."""
    run = wandb.init(
        project=PROJECT_NAME,
        name=model_name,
        job_type=job_type,
        config=config,
        reinit="finish_previous",
    )
    wandb.log(metrics)

    if figure_path is not None and figure_path.exists():
        wandb.log({"forecast_plot": wandb.Image(str(figure_path))})

    run.finish()
