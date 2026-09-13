"""
FastAPI entry point for the hospitality forecasting service.

Run locally with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.api.history import HistoryCache
from src.api.inference import ModelService
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hospitality_api")

REFRESH_INTERVAL_SECONDS = 24 * 60 * 60

history_cache = HistoryCache()
model_service: ModelService | None = None


async def _refresh_loop() -> None:
    """Reloads the historical dataset once a day so the history-based inputs
    and fallback lookups stay current if the pipeline is re-run with new data."""
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        try:
            history_cache.refresh()
            logger.info("History cache refreshed: %d rows", len(history_cache.df))
        except Exception:
            logger.exception("Failed to refresh history cache")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_service

    history_cache.refresh()
    model_service = ModelService(history_cache)
    logger.info(
        "Startup complete: model loaded, %d historical rows cached",
        len(history_cache.df),
    )

    refresh_task = asyncio.create_task(_refresh_loop())
    yield
    refresh_task.cancel()


app = FastAPI(title="Hospitality Forecasting API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    return HealthResponse(
        status="ok",
        model=model_service.model_name,
        model_trained_through=model_service.trained_through,
        historical_rows=len(history_cache.df),
        history_last_refreshed=history_cache.last_refreshed.isoformat(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    try:
        results = [
            model_service.predict_one(item.date, item.forecast_sales)
            for item in payload.requests
        ]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(results=results)
