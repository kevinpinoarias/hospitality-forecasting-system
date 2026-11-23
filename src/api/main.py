"""
FastAPI entry point for the hospitality forecasting service.

Run locally with:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from src.api.error_ranking import DEFAULT_RANKING_DAYS, MAX_RANKING_DAYS, rank_errors
from src.api.history import HistoryCache
from src.api.inference import ModelService
from src.api.model_versions import STANDARD, VersionService, VersionUnavailable
from src.api.schemas import (
    ErrorRankingResponse, HealthResponse, PredictRequest, PredictResponse, ScenarioPredictions, VersionsResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hospitality_api")

REFRESH_INTERVAL_SECONDS = 24 * 60 * 60

history_cache = HistoryCache()
model_service: ModelService | None = None
version_service: VersionService | None = None  # the model picker; versions load the first time they are picked


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
    global model_service, version_service

    history_cache.refresh()
    model_service = ModelService(history_cache)
    version_service = VersionService(history_cache)
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

    asked = {item.model_version for item in payload.requests if item.model_version not in (None, STANDARD)}
    for choice in sorted(asked):
        if version_service is None or not version_service.is_known(choice):
            raise HTTPException(status_code=422, detail=f"unknown model_version {choice!r}; GET /versions lists the choices")
        if not version_service.is_available(choice):
            raise HTTPException(status_code=422, detail=f"model_version {choice!r} is not installed on this server")

    try:
        results = [
            model_service.predict_one(item.date, item.forecast_sales)
            for item in payload.requests
        ]
        for choice in sorted(asked):
            mine = [i for i, item in enumerate(payload.requests) if item.model_version == choice]
            estimates = {payload.requests[i].date: payload.requests[i].forecast_sales for i in mine if payload.requests[i].forecast_sales is not None}
            answers = version_service.forecast(choice, sorted({payload.requests[i].date for i in mine}), manager_estimates=estimates)
            for i in mine:
                answer = answers[payload.requests[i].date]
                if answer.best is None:  # the version cannot answer this date: the standard figures stay, with the reason
                    results[i].model_version_note = answer.note
                    continue
                results[i].predictions = ScenarioPredictions(dry_scenario=answer.dry, heavy_rain_scenario=answer.heavy, best_estimate=answer.best)
                results[i].model_version = choice
    except VersionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(results=results)


@app.get("/versions", response_model=VersionsResponse)
def versions() -> VersionsResponse:
    """The models a forecast can come from: the standard (live) model and the forecast log's versions,
    each with a one-line legend and whether it is installed on this server."""
    if version_service is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    return VersionsResponse(choices=version_service.menu())


@app.get("/errors/ranking", response_model=ErrorRankingResponse)
def error_ranking(
    kind: Literal["worst", "over", "under", "best"] = "worst",
    forecast: Literal["model", "manager"] = "model",
    limit: int = Query(DEFAULT_RANKING_DAYS, ge=1, le=MAX_RANKING_DAYS),
) -> ErrorRankingResponse:
    """The days a forecast missed by the most: `kind` is the biggest misses
    in either direction (worst), the days it was too high (over) or too low
    (under), or the days it came closest (best); `forecast` is whose forecast
    to judge - the model's or the manager's own."""
    if model_service is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded")

    try:
        return rank_errors(history_cache.df, model_service.backtest_predictions, kind, forecast, limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
