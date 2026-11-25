"""
The code that MAKES a forecast with the weekend package and the forecast-log versions, in one light place.

It was spread over the experiment scripts (Series 24, 26, 29 and 35) and the forecast log, whose imports pull
in the training and plotting stack (xgboost, scikit-learn, wandb, matplotlib). Moved here unchanged in
November 2026 so the live API can serve the versions without that stack; the experiments and the forecast
log import the same functions from here, so the log, the experiments and the API compute exactly the same
thing. Nothing here may import from src.experiments or src.evaluation (tests/test_forecast_core.py checks).

  bags        a CatBoost bag over seeds, loading it, file fingerprints
  rows        the feature rows for a forecast: calendars, weekend inputs, the Hydro flag, the served rows
  correction  the Saturday and Sunday correction from recent errors
  routing     which piece forecasts which day, and the year-end specialist windows
"""
