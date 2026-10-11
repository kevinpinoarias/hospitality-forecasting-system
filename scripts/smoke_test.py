"""
Check that a running API container actually works, using only the standard
library so it needs nothing installed.

Waits for /health to report ok, then asks /predict for a day inside the
backtest period and checks the answer is a sensible number.
Run:  python scripts/smoke_test.py http://localhost:8000
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

PREDICT_DATE = "2025-06-14"  # inside the backtest period, so no weather download is needed
SANE_RANGE_GBP = (1_000, 40_000)  # daily sales are about 6,000-16,000
START_UP_SECONDS = 90


def call(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def wait_for_health(base: str) -> dict:
    deadline = time.time() + START_UP_SECONDS
    while True:
        try:
            health = call(f"{base}/health")
            if health.get("status") == "ok":
                return health
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.time() > deadline:
            raise SystemExit(f"SMOKE TEST FAILED: /health not ok after {START_UP_SECONDS}s")
        time.sleep(2)


def main(base: str) -> None:
    health = wait_for_health(base)
    print(f"/health ok - model {health['model']}, {health['historical_rows']} historical rows")

    result = call(f"{base}/predict", {"requests": [{"date": PREDICT_DATE}]})["results"][0]
    estimate = result["predictions"]["best_estimate"]
    print(f"/predict {PREDICT_DATE}: best estimate {estimate:.0f}, source {result['prediction_source']}")

    low, high = SANE_RANGE_GBP
    if not low <= estimate <= high:
        raise SystemExit(f"SMOKE TEST FAILED: {estimate:.0f} is outside {low}-{high}")
    if result["prediction_source"] != "made_before_the_day":
        raise SystemExit(f"SMOKE TEST FAILED: unexpected prediction_source {result['prediction_source']}")

    ranking = call(f"{base}/errors/ranking?kind=worst&limit=3")
    if len(ranking["days"]) != 3:
        raise SystemExit("SMOKE TEST FAILED: /errors/ranking did not return 3 days")
    print("/errors/ranking ok")
    print("Smoke test passed.")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000")
