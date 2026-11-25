"""
src/forecast_core holds the code that makes a version's forecast, moved out of the experiment scripts so the
live API can use it. These tests keep it light (none of the training or plotting stack, nothing from the
experiments) and check the old modules still hand out the very same objects, so the forecast log, the
experiments and the API cannot drift apart.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEAVY = ("xgboost", "sklearn", "wandb", "matplotlib", "torch", "statsmodels", "scipy")


def test_importing_the_core_pulls_in_no_training_or_plotting_stack():
    code = (
        "import sys; import src.forecast_core.bags, src.forecast_core.rows, src.forecast_core.correction, src.forecast_core.routing; "
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]; "
        "exp = [m for m in sys.modules if m.startswith(('src.experiments', 'src.evaluation'))]; "
        "print(heavy, exp); sys.exit(1 if heavy or exp else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_old_modules_hand_out_the_same_objects():
    from src.experiments import run_package_validation as pv
    from src.experiments import run_series24_weekend_correction as s24
    from src.experiments import run_series26_weekend_features as s26
    from src.experiments import run_series35_holdout as h
    from src.forecast_core import bags, correction, routing, rows
    from src.forecast_log import predict, v1, v1_13

    assert h.Bag is bags.Bag and h.fingerprint is bags.fingerprint
    assert h.correct_new_block is correction.correct_new_block and s24.correct is correction.correct
    assert pv.CORRECTION is correction.CORRECTION == {"half_life_weeks": 8, "strength": 0.5, "scope": ("Saturday", "Sunday")}
    assert s26.add_weekend_features is rows.add_weekend_features and s26.VARIANTS is rows.WEEKEND_VARIANTS
    assert h.EXTRA is rows.EXTRA and v1.add_hydro_thu is rows.add_hydro_thu
    assert predict.Models is routing.Models and predict.route_post is routing.route_post and predict.served_rows is rows.served_rows
    assert v1_13.in_window is routing.in_window and v1_13.INPUTS is routing.INPUTS


def test_the_year_end_windows():
    import pandas as pd

    from src.forecast_core.routing import COMPANION_WINDOW, MASK_WINDOW, in_window

    days = pd.date_range("2026-12-24", "2027-01-10")
    companion = [d.strftime("%m-%d") for d, on in zip(days, in_window(days, *COMPANION_WINDOW)) if on]
    masked = [d.strftime("%m-%d") for d, on in zip(days, in_window(days, *MASK_WINDOW)) if on]
    assert companion == ["12-26", "12-27", "12-28", "12-29", "12-30", "12-31"]
    assert masked == [f"01-0{i}" for i in range(1, 9)]
