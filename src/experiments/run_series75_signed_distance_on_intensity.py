"""
Series 75 - does the signed distance to payday add anything on top of the triangle intensity, in the Saturday piece?

The signed distance (days to the nearest payday, negative before and positive after, clipped to -7..+7) was tried alone in Series 61 (`mix_signed_distance`, the nearest of the last working day, 25th and 28th) and 64
(`lwd_signed_distance`, the last working day only): level in the backtest, better on the 56 newest days. The triangle intensity (`intensity_sym`) has since taken over most of that ground: in the Saturday piece it is
-16.06 pounds a day before the correction (post model -2.99 after it), against -1.69 for the lwd signed distance, and forecast-log version v1.11 runs it. The open question is whether the signed distance still adds
something ON TOP of the intensity. We expect little: both describe where a day sits in the pay cycle.

The test, built so that "adds something" cannot be an artefact of adding any extra column (Series 63 showed a redundant column moves a seeded fit by a few pounds, and Series 72-73 how large the placebo spread is):

  A      the Saturday piece with `intensity_sym` (v1.11's Saturday piece; Series 70's fits on seeds 200-299, Series 73's on seeds 300-399: reused)
  B1     A plus `sd_lwd`, the signed distance to the nearest last-working-day payday (Series 64's definition)
  B2     A plus `sd_mix`, the signed distance to the nearest of the last working day, 25th and 28th (Series 61's definition)
  P1-12  twelve placebo add-ons: A plus a signed distance built the same way (nearest payday, signed, clipped to -7..+7) but to paydays at the WRONG dates, nine single-day rules (the 3rd to the 10th and the 20th of the month, moved
         back off weekends and bank holidays) and three triples of such days (seed 75). They show what adding a signed-distance-shaped column does when it carries no payday information.

Everything is judged against A, not against the deployed piece. Saturday piece only (the piece v1.11 puts the column in), 585 backtest days, folds and settings as Series 54. 320 fits: B1 and B2 and the 12 placebos on seeds 200-299,
B1 and B2 again on seeds 300-399 (fresh), 20 chunks each, backtest only.

READING (set before any result was seen)
  measures    B minus A: the Saturday piece's change on the 84 Saturdays before the correction (pounds a day; negative = B better), and the post model's change after the correction on all 585 days
  adds value  for B1 or B2: better than at least 11 of the 12 placebo add-ons on BOTH measures, the post model's change at most -0.5, AND on the fresh seeds 300-399 the Saturday piece's change has the same sign and at least half the size
  adds nothing   the Saturday piece's change is above -1.0, or at least 4 of the 12 placebo add-ons are as good on either measure
  in between  reported as it is
  expected    adds nothing (the intensity already carries the pay-cycle position; the signed distance's earlier gain was on the newest days and not in the backtest)
  limits      same days and folds throughout; 12 placebos give a smallest p of 0.08; Saturday piece only; no pass/fail gate; nothing in production changes; v1.11 is not changed by this

Run:  python -m src.experiments.run_series75_signed_distance_on_intensity list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import holidays
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series61_payday_rules as s61
from src.experiments import run_series64_payday_plan_rest as s64
from src.experiments import run_series70_payday_intensity as s70
from src.experiments import run_series73_intensity_stress_tests as s73
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series75"
RESULTS = OUT_DIR / "series75_signed_distance_on_intensity.jsonl"
PIECE = "Saturday"
FRESH_SEEDS = list(range(300, 400))
POOL = [3, 4, 5, 6, 7, 8, 9, 10, 20]
_rng = np.random.default_rng(75)
PLACEBO_RULES = {f"sd_plc{i}": [d] for i, d in enumerate(POOL)}
PLACEBO_RULES.update({f"sd_plc{len(POOL) + i}": sorted(_rng.choice(POOL, size=3, replace=False).tolist()) for i in range(3)})
REAL_RULES = {"sd_lwd": ["last_working_day"], "sd_mix": ["last_working_day", 25, 28]}
REAL = list(REAL_RULES)
PLACEBOS = list(PLACEBO_RULES)
NEW_COLUMNS = REAL + PLACEBOS + [pi.COLUMN]


def signed_distance(dates: pd.Series, rules: list) -> np.ndarray:
    """Days to the nearest payday over `rules`: negative before it, positive after, clipped to -7..+7, a tie going to 'after' (Series 61 and 64's definition; one rule gives the single-rule version)."""
    calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
    pairs = [s61.days_around(dates, rule, calendar) for rule in rules]
    since, to = np.min([p[0] for p in pairs], axis=0), np.min([p[1] for p in pairs], axis=0)
    return np.clip(np.where(since <= to, since, -to), -7, 7)


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = pi.add_payday_intensity(frame, DATE_COL)
    d = out[DATE_COL]
    for name, rules in {**REAL_RULES, **PLACEBO_RULES}.items():
        out[name] = signed_distance(d, rules)
    return out


def piece_features(extra: str) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[PIECE][0] + [pi.COLUMN, extra]


def init_worker75() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []

    def add(variant: str, column: str, seeds: list[int]) -> None:
        features = piece_features(column)
        for i in range(0, len(seeds), s54.BACKTEST_CHUNK):
            chunk = seeds[i:i + s54.BACKTEST_CHUNK]
            out.append({"id": f"backtest_{PIECE}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": PIECE, "variant": variant, "seeds": chunk, "features": features})

    for name in REAL + PLACEBOS:
        add(name, name, SEEDS)
    for name in REAL:
        add(f"{name}_fresh", name, FRESH_SEEDS)
    return out


def mine(done: dict, variant: str, seeds: list[int]) -> np.ndarray:
    rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", PIECE, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(seeds), (variant, per_seed.shape)
    return per_seed.mean(axis=0)


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    sat = dow == 5
    base = s54.load_bags("backtest", s70.RESULTS, {"none": [], "intensity_sym": []}, s54.RESULTS)
    other = {p: base[(p, "none")] for p in s54.PIECES}
    done = read_done(RESULTS)
    done73 = read_done(s73.RESULTS)
    pd.set_option("display.width", 250)

    def package(sat_pred: np.ndarray) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [other["Friday"], sat_pred, other["Sunday"]], default=other["Monday-Thursday"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def against(sat_pred: np.ndarray, reference: np.ndarray) -> dict:
        d = np.abs(sat_pred - a) - np.abs(reference - a)
        diff = np.abs(package(sat_pred) - a) - np.abs(package(reference) - a)
        return {"Saturday piece, before correction": float(d[sat].mean()), "post model, after correction": float(diff.mean()), "Saturdays improved": int((d[sat] < 0).sum())}

    reference = base[(PIECE, "intensity_sym")]  # A, seeds 200-299
    none = base[(PIECE, "none")]
    rows = {name: against(mine(done, name, SEEDS), reference) for name in REAL + PLACEBOS}
    table = pd.DataFrame(rows).T
    table["rules (days of month)"] = [", ".join(map(str, (REAL_RULES | PLACEBO_RULES)[n])) for n in table.index]
    print("A = THE SATURDAY PIECE WITH THE TRIANGLE INTENSITY. EACH ROW IS A PLUS ONE EXTRA COLUMN, minus A (negative = better than A; seeds 200-299)\n")
    print(table.round(2).to_string())
    print(f"\nfor reference, A itself against the deployed Saturday piece: {against(reference, none)}")
    plc = table.loc[PLACEBOS]
    verdicts = {}
    fresh_ref = s73.mine(done73, PIECE, "real_fresh", FRESH_SEEDS)
    for name in REAL:
        row = table.loc[name]
        results = {}
        print(f"\n{name}")
        for measure in ("Saturday piece, before correction", "post model, after correction"):
            as_good = int((plc[measure].astype(float) <= float(row[measure])).sum())
            results[measure] = as_good
            print(f"  {measure}: {float(row[measure]):+.2f}; placebo add-ons at least as good: {as_good} of {len(plc)} (p = {(1 + as_good) / (1 + len(plc)):.3f}); placebo mean {plc[measure].astype(float).mean():+.2f}, best {plc[measure].astype(float).min():+.2f}")
        fresh = against(mine(done, f"{name}_fresh", FRESH_SEEDS), fresh_ref)
        print(f"  on fresh seeds 300-399 (against the intensity-only piece on the same seeds): Saturday piece {fresh['Saturday piece, before correction']:+.2f}, post model {fresh['post model, after correction']:+.2f}")
        sat_change, pkg = float(row["Saturday piece, before correction"]), float(row["post model, after correction"])
        adds = (all(v <= 1 for v in results.values()) and pkg <= -0.5 and fresh["Saturday piece, before correction"] <= 0.5 * sat_change and fresh["Saturday piece, before correction"] < 0)
        nothing = sat_change > -1.0 or any(v >= 4 for v in results.values())
        verdicts[name] = "ADDS VALUE" if adds else ("ADDS NOTHING" if nothing else "IN BETWEEN")
        print(f"  READING SET BEFOREHAND: {verdicts[name]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "add_ons.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(REAL)} real add-ons + {len(PLACEBOS)} placebo add-ons on seeds 200-299 and the {len(REAL)} real ones on fresh seeds 300-399: {len(tasks)} tasks (backtest only, Saturday piece)")
        d, _ = load_protocol()
        frame = add_columns(d)
        assert frame[NEW_COLUMNS].notna().all().all(), "a column has missing values"
        assert np.array_equal(frame["sd_lwd"], s64.add_plan_columns(d)["lwd_signed_distance"]), "sd_lwd differs from Series 64's lwd_signed_distance"
        assert np.array_equal(frame["sd_mix"], s61.add_payday_rule_columns(d)["mix_signed_distance"]), "sd_mix differs from Series 61's mix_signed_distance"
        assert np.allclose(frame[pi.COLUMN], s70.add_intensity_columns(d)["intensity_sym"]), "the intensity differs from Series 70's"
        sat = frame[DATE_COL].dt.dayofweek == 5
        real_intensity = frame.loc[sat, pi.COLUMN]
        print(f"{'column':10s} {'days of month':>14s} {'corr with intensity (Saturdays)':>34s} {'distinct values':>16s}")
        for name in REAL + PLACEBOS:
            rules = (REAL_RULES | PLACEBO_RULES)[name]
            print(f"{name:10s} {', '.join(map(str, rules))[:14]:>14s} {float(np.corrcoef(frame.loc[sat, name], real_intensity)[0, 1]):34.2f} {frame[name].nunique():16d}")
        print("the 12 placebo rule sets are distinct:", len({tuple(v) for v in PLACEBO_RULES.values()}) == len(PLACEBOS))
        print("sd_lwd and sd_mix equal Series 64's and Series 61's columns on every day")
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker75, "series75")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
