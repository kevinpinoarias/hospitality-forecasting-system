"""
Series 73 - three stress tests of the triangle payday intensity in the Saturday piece: pay rules at the wrong dates, one rule at a time, and fresh seeds for the real column.

Series 72 found the real `intensity_sym` better than 16 placebo columns in the Saturday piece (Saturday piece -16.06 pounds a day before the correction, post model -2.99 after it, against a best placebo of
-12.76 and -1.65), but its placebos were not clean nulls (a shift of 7 or 14 days often lands near another rule's real payday), 16 placebos give a smallest p of 0.06, and the real column was fitted on the
same seeds as the placebos. These three tests tighten that. All use the Saturday piece (the piece forecast-log version v1.11 puts the column in), 585 backtest days, folds and settings as Series 54.

  1. Cleaner placebos: pay rules at the wrong dates.   16 columns built exactly like the real one (four rules, the same triangle of width 4, the same weights 0.34 / 0.30 / 0.30 / 0.18, the same rolling back from weekends
     and Scottish bank holidays) but with the four paydays on days of the month drawn from {3, 4, 5, 6, 7, 8, 9, 10, 20}, which keeps them clear of the real heat (the 15th rule's 11-19 and the 22-31 and 1-2 of the
     last working day, the 25th and the 28th). Seed 73 draws the sets. This is the null "four pay rules, wrong dates", fitted on seeds 200-299 like the real column. Smallest p with 16 placebos: 0.06; pooled with
     Series 72's 16 placebos, 0.03.
  2. One rule at a time.   The real triangle built from the last working day only, the 25th only, the 28th only, the 15th only, and the three month-end rules without the 15th (`month_end3`), each with its survey
     weight. It shows which pay dates carry the gain.
  3. Fresh seeds for the real column.   The deployed Saturday piece and the Saturday piece with `intensity_sym`, and the same two for the raw Saturday fallback, refitted on seeds 300-399, which no earlier series used.

READING (set before any result was seen)
  1 supports payday dates   the real column is better than at least 15 of the 16 wrong-date placebos on BOTH measures (the Saturday piece's change before the correction on the 84 Saturdays, and the post model's change after
                            the correction on all 585 days)
  1 generic effect          at least 4 of the 16 wrong-date placebos are as good as the real column on either measure
  2 month-end paydays carry it   `month_end3` keeps at least 80% of the real column's gain on both measures, and `only_15` gains less than each of `only_lwd`, `only_25` and `only_28`
  2 the 15th carries it     `only_15` gains at least as much as the best of the other three single rules: the signal is not about month-end pay (Series 61 found the 15th flat)
  3 replicates              on seeds 300-399 the Saturday piece's change is at least 8.0 pounds better than the deployed Saturday piece and the post model's change is at least 1.5 better
  3 seed luck               the Saturday piece's change is less than 4.0 better, or the post model's less than 0.75 better
  expected    1 supports; 2 the month-end rules carry it; 3 replicates (from Series 72)
  limits      same days and folds throughout; 16 placebos is a rough null; the gain is about 0.3% of the daily error; no pass/fail gate; nothing in production changes

Run:  python -m src.experiments.run_series73_intensity_stress_tests list | run [--workers 10] | report
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
from src.experiments import run_series70_payday_intensity as s70
from src.experiments import run_series72_intensity_placebo as s72
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import SEEDS
from src.experiments.series19_21_common import _STATE, DATE_COL, load_protocol, read_done, run_parallel, test_days
from src.features import payday_intensity as pi
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series73"
RESULTS = OUT_DIR / "series73_intensity_stress_tests.jsonl"
PIECE, FALLBACK = "Saturday", "Saturday fallback (raw)"
FRESH_SEEDS = list(range(300, 400))
POOL = [3, 4, 5, 6, 7, 8, 9, 10, 20]
N_WRONG = 16
WEIGHTS = [pi.SHARES[rule] for rule in pi.RULES]  # 0.34, 0.30, 0.30, 0.18, in the order the real rules are listed
_rng = np.random.default_rng(73)
WRONG = {f"wd_plc{i}": sorted(_rng.choice(POOL, size=4, replace=False).tolist()) for i in range(N_WRONG)}
RULE_ONLY = {"only_lwd": ["last_working_day"], "only_25": [25], "only_28": [28], "only_15": [15], "month_end3": ["last_working_day", 25, 28]}
NEW_COLUMNS = list(WRONG) + list(RULE_ONLY) + [pi.COLUMN]


def piece_features(piece: str, column: str | None) -> list[str]:
    return list(s54.FINAL_FEATURES) + s54.PIECES[piece][0] + ([column] if column else [])


def triangle(dates, rules_weights: list[tuple]) -> np.ndarray:
    calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
    total = np.zeros(len(dates))
    for rule, weight in rules_weights:
        total += weight * np.clip(1.0 - pi.distance_to_nearest_payday(dates, rule, calendar) / pi.WIDTH, 0.0, None)
    return total


def add_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    d = out[DATE_COL]
    for name, days in WRONG.items():
        out[name] = triangle(d, list(zip(days, WEIGHTS)))
    for name, rules in RULE_ONLY.items():
        out[name] = triangle(d, [(rule, pi.SHARES[rule]) for rule in rules])
    return pi.add_payday_intensity(out, DATE_COL)


def init_worker73() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []

    def add(piece: str, variant: str, column: str | None, seeds: list[int]) -> None:
        features = piece_features(piece, column)
        for i in range(0, len(seeds), s54.BACKTEST_CHUNK):
            chunk = seeds[i:i + s54.BACKTEST_CHUNK]
            out.append({"id": f"backtest_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": "backtest", "piece": piece, "variant": variant, "seeds": chunk, "features": features})

    for name in list(WRONG) + list(RULE_ONLY):
        add(PIECE, name, name, SEEDS)
    for piece in (PIECE, FALLBACK):
        add(piece, "none_fresh", None, FRESH_SEEDS)
        add(piece, "real_fresh", pi.COLUMN, FRESH_SEEDS)
    return out


def mine(done: dict, piece: str, variant: str, seeds: list[int]) -> np.ndarray:
    rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == ("backtest", piece, variant)]
    per_seed = np.concatenate(rows, axis=0)
    assert per_seed.shape[0] == len(seeds), (piece, variant, per_seed.shape)
    return per_seed.mean(axis=0)


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow, fold = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy(), days["fold_id"].to_numpy()
    sat = dow == 5
    y25, y26 = (days["date"] < "2026-01-01").to_numpy(), (days["date"] >= "2026-01-01").to_numpy()
    base = s54.load_bags("backtest", s70.RESULTS, {"none": [], "intensity_sym": []}, s54.RESULTS)
    other = {p: base[(p, "none")] for p in s54.PIECES}
    done = read_done(RESULTS)
    pd.set_option("display.width", 250)

    def package(sat_pred: np.ndarray) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [other["Friday"], sat_pred, other["Sunday"]], default=other["Monday-Thursday"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    def measures(sat_pred: np.ndarray, sat_none: np.ndarray) -> dict:
        d = np.abs(sat_pred - a) - np.abs(sat_none - a)
        diff = np.abs(package(sat_pred) - a) - np.abs(package(sat_none) - a)
        return {"Saturday piece, before correction": float(d[sat].mean()), "post model, after correction": float(diff.mean()), "Saturdays improved": int((d[sat] < 0).sum()),
                "2025": float(diff[y25].mean()), "2026": float(diff[y26].mean()), "folds better": int(sum(diff[fold == k].mean() < 0 for k in range(10)))}

    none_orig, real = base[(PIECE, "none")], base[(PIECE, "intensity_sym")]
    real_m = measures(real, none_orig)
    # ---- part 1
    wrong_rows = {name: measures(mine(done, PIECE, name, SEEDS), none_orig) for name in WRONG}
    t1 = pd.DataFrame([{"column": "REAL intensity_sym", "days of month": "28/25/last/15", **real_m}] +
                      [{"column": n, "days of month": "/".join(map(str, WRONG[n])), **m} for n, m in wrong_rows.items()])
    print("PART 1. THE REAL COLUMN AGAINST 16 PLACEBOS WITH THE PAY RULES AT THE WRONG DATES (Saturday piece, seeds 200-299; negative = better)\n")
    print(t1.sort_values("Saturday piece, before correction").round(2).to_string(index=False))
    plc = pd.DataFrame(wrong_rows).T
    s72_bags = s72.saturday_bags("backtest")
    s72_rows = {k: measures(v, s72_bags["none"]) for k, v in s72_bags.items() if k.startswith("plc_")}
    pooled = pd.concat([plc, pd.DataFrame(s72_rows).T])
    verdict1 = {}
    for measure in ("Saturday piece, before correction", "post model, after correction"):
        as_good = int((plc[measure].astype(float) <= real_m[measure]).sum())
        as_good_pooled = int((pooled[measure].astype(float) <= real_m[measure]).sum())
        verdict1[measure] = as_good
        print(f"\n{measure}: the real column ({real_m[measure]:+.2f}); wrong-date placebos at least as good: {as_good} of {len(plc)} (p = {(1 + as_good) / (1 + len(plc)):.3f}); "
              f"pooled with Series 72's {len(s72_rows)} placebos: {as_good_pooled} of {len(pooled)} (p = {(1 + as_good_pooled) / (1 + len(pooled)):.3f}); "
              f"wrong-date mean {plc[measure].astype(float).mean():+.2f}, best {plc[measure].astype(float).min():+.2f}")
    print("  READING 1:", "SUPPORTS PAYDAY DATES" if all(v <= 1 for v in verdict1.values()) else ("GENERIC EFFECT" if any(v >= 4 for v in verdict1.values()) else "IN BETWEEN"), verdict1)
    # ---- part 2
    rule_rows = {"REAL (all four rules)": real_m}
    for name in RULE_ONLY:
        rule_rows[name] = measures(mine(done, PIECE, name, SEEDS), none_orig)
    t2 = pd.DataFrame(rule_rows).T
    for col, tag in (("Saturday piece, before correction", "share of real gain, Saturday piece"), ("post model, after correction", "share of real gain, post model")):
        t2[tag] = t2[col].astype(float) / real_m[col]
    print("\n\nPART 2. ONE PAY RULE AT A TIME (Saturday piece, seeds 200-299; negative = better; a share of 1.00 is as much gain as all four rules)\n")
    print(t2.drop(columns=["2025", "2026", "folds better"]).round(2).to_string())
    sat_gain = {n: float(rule_rows[n]["Saturday piece, before correction"]) for n in ("only_lwd", "only_25", "only_28", "only_15")}
    pkg_gain = {n: float(rule_rows[n]["post model, after correction"]) for n in ("only_lwd", "only_25", "only_28", "only_15")}
    me3_keeps = all(float(rule_rows["month_end3"][m]) <= 0.8 * real_m[m] for m in ("Saturday piece, before correction", "post model, after correction"))
    fifteenth_weakest = all(sat_gain["only_15"] > sat_gain[n] and pkg_gain["only_15"] > pkg_gain[n] for n in ("only_lwd", "only_25", "only_28"))
    fifteenth_best = sat_gain["only_15"] <= min(sat_gain[n] for n in ("only_lwd", "only_25", "only_28")) and pkg_gain["only_15"] <= min(pkg_gain[n] for n in ("only_lwd", "only_25", "only_28"))
    print("  READING 2:", "MONTH-END PAYDAYS CARRY IT" if (me3_keeps and fifteenth_weakest) else ("THE 15TH CARRIES IT" if fifteenth_best else "IN BETWEEN"),
          f"(month_end3 keeps >= 80% of the gain on both measures: {me3_keeps}; the 15th alone gains least: {fifteenth_weakest})")
    # ---- part 3
    fresh_none, fresh_real = mine(done, PIECE, "none_fresh", FRESH_SEEDS), mine(done, PIECE, "real_fresh", FRESH_SEEDS)
    fm = measures(fresh_real, fresh_none)
    fb_none, fb_real = mine(done, FALLBACK, "none_fresh", FRESH_SEEDS), mine(done, FALLBACK, "real_fresh", FRESH_SEEDS)
    fb_change = float((np.abs(fb_real - a) - np.abs(fb_none - a))[sat].mean())
    print("\n\nPART 3. THE REAL COLUMN ON FRESH SEEDS 300-399 (the deployed Saturday piece refitted on the same fresh seeds; negative = better)\n")
    print(pd.DataFrame({"seeds 200-299 (Series 70)": real_m, "seeds 300-399": fm}).round(2).to_string())
    print(f"  raw Saturday fallback piece, 84 Saturdays, before correction: seeds 200-299 {float((np.abs(base[(FALLBACK, 'intensity_sym')] - a) - np.abs(base[(FALLBACK, 'none')] - a))[sat].mean()):+.2f}, seeds 300-399 {fb_change:+.2f}")
    replicates = fm["Saturday piece, before correction"] <= -8.0 and fm["post model, after correction"] <= -1.5
    luck = fm["Saturday piece, before correction"] > -4.0 or fm["post model, after correction"] > -0.75
    print("  READING 3:", "REPLICATES" if replicates else ("SEED LUCK" if luck else "IN BETWEEN"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1.to_csv(OUT_DIR / "part1_wrong_date_placebos.csv", index=False)
    t2.to_csv(OUT_DIR / "part2_rule_by_rule.csv")
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(WRONG)} wrong-date placebos + {len(RULE_ONLY)} rule-only columns (seeds 200-299) and the real column on fresh seeds 300-399 (Saturday and the raw fallback): {len(tasks)} tasks (backtest only)")
        d, _ = load_protocol()
        frame = add_columns(d)
        assert frame[NEW_COLUMNS].notna().all().all(), "a column has missing values"
        sat = frame[DATE_COL].dt.dayofweek == 5
        real = frame.loc[sat, pi.COLUMN]
        assert np.allclose(frame[pi.COLUMN], s70.add_intensity_columns(d)["intensity_sym"]), "the production column differs from Series 70's"
        calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
        fifteenth = pi.SHARES[15] * np.clip(1.0 - pi.distance_to_nearest_payday(frame[DATE_COL], 15, calendar) / pi.WIDTH, 0.0, None)
        assert np.allclose(frame["month_end3"] + fifteenth, frame[pi.COLUMN]), "month_end3 plus the 15th rule is not the real column"
        print(f"{'column':12s} {'days of month':>14s} {'mean':>6s} {'max':>5s} {'corr with real (Saturdays)':>27s}")
        for name in list(WRONG) + list(RULE_ONLY):
            x = frame.loc[sat, name]
            label = "/".join(map(str, WRONG[name])) if name in WRONG else ",".join(map(str, RULE_ONLY[name]))[:14]
            print(f"{name:12s} {label:>14s} {x.mean():6.3f} {x.max():5.2f} {float(np.corrcoef(x, real)[0, 1]):27.2f}")
        print("the 16 wrong-date sets are distinct:", len({tuple(v) for v in WRONG.values()}) == len(WRONG))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker73, "series73")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
