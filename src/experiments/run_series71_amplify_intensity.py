"""
Series 71 - amplifying the triangle payday intensity, with the choice made on the first half of the days and judged on the second.

Series 70's triangle intensity (`intensity_sym`: each pay rule adds a triangle of heat around its payday, 1 - k/4 for the nearest payday at k = 0 to 3 days either side, weighted by the
CIPP survey shares 0.34 / 0.30 / 0.30 / 0.18 for the last working day, the 25th, the 28th and the 15th) was 3.34 pounds a day better for the post model and 3.80 for the hydro model after the
correction (intervals -7.7 to +0.9 and -8.3 to +0.4, 7 and 8 folds of 10, better in 94% and 96% of bootstrap resamples), and better on the 56 new days (-2.9, -1.3). Nearly all of it is the Saturday
piece (-20.8 pounds a day, -2.99 of the -3.34; every other day about flat), and it fades in 2026 (-1.4 and -0.4 against -4.8 and -6.3 in 2025; Saturdays -6.1 against -31.8). It was also the best of
four variants after about forty payday variants tried, which chance alone could produce. This series asks whether the signal can be made stronger, with rules fixed first so the result is not overfit.

A tree only asks "above or below a cut", so scaling or squaring the column changes nothing; only a different shape or place changes what the model can use. Four fixed variants, each on its own:

  sym_saturday_only   `intensity_sym` in the Saturday pieces only; every other piece as deployed. The Saturday piece is where the gain is, and the other pieces only spend splits on it. These are the
                      same fits as Series 70's Saturday pieces (same columns, same seeds), so they are reused, not refitted.
  sym_w2              the triangle narrowed to reach 1 day either side of payday (1 - k/2): the payday itself matters most
  sym_w7              the triangle widened to reach 7 days either side (1 - k/8): the whole stretch around payday matters
  sym_week            the sum of `intensity_sym` over the 7 days ending on the day: the money still in circulation days after payday

Series 54's design (every variant in every piece of the weekly log that it changes, 100 seeds 200-299, 585 days and the 56 new days, the two models assembled and corrected; 540 fits).

THE RULE (set before any result was seen; the choice is made on the first half of the backtest and judged on the second)
  candidates   `intensity_sym` (Series 70), the four above
  choose       the candidate with the largest gain on the first five folds (folds 0-4, 285 days) for the post model, after the correction; one choice for both models
  judge        on the last five folds (folds 5-9, 300 days) and the 56 new days. The choice is SUPPORTED if, on folds 5-9, it is at least 1.5 pounds a day better than the deployed pieces in both
               models and better in at least 3 of the 5 folds in both, and it is not worse on the new days (before the correction) in either model
  amplified    the choice is a new variant (not Series 70's) and is at least 1 pound a day better than `intensity_sym` on folds 5-9 for the post model
  also print   every candidate on both halves, on 2025 and 2026, and on the new days, so a result can be read in full
  expected     the 2026 fade stays: no candidate is clearly better than `intensity_sym` on folds 5-9, and the choice is not supported
  limits       no pass/fail gate beyond the rule above; nothing in production changes; the new days are few and have been read across many series, so they are a weak check

Run:  python -m src.experiments.run_series71_amplify_intensity list | run [--workers 10] | report
"""

from __future__ import annotations

import argparse

import holidays
import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series23_log_vs_raw_peaks as s23
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series47_weekend_flags_in_pieces as s47
from src.experiments import run_series54_day_of_month as s54
from src.experiments import run_series70_payday_intensity as s70
from src.experiments.run_package_validation import CORRECTION
from src.experiments.run_series46_calendar_confirmation import CUTOFF, SEEDS
from src.experiments.run_series61_payday_rules import days_around
from src.experiments.series19_21_common import _STATE, DATE_COL, TARGET_COL, load_protocol, read_done, run_parallel, test_days
from src.forecast_log.v1 import add_hydro_thu

OUT_DIR = CONFIG.path("experiments_dir") / "series71"
RESULTS = OUT_DIR / "series71_amplify_intensity.jsonl"
FITTED = ["sym_w2", "sym_w7", "sym_week"]
CANDIDATES = ["intensity_sym", "sym_saturday_only", *FITTED]
SATURDAY_PIECES = ("Saturday", "Saturday fallback (raw)")
NEW_COLUMNS = list(FITTED)
WIDTH = {"intensity_sym": 4.0, "sym_w2": 2.0, "sym_w7": 8.0}


def piece_features(piece: str, variant: str) -> list[str]:
    cols = list(s54.FINAL_FEATURES) + s54.PIECES[piece][0]
    return cols + ([variant] if variant != "none" else [])


def triangle_intensity(dates: pd.Series, width: float) -> np.ndarray:
    """Sum over the four pay rules of survey share x max(0, 1 - k / width), k = days to the rule's nearest payday (Series 70's `intensity_sym` at width 4)."""
    calendar = holidays.UK(subdiv=CONFIG.project.uk_subdivision)
    total = np.zeros(len(dates))
    for rule in s70.RULES:
        since, to = days_around(dates, rule, calendar)
        total += s70.SHARES[rule] * np.clip(1.0 - np.minimum(since, to) / width, 0.0, None)
    return total


def add_amplified_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    d = out[DATE_COL]
    out["sym_w2"], out["sym_w7"] = triangle_intensity(d, WIDTH["sym_w2"]), triangle_intensity(d, WIDTH["sym_w7"])
    days = pd.date_range(d.min().normalize() - pd.Timedelta(days=6), d.max().normalize(), freq="D")
    daily = pd.Series(triangle_intensity(pd.Series(days), WIDTH["intensity_sym"]), index=days)
    out["sym_week"] = daily.rolling(7).sum().reindex(pd.DatetimeIndex(d.dt.normalize())).to_numpy()
    return out


def init_worker71() -> None:
    s47.init_worker47()
    clash = set(NEW_COLUMNS) & set(_STATE["df"].columns)
    assert not clash, f"columns already exist: {clash}"
    _STATE["df"] = add_amplified_columns(add_hydro_thu(_STATE["df"]))


def make_tasks() -> list[dict]:
    out = []
    for piece in s54.PIECES:
        for variant in FITTED:
            features = piece_features(piece, variant)
            for part, size in (("backtest", s54.BACKTEST_CHUNK), ("new_days", s54.NEW_DAYS_CHUNK)):
                for i in range(0, len(SEEDS), size):
                    chunk = SEEDS[i:i + size]
                    out.append({"id": f"{part}_{piece}_{variant}_s{chunk[0]}-{chunk[-1]}", "part": part, "piece": piece, "variant": variant,
                                "seeds": chunk, "features": features})
    return out


def all_bags(part: str) -> dict[tuple[str, str], np.ndarray]:
    """(piece, variant) -> the 100-seed bag, for `none`, the candidates and Series 70's `intensity_sym`."""
    base = s54.load_bags(part, s70.RESULTS, {"none": [], "intensity_sym": []}, s54.RESULTS)
    done = read_done(RESULTS)
    out = dict(base)
    for piece in s54.PIECES:
        out[(piece, "sym_saturday_only")] = base[(piece, "intensity_sym")] if piece in SATURDAY_PIECES else base[(piece, "none")]
        for variant in FITTED:
            rows = [np.array(r["per_seed"]) for r in done.values() if (r["part"], r["piece"], r["variant"]) == (part, piece, variant)]
            per_seed = np.concatenate(rows, axis=0)
            assert per_seed.shape[0] == len(SEEDS), (part, piece, variant, per_seed.shape)
            out[(piece, variant)] = per_seed.mean(axis=0)
    return out


def report() -> None:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, fold, dow = days["actual"].to_numpy(), days["fold_id"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    new = df[df[DATE_COL] >= CUTOFF]
    a_new, dow_new = new[TARGET_COL].to_numpy(), new[DATE_COL].dt.dayofweek.to_numpy()
    bt, nd = all_bags("backtest"), all_bags("new_days")
    first, second = fold < 5, fold >= 5
    y2025, y2026 = (days["date"] < "2026-01-01").to_numpy(), (days["date"] >= "2026-01-01").to_numpy()
    pd.set_option("display.width", 250)

    def package(bags: dict, variant: str, mon_thu: str, dows: np.ndarray) -> np.ndarray:
        return np.select([dows == 4, dows == 5, dows == 6], [bags[("Friday", variant)], bags[("Saturday", variant)], bags[("Sunday", variant)]], default=bags[(mon_thu, variant)])

    print("EACH PIECE ON ITS OWN WEEKDAYS (change = with the variant minus as deployed; negative = better; 585 days, 100 seeds)")
    for variant in CANDIDATES:
        rows = []
        for piece, (_, _, weekdays, _) in s54.PIECES.items():
            own, own_new = np.isin(dow, weekdays), np.isin(dow_new, weekdays)
            d = np.abs(bt[(piece, variant)] - a) - np.abs(bt[(piece, "none")] - a)
            change, lo, hi = s23.masked_block_bootstrap(d, own.astype(float))
            d_new = np.abs(nd[(piece, variant)] - a_new) - np.abs(nd[(piece, "none")] - a_new)
            rows.append({"piece": piece, "change": change, "lo": lo, "hi": hi, "folds_better": int(sum(d[own & (fold == k)].mean() < 0 for k in range(10))),
                         "new_days_change": float(d_new[own_new].mean())})
        print(f"\n{variant}"); print(pd.DataFrame(rows).round(2).to_string(index=False))

    results: dict[tuple[str, str], dict] = {}
    print("\n\nTHE WHOLE MODELS, after the correction (negative = better than the deployed pieces)")
    for model, mon_thu in (("post model", "Monday-Thursday"), ("hydro model", "Monday-Thursday (hydro)")):
        base = s24.correct(days, package(bt, "none", mon_thu, dow), CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])
        base_new = package(nd, "none", mon_thu, dow_new)
        rows = []
        for variant in CANDIDATES:
            cand = s24.correct(days, package(bt, variant, mon_thu, dow), CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])
            diff = np.abs(cand - a) - np.abs(base - a)
            change, lo, hi = s23.masked_block_bootstrap(diff, np.ones(len(a)))
            new_change = float((np.abs(package(nd, variant, mon_thu, dow_new) - a_new) - np.abs(base_new - a_new)).mean())
            r = {"variant": variant, "all 585": change, "lo": lo, "hi": hi, "folds 0-4": float(diff[first].mean()), "folds 5-9": float(diff[second].mean()),
                 "2025": float(diff[y2025].mean()), "2026": float(diff[y2026].mean()), "Saturdays": float(diff[dow == 5].mean()),
                 "folds better 5-9": int(sum(diff[fold == k].mean() < 0 for k in range(5, 10))), "new days": new_change}
            rows.append(r)
            results[(model, variant)] = r
        print(f"\n{model}"); print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n\nTHE RULE SET BEFOREHAND")
    winner = min(CANDIDATES, key=lambda v: results[("post model", v)]["folds 0-4"])
    print(f"  choice on folds 0-4 (post model, after the correction): {winner}  ({results[('post model', winner)]['folds 0-4']:+.2f} pounds a day)")
    checks = []
    for model in ("post model", "hydro model"):
        r = results[(model, winner)]
        ok = r["folds 5-9"] <= -1.5 and r["folds better 5-9"] >= 3 and r["new days"] <= 0
        checks.append(ok)
        print(f"  {model:12s} on folds 5-9: {r['folds 5-9']:+.2f}, folds better {r['folds better 5-9']} of 5; new days {r['new days']:+.2f}  ->  {'meets' if ok else 'does not meet'} the bar")
    print("  SUPPORTED" if all(checks) else "  NOT SUPPORTED", "(the choice is judged on the second half and the new days)")
    sym = results[("post model", "intensity_sym")]["folds 5-9"]
    gain_vs_sym = sym - results[("post model", winner)]["folds 5-9"]
    print(f"  amplified: {'yes' if winner != 'intensity_sym' and gain_vs_sym >= 1.0 else 'no'} (the choice is {gain_vs_sym:+.2f} pounds a day better than `intensity_sym` on folds 5-9, post model)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": m, **r} for (m, _), r in results.items()]).to_csv(OUT_DIR / "packages.csv", index=False)
    print(f"\nSaved to {OUT_DIR}")


def main(step: str, workers: int) -> None:
    if step == "list":
        tasks = make_tasks()
        print(f"{len(s54.PIECES)} pieces x {len(FITTED)} fitted variants, {len(tasks)} tasks (`sym_saturday_only` reuses Series 70's Saturday fits)")
        for piece in s54.PIECES:
            print(f"  {piece:26s} none {len(piece_features(piece, 'none'))} columns, each variant {len(piece_features(piece, FITTED[0]))}")
        d, _ = load_protocol()
        frame = add_amplified_columns(s70.add_intensity_columns(d))
        assert np.allclose(triangle_intensity(frame[DATE_COL], 4.0), frame["intensity_sym"]), "the width-4 triangle is not Series 70's intensity_sym"
        assert frame[NEW_COLUMNS].notna().all().all(), "a column has missing values"
        for name in ["intensity_sym", *NEW_COLUMNS]:
            x = frame[name]
            print(f"{name:14s} nonzero on {int((x > 0).sum()):4d} of {len(frame)} days; max {x.max():.2f}; mean {x.mean():.3f}")
        show = frame[frame[DATE_COL].between("2025-09-22", "2025-10-04")]
        print(show[[DATE_COL, "intensity_sym", "sym_w2", "sym_w7", "sym_week"]].assign(weekday=show[DATE_COL].dt.strftime("%a")).to_string(index=False))
    elif step == "run":
        run_parallel(make_tasks(), s54.fit_task, RESULTS, workers, init_worker71, "series71")
        report()
    else:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "run", "report"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    main(args.step, args.workers)
