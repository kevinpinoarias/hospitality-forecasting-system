"""
Series 93 - the form of v1.14 (the year-ago sales input) measured against v1.13, its parent. No new fits: composed from stored forecasts.

The version plan (docs/JANUARY_DECEMBER_CANDIDATES_2026_11.md): v1.13 is the hydro arm of v1.12 with the year-end specialist models (a companion on 26-31 December, a masked model on 1-8 January, everything else v1.12's own
files); v1.14 is v1.13 plus one input in the pieces, `sales_364` (the sales on the same weekday 364 days earlier; 655 of the 1,006 training rows have it). Series 90/92 measured the input against the unmasked package and found a modest,
uneven gain (-6.4 a day on 100 seeds, -8.3 on fresh seeds; Monday-Thursday -6 to -16, Saturday -8, Sunday +18.6 WORSE). Kevin (9 November 2026): run it for Monday to Saturday, and show the Sunday forecast of v1.13 against v1.14.
v1.14 is read against its PARENT v1.13, which is what the log's Q8 would compare.

Packages, all served with the S6 stand-in, the deployed weekend correction, 585 days; the specialist routing (the companion on 26-31 December, the masked model on 1-8 January) is v1.13's in every one:
  V13          v1.13: the unmasked pieces outside the two windows (Series 89's fits)
  V14 all      `sales_364` in all four pieces (Series 90's 25 seeds + Series 92's 75, and the fresh seeds)
  V14 Mon-Sat  `sales_364` in the Monday-Thursday, Friday and Saturday pieces; the Sunday piece is v1.13's own
  V14 Mon-Fri  in the Monday-Thursday and Friday pieces only
  V14 Mon-Thu  in the Monday-Thursday piece only
The specialist models themselves are not refitted with the input here (they have no recent-sales inputs; whether the year-ago input belongs in them is a registration question).

READING (written before the numbers were computed)
  measure     the net change of each V14 form against V13 over the 585 days (pounds a day; negative = better), on (a) all 100 seeds and (b) the fresh seeds 300-399, by weekday beside it
  helps       the net change is at most -2.0 on BOTH seed sets (the 25-seed noise is about 2; with 100 seeds it is smaller)
  Sunday      leaving the Sunday piece out is BETTER than putting the input in every piece if V14 Mon-Sat is lower than V14 all by more than 2.0 on both seed sets
  also        the Sunday forecasts of V13, V14 all and V14 Mon-Sat are drawn (84 Sundays, 100 seeds); Sundays of V14 Mon-Sat may differ slightly from V13's because the weekend correction uses the other days' errors
  limits      the same 585 days (one December, one January) so this checks the form, not the days; the specialists are not refitted with the input; the gain is uneven over the year (winter and summer better, early 2025,
              when the training rows had no year-ago values, and spring 2026 worse); any registration is Kevin's decision, after an amendment and a refit, with forecasts back-run from 2 November

Run:  python -m src.experiments.run_series93_v114_form report
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.experiments import run_series24_weekend_correction as s24
from src.experiments import run_series82_mask_windows as s82
from src.experiments import run_series89_standin_repeat_hybrid as s89
from src.experiments import run_series90_year_end_candidates as s90
from src.experiments import run_series92_year_end_seeds_repeat as s92
from src.experiments.run_package_validation import CORRECTION
from src.experiments.series19_21_common import load_protocol, read_done, test_days

OUT_DIR = CONFIG.path("experiments_dir") / "series93"
PIECES = s90.PIECES
FORMS = {"V14 all": PIECES, "V14 Mon-Sat": ["Monday-Thursday (hydro)", "Friday", "Saturday"], "V14 Mon-Fri": ["Monday-Thursday (hydro)", "Friday"], "V14 Mon-Thu": ["Monday-Thursday (hydro)"]}


def build() -> dict:
    df, folds = load_protocol()
    assert len(folds) == 10, "the protocol no longer has the 10 folds that end on 6 September 2026"
    days = test_days(df, folds)
    days["date"] = pd.to_datetime(days["date"])
    days["weekday"] = days["date"].dt.day_name()
    a, dow = days["actual"].to_numpy(), days["date"].dt.dayofweek.to_numpy()
    six = s82.in_window(days["date"], (12, 26), (12, 31))
    jan8 = s82.in_window(days["date"], (1, 1), (1, 8))
    done89, done90, done92 = read_done(s89.RESULTS), read_done(s90.RESULTS), read_done(s92.RESULTS)

    def package(p: dict) -> np.ndarray:
        pred = np.select([dow == 4, dow == 5, dow == 6], [p["Friday"], p["Saturday"], p["Sunday"]], default=p["Monday-Thursday (hydro)"])
        return s24.correct(days, pred, CORRECTION["half_life_weeks"], CORRECTION["strength"], scope=CORRECTION["scope"])

    out = {}
    for label, suffix in (("all 100 seeds", ""), ("fresh seeds 300-399", "_fresh")):
        base = {pc: s89.mine(done89, pc, "sd_base" + suffix, "S6") for pc in PIECES}
        msk = {pc: s89.mine(done89, pc, "sd_j1_8" + suffix, "S6") for pc in PIECES}
        comp = {pc: s89.mine(done89, pc, "sd_companion" + suffix, "S6") for pc in PIECES}
        if suffix == "":
            B = {pc: (s90.mine(done90, pc, "B") * 25 + s92.collect(done92, pc, "B", 75) * 75) / 100 for pc in PIECES}
        else:
            B = {pc: s92.collect(done92, pc, "B_fresh", 100) for pc in PIECES}

        def routed(main: dict) -> dict:
            return {pc: np.where(six, comp[pc], np.where(jan8, msk[pc], main[pc])) for pc in PIECES}

        packs = {"V13": package(routed(base))}
        for name, used in FORMS.items():
            packs[name] = package(routed({pc: (B[pc] if pc in used else base[pc]) for pc in PIECES}))
        out[label] = packs
    return {"days": days, "a": a, "dow": dow, "six": six, "jan8": jan8, "packs": out}


def report() -> None:
    r = build()
    a, dow = r["a"], r["dow"]
    pd.set_option("display.width", 250)
    tables, nets = {}, {}
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for label, packs in r["packs"].items():
        rows = {}
        for name in FORMS:
            d = np.abs(packs[name] - a) - np.abs(packs["V13"] - a)
            rows[name] = {"net, 585 days": float(d.mean()), **{n: float(d[dow == i].mean()) for i, n in enumerate(names)}}
        tables[label] = pd.DataFrame(rows).T
        nets[label] = tables[label]["net, 585 days"]
        print(f"{label.upper()}: EACH V14 FORM MINUS V13 (the parent), served with S6 (pounds a day; negative = better)\n")
        print(tables[label].round(2).to_string())
        print(f"   v1.13's own average miss: {np.abs(packs['V13'] - a).mean():.0f}\n")
    for name in FORMS:
        helps = all(nets[k][name] <= -2.0 for k in nets)
        print(f"READING SET BEFOREHAND, {name}: " + ", ".join(f"{k}: {nets[k][name]:+.2f}" for k in nets) + f"  ->  {'HELPS' if helps else 'does not clear -2.0 on both seed sets'}")
    sunday = all(nets[k]["V14 all"] - nets[k]["V14 Mon-Sat"] > 2.0 for k in nets)
    print("READING SET BEFOREHAND, leaving Sunday out: Mon-Sat minus all = " + ", ".join(f"{k}: {nets[k]['V14 Mon-Sat'] - nets[k]['V14 all']:+.2f}" for k in nets) + f"  ->  {'BETTER (more than 2.0 on both)' if sunday else 'not clearly better'}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(OUT_DIR / f"forms_{'fresh' if 'fresh' in k else 'seeds100'}.csv")
    chart(r)


def chart(r: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days, a, dow = r["days"], r["a"], r["dow"]
    packs = r["packs"]["all 100 seeds"]
    sun = dow == 6
    dates = days["date"][sun]
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.6), facecolor=SURFACE, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.3}, sharex=True)
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.plot(dates, a[sun], color="#2a78d6", lw=1.8, marker="o", ms=3.5, label="Actual sales")
    ax.plot(dates, packs["V13"][sun], color="#898781", lw=1.5, label="v1.13 (no year-ago input)")
    ax.plot(dates, packs["V14 all"][sun], color="#eb6834", lw=1.5, label="v1.14 with the input in every piece (Sunday piece too)")
    ax.plot(dates, packs["V14 Mon-Sat"][sun], color="#1baf7a", lw=1.2, ls="--", label="v1.14 Monday-Saturday only (Sunday piece is v1.13's)")
    ax.set_ylabel("Sunday sales, pounds")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    e13, e_all, e_ms = (np.abs(packs[k][sun] - a[sun]) for k in ("V13", "V14 all", "V14 Mon-Sat"))
    ax.set_title(f"Sunday forecasts, the 84 Sundays of the backtest (100 seeds, served with the S6 stand-in). Average miss: v1.13 {e13.mean():,.0f}, v1.14 all pieces {e_all.mean():,.0f}, v1.14 Mon-Sat {e_ms.mean():,.0f}", loc="left", fontsize=10.5, color=INK)
    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.grid(axis="y", color=GRID)
    diff = e_all - e13
    ax2.bar(dates, diff, width=5, color=np.where(diff > 0, "#eb6834", "#1baf7a"))
    ax2.axhline(0, color=INK2, lw=0.8)
    ax2.set_ylabel("miss of v1.14 (all) minus v1.13\n(orange = worse)")
    ax2.set_title(f"How each Sunday's miss changes when the Sunday piece gets the year-ago input: average {diff.mean():+.1f} a day, {int((diff > 0).sum())} Sundays worse, {int((diff < 0).sum())} better", loc="left", fontsize=10.5, color=INK)
    fig.autofmt_xdate()
    out = CONFIG.path("reports_dir") / "versions" if "reports_dir" in CONFIG.paths else None
    path = (out or (OUT_DIR.parent.parent / "versions"))
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / "sunday_v13_v14.png", dpi=140, facecolor=SURFACE, bbox_inches="tight")
    print(f"chart: {path / 'sunday_v13_v14.png'}")


if __name__ == "__main__":
    report()
