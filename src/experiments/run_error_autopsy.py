"""
Error autopsy - where does the final model miss, and what might the model not know?

Takes the final model's out-of-sample predictions on its 585 backtest days (the same
protocol as every series in the experiment log) and answers three questions:

  1. Which days were missed by the most - split into days far BUSIER than forecast
     (surges) and days far QUIETER (lulls)? Each is listed with the context a manager
     can recall it by: weekday, bank-holiday proximity, school break, weather, how it
     compared with the same weekday in the previous four weeks, and the manager's own
     forecast - with a blank column for "what happened that day?".
  2. Is the error spread evenly, or does it concentrate in kinds of day (weekdays,
     months, holidays, weather)? Compared with each kind's share of days.
  3. Did the manager know something the model did not? If the manager's forecast is
     right where the model is wrong, someone had information the features lack; if both
     miss together, the cause is outside both.

Everything here is descriptive: it shows where to look, not why. Closure days (Christmas
Day, New Year's Day) are excluded, as in the API's error ranking.

Run:  python -m src.experiments.run_error_autopsy
Writes reports/experiments/error_autopsy/{error_autopsy.html, worst_days.csv, *.png}
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import holidays
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from scipy.optimize import minimize  # noqa: E402

from src.config import CONFIG  # noqa: E402
from src.features.build_features import SCHOOL_BREAKS  # noqa: E402
from src.experiments.report_series19_21 import CSS  # noqa: E402
from src.experiments.series19_21_common import OUT_DIR as SERIES_DIR  # noqa: E402
from src.experiments.series19_21_common import load_protocol, test_days  # noqa: E402
from src.preprocessing.build_daily_sales import KNOWN_CLOSURE_MONTH_DAYS  # noqa: E402

OUT = CONFIG.path("experiments_dir") / "error_autopsy"
NAVY, GREEN, ORANGE, RED, GREY = "#1a2b45", "#2f6f4e", "#c8742b", "#b23a48", "#8a97a8"
WORST_EACH = 20  # days listed in each direction
FIGS: dict[str, Path] = {}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9.5, "axes.titleweight": "bold"})


# --------------------------------------------------------------------------- building blocks


def load_predictions() -> pd.DataFrame:
    """date, fold_id, actual, manual (the manager's forecast) and model (the final model's
    out-of-sample forecast) for the 585 backtest days."""
    days = test_days(*load_protocol())
    preds = None
    for candidate in (CONFIG.path("results_dir") / "final_model_backtest_predictions.csv",):
        if candidate.exists():
            preds = pd.read_csv(candidate).rename(columns={"final_model_prediction": "model"})
            preds["date"] = pd.to_datetime(preds["date"]).dt.strftime("%Y-%m-%d")
            preds = preds[["date", "model"]]
    if preds is None:  # fall back to the baseline run of Series 19 (identical figures)
        rec = json.loads(next(line for line in (SERIES_DIR / "series19_column_order.jsonl").read_text(encoding="utf-8").splitlines()
                              if json.loads(line)["id"] == "baseline_order"))
        preds = pd.DataFrame({"date": days["date"], "model": rec["pred_bag"]})
    out = days.merge(preds, on="date")
    out["date"] = pd.to_datetime(out["date"])
    return out


def holiday_context(date: pd.Timestamp, calendar: dict, window: int = 3) -> str:
    """'' or e.g. 'Spring Bank Holiday' / '2 days after Good Friday' / '1 day before May Day'."""
    d = date.date()
    if d in calendar:
        return calendar[d]
    best = None
    for hd, name in calendar.items():
        gap = (d - hd).days
        if 0 < abs(gap) <= window and (best is None or abs(gap) < abs(best[0])):
            best = (gap, name)
    if best is None:
        return ""
    gap, name = best
    n = abs(gap)
    return f"{n} day{'s' if n > 1 else ''} {'after' if gap > 0 else 'before'} {name}"


def same_weekday_average(date: pd.Timestamp, sales: pd.Series, weeks: int = 4) -> float:
    """Mean sales of the same weekday in the `weeks` weeks before `date` (NaN if fewer than 2 exist)."""
    prev = [sales.get(date - pd.Timedelta(days=7 * k), np.nan) for k in range(1, weeks + 1)]
    prev = [p for p in prev if not np.isnan(p)]
    return float(np.mean(prev)) if len(prev) >= 2 else float("nan")


def category_table(df: pd.DataFrame, column: str, order: list | None = None) -> pd.DataFrame:
    """For each value of `column`: days, MAE, bias (forecast - actual), share of all absolute error
    and share of days. An index above 1 means the group carries more error than its share of days."""
    total = df["abs_err"].sum()
    g = df.groupby(column, observed=True).agg(days=("abs_err", "size"), MAE=("abs_err", "mean"), bias=("gap", "mean"),
                                              error_total=("abs_err", "sum")).reset_index()
    g["share_of_days_%"] = g["days"] / len(df) * 100
    g["share_of_error_%"] = g["error_total"] / total * 100
    g["error_index"] = g["share_of_error_%"] / g["share_of_days_%"]
    g = g.drop(columns="error_total").rename(columns={column: "group"})
    if order:
        g["o"] = g["group"].map({k: i for i, k in enumerate(order)})
        g = g.sort_values("o").drop(columns="o")
    return g.reset_index(drop=True)


def savefig(fig, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    FIGS[name] = path
    return name


def img(name: str, caption: str = "") -> str:
    data = base64.b64encode(FIGS[name].read_bytes()).decode()
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f'<figure><img src="data:image/png;base64,{data}" alt="{name}">{cap}</figure>'


def tbl(df: pd.DataFrame, fmt: str = "{:,.0f}") -> str:
    return df.to_html(index=False, border=0, classes="t", float_format=lambda v: fmt.format(v), na_rep="–", escape=True)


# --------------------------------------------------------------------------- the analysis

def school_phase(date: pd.Timestamp, breaks: pd.DataFrame = SCHOOL_BREAKS) -> str:
    """Where a day sits relative to Glasgow's school breaks: the last week of term before a break,
    the first week of a break, its middle, its last week, or ordinary term time."""
    d = date.normalize()
    for b in breaks.itertuples():
        start, end = pd.Timestamp(b.start), pd.Timestamp(b.end)
        if start <= d <= end:
            if (d - start).days < 7:
                return "first week of a break"
            if (end - d).days < 7:
                return "last week of a break"
            return "middle of a break"
        if 0 < (start - d).days <= 7:
            return "last week of term"
    return "other term time"


def weekday_bias_by_block(ranked: pd.DataFrame) -> pd.DataFrame:
    """Average forecast-minus-actual for each weekday in each 60-day test block."""
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    t = ranked.pivot_table(index="weekday", columns="fold_id", values="gap", aggfunc="mean").reindex(order)
    t.columns = [f"block {c}" for c in t.columns]
    return t.reset_index().rename(columns={"weekday": "weekday (+ forecast too high)"})


def forward_chained_corrections(ranked: pd.DataFrame) -> pd.DataFrame:
    """Would correcting the forecast for a weekday's past bias have helped? Each 60-day block is
    corrected using ONLY the blocks before it (never future data), and scored from block 1 on.
    Several variants are tried, so treat any gain as a hypothesis to confirm on new data, not a result."""
    a, c, f, wd = (ranked[k].to_numpy() for k in ("actual", "model", "fold_id", "weekday"))
    folds = np.unique(f)

    def run(correct):
        out = c.copy()
        for k in folds[1:]:
            tr, te = f < k, f == k
            out[te] = correct(tr, te)
        return out

    def add_weekday(strength):
        def fn(tr, te):
            p = c[te].copy()
            for day in np.unique(wd[te]):
                mt, me = tr & (wd == day), wd[te] == day
                if mt.sum() >= 4:
                    p[me] = c[te][me] - strength * np.mean(c[mt] - a[mt])
            return p
        return fn

    def saturday(tr, te):
        p = c[te].copy()
        mt, me = tr & (wd == "Saturday"), wd[te] == "Saturday"
        p[me] = c[te][me] * (np.sum(a[mt]) / np.sum(c[mt]))
        return p

    def linear(tr, te):
        res = minimize(lambda x: np.mean(np.abs(x[0] + x[1] * c[tr] - a[tr])), [0.0, 1.0], method="Nelder-Mead")
        return res.x[0] + res.x[1] * c[te]

    mask = f >= 1
    base = float(np.mean(np.abs(c[mask] - a[mask])))
    rows = []
    for name, fn in (("Each weekday shifted by its past average bias (full strength)", add_weekday(1.0)),
                     ("Each weekday shifted by half its past average bias", add_weekday(0.5)),
                     ("Saturdays only, scaled by their past actual/forecast ratio", saturday),
                     ("All days stretched by a fitted line (actual = a + b x forecast)", linear)):
        mae = float(np.mean(np.abs(run(fn)[mask] - a[mask])))
        rows.append({"correction (learned from earlier blocks only)": name, "MAE": mae, "uncorrected MAE": base, "change": mae - base})
    return pd.DataFrame(rows)



def build_frame() -> pd.DataFrame:
    base = load_predictions()
    feats = pd.read_csv(CONFIG.path("engineered_features"), parse_dates=["date"])
    sales = feats.set_index("date")["total_sales"]
    df = base.merge(feats[["date", "is_school_holiday", "is_payday_window_pm3", "is_christmas_period", "max_temp", "rain_mm",
                           "sun_hours", "is_heavy_rain", "is_hot_for_scotland", "is_long_weekend"]], on="date", how="left")
    cal = holidays.UK(subdiv=CONFIG.project.uk_subdivision, years=range(2024, 2028))
    cal = {d: n for d, n in cal.items()}
    df["closure"] = [(d.month, d.day) in KNOWN_CLOSURE_MONTH_DAYS for d in df["date"]]
    df["weekday"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.strftime("%b %Y")
    df["gap"] = df["model"] - df["actual"]                 # + : forecast too high (a lull), - : too low (a surge)
    df["abs_err"] = df["gap"].abs()
    df["manager_gap"] = df["manual"] - df["actual"]
    df["manager_abs"] = df["manager_gap"].abs()
    df["pct_of_actual"] = np.where(df["actual"] > 0, df["abs_err"] / df["actual"] * 100, np.nan)
    df["holiday"] = [holiday_context(d, cal) for d in df["date"]]
    df["same_weekday_4wk_avg"] = [same_weekday_average(d, sales) for d in df["date"]]
    df["vs_usual_%"] = (df["actual"] / df["same_weekday_4wk_avg"] - 1) * 100
    df["rain"] = pd.cut(df["rain_mm"], [-0.01, 1, 5, 1e9], labels=["dry (<1 mm)", "light rain (1-5 mm)", "heavy rain (>5 mm)"])
    t_warm, t_hot = CONFIG.weather.warm_temp_c, CONFIG.weather.hot_scotland_c
    df["temperature"] = pd.cut(df["max_temp"], [-50, 12, t_warm, t_hot, 60],
                               labels=["cool (<12C)", f"mild (12-{t_warm:.0f}C)", f"warm ({t_warm:.0f}-{t_hot:.0f}C)", f"hot (>{t_hot:.0f}C)"])
    df["school_break"] = np.where(df["is_school_holiday"] == 1, "school holidays", "term time")
    df["bank_holiday_week"] = np.where(df["holiday"] != "", "bank holiday or within 3 days", "ordinary")
    df["payday"] = np.where(df["is_payday_window_pm3"] == 1, "within 3 days of payday", "other")
    df["school_phase"] = [school_phase(d) for d in df["date"]]
    df["direction"] = np.where(df["gap"] < 0, "busier than forecast", "quieter than forecast")
    return df


def worst_table(df: pd.DataFrame, direction: str, n: int) -> pd.DataFrame:
    sub = df[~df["closure"]]
    sub = sub[sub["gap"] < 0] if direction == "under" else sub[sub["gap"] > 0]
    sub = sub.sort_values("abs_err", ascending=False).head(n)
    out = pd.DataFrame({
        "Date": sub["date"].dt.strftime("%a %d %b %Y"),
        "Actual sales": sub["actual"], "Our forecast": sub["model"], "Miss": sub["abs_err"], "Miss %": sub["pct_of_actual"],
        "Manager's forecast": sub["manual"], "Usual for this weekday (4 wk)": sub["same_weekday_4wk_avg"], "vs usual %": sub["vs_usual_%"],
        "Holiday context": sub["holiday"].replace("", "-"), "School": np.where(sub["is_school_holiday"] == 1, "holidays", "term"),
        "Rain mm": sub["rain_mm"], "Max temp C": sub["max_temp"], "What happened?": "",
    })
    out.insert(0, "Direction", "busier than forecast" if direction == "under" else "quieter than forecast")
    return out.reset_index(drop=True)


def run() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_frame()
    ranked = df[~df["closure"]].copy()
    total_err = ranked["abs_err"].sum()
    n = len(ranked)
    mae = ranked["abs_err"].mean()
    top5, top10 = (ranked["abs_err"].nlargest(max(1, int(n * p))).sum() / total_err * 100 for p in (0.05, 0.10))

    under = worst_table(df, "under", WORST_EACH)
    over = worst_table(df, "over", WORST_EACH)
    pd.concat([under, over]).to_csv(OUT / "worst_days.csv", index=False)

    # ---- who knew what: the manager's forecast on the model's worst days
    worst_all = ranked.nlargest(30, "abs_err")
    mgr_better = float((worst_all["manager_abs"] < worst_all["abs_err"]).mean() * 100)
    rho = stats.spearmanr(ranked["abs_err"], ranked["manager_abs"]).statistic
    both_miss = ranked[(ranked["abs_err"] > 1500) & (ranked["manager_abs"] > 1500) & (np.sign(ranked["gap"]) == np.sign(ranked["manager_gap"]))]
    big = ranked[ranked["abs_err"] > 1500]

    # ---- error concentration by kind of day
    wk_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    tables = {
        "Weekday": category_table(ranked, "weekday", wk_order),
        "School calendar": category_table(ranked, "school_break"),
        "School break phase": category_table(ranked, "school_phase", ["last week of term", "first week of a break", "middle of a break", "last week of a break", "other term time"]),
        "Bank holidays": category_table(ranked, "bank_holiday_week"),
        "Payday": category_table(ranked, "payday"),
        "Rain": category_table(ranked.dropna(subset=["rain"]), "rain"),
        "Temperature": category_table(ranked.dropna(subset=["temperature"]), "temperature"),
        "Month": category_table(ranked, "month", list(dict.fromkeys(ranked.sort_values("date")["month"]))),
    }
    for name, t in tables.items():
        t.insert(0, "kind of day", name)
    all_cats = pd.concat(tables.values(), ignore_index=True)
    all_cats.to_csv(OUT / "error_by_kind_of_day.csv", index=False)
    notable = all_cats[(all_cats["days"] >= 12) & (all_cats["error_index"] >= 1.2)].sort_values("error_index", ascending=False)

    wd_blocks = weekday_bias_by_block(ranked)
    chained = forward_chained_corrections(ranked)
    sat_blocks = ranked[ranked['weekday'] == 'Saturday'].groupby('fold_id')['gap'].mean()
    # ---- runs of consecutive big misses in the same direction
    runs, cur = [], []
    for _, r in ranked.sort_values("date").iterrows():
        big_miss = r["abs_err"] >= 1500
        if big_miss and cur and (r["date"] - cur[-1]["date"]).days == 1 and np.sign(r["gap"]) == np.sign(cur[-1]["gap"]):
            cur.append(r)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [r] if big_miss else []
    if len(cur) >= 2:
        runs.append(cur)
    runs_df = pd.DataFrame([{"From": r[0]["date"].strftime("%a %d %b %Y"), "To": r[-1]["date"].strftime("%a %d %b %Y"), "Days": len(r),
                             "Direction": "busier than forecast" if r[0]["gap"] < 0 else "quieter than forecast",
                             "Total miss": sum(x["abs_err"] for x in r), "Holiday context": "; ".join(sorted({x["holiday"] for x in r if x["holiday"]})) or "-"}
                            for r in runs])

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(ranked["date"], ranked["actual"], color=NAVY, lw=1, label="actual sales")
    ax.plot(ranked["date"], ranked["model"], color=GREEN, lw=1, alpha=0.8, label="final model (out-of-sample)")
    marks = ranked.nlargest(14, "abs_err")
    ax.scatter(marks["date"], marks["actual"], color=RED, s=26, zorder=3, label="14 largest misses")
    for _, r in marks.iterrows():
        ax.annotate(r["date"].strftime("%d %b %y"), (r["date"], r["actual"]), fontsize=7, xytext=(3, 4), textcoords="offset points")
    ax.set_ylabel("daily sales (£)")
    ax.set_ylim(top=float(ranked["actual"].max()) * 1.14)
    ax.set_title("Actual sales and the model's forecast, with the days it missed most")
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    h1 = savefig(fig, "timeline")

    wk = ranked.copy()
    wk["week"] = wk["date"].dt.to_period("W-SUN").dt.start_time
    grid = wk.pivot_table(index="weekday", columns="week", values="gap", aggfunc="mean").reindex(wk_order)
    fig, ax = plt.subplots(figsize=(12, 3.4))
    lim = float(np.nanpercentile(np.abs(grid.to_numpy()), 98))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_yticks(range(7), [d[:3] for d in wk_order])
    ticks = list(range(0, len(grid.columns), 8))
    ax.set_xticks(ticks, [grid.columns[i].strftime("%b %y") for i in ticks])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="forecast minus actual (£)\nred: sold MORE than forecast (a surge)\nblue: sold LESS (a lull)", shrink=0.9)
    ax.set_title("Calendar of misses - each cell is one day (columns are weeks, rows are weekdays)")
    h2 = savefig(fig, "calendar")

    wd = tables["Weekday"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(wd))
    ax.bar(x, wd["MAE"], color=GREEN, alpha=0.85, label="average miss")
    ax.plot(x, wd["bias"].abs(), "o", color=RED, label="|average bias|")
    ax.set_xticks(x, [d[:3] for d in wd["group"]])
    for i, share in enumerate(wd["share_of_error_%"]):
        ax.annotate(f"{share:.0f}% of\nall error", (i, wd["MAE"].iloc[i]), ha="center", va="bottom", fontsize=7.5, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, wd["MAE"].max() * 1.22)
    ax.set_ylabel("£")
    ax.set_title("Where the error lives: by weekday")
    ax.legend(fontsize=8)
    h3 = savefig(fig, "weekday")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(ranked["manager_abs"], ranked["abs_err"], s=10, color=GREY, alpha=0.6)
    ax.scatter(worst_all["manager_abs"], worst_all["abs_err"], s=22, color=RED, label="model's 30 worst days")
    lim = max(ranked["manager_abs"].max(), ranked["abs_err"].max()) * 1.03
    ax.plot([0, lim], [0, lim], color=NAVY, ls="--", lw=1)
    ax.set_xlabel("manager's miss (£)")
    ax.set_ylabel("model's miss (£)")
    ax.set_title(f"Did the manager know something? (rank correlation {rho:.2f})\nbelow the line: the manager was closer")
    ax.legend(fontsize=8)
    h4 = savefig(fig, "manager_vs_model")

    folds = ranked.groupby("fold_id").agg(bias=("gap", "mean"), start=("date", "min"), end=("date", "max"), MAE=("abs_err", "mean"))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.bar(folds.index, folds["bias"], color=[RED if b > 0 else GREEN for b in folds["bias"]])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(folds.index, [f"{s:%b %y}" for s in folds["start"]], fontsize=8)
    ax.set_ylabel("average forecast minus actual (£)")
    ax.set_title("Bias in each 60-day test block: red = forecast too high on average, green = too low")
    savefig(fig, "bias_by_block")

    if len(notable):
        top = notable.head(10)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        labels = [f"{k}: {g}" for k, g in zip(top["kind of day"], top["group"])]
        ax.barh(labels[::-1], top["error_index"].to_numpy()[::-1], color=ORANGE)
        ax.axvline(1, color=NAVY, ls="--", lw=1)
        ax.set_xlabel("error index (share of all error / share of days; 1 = average)")
        ax.set_title("Kinds of day that carry more than their share of the error")
        h6 = savefig(fig, "over_represented")
    else:
        h6 = None

    # ------------------------------------------------------------------ page
    d = ranked["gap"]
    surge_n, lull_n = int((d < 0).sum()), int((d > 0).sum())
    findings = [
        f"<li><b>The model's average miss is £{mae:,.0f}</b> over {n} days. The worst 5% of days carry <b>{top5:.0f}%</b> of all the error, "
        f"the worst 10% carry <b>{top10:.0f}%</b> - a heavy tail, but not a dominant one: most error is spread across ordinary days.</li>",
        f"<li><b>Surges and lulls:</b> {surge_n} days sold more than forecast and {lull_n} less. The biggest single misses are "
        f"£{under['Miss'].iloc[0]:,.0f} too low ({under['Date'].iloc[0]}) and £{over['Miss'].iloc[0]:,.0f} too high ({over['Date'].iloc[0]}).</li>",
        f"<li><b>Weekday:</b> Friday to Sunday carry {tables['Weekday'].set_index('group').loc[['Friday', 'Saturday', 'Sunday'], 'share_of_error_%'].sum():.0f}% of the error "
        f"on {tables['Weekday'].set_index('group').loc[['Friday', 'Saturday', 'Sunday'], 'share_of_days_%'].sum():.0f}% of days - the busy days are the hard days.</li>",
        f"<li><b>Did the manager know something the model did not?</b> Sometimes: on the model's 30 worst days the manager's forecast was closer on {mgr_better:.0f}%. "
        f"But the two forecasts' misses are correlated (rank correlation {rho:.2f}), and on {len(both_miss)} of the {len(big)} days the model missed by more than £1,500 "
        f"the manager also missed by more than £1,500 in the same direction - for most big misses the cause was in neither forecast.</li>",
    ]
    if len(runs_df):
        findings.append(f"<li><b>Runs of big misses:</b> {len(runs_df)} stretches of two or more consecutive days missed by £1,500+ in the same direction "
                        "(table below) - event-like patterns are most likely to show up here.</li>")
    notable_html = ""
    if len(notable):
        names = "; ".join(f"{k} - {g} (index {i:.2f}, {int(dys)} days)" for k, g, i, dys in
                          zip(notable["kind of day"], notable["group"], notable["error_index"], notable["days"]))
        notable_html = f"<p>Kinds of day (at least 12 days) carrying 20% or more than their share of the error: {html.escape(names)}.</p>"

    cat_show = all_cats.copy()
    cat_show.columns = ["kind of day", "group", "days", "avg miss £", "bias £", "share of days %", "share of error %", "error index"]

    def table_with_fmt(frame: pd.DataFrame) -> str:
        return frame.to_html(index=False, border=0, classes="t", escape=True, na_rep="-",
                             formatters={c: (lambda v: f"{v:,.0f}") for c in frame.columns if frame[c].dtype.kind == "f" and c not in ("Miss %", "vs usual %", "Rain mm", "Max temp C", "MAE", "uncorrected MAE", "change")}
                             | {c: (lambda v: f"{v:,.1f}") for c in ("Miss %", "vs usual %", "Rain mm", "Max temp C") if c in frame.columns}
                             | {c: (lambda v: f"{v:,.2f}") for c in ("MAE", "uncorrected MAE", "change") if c in frame.columns})

    questions = (
        "<ol><li><b>Surges</b> - for each day in the first table: what made it busy? (an event, a match, a concert, a function booking, a group, "
        "good weather, a promotion, a change in opening hours, a competitor closing?)</li>"
        "<li><b>Lulls</b> - for each day in the second table: what made it quiet? (a closure, a staffing problem, bad news, roadworks, a rival event, "
        "a long-weekend exodus?)</li>"
        "<li><b>Recurring patterns</b> - do the surges cluster around the same causes (home matches, a type of function, a time of year)? Is any of it "
        "<i>known days in advance</i> - because those are the things that can become model inputs.</li>"
        "<li><b>What is known the day before?</b> Bookings, covers, function diary, event listings: which of these exist as data, and from when?</li></ol>")

    parts = [
        f"<h1>Error autopsy - where does the final model miss?</h1><p class='small'>The final model's out-of-sample forecasts on its {len(df)} backtest days "
        f"({df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y}); {n} ranked after leaving out closure days. Descriptive only: it shows where to look, not why.</p>",
        "<h2>What the data says</h2><ul>" + "".join(findings) + "</ul>",
        img(h1), img(h2, "Red: the day sold MORE than forecast (a surge). Blue: it sold LESS (a lull). Look for vertical stripes (a weekday that is regularly off) and horizontal clumps (stretches of weeks)."),
        img(h3), img(h4),
    ]
    if h6:
        parts += [img(h6), notable_html]
    parts += [
        "<h2>Patterns worth testing (exploratory)</h2>",
        f"<p><b>1. Saturdays are usually under-forecast.</b> On average the model's Saturday forecast is £{-tables['Weekday'].set_index('group').loc['Saturday', 'bias']:,.0f} "
        f"too low, and too low in {int((sat_blocks < 0).sum())} of {len(sat_blocks)} test blocks - worst on the busiest Saturdays (the usual 'peak shrinkage' of a model that averages). "
        f"But the pattern fades in the latest blocks, so it is not a constant offset. Forward-chained, using only earlier blocks to correct each block:</p>",
        table_with_fmt(chained),
        "<p class='small'>Several corrections were tried after seeing the pattern, so a gain here is a hypothesis to confirm on new data, not a result. "
        "A full-strength correction does no good because the bias changes sign over time; a gentler one helps. Down-weighting old data (recency weighting) aims to track such changes properly.</p>",
        table_with_fmt(wd_blocks),
        img("bias_by_block"),
        "<p><b>2. The start of the school summer holidays.</b> Both years, the biggest surges cluster in the last days of June and first days of July "
        "(28 Jun and 1 Jul 2025; 28 and 30 Jun 2026) - exactly when Glasgow's summer break begins (30 Jun 2025, 29 Jun 2026). The model knows 'school holidays' but not "
        "'the first week of them'; see the 'School break phase' rows in the detail table below.</p>",
        f"<p><b>3. Hot days are quieter than the model expects.</b> On days above {CONFIG.weather.hot_scotland_c:.0f} degrees C the forecast is on average "
        f"£{tables['Temperature'].set_index('group').iloc[-1]['bias']:,.0f} too high, while on cooler days it is too low.</p>",
        "<h2>For the venue manager</h2>", questions,
        "<p>Please fill in the last column and send it back. <b>The table is also saved as <code>worst_days.csv</code></b> (open in Excel).</p>",
        f"<h3>The {WORST_EACH} days that were much BUSIER than forecast</h3>", table_with_fmt(under),
        f"<h3>The {WORST_EACH} days that were much QUIETER than forecast</h3>", table_with_fmt(over),
    ]
    if len(runs_df):
        parts += ["<h3>Stretches of consecutive big misses (£1,500+ each day, same direction)</h3>", table_with_fmt(runs_df)]
    parts += ["<h2>Detail: error by kind of day</h2>",
              "<p>The error index compares a group's share of all error with its share of days (1.0 = average).</p>", table_with_fmt(cat_show)]
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Error autopsy</title><style>{CSS}</style></head><body>{''.join(parts)}</body></html>")
    out = OUT / "error_autopsy.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")
    print(f"mean miss £{mae:,.0f}; worst 5% = {top5:.0f}% and worst 10% = {top10:.0f}% of error; manager closer on {mgr_better:.0f}% of model's 30 worst days; rho {rho:.2f}")
    return out


if __name__ == "__main__":
    run()
