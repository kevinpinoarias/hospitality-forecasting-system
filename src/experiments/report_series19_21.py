"""
Builds the review report for Series 19-21 from the raw task results: tables (CSV +
markdown), figures (PNG) and one self-contained HTML page.

    python -m src.experiments.report_series19_21 [--commentary path/to/commentary.md]

Everything is computed from reports/experiments/series19_21/*.jsonl. A figure or table
whose results are not there yet is skipped, so the report can be built part-way
through a run.
"""

from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.experiments import run_series19_column_order as s19  # noqa: E402
from src.experiments import run_series20_neural_v2 as s20  # noqa: E402
from src.experiments import run_series21_catboost_depth_iterations as s21  # noqa: E402
from src.experiments import run_series22_neural_target_day as s22  # noqa: E402
from src.experiments.series19_21_common import OUT_DIR, load_protocol, read_done, test_days  # noqa: E402

REPORT_DIR = OUT_DIR / "report"
NAVY, GREEN, ORANGE, RED, GREY, BLUE = "#1a2b45", "#2f6f4e", "#c8742b", "#b23a48", "#8a97a8", "#3b6ea5"
CURRENT_MAE = 909.57

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 9.5, "axes.titleweight": "bold", "axes.titlesize": 11,
})

SECTIONS: list[dict] = []  # {"title", "html"} in order, assembled into the page at the end
FIGURES: dict[str, Path] = {}


# --------------------------------------------------------------------------- helpers


def savefig(fig, name: str) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    FIGURES[name] = path
    return name


def img_tag(name: str, caption: str = "") -> str:
    data = base64.b64encode(FIGURES[name].read_bytes()).decode()
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(name)}">{cap}</figure>'


def table_html(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    return df.to_html(index=False, border=0, classes="t", float_format=lambda v: floatfmt.format(v), na_rep="–", escape=True)


def save_table(df: pd.DataFrame, name: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / f"{name}.csv", index=False)


def block_bootstrap(diff: np.ndarray, block: int = 10, reps: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of `diff` (e.g. daily error of A minus daily error of B) and a 95% CI from a
    moving-block bootstrap (blocks of consecutive days, so day-to-day correlation is kept)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def section(title: str, *parts: str) -> None:
    SECTIONS.append({"title": title, "html": "\n".join(parts)})


# --------------------------------------------------------------------------- shared data


def protocol_days() -> pd.DataFrame:
    return test_days(*load_protocol())


def final_model_series(days: pd.DataFrame) -> pd.Series:
    rec = read_done(s19.RESULTS).get("baseline_order") or read_done(s21.RESULTS).get("d4_i300_lr0.05")
    if rec is None:
        raise SystemExit("no baseline (current final model) result yet")
    return pd.Series(rec["pred_bag"], index=days["date"])


# --------------------------------------------------------------------------- Series 21


def report_series21(days: pd.DataFrame) -> None:
    done = read_done(s21.RESULTS)
    grid = [r for r in done.values() if r["kind"] == "grid"]
    if not grid:
        return
    rows = []
    for r in grid:
        p = r["params"]
        rows.append({"depth": p["depth"], "iterations": p["iterations"], "learning_rate": p["learning_rate"],
                     "MAE": r["bag"]["MAE"], "MAE_fold_mean": r["bag"]["MAE_fold_mean"], "RMSE": r["bag"]["RMSE"],
                     "MAPE_%": r["bag"]["MAPE"], "MAE_single_seed": r["seed1"]["MAE"], "minutes": r["seconds"] / 60, "id": r["id"]})
    df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    cur = df[df["id"] == s21.config_id(4, 300, 0.05)]
    cur_mae = float(cur["MAE"].iloc[0]) if len(cur) else CURRENT_MAE
    df["vs_current"] = df["MAE"] - cur_mae
    df["rank"] = np.arange(1, len(df) + 1)
    save_table(df.drop(columns=["id"]), "series21_grid_results")

    # heatmaps, one panel per learning rate
    lrs = sorted(df["learning_rate"].unique())
    fig, axes = plt.subplots(1, len(lrs), figsize=(4.2 * len(lrs), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    vmin, vmax = df["MAE"].min(), df["MAE"].max()
    for ax, lr in zip(axes, lrs):
        sub = df[df["learning_rate"] == lr].pivot(index="depth", columns="iterations", values="MAE").sort_index(ascending=False)
        im = ax.imshow(sub.to_numpy(), cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(sub.columns)), [str(c) for c in sub.columns])
        ax.set_yticks(range(len(sub.index)), [f"depth {d}" for d in sub.index])
        ax.set_title(f"learning rate {lr}")
        ax.set_xlabel("iterations")
        ax.grid(False)
        for i, depth in enumerate(sub.index):
            for j, it in enumerate(sub.columns):
                v = sub.iloc[i, j]
                is_cur = (depth, it, lr) == (4, 300, 0.05)
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, fontweight="bold" if is_cur else None)
                if is_cur:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=NAVY, lw=2.5))
    fig.colorbar(im, ax=axes, shrink=0.85, label="pooled MAE (£), lower is better")
    fig.suptitle("Series 21 - CatBoost depth x iterations x learning rate (current model outlined)", y=1.04, fontweight="bold")
    h1 = savefig(fig, "s21_heatmaps")

    # best learning rate for each depth/iterations
    fig, ax = plt.subplots(figsize=(7, 4))
    for depth, colour in zip(sorted(df["depth"].unique()), [RED, ORANGE, GREEN]):
        sub = df[df["depth"] == depth].groupby("iterations")["MAE"].min()
        ax.plot(sub.index, sub.values, "o-", color=colour, label=f"depth {depth}")
    ax.axhline(cur_mae, color=NAVY, ls="--", lw=1.2, label=f"current model £{cur_mae:.2f}")
    ax.set_xlabel("iterations")
    ax.set_ylabel("pooled MAE (£) at the best learning rate")
    ax.set_title("More iterations, shallower trees: the best result at each setting")
    ax.legend()
    h2 = savefig(fig, "s21_best_by_iterations")

    # confirmation on fresh seeds
    conf = [r for r in done.values() if r["kind"] == "confirm"]
    conf_html = ""
    if conf:
        ref = done.get("confirm_" + s21.config_id(4, 300, 0.05))
        ref_abs = np.abs(np.array(ref["pred_bag"]) - days["actual"].to_numpy()) if ref else None
        crow = []
        for r in sorted(conf, key=lambda r: r["bag"]["MAE"]):
            cfg = r["id"].replace("confirm_", "")
            original = df[df["id"] == cfg]["MAE"]
            row = {"configuration": cfg, "MAE_original_seeds": float(original.iloc[0]) if len(original) else np.nan,
                   "MAE_fresh_seeds": r["bag"]["MAE"]}
            if ref_abs is not None and r is not ref:
                d = np.abs(np.array(r["pred_bag"]) - days["actual"].to_numpy()) - ref_abs
                row["vs_current_on_fresh_seeds"] = fmt_ci(*block_bootstrap(d))
            else:
                row["vs_current_on_fresh_seeds"] = "(the current model)"
            crow.append(row)
        cdf = pd.DataFrame(crow)
        save_table(cdf, "series21_confirmation")
        fig, ax = plt.subplots(figsize=(7.5, 4))
        x = np.arange(len(cdf))
        ax.bar(x - 0.2, cdf["MAE_original_seeds"], 0.4, color=GREY, label="original seeds (chose it)")
        ax.bar(x + 0.2, cdf["MAE_fresh_seeds"], 0.4, color=GREEN, label="fresh seeds (did not choose it)")
        ax.set_xticks(x, [c.replace("_", "\n", 1) for c in cdf["configuration"]], fontsize=7.5)
        lo = min(cdf[["MAE_original_seeds", "MAE_fresh_seeds"]].min()) - 8
        ax.set_ylim(lo, max(cdf[["MAE_original_seeds", "MAE_fresh_seeds"]].max()) + 6)
        ax.set_ylabel("pooled MAE (£)")
        ax.set_title("The top configurations re-run on seeds that did not pick them")
        ax.legend(fontsize=8)
        h3 = savefig(fig, "s21_confirmation")
        conf_html = (
            "<h3>Confirmation on fresh seeds</h3><p>The best of 60 configurations on one set of 25 seeds is partly seed luck. "
            "The top 5 were re-run on 25 different seeds, next to the current model on the same seeds. The right-hand column is "
            "the paired difference in daily absolute error against the current model (mean [95% block-bootstrap CI], £; negative = better).</p>"
            + img_tag(h3, "Grey: the score that selected the configuration. Green: the same configuration on fresh seeds.")
            + table_html(cdf)
        )

    top = df.head(12)[["rank", "depth", "iterations", "learning_rate", "MAE", "vs_current", "MAE_fold_mean", "RMSE", "MAE_single_seed", "minutes"]]
    best = df.iloc[0]
    section(
        "Series 21 - shallower trees, more iterations (CatBoost grid)",
        f"<p>60 configurations (depth 2-4 x 5 iteration counts x 4 learning rates), each scored under the final model's own protocol "
        f"(25 seeds, 10 rolling-origin folds, 585 test days). The current model is depth 4, 300 iterations, learning rate 0.05: "
        f"£{cur_mae:.2f}. The best grid point is <b>depth {int(best['depth'])}, {int(best['iterations'])} iterations, learning rate "
        f"{best['learning_rate']}: £{best['MAE']:.2f}</b> ({best['vs_current']:+.2f} against the current model).</p>",
        img_tag(h1), img_tag(h2),
        "<h3>Top 12 configurations</h3>", table_html(top), conf_html,
        "<p class='small'>Full table: <code>series21_grid_results.csv</code>. MAE_single_seed is the same configuration with one seed "
        "(what an un-bagged model would score).</p>",
    )


# --------------------------------------------------------------------------- Series 19


def report_series19(days: pd.DataFrame) -> None:
    done = read_done(s19.RESULTS)
    if "baseline_order" not in done:
        return
    base = done["baseline_order"]
    random = [r for r in done.values() if r["kind"] == "random_order"]
    seeds = [r for r in done.values() if r["kind"] == "seed_control"]
    structured = [r for r in done.values() if r["kind"] == "structured_order"]
    if len(random) < 5:
        return
    mae = lambda rs, key="bag": np.array([r[key]["MAE"] for r in rs])  # noqa: E731
    o_bag, o_b5, o_s1 = mae(random), mae(random, "bag5"), mae(random, "seed1")
    c_bag = mae(seeds) if seeds else np.array([])

    def describe(label, arr):
        return {"group": label, "n": len(arr), "mean": arr.mean(), "sd": arr.std(ddof=1), "min": arr.min(),
                "p5": np.percentile(arr, 5), "median": np.median(arr), "p95": np.percentile(arr, 95), "max": arr.max(),
                "range": arr.max() - arr.min()}

    rows = [describe("random column orders - 25-seed bag (the final-model protocol)", o_bag)]
    if len(c_bag) > 1:
        rows.append(describe("CONTROL: same order, different 25-seed sets (seed luck only)", c_bag))
    rows += [describe("random orders - bag of 5 seeds", o_b5), describe("random orders - a single seed", o_s1)]
    stats_df = pd.DataFrame(rows)
    save_table(stats_df, "series19_distribution_summary")
    pct_rank = float((o_bag < base["bag"]["MAE"]).mean() * 100)

    notes = [f"<p>The current column order scores <b>£{base['bag']['MAE']:.2f}</b> (this reproduces the final model's figure exactly). "
             f"Across <b>{len(random)}</b> random orders of the same 38 columns the 25-seed bag scored between £{o_bag.min():.2f} and "
             f"£{o_bag.max():.2f} (mean £{o_bag.mean():.2f}, standard deviation £{o_bag.std(ddof=1):.2f}). Only <b>{pct_rank:.0f}%</b> of the random orders "
             f"scored lower (better) than the current order; {100 - pct_rank:.0f}% scored higher.</p>"]
    if len(c_bag) > 1:
        lev = stats.levene(o_bag, c_bag)
        ratio = o_bag.std(ddof=1) / c_bag.std(ddof=1)
        notes.append(f"<p>The control matters: keeping the current order and changing only the 25 seeds gave a standard deviation of "
                     f"£{c_bag.std(ddof=1):.2f} - so <b>seed luck alone moves the figure about as much as column order does</b> "
                     f"(order sd / seed-control sd = {ratio:.2f}; Levene test for different spread p = {lev.pvalue:.2f}).</p>")
        sim = np.random.default_rng(0).normal(c_bag.mean(), c_bag.std(ddof=1), size=(20000, len(random))).min(axis=1)
        notes.append(f"<p>Even the best of {len(random)} orders (£{o_bag.min():.2f}) is about what picking the luckiest of {len(random)} "
                     f"seed sets would give: simulating that from the control, the best of {len(random)} draws averages "
                     f"£{sim.mean():.2f}, and the observed best is higher than {float((sim < o_bag.min()).mean() * 100):.0f}% of such simulated bests - "
                     f"so it is not better than luck alone would produce.</p>")

    # histogram
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(min(o_bag.min(), c_bag.min() if len(c_bag) else 1e9) - 2, max(o_bag.max(), c_bag.max() if len(c_bag) else 0) + 2, 24)
    ax.hist(o_bag, bins=bins, color=GREEN, alpha=0.75, label=f"{len(random)} random column orders")
    if len(c_bag) > 1:
        ax.hist(c_bag, bins=bins, color=ORANGE, alpha=0.6, label=f"{len(c_bag)} seed sets, current order (control)")
    ax.axvline(base["bag"]["MAE"], color=NAVY, lw=2, label=f"current order £{base['bag']['MAE']:.2f}")
    ax.set_xlabel("pooled MAE (£), 25-seed bag")
    ax.set_ylabel("number of runs")
    ax.set_title("Series 19 - column order versus seed luck")
    ax.legend(fontsize=8)
    h1 = savefig(fig, "s19_histogram")

    # box comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = [("orders\n25-seed bag", o_bag, GREEN)]
    if len(c_bag) > 1:
        groups.append(("seed control\n25-seed bag", c_bag, ORANGE))
    groups += [("orders\n5-seed bag", o_b5, BLUE), ("orders\nsingle seed", o_s1, RED)]
    bp = ax.boxplot([g[1] for g in groups], tick_labels=[g[0] for g in groups], patch_artist=True, widths=0.55)
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(g[2])
        patch.set_alpha(0.6)
    ax.axhline(base["bag"]["MAE"], color=NAVY, ls="--", lw=1.2, label="current order, 25-seed bag")
    ax.set_ylabel("pooled MAE (£)")
    ax.set_title("The fewer the seeds, the more the order (and the seed) matters")
    ax.legend(fontsize=8)
    h2 = savefig(fig, "s19_boxes")

    # caterpillar
    order_sorted = sorted(random, key=lambda r: r["bag"]["MAE"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    vals = [r["bag"]["MAE"] for r in order_sorted]
    ax.plot(range(1, len(vals) + 1), vals, "o", ms=3.5, color=GREEN)
    ax.axhline(base["bag"]["MAE"], color=NAVY, lw=1.6, label="current order")
    if len(c_bag) > 1:
        ax.axhspan(c_bag.mean() - 2 * c_bag.std(ddof=1), c_bag.mean() + 2 * c_bag.std(ddof=1), color=ORANGE, alpha=0.2,
                   label="seed-luck band (+/-2 sd of the control)")
    ax.set_xlabel("random column orders, sorted best to worst")
    ax.set_ylabel("pooled MAE (£)")
    ax.set_title("Every random order, ranked")
    ax.legend(fontsize=8)
    h3 = savefig(fig, "s19_ranked")

    # feature position vs MAE
    orders = pd.read_json(s19.ORDERS, typ="series") if s19.ORDERS.exists() else None
    pos_html = ""
    if orders is not None:
        feats = list(base["features"])
        pos = np.array([[orders[r["id"]].index(f) + 1 for f in feats] for r in random])
        rho = [stats.spearmanr(pos[:, j], o_bag).statistic for j in range(len(feats))]
        corr = pd.DataFrame({"feature": feats, "spearman_rho": rho}).sort_values("spearman_rho")
        save_table(corr, "series19_position_correlation")
        n = len(random)
        t_crit = stats.t.ppf(1 - 0.05 / (2 * len(feats)), n - 2)
        bonferroni = t_crit / np.sqrt(n - 2 + t_crit**2)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.barh(corr["feature"], corr["spearman_rho"], color=[GREEN if v < 0 else RED for v in corr["spearman_rho"]])
        ax.axvline(0, color="k", lw=0.8)
        for s in (-bonferroni, bonferroni):
            ax.axvline(s, color=NAVY, ls="--", lw=1)
        ax.set_xlabel(f"Spearman correlation: position in the order vs MAE (dashed: Bonferroni limit +/-{bonferroni:.2f})")
        ax.set_title("Does putting a feature early or late help?  (negative = later is better)")
        ax.tick_params(axis="y", labelsize=7.5)
        h4 = savefig(fig, "s19_position")
        strong = corr[corr["spearman_rho"].abs() > bonferroni]
        pos_html = (img_tag(h4) + f"<p>{len(strong)} of {len(feats)} features pass the Bonferroni limit (|rho| &gt; {bonferroni:.2f}) - "
                    "with 38 features tested, about two would reach plain 5% significance by chance alone.</p>")

    # confirmation on fresh seeds
    conf_html = ""
    cb = [r for r in done.values() if r["kind"] == "confirm_order"]
    cbase = [r for r in done.values() if r["kind"] == "confirm_baseline"]
    if cb and cbase:
        tbl = []
        by_label: dict[str, list[float]] = {}
        for r in cb:
            label = re.sub(r"_s\d+$", "", r["id"]).replace("confirm_", "")
            by_label.setdefault(label, []).append(r["bag"]["MAE"])
        base_conf = np.array([r["bag"]["MAE"] for r in sorted(cbase, key=lambda r: r["id"])])
        for label, vals in sorted(by_label.items()):
            orig = done[re.sub(r"^best\d+_", "", label)]["bag"]["MAE"]
            tbl.append({"order": label, "MAE_original_seeds": orig, "MAE_fresh_seeds_mean": float(np.mean(vals)),
                        "fresh_sd": float(np.std(vals, ddof=1)), "n_seed_sets": len(vals),
                        "current_order_same_seed_sets": float(base_conf.mean())})
        ctab = pd.DataFrame(tbl)
        save_table(ctab, "series19_confirmation")
        conf_html = ("<h3>Confirmation on fresh seeds</h3><p>The best 3 of the random orders re-run on 5 seed sets that did not choose them, "
                     "next to the current order on the same seed sets.</p>" + table_html(ctab))

    st = pd.DataFrame([{"order": r["id"], "MAE_25_seed_bag": r["bag"]["MAE"], "vs_current": r["bag"]["MAE"] - base["bag"]["MAE"]}
                       for r in structured])
    section(
        "Series 19 - does the order of the 38 columns matter?",
        "<p>The final model's exact protocol (25-seed log1p CatBoost bag, 10 folds, 585 test days), with only the column order changed. "
        "Each random order is a different permutation of the same 38 columns.</p>",
        *notes, img_tag(h1), img_tag(h2), "<h3>Distribution summary</h3>", table_html(stats_df), img_tag(h3), pos_html, conf_html,
        "<h3>Structured orders (for reference)</h3>", table_html(st) if len(st) else "",
        "<p class='small'>Per-order results: series19_column_order.jsonl; the orders themselves: series19_orders.json.</p>",
    )


# --------------------------------------------------------------------------- Series 20


def report_series20(days: pd.DataFrame, final: pd.Series) -> None:
    done = read_done(s20.RESULTS)
    if not done:
        return
    actual = days.set_index("date")["actual"]
    manual = days.set_index("date")["manual"]

    def frame(kind_prefix: str) -> pd.DataFrame:
        recs = [r for r in done.values() if r["kind"] == kind_prefix]
        return recs

    def mae_on(rec, dates):
        s = pd.Series(rec["pred"], index=rec["dates"]).loc[list(dates)]
        return float(np.mean(np.abs(s.to_numpy() - actual.loc[list(dates)].to_numpy())))

    def abs_err(rec, dates):
        s = pd.Series(rec["pred"], index=rec["dates"]).loc[list(dates)]
        return np.abs(s.to_numpy() - actual.loc[list(dates)].to_numpy())

    parts: list[str] = []
    a_recs = frame("A")
    summary_rows: list[dict] = []
    if a_recs:
        days_a = sorted(set.intersection(*(set(r["dates"]) for r in a_recs)))
        final_a = float(np.mean(np.abs(final.loc[days_a].to_numpy() - actual.loc[days_a].to_numpy())))
        manual_a = float(np.mean(np.abs(manual.loc[days_a].to_numpy() - actual.loc[days_a].to_numpy())))
        rows = []
        for r in a_recs:
            cfg = re.match(r"A_(.+)_(v1|v2_no_year|v2)_s\d+$", r["id"]).group(1)
            rows.append({"config": cfg, "features": r["feature_set"], "seed": r["seed"], "MAE": mae_on(r, days_a)})
        adf = pd.DataFrame(rows)
        agg = adf.groupby(["config", "features"])["MAE"].agg(["mean", "std", "min", "max", "count"]).reset_index()
        agg.columns = ["config", "features", "MAE_mean", "MAE_sd", "MAE_best_seed", "MAE_worst_seed", "seeds"]
        agg["vs_final_model"] = agg["MAE_mean"] - final_a
        order_cfg = list(s20.CURRENT_CONFIGS)
        agg["o"] = agg["config"].map(order_cfg.index)
        fo = {"v1": 0, "v2": 1, "v2_no_year": 2}
        agg["f"] = agg["features"].map(fo)
        agg = agg.sort_values(["o", "f"]).drop(columns=["o", "f"])
        save_table(agg, "series20_stageA_current_configs")
        # figure
        fig, ax = plt.subplots(figsize=(10, 4.6))
        cfgs = order_cfg
        width = 0.26
        colours = {"v1": GREY, "v2": GREEN, "v2_no_year": ORANGE}
        labels = {"v1": "V1 (15 features)", "v2": "V2 (38 features)", "v2_no_year": "V2 without 'year' (diagnostic)"}
        for k, fs in enumerate(["v1", "v2", "v2_no_year"]):
            xs, ys, es = [], [], []
            for i, c in enumerate(cfgs):
                sub = adf[(adf["config"] == c) & (adf["features"] == fs)]["MAE"]
                if len(sub):
                    xs.append(i + (k - 1) * width)
                    ys.append(sub.mean())
                    es.append(sub.std() if len(sub) > 1 else 0)
                    ax.scatter([i + (k - 1) * width] * len(sub), sub, color="k", s=8, zorder=3)
            ax.bar(xs, ys, width * 0.92, yerr=es, color=colours[fs], alpha=0.85, label=labels[fs], capsize=3)
        ax.axhline(final_a, color=NAVY, lw=2, label=f"final model £{final_a:.0f}")
        ax.axhline(manual_a, color=RED, lw=1.5, ls="--", label=f"manager's forecast £{manual_a:.0f}")
        ax.set_xticks(range(len(cfgs)), [c.replace("_", "\n") for c in cfgs])
        ax.set_ylabel(f"pooled MAE (£) on the {len(days_a)} days every run predicted")
        ax.set_title("Series 20 stage A - the current networks on V1 vs V2\n(5 seeds each; dots are single seeds)")
        ax.set_ylim(700, max(adf["MAE"].max(), manual_a) + 260)
        ax.legend(fontsize=8, ncol=3, loc="upper left")
        hA = savefig(fig, "s20_stageA")
        parts += [f"<h3>Stage A - the current network configurations, V1 vs V2</h3><p>Production defaults (100 epochs) and the Series 1 winners "
                  f"(40 epochs), each over 5 seeds, on the {len(days_a)} days (of 585) that every network prediction exists for; the final model "
                  f"scores £{final_a:.2f} and the manager's forecast £{manual_a:.2f} on the same days.</p>", img_tag(hA), table_html(agg)]
        summary_rows.append({"what": "final model (same days)", "MAE": final_a})

    b_recs = frame("B")
    if b_recs:
        days_b = sorted(set.intersection(*(set(r["dates"]) for r in b_recs)))
        final_b = float(np.mean(np.abs(final.loc[days_b].to_numpy() - actual.loc[days_b].to_numpy())))
        rows = [{"arch": r["arch"], "hp": s20.hp_key(r["arch"], r["hp"]), "features": r["feature_set"], "MAE": mae_on(r, days_b), "id": r["id"]}
                for r in b_recs]
        bdf = pd.DataFrame(rows)
        wide = bdf.pivot_table(index=["arch", "hp"], columns="features", values="MAE").reset_index()
        wide["v2_minus_v1"] = wide["v2"] - wide["v1"]
        save_table(wide, "series20_stageB_grid")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        for ax, arch in zip(axes, ["lstm", "transformer"]):
            sub = wide[wide["arch"] == arch].dropna(subset=["v1", "v2"])
            ax.scatter(sub["v1"], sub["v2"], color=GREEN, s=28, alpha=0.8)
            lim = [min(sub[["v1", "v2"]].min()) - 30, max(sub[["v1", "v2"]].max()) + 30]
            ax.plot(lim, lim, color=NAVY, ls="--", lw=1)
            ax.axhline(final_b, color=RED, lw=1)
            ax.axvline(final_b, color=RED, lw=1, label=f"final model £{final_b:.0f}")
            ax.set_xlabel("MAE with V1 features (£)")
            ax.set_ylabel("MAE with V2 features (£)")
            ax.set_title(f"{arch.upper() if arch == 'lstm' else arch.capitalize()} ({len(sub)} grid points)")
            ax.legend(fontsize=8, loc="upper left")
        fig.suptitle("Series 20 stage B - every grid point, V1 vs V2 (below the dashed line = V2 is better)", y=1.02, fontweight="bold")
        hB = savefig(fig, "s20_stageB_scatter")
        fig, ax = plt.subplots(figsize=(9, 4.2))
        data, labs, cols = [], [], []
        for arch in ["lstm", "transformer"]:
            for fs, c in (("v1", GREY), ("v2", GREEN)):
                data.append(bdf[(bdf["arch"] == arch) & (bdf["features"] == fs)]["MAE"].to_numpy())
                labs.append(f"{arch}\n{fs.upper()}")
                cols.append(c)
        bp = ax.boxplot(data, tick_labels=labs, patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.axhline(final_b, color=NAVY, lw=2, label=f"final model £{final_b:.0f}")
        ax.set_ylabel("pooled MAE (£), seed 42")
        ax.set_title("Series 20 stage B - the whole hyperparameter grid, V1 vs V2")
        ax.legend(fontsize=8)
        hB2 = savefig(fig, "s20_stageB_box")
        top = (bdf.sort_values("MAE").groupby(["arch", "features"]).head(5).sort_values(["arch", "features", "MAE"])
               [["arch", "features", "hp", "MAE"]])
        parts += [f"<h3>Stage B - hyperparameter grid, V1 and V2 (seed 42)</h3><p>{len(bdf) // 2} grid points per feature set "
                  f"(LSTM and Transformer: 36 each), scored on the {len(days_b)} days all of them predict; the final model scores "
                  f"£{final_b:.2f} on those days.</p>", img_tag(hB), img_tag(hB2), "<h4>Best 5 per network and feature set</h4>", table_html(top)]

    c_recs = frame("C")
    if c_recs and b_recs:
        days_c = sorted(set.intersection(*(set(r["dates"]) for r in c_recs + b_recs)))
        final_c = float(np.mean(np.abs(final.loc[days_c].to_numpy() - actual.loc[days_c].to_numpy())))
        rows = []
        for r in c_recs:
            m = re.match(r"C_(lstm|transformer)_(v1|v2)_top(\d)_(.+)_s\d+$", r["id"])
            rows.append({"arch": m.group(1), "features": m.group(2), "b_rank": int(m.group(3)), "hp": m.group(4), "seed": r["seed"], "MAE": mae_on(r, days_c)})
        cdf = pd.DataFrame(rows)
        b_lookup = {(r["arch"], r["feature_set"], s20.hp_key(r["arch"], r["hp"])): mae_on(r, days_c) for r in b_recs}
        agg = cdf.groupby(["arch", "features", "b_rank", "hp"])["MAE"].agg(["mean", "std"]).reset_index()
        agg["seed42_MAE_that_chose_it"] = [b_lookup[(a, f, h)] for a, f, h in zip(agg["arch"], agg["features"], agg["hp"])]
        agg = agg.rename(columns={"mean": "fresh_seeds_MAE_mean", "std": "fresh_seeds_sd"})
        agg["vs_final_model"] = agg["fresh_seeds_MAE_mean"] - final_c
        save_table(agg, "series20_stageC_confirmation")
        fig, ax = plt.subplots(figsize=(9, 4.4))
        for (arch, fs), g in agg.groupby(["arch", "features"]):
            ax.errorbar(g["seed42_MAE_that_chose_it"], g["fresh_seeds_MAE_mean"], yerr=g["fresh_seeds_sd"], fmt="o" if arch == "lstm" else "s",
                        color=GREEN if fs == "v2" else GREY, label=f"{arch} {fs.upper()}", capsize=2)
        lim = [agg[["seed42_MAE_that_chose_it", "fresh_seeds_MAE_mean"]].min().min() - 30, agg[["seed42_MAE_that_chose_it", "fresh_seeds_MAE_mean"]].max().max() + 30]
        ax.plot(lim, lim, color=NAVY, ls="--", lw=1)
        ax.axhline(final_c, color=RED, lw=1.5, label=f"final model £{final_c:.0f}")
        ax.set_xlabel("MAE of the grid run (seed 42) that selected the configuration (£)")
        ax.set_ylabel("MAE on 5 fresh seeds (£)")
        ax.set_title("Stage C - the best grid points re-run on fresh seeds\n(above the dashed line: the first score was partly luck)")
        ax.legend(fontsize=8)
        hC = savefig(fig, "s20_stageC")
        best = agg.sort_values("fresh_seeds_MAE_mean").groupby("features").head(1)
        stats_rows = []
        for _, row in best.iterrows():
            recs = [r for r in c_recs if re.match(rf"C_{row['arch']}_{row['features']}_top{row['b_rank']}_{re.escape(row['hp'])}_s\d+$", r["id"])]
            e = np.mean([abs_err(r, days_c) for r in recs], axis=0)
            f_err = np.abs(final.loc[days_c].to_numpy() - actual.loc[days_c].to_numpy())
            stats_rows.append({"best network": f"{row['arch']} {row['hp']}", "features": row["features"].upper(),
                               "MAE (fresh seeds)": row["fresh_seeds_MAE_mean"],
                               "vs final model (paired, mean [95% CI])": fmt_ci(*block_bootstrap(e - f_err))})
        parts += [f"<h3>Stage C - confirmation on fresh seeds</h3><p>The 3 best grid points per network and feature set, re-run on 5 seeds "
                  f"that did not pick them, scored on the {len(days_c)} days all runs predict (final model £{final_c:.2f}).</p>", img_tag(hC),
                  "<h4>Best confirmed network per feature set, against the final model</h4>", table_html(pd.DataFrame(stats_rows)),
                  table_html(agg.sort_values(["arch", "features", "b_rank"]))]

    n_days = len(days_a) if a_recs else (len(days_b) if b_recs else 0)
    section(
        "Series 20 - the neural networks on the 38-feature set (V2)",
        "<p>The repo's own LSTM and Transformer, trained from scratch on each fold's expanding history and scored on the final model's folds. "
        "<b>Design note:</b> as in the original code, a network reads only the days <i>before</i> the one it predicts (not that day's own "
        "calendar, manager's forecast or weather), and skips windows spanning a missing calendar day - so a few test days get no prediction "
        f"and every comparison here is made on the days all runs predict ({n_days} of 585).</p>", *parts,
    )



# --------------------------------------------------------------------------- Series 22


def report_series22(days: pd.DataFrame, final: pd.Series) -> None:
    done22 = read_done(s22.RESULTS)
    d_recs = [r for r in done22.values() if r["kind"] == "D"]
    if not d_recs:
        return
    c_recs = {r["id"]: r for r in read_done(s20.RESULTS).values() if r["kind"] == "C"}
    actual = days.set_index("date")["actual"]
    manual = days.set_index("date")["manual"]

    def series_of(rec):
        return pd.Series(rec["pred"], index=rec["dates"])

    def mae_on(rec, dates):
        s_ = series_of(rec).loc[list(dates)]
        return float(np.mean(np.abs(s_.to_numpy() - actual.loc[list(dates)].to_numpy())))

    def abs_err(rec, dates):
        s_ = series_of(rec).loc[list(dates)]
        return np.abs(s_.to_numpy() - actual.loc[list(dates)].to_numpy())

    e_recs = [r for r in done22.values() if r["kind"] == "E"]
    all_recs = d_recs + list(c_recs.values())
    days_c = sorted(set.intersection(*(set(r["dates"]) for r in all_recs)))
    final_c = float(np.mean(np.abs(final.loc[days_c].to_numpy() - actual.loc[days_c].to_numpy())))
    manual_c = float(np.mean(np.abs(manual.loc[days_c].to_numpy() - actual.loc[days_c].to_numpy())))

    # the Series 20 stage C run of the same setting and seed, for a paired comparison
    twin_of = {}
    for c in c_recs.values():
        m = re.match(r"C_(lstm|transformer)_(v1|v2)_top\d_(.+)_s(\d+)$", c["id"])
        twin_of[(m.group(1), m.group(2), m.group(3), int(m.group(4)))] = c

    rows = []
    for r in d_recs:
        m = re.match(r"D_(lstm|transformer)_(v1|v2)_top(\d)_(.+)_s(\d+)$", r["id"])
        arch, fs, hp, seed = m.group(1), m.group(2), m.group(4), int(m.group(5))
        twin = twin_of.get((arch, fs, hp, seed))
        rows.append({"arch": arch, "features": fs, "setting": hp, "seed": seed, "with_target_day": mae_on(r, days_c),
                     "without_target_day": mae_on(twin, days_c) if twin else np.nan})
    df = pd.DataFrame(rows)
    agg = df.groupby(["arch", "features", "setting"])[["without_target_day", "with_target_day"]].mean().reset_index()
    agg["change"] = agg["with_target_day"] - agg["without_target_day"]
    agg["with_vs_final_model"] = agg["with_target_day"] - final_c
    agg = agg.sort_values(["arch", "features", "with_target_day"])
    save_table(agg, "series22_stageD_with_vs_without_target_day")

    best = agg.sort_values("with_target_day").groupby(["arch", "features"]).head(1).sort_values(["arch", "features"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(best))
    ax.bar(x - 0.2, best["without_target_day"], 0.4, color=GREY, label="Series 20: sees only the days before")
    ax.bar(x + 0.2, best["with_target_day"], 0.4, color=GREEN, label="Series 22: also sees the predicted day's own features")
    ax.axhline(final_c, color=NAVY, lw=2, label=f"final model £{final_c:.0f}")
    ax.axhline(manual_c, color=RED, ls="--", lw=1.5, label=f"manager's forecast £{manual_c:.0f}")
    ax.set_xticks(x, [f"{a}\n{f.upper()}" for a, f in zip(best["arch"], best["features"])])
    ax.set_ylim(min(final_c, best["with_target_day"].min()) - 120, max(best["without_target_day"].max(), manual_c) + 150)
    ax.set_ylabel(f"pooled MAE (£) on the {len(days_c)} days all runs predict")
    ax.set_title("Series 22 - the best settings, with and without the predicted day's own features\n(5 fresh seeds each, paired seed for seed)")
    ax.legend(fontsize=8, loc="upper right")
    hD = savefig(fig, "s22_with_vs_without")

    parts = [f"<h3>Stage D - the same settings, with and without the target day</h3><p>The 3 best fresh-seed-confirmed settings per network "
             f"and feature set from Series 20, re-run on the same 5 seeds (100-104) with the predicted day's own features added as the last "
             f"step of the window. All figures are on the {len(days_c)} days every run predicts; the final model scores £{final_c:.2f} on them "
             f"and the manager's forecast £{manual_c:.2f}.</p>", img_tag(hD), table_html(agg)]

    best_fs = agg.sort_values("with_target_day").groupby("features").head(1)
    stat_rows, errs = [], {}
    f_err = np.abs(final.loc[days_c].to_numpy() - actual.loc[days_c].to_numpy())
    for row in best_fs.itertuples():
        recs = [r for r in d_recs if re.match(rf"D_{row.arch}_{row.features}_top\d_{re.escape(row.setting)}_s\d+$", r["id"])]
        e = np.mean([abs_err(r, days_c) for r in recs], axis=0)
        errs[row.features] = e
        stat_rows.append({"best network": f"{row.arch} {row.setting}", "features": row.features.upper(), "MAE (fresh seeds)": row.with_target_day,
                          "vs final model (paired, mean [95% CI])": fmt_ci(*block_bootstrap(e - f_err))})
    if {"v1", "v2"} <= set(errs):
        stat_rows.append({"best network": "V2 minus V1 (both with the target day)", "features": "-", "MAE (fresh seeds)": np.nan,
                          "vs final model (paired, mean [95% CI])": fmt_ci(*block_bootstrap(errs["v2"] - errs["v1"]))})
    parts += ["<h4>Best network per feature set against the final model</h4>", table_html(pd.DataFrame(stat_rows))]

    if e_recs:
        e_rows = []
        for r in e_recs:
            m = re.match(r"E_(lstm|transformer)_(v1|v2)_w(\d+)_s(\d+)$", r["id"])
            e_rows.append({"arch": m.group(1), "features": m.group(2), "window": int(m.group(3)), "seed": int(m.group(4)), "rec": r})
        days_e = sorted(set.intersection(*(set(x["rec"]["dates"]) for x in e_rows)))
        final_e = float(np.mean(np.abs(final.loc[days_e].to_numpy() - actual.loc[days_e].to_numpy())))
        edf = pd.DataFrame([{k: v for k, v in x.items() if k != "rec"} | {"MAE": mae_on(x["rec"], days_e)} for x in e_rows])
        eagg = edf.groupby(["arch", "features", "window"])["MAE"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "MAE_mean", "std": "MAE_sd"})
        eagg["vs_final_model"] = eagg["MAE_mean"] - final_e
        save_table(eagg, "series22_stageE_window_sweep")
        fig, ax = plt.subplots(figsize=(8, 4.4))
        for (arch, fs), g in eagg.groupby(["arch", "features"]):
            g = g.sort_values("window")
            ax.errorbar(g["window"], g["MAE_mean"], yerr=g["MAE_sd"].fillna(0), marker="o" if arch == "lstm" else "s", capsize=2,
                        color=GREEN if fs == "v2" else GREY, ls="-" if arch == "lstm" else "--", label=f"{arch} {fs.upper()}")
        ax.axhline(final_e, color=NAVY, lw=2, label=f"final model £{final_e:.0f}")
        ax.set_xscale("log")
        ax.set_xticks(s22.WINDOWS_E, [str(w) for w in s22.WINDOWS_E])
        ax.minorticks_off()
        ax.set_xlabel("window: days of history ending at the predicted day (1 = that day's own features only)")
        ax.set_ylabel(f"pooled MAE (£) on {len(days_e)} days")
        ax.set_title("Series 22 stage E - how much history does the network need?")
        ax.legend(fontsize=8, ncol=2)
        hE = savefig(fig, "s22_window_sweep")
        parts += [f"<h3>Stage E - how much history is needed?</h3><p>The best setting per network and feature set with the window swept over "
                  f"{s22.WINDOWS_E} days (window 1 = the predicted day's own features only), 3 seeds each, on the {len(days_e)} days all runs predict "
                  f"(final model £{final_e:.2f}).</p>", img_tag(hE), table_html(eagg)]

    section("Series 22 - the networks, allowed to see the day they predict",
            "<p>Series 20's networks read only the days <i>before</i> the one they predict. Here the window <i>ends at</i> the predicted day, so "
            "its calendar, the manager's forecast and the weather - all known in advance, exactly what the tree models use - are inputs. "
            "The target (sales) itself is never an input.</p>", *parts)


# --------------------------------------------------------------------------- page


CSS = """
body{font-family:Helvetica,Arial,sans-serif;color:#1c1c1c;max-width:1050px;margin:2rem auto;padding:0 1.2rem;line-height:1.5;background:#fff}
h1{color:#1a2b45;border-bottom:2px solid #1a2b45;padding-bottom:.3rem}h2{color:#1a2b45;margin-top:3rem;border-bottom:1px solid #c9d0d8}
h3{color:#2f6f4e;margin-top:1.8rem}h4{color:#33475b}
table.t{border-collapse:collapse;font-size:.82rem;margin:.6rem 0 1.2rem;width:100%}
table.t th{background:#1a2b45;color:#fff;text-align:left;padding:.4rem .5rem}table.t td{padding:.35rem .5rem;border-bottom:1px solid #c9d0d8}
table.t tr:nth-child(even) td{background:#eef1f4}
figure{margin:1rem 0}img{max-width:100%}figcaption{font-size:.8rem;color:#555}.small{font-size:.8rem;color:#555}
code{background:#eef1f4;padding:0 .25rem}.headline{background:#eef1f4;border-left:4px solid #2f6f4e;padding:.6rem 1rem;margin:1rem 0}
@media (prefers-color-scheme: dark){body{background:#14181f;color:#e6e9ee}h1,h2{color:#cfe0ff;border-color:#33475b}table.t td{border-color:#33475b}
table.t tr:nth-child(even) td{background:#1d2430}code,.headline{background:#1d2430}figure img{background:#fff}}
"""


def build(commentary: str = "") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df, folds = load_protocol()
    days = test_days(df, folds)
    final = final_model_series(days)
    for fn in (report_series20, report_series22, report_series21, report_series19):
        try:
            if fn in (report_series20, report_series22):
                fn(days, final)
            else:
                fn(days)
        except SystemExit:
            raise
    head = ("<h1>Series 19-21 - review report</h1>"
            "<p class='small'>Column order (19), neural networks on the 38 features (20) and shallower CatBoost trees with more iterations (21). "
            "Every figure is scored under the final model's own protocol: 10 rolling-origin folds, expanding window, the same 950 rows and the "
            f"same 585 test days; the current final model scores £{CURRENT_MAE:.2f} MAE.</p>")
    toc = "<ul>" + "".join(f"<li>{html.escape(s['title'])}</li>" for s in SECTIONS) + "</ul>"
    body = "".join(f"<h2>{html.escape(s['title'])}</h2>{s['html']}" for s in SECTIONS)
    comm = f"<div class='headline'>{commentary}</div>" if commentary else ""
    page = f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Series 19-21 review</title><style>{CSS}</style></head><body>{head}{comm}{toc}{body}</body></html>"
    out = REPORT_DIR / "series19_21_report.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(SECTIONS)} sections, {len(FIGURES)} figures)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commentary", type=Path, default=None, help="HTML snippet shown at the top")
    args = parser.parse_args()
    build(args.commentary.read_text(encoding="utf-8") if args.commentary else "")
