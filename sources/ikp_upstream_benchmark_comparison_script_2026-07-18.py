#!/usr/bin/env python3
"""Compare official knowledge benchmark scores vs IKP as a parameter-count proxy.

Joins data/benchmarks/benchmark_scores.csv (vendor-published official numbers)
with data/densing_analysis_data.csv (params + IKP penalized accuracy).

For each benchmark we fit OLS against log10(total params) and report R^2, slope,
N. Crucially we also fit IKP on the SAME subset so the comparison is apples to
apples (controls for which models happen to publish that benchmark).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IKP_CSV = ROOT / "data" / "densing_analysis_data.csv"
BENCH_CSV = ROOT / "data" / "benchmarks" / "benchmark_scores.csv"
OUT_DIR = ROOT / "data" / "benchmarks"
FIG_DIR = ROOT / "paper" / "figures"


# Same calibration excludes used by loo_cv_analysis.py (broken APIs + post-training outliers)
CALIBRATION_EXCLUDE = {
    "minimax-m1-think",
    "hunyuan-a13b",
    "hunyuan-a13b-think",
    "hermes-3-405b",
    "ling-2.6-flash",
    "deepseek-v3.1-nex-n1",
    "intellect-3-think",
}


def fit_ols(x: np.ndarray, y: np.ndarray):
    res = stats.linregress(x, y)
    return {
        "n": len(x),
        "r2": res.rvalue ** 2,
        "slope": res.slope,
        "intercept": res.intercept,
        "p": res.pvalue,
        "stderr": res.stderr,
    }


def fit_with_time(x_params: np.ndarray, x_months: np.ndarray, y: np.ndarray):
    """OLS y ~ log_params + months. Returns slope on params, slope on time, R^2, n."""
    X = np.column_stack([np.ones_like(x_params), x_params, x_months])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return {"n": len(y), "intercept": beta[0], "slope_params": beta[1], "slope_months": beta[2], "r2": r2}


def main() -> int:
    ikp = pd.read_csv(IKP_CSV)
    bench = pd.read_csv(BENCH_CSV)

    df = ikp.merge(bench, on="model", how="left")
    print(f"IKP models: {len(ikp)}, benchmark rows: {len(bench)}, joined: {len(df)}")
    df = df[~df["model"].isin(CALIBRATION_EXCLUDE)].reset_index(drop=True)
    print(f"After calibration excludes: {len(df)}")

    benchmarks = ["mmlu", "mmlu_pro", "gpqa_diamond", "simpleqa"]
    bench_labels = {
        "mmlu": "MMLU",
        "mmlu_pro": "MMLU-Pro",
        "gpqa_diamond": "GPQA Diamond",
        "simpleqa": "SimpleQA",
    }

    rows = []
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()

    # --- IKP reference panel: ALL models with params ---
    ikp_x = df["log10_params"].to_numpy()
    ikp_y = (df["pen_acc"] * 100).to_numpy()
    ikp_fit = fit_ols(ikp_x, ikp_y)
    rows.append({"metric": "IKP (full set)", **ikp_fit})

    ax = axes[0]
    ax.scatter(ikp_x, ikp_y, s=22, alpha=0.65, color="tab:blue", edgecolor="white", linewidth=0.4)
    xs = np.linspace(ikp_x.min(), ikp_x.max(), 50)
    ax.plot(xs, ikp_fit["slope"] * xs + ikp_fit["intercept"], color="black", lw=1.6)
    ax.set_title(f"IKP (ours, full set)\n$R^2$={ikp_fit['r2']:.3f},  N={ikp_fit['n']}")
    ax.set_xlabel(r"$\log_{10}(\mathrm{total\ params,\ B})$")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    # --- Per-benchmark panels ---
    for i, b in enumerate(benchmarks, start=1):
        sub = df.dropna(subset=[b]).copy()
        if len(sub) < 4:
            print(f"  {b}: only {len(sub)} models, skipping")
            continue
        x = sub["log10_params"].to_numpy()
        y_b = sub[b].to_numpy()  # already in 0-100
        y_ikp = (sub["pen_acc"] * 100).to_numpy()

        fit_b = fit_ols(x, y_b)
        fit_ikp_sub = fit_ols(x, y_ikp)
        rows.append({"metric": f"{b}", **fit_b})
        rows.append({"metric": f"IKP (subset matching {b})", **fit_ikp_sub})

        ax = axes[i]
        label = bench_labels.get(b, b)
        ax.scatter(x, y_b, s=26, alpha=0.75, color="tab:red", edgecolor="white", linewidth=0.4, label=label)
        ax.scatter(x, y_ikp, s=22, alpha=0.45, color="tab:blue", marker="x", label="IKP (same models)")
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, fit_b["slope"] * xs + fit_b["intercept"], color="darkred", lw=1.6)
        ax.plot(xs, fit_ikp_sub["slope"] * xs + fit_ikp_sub["intercept"], color="navy", lw=1.6, ls="--")
        ax.set_title(f"{label}\n{label} $R^2$={fit_b['r2']:.3f}  vs  IKP $R^2$={fit_ikp_sub['r2']:.3f},  N={fit_b['n']}")
        ax.set_xlabel(r"$\log_{10}(\mathrm{total\ params,\ B})$")
        if i % 3 == 0:
            ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    # hide unused subplot (5 panels in a 2x3 grid)
    for j in range(len(benchmarks) + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Official knowledge benchmarks vs incompressible knowledge probes",
        y=1.00, fontsize=13,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "benchmark_comparison.pdf"
    out_png = FIG_DIR / "benchmark_comparison.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"saved {out_pdf}")
    print(f"saved {out_png}")

    summary = pd.DataFrame(rows)
    summary["r2"] = summary["r2"].round(4)
    summary["slope"] = summary["slope"].round(4)
    summary["p"] = summary["p"].apply(lambda v: f"{v:.2e}")
    out_csv = OUT_DIR / "regression_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"saved {out_csv}")
    print()
    print(summary.to_string(index=False))

    # ── Time-coefficient analysis ────────────────────────────────────────────
    # Densing-Law style fit: metric ~ log_params + months. The time slope tells
    # us whether the benchmark drifts over time at fixed parameter count
    # (compressible) or stays flat (incompressible, like IKP).
    print("\n=== Time-coefficient analysis (Densing-style) ===")
    time_rows = []
    for b in benchmarks:
        sub = df.dropna(subset=[b]).copy()
        if len(sub) < 6:
            continue
        x_p = sub["log10_params"].to_numpy()
        x_m = sub["months"].to_numpy()
        y_b = sub[b].to_numpy()
        y_ikp = (sub["pen_acc"] * 100).to_numpy()
        time_rows.append({"metric": b, **fit_with_time(x_p, x_m, y_b)})
        time_rows.append({"metric": f"IKP (subset {b})", **fit_with_time(x_p, x_m, y_ikp)})

    # Full IKP set
    time_rows.append({
        "metric": "IKP (full set)",
        **fit_with_time(df["log10_params"].to_numpy(), df["months"].to_numpy(), (df["pen_acc"] * 100).to_numpy()),
    })

    tdf = pd.DataFrame(time_rows)
    for c in ["intercept", "slope_params", "slope_months", "r2"]:
        tdf[c] = tdf[c].round(4)
    tdf.to_csv(OUT_DIR / "time_coefficients.csv", index=False)
    print(tdf.to_string(index=False))

    # also dump the joined per-model table for the appendix
    keep = [
        "model", "params_B", "log10_params", "release_date", "pen_acc",
        *benchmarks,
    ]
    joined = df[keep].copy()
    joined.to_csv(OUT_DIR / "joined_per_model.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
