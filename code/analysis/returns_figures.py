#!/usr/bin/env python3
"""
Figures for the return test. Reads the output folder of returns_backtest.py
(returns_by_month.csv and returns_cost_grid.csv) and writes PNG + PDF figures
(PDF is vector, for LaTeX) to <folder>/figs/.

  fig1_wealth          growth of 1 dollar, frictionless and after costs (log scale)
  fig2_sharpe_vs_cost  Sharpe ratio of every rule as one-way cost rises 0 -> 100 bp
  fig3_risk_return     annual return vs volatility, frictionless, with equal-weight
                       iso-Sharpe line (points above the line beat equal weight on Sharpe)
  fig4_drawdown        drawdowns of equal weight, HMM tilt, HMM top (frictionless)

Run from the repository root:
  python code/analysis/returns_figures.py results/monthly_returns_test
  python code/analysis/returns_figures.py results/monthly_returns_test --out docs/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RULES = ["hmm_tilt", "hmm_top", "hmm_expected", "chain_expected",
         "last_winner", "best_so_far", "equal_weight"]
LABEL = {"hmm_tilt": "HMM tilt", "hmm_top": "HMM top", "hmm_expected": "HMM expected",
         "chain_expected": "Chain expected", "last_winner": "Last winner",
         "best_so_far": "Best so far", "equal_weight": "Equal weight"}
STYLE = {  # colour, linestyle
    "hmm_tilt": ("#1b6ca8", "-"), "hmm_top": ("#d95f02", "-"),
    "hmm_expected": ("#e7298a", "-"), "chain_expected": ("#1b9e77", "--"),
    "last_winner": ("#7570b3", "--"), "best_so_far": ("#a6761d", ":"),
    "equal_weight": ("black", "-"),
}


def save(fig, folder, name):
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def net(df, rule, bp):
    col = f"net_{rule}_{bp:g}bp"
    return df[col].to_numpy() if col in df else df[f"gross_{rule}"].to_numpy() - \
        df[f"traded_{rule}"].to_numpy() * bp / 1e4


def sharpe(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(12)


def fig_wealth(df, dates, folder, cost_bp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, bp, title in [(axes[0], 0, "Frictionless (0 bp)"),
                          (axes[1], cost_bp, f"After costs ({cost_bp:g} bp one-way)")]:
        for r in RULES:
            c, ls = STYLE[r]
            ax.plot(dates, np.cumprod(1 + net(df, r, bp)), color=c, ls=ls,
                    lw=2.2 if r == "equal_weight" else 1.4, label=LABEL[r])
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Growth of $1 (log scale)")
    axes[1].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle("Cumulative wealth of each rule, walk-forward test period", y=1.02)
    save(fig, folder, "fig1_wealth")


def fig_sharpe_cost(df, folder):
    grid = np.arange(0, 101, 1.0)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for r in RULES:
        g, t = df[f"gross_{r}"].to_numpy(), df[f"traded_{r}"].to_numpy()
        s = [sharpe(g - t * c / 1e4) for c in grid]
        c_, ls = STYLE[r]
        ax.plot(grid, s, color=c_, ls=ls, lw=2.4 if r == "equal_weight" else 1.5, label=LABEL[r])
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("One-way trading cost (bp per unit of weight traded)")
    ax.set_ylabel("Annualised Sharpe ratio")
    ax.set_title("Sharpe ratio versus transaction cost")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.3)
    save(fig, folder, "fig2_sharpe_vs_cost")


def fig_risk_return(grid, folder):
    z = grid[grid.cost_bp == grid.cost_bp.min()].set_index("rule")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    xmax = z["ann_vol_pct"].max() * 1.15
    xs = np.array([0, xmax])
    ax.plot(xs, z.loc["equal_weight", "sharpe"] * xs, color="black", lw=0.9, ls="--",
            label="Equal-weight Sharpe (iso-Sharpe line)")
    # (dx, dy, horizontal alignment) so neighbouring labels do not overlap
    off = {"chain_expected": (-8, 5, "right"), "last_winner": (-8, -10, "right"),
           "hmm_top": (-6, 9, "left"), "best_so_far": (8, -4, "left"),
           "hmm_expected": (8, -4, "left"), "hmm_tilt": (8, 2, "left"),
           "equal_weight": (8, -10, "left")}
    for r in RULES:
        c, _ = STYLE[r]
        ax.scatter(z.loc[r, "ann_vol_pct"], z.loc[r, "ann_return_pct"], color=c, s=55, zorder=3)
        dx, dy, ha = off[r]
        ax.annotate(f"{LABEL[r]} ({z.loc[r, 'sharpe']:.3f})",
                    (z.loc[r, "ann_vol_pct"], z.loc[r, "ann_return_pct"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=8)
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Annualised return (%)")
    ax.set_title("Risk and return, frictionless (Sharpe in brackets)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.3)
    save(fig, folder, "fig3_risk_return")


def fig_drawdown(df, dates, folder):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for r in ["equal_weight", "hmm_tilt", "hmm_top"]:
        w = np.cumprod(1 + net(df, r, 0))
        c, ls = STYLE[r]
        ax.plot(dates, (w / np.maximum.accumulate(w) - 1) * 100, color=c, ls=ls,
                lw=2 if r == "equal_weight" else 1.3, label=LABEL[r])
    ax.set_ylabel("Drawdown from previous peak (%)")
    ax.set_title("Drawdowns, frictionless")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(alpha=0.3)
    save(fig, folder, "fig4_drawdown")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="output folder of returns_backtest.py")
    ap.add_argument("--out", default=None, help="output folder (default: <folder>/figs)")
    ap.add_argument("--cost-bp", type=float, default=10, help="cost for the right panel of fig 1")
    args = ap.parse_args()
    folder = Path(args.folder)
    df = pd.read_csv(folder / "returns_by_month.csv")
    grid = pd.read_csv(folder / "returns_cost_grid.csv")
    dates = pd.to_datetime(df["period_ending"])
    figs = Path(args.out) if args.out else folder / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    fig_wealth(df, dates, figs, args.cost_bp)
    fig_sharpe_cost(df, figs)
    fig_risk_return(grid, figs)
    fig_drawdown(df, dates, figs)
    print(f"Wrote 4 figures (png + pdf) to {figs}/")


if __name__ == "__main__":
    main()
