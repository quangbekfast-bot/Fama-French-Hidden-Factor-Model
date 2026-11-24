#!/usr/bin/env python3
"""
Return test with a transaction-cost grid for the factor-leadership HMM (monthly).

This script does NOT re-implement the model. It imports code/sta348_hmm_v1.py and
calls its own soft_init / fit_with_restarts / walk_forward, so the HMM, the
warm-started monthly refits, and all seven holding rules are exactly the main
specification: every decision for month t uses data through t-1 only.

What it adds on top of the main script's single 10 bp return table:
  * a grid of one-way trading costs (default 0 = frictionless, 5, 10, 25, 50, 100 bp)
  * net return, volatility, Sharpe, drawdown, turnover for every rule at every cost
  * the cost at which each rule stops beating equal weight (Sharpe basis)
  * paired t-statistic of the net return difference versus equal weight

Cost convention (same as the main script): cost = bps/1e4 x sum_k |w_{k,t} - w_{k,t-1}|
per period (a full switch from one factor to another trades 2 units), no cost for the
first period. Optional: --drift measures turnover against weights that have drifted
with returns, --entry-cost charges the initial purchase.

Run from the repository root:

  # full run (refits the HMM every month, about 2 minutes)
  python code/analysis/returns_backtest.py --out results/monthly_returns_test

  # fast: reuse the main run's weights, only apply the cost grid
  python code/analysis/returns_backtest.py --wf results/monthly_main/walk_forward.csv \
      --out results/monthly_returns_test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the code/ folder
import sta348_hmm_v1 as main_model  # noqa: E402

FACTORS = main_model.FACTORS
RULES = list(main_model.STRATEGIES)  # hmm_expected, hmm_top, hmm_tilt, chain_expected, ...


# ------------------------------------------------------------------ data ----
def load_clean(path):
    df = pd.read_csv(path)
    dates = pd.DatetimeIndex(pd.to_datetime(df["month_end"]))
    obs = df["top_code"].to_numpy(int)
    R = df[FACTORS].to_numpy(float)
    bad = int((R.argmax(axis=1) != obs).sum())
    if bad:
        print(f"WARNING: top_code differs from argmax of factor returns in {bad} rows.")
    return df, dates, obs, R


def train_test_split(df, T, train_frac):
    if "window" in df.columns and (df["window"] == "test").any():
        split = int((df["window"] != "test").to_numpy().sum())
    else:
        split = int(round(T * train_frac))
    if split != int(round(T * train_frac)):
        print(f"NOTE: split from 'window' column ({split}) differs from "
              f"round(T*train_frac) = {int(round(T * train_frac))}.")
    return split


# ----------------------------------------------------- main-model walk-forward ----
def run_main_model(obs, dates, R, split, args):
    train = obs[:split]
    base = main_model.base_rates(train)
    pi = np.full(main_model.N, 1 / main_model.N) if args.pi == "uniform" else base.copy()
    A0, B0 = main_model.soft_init(base, args.init_stay, args.init_boost)
    model, fit_info = main_model.fit_with_restarts(
        train, pi, A0, B0, args.restarts, args.jitter, args.n_iter, args.tol, False, args.seed)
    print(f"Initial fit on {split} training months: best log-likelihood {fit_info['best_loglik']:.2f}")
    wf, info = main_model.walk_forward(obs, split, model, pi, args.refit_every, args.n_iter,
                                       args.tol, False, args.smoothing, R)
    wf.insert(0, "period_ending", dates[wf["t"].to_numpy()])
    print(f"Walk-forward done: {len(wf)} test months, {info['refits']} refits.")
    return wf


# ---------------------------------------------------------------- costs ----
def traded_units(W, R, drift, entry_cost):
    """Per-period sum_k |w_t - w_{t-1}|. drift: compare with previous weights after
    they have moved with that period's returns. entry_cost: charge the first purchase."""
    to = np.zeros(len(W))
    prev = None
    for t in range(len(W)):
        if prev is None:
            to[t] = np.abs(W[t]).sum() if entry_cost else 0.0
        else:
            to[t] = np.abs(W[t] - prev).sum()
        prev = W[t] * (1 + R[t]) / (1 + W[t] @ R[t]) if drift else W[t].copy()
    return to


def perf(net, to, P=12):
    sd = net.std(ddof=1)
    wealth = (1 + net).cumprod()
    return {
        "ann_return_pct": net.mean() * P * 100,
        "ann_vol_pct": sd * np.sqrt(P) * 100,
        "sharpe": net.mean() / sd * np.sqrt(P) if sd > 0 else np.nan,
        "max_drawdown_pct": (wealth / np.maximum.accumulate(wealth) - 1).min() * 100,
        "avg_traded_per_month": to.mean(),
    }


def sharpe_path(gross, to, grid):
    net = gross[:, None] - to[:, None] * grid[None, :] / 1e4
    return net.mean(axis=0) / net.std(axis=0, ddof=1) * np.sqrt(12)


def beats_ew_up_to(gross, to, g_ew, to_ew, cap=500):
    """Largest one-way cost (bp, integer grid) up to which the rule's Sharpe still exceeds
    equal weight's at the same cost. NaN = never beats equal weight, even frictionless."""
    grid = np.arange(0, cap + 1, dtype=float)
    beats = sharpe_path(gross, to, grid) > sharpe_path(g_ew, to_ew, grid)
    if not beats[0]:
        return np.nan
    if beats.all():
        return float(cap)
    return float(grid[int(np.argmax(~beats)) - 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default="data/clean/sta348_factor_data_monthly.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wf", default=None,
                    help="existing walk_forward.csv from the main script; skips the HMM run")
    ap.add_argument("--costs", type=float, nargs="+", default=[0, 5, 10, 25, 50, 100],
                    help="one-way cost in bp per unit traded (0 = frictionless)")
    ap.add_argument("--drift", action="store_true", help="turnover vs drifted weights")
    ap.add_argument("--entry-cost", action="store_true", help="charge the first purchase")
    # model settings = main script defaults
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--pi", choices=["uniform", "train"], default="uniform")
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.2)
    ap.add_argument("--init-stay", type=float, default=0.8)
    ap.add_argument("--init-boost", type=float, default=2.0)
    ap.add_argument("--refit-every", type=int, default=1, help="0 = freeze A, B after training")
    ap.add_argument("--n-iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--smoothing", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=348)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, dates, obs, R = load_clean(args.clean)
    T = len(obs)
    split = train_test_split(df, T, args.train_frac)
    print(f"Panel {dates[0].date()} .. {dates[-1].date()} ({T} months); "
          f"train {split}, test {T - split} (from {dates[split].date()})")

    if args.wf:
        wf = pd.read_csv(args.wf)
        print(f"Using weights from {args.wf} ({len(wf)} test months); no refitting.")
    else:
        wf = run_main_model(obs, dates, R, split, args)
        wf.to_csv(out / "walk_forward.csv", index=False)

    # factor returns for the test months (needed only for --drift)
    Rt = (df.assign(month_end=pd.to_datetime(df["month_end"])).set_index("month_end")
            .loc[pd.to_datetime(wf["period_ending"]), FACTORS].to_numpy())

    rules = [k for k in RULES if f"w_{k}_{FACTORS[0]}" in wf.columns]
    gross, to, net_cols = {}, {}, {}
    for k in rules:
        W = wf[[f"w_{k}_{f}" for f in FACTORS]].to_numpy()
        gross[k] = wf[f"ret_{k}"].to_numpy()
        to[k] = traded_units(W, Rt, args.drift, args.entry_cost)

    rows = []
    for k in rules:
        for bps in args.costs:
            net = gross[k] - to[k] * bps / 1e4
            d = net - (gross["equal_weight"] - to["equal_weight"] * bps / 1e4)
            t_stat = (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                      if k != "equal_weight" and d.std() > 0 else np.nan)
            net_cols[f"net_{k}_{bps:g}bp"] = net
            rows.append({"rule": k, "cost_bp": bps, **perf(net, to[k]), "t_vs_equal_weight": t_stat})
    summ = pd.DataFrame(rows)
    summ["beats_ew_up_to_bp"] = summ["rule"].map(
        {k: (beats_ew_up_to(gross[k], to[k], gross["equal_weight"], to["equal_weight"])
             if k != "equal_weight" else np.nan) for k in rules})

    ret_out = pd.DataFrame({"period_ending": wf["period_ending"]})
    for k in rules:
        ret_out[f"gross_{k}"] = gross[k]
        ret_out[f"traded_{k}"] = to[k]
    ret_out = pd.concat([ret_out, pd.DataFrame(net_cols)], axis=1)
    ret_out.to_csv(out / "returns_by_month.csv", index=False)
    summ.to_csv(out / "returns_cost_grid.csv", index=False)
    (out / "settings.json").write_text(json.dumps(
        {**vars(args), "test_months": len(wf), "test_start": str(wf["period_ending"].iloc[0]),
         "test_end": str(wf["period_ending"].iloc[-1])}, indent=2, default=str))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    order = [k for k in rules]
    cap = "with drift" if args.drift else "no drift (main-script convention)"
    print(f"\nTurnover: {cap}; entry cost {'charged' if args.entry_cost else 'not charged'}")
    print("\nSharpe ratio by rule and one-way cost (bp); 0 = frictionless:")
    print(summ.pivot(index="rule", columns="cost_bp", values="sharpe").loc[order].round(3))
    print("\nNet annual return (%) by rule and cost:")
    print(summ.pivot(index="rule", columns="cost_bp", values="ann_return_pct").loc[order].round(2))
    print("\nFrictionless detail:")
    z = summ[summ.cost_bp == min(args.costs)].set_index("rule").loc[order]
    print(z[["ann_return_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct",
             "avg_traded_per_month", "t_vs_equal_weight", "beats_ew_up_to_bp"]].round(3))
    print("\nbeats_ew_up_to_bp: highest one-way cost at which the rule's Sharpe still exceeds equal "
          "weight's (NaN = never beats it, even with no costs; 500 = still ahead at 500 bp).")
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
