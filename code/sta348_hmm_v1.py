"""
STA348 factor-leadership HMM, version 1.

Implements the design in "Sta348 Model":

  * Hidden states  Q = [SMB, HML, MOM]  ("factor k is currently favoured")
  * Observations   o_t = argmax(SMB_t, HML_t, MOM_t), the factor that topped week t
  * Parameters     lambda = (A, B, pi), A 3x3 transitions, B 3x3 emissions
  * Learning       Baum-Welch (forward-backward) via hmmlearn's CategoricalHMM,
                   soft-named start (each state's B row tilted towards its own factor),
                   several jittered restarts, best log-likelihood kept
  * Decoding       Viterbi on the training window (describes history, uses hindsight)
  * Forecasting    forward algorithm only (filtered probabilities, no look-ahead)
  * Validation     walk-forward over the test window: predict next week's top factor,
                   observe it, refit, repeat. Benchmarked against the raw MLE
                   transition matrix (observable chain), "same as last week" and
                   "most frequent factor".
  * Returns test   the return earned by following each rule (and equal weight)
                   over the test window, gross and after trading costs.

Data sources (pick one sub-command):

  french    Ken French CSVs (or the .zip downloads). Weekly: the 3-factor daily or
            weekly file plus the momentum daily file (compounded into weeks).
            Monthly: the monthly 3-factor file plus the monthly momentum file.
            The frequency is detected from the dates.
  sp        S&P DJI "PerformanceGraphExport" .xls index-level files. Index names
            are matched to factors by keyword (SmallCap -> SMB, Value -> HML,
            Momentum -> MOM). These are long-only index levels, not long-short
            factors, so treat results as a pipeline test unless you have a reason
            to believe they proxy the factors.
  simulate  Generate data from a known A and B, fit, and check the estimator
            recovers them (validation step: does the code work?).

Examples:

  python sta348_hmm_v1.py french --ff3 F-F_Research_Data_Factors_daily_CSV.zip \
         --mom F-F_Momentum_Factor_daily_CSV.zip --out results_french
  python sta348_hmm_v1.py french --ff3 F-F_Research_Data_Factors_CSV.zip \
         --mom F-F_Momentum_Factor_CSV.zip --out results_monthly
  python sta348_hmm_v1.py sp --xls PerformanceGraphExport*.xls --out results_sp
  python sta348_hmm_v1.py simulate --weeks 3000 --out results_sim

Requires: numpy, pandas, hmmlearn (>= 0.3), xlrd (only for the sp sub-command).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import CategoricalHMM

FACTORS = ["SMB", "HML", "MOM"]
N = len(FACTORS)

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def _read_text(path: str) -> str:
    """Read a CSV, or the first CSV inside a .zip (Ken French downloads are zipped)."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            member = next(n for n in zf.namelist() if n.lower().endswith((".csv", ".txt")))
            return zf.read(member).decode("latin-1")
    return p.read_text(encoding="latin-1")


def read_french_daily(path: str) -> pd.DataFrame:
    """Parse a Ken French CSV: text preamble, a header row starting with a comma,
    then rows 'YYYYMMDD, ...' (daily/weekly) or 'YYYYMM, ...' (monthly, dated at
    month end), values in percent. The annual block appended to monthly files is
    skipped. Returns decimals."""
    lines = _read_text(path).splitlines()
    header, rows = None, []
    for line in lines:
        parts = [x.strip() for x in line.split(",")]
        if re.fullmatch(r"\d{8}|\d{6}", parts[0] or "") and (not rows or len(parts[0]) == len(rows[0][0])):
            rows.append(parts)
        elif rows:            # first non-data line after the data block ends it
            break
        elif line.startswith(","):
            header = parts[1:]
    if header is None or not rows:
        raise ValueError(f"Could not find a daily data block in {path}")
    df = pd.DataFrame([r[1:len(header) + 1] for r in rows], columns=header)
    if len(rows[0][0]) == 6:
        df.index = pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        df.index = pd.to_datetime([r[0] for r in rows], format="%Y%m%d")
    df = df.apply(pd.to_numeric, errors="coerce").replace([-99.99, -999], np.nan)
    return df / 100.0


def load_french(ff3_path: str, mom_path: str) -> pd.DataFrame:
    """Return WEEKLY or MONTHLY decimal returns for SMB, HML, MOM.

    Monthly: pass the monthly 3-factor and monthly momentum files; they are
    joined on month.
    Weekly: the 3-factor file may be daily or weekly; the momentum file is
    daily (there is no weekly momentum file).
      * daily 3-factor file: everything is compounded into weeks ending Friday.
      * weekly 3-factor file: daily momentum is compounded into French's own
        weeks, i.e. each day d goes to the first weekly date on or after d, so
        all three factors cover exactly the same trading days.
    """
    ff3 = read_french_daily(ff3_path)
    mom = read_french_daily(mom_path)
    mom.columns = ["MOM" if c.lower().startswith("mom") else c for c in mom.columns]
    mom = mom["MOM"].dropna()

    gap = np.median(np.diff(ff3.index.values).astype("timedelta64[D]").astype(int))
    if gap >= 25:                                   # monthly files
        out = ff3[["SMB", "HML"]].join(mom.rename("MOM"), how="inner").dropna()
        return out[FACTORS]
    weekly_ff3 = gap >= 5
    if not weekly_ff3:
        daily = ff3[["SMB", "HML"]].join(mom, how="inner").dropna()
        return to_weekly(daily[FACTORS])

    weeks = ff3.index
    mom = mom[(mom.index > weeks[0] - pd.Timedelta(days=7)) & (mom.index <= weeks[-1])]
    week_of_day = weeks[np.searchsorted(weeks.values, mom.index.values, side="left")]
    mom_weekly = (1.0 + mom).groupby(week_of_day).prod() - 1.0
    days_per_week = mom.groupby(week_of_day).size()
    out = ff3[["SMB", "HML"]].join(mom_weekly.rename("MOM"), how="inner").dropna()
    # the first week can be partial if momentum starts mid-week; drop it
    first = out.index[0]
    if days_per_week.get(first, 0) < 4:
        out = out.iloc[1:]
    return out[FACTORS]


def load_sp_xls(paths: list[str]) -> pd.DataFrame:
    """Read S&P DJI PerformanceGraphExport .xls files (one index per file) and
    return daily simple returns, columns mapped to SMB/HML/MOM by keyword."""
    import xlrd

    keyword = {"SMB": ["smallcap", "small cap", "small-cap"],
               "HML": ["value"],
               "MOM": ["momentum"]}
    series = {}
    for path in paths:
        sh = xlrd.open_workbook(path).sheet_by_index(0)
        name, dates, levels = None, [], []
        for r in range(sh.nrows):
            v = sh.row_values(r)
            if isinstance(v[0], str) and v[0].strip().lower().startswith("effective date"):
                name = str(v[1]).strip()
            elif name and isinstance(v[0], float) and isinstance(v[1], float):
                dates.append(xlrd.xldate_as_datetime(v[0], sh.book.datemode))
                levels.append(v[1])
        if name is None:
            raise ValueError(f"No 'Effective date' header found in {path}")
        factor = next((f for f, kws in keyword.items()
                       if any(k in name.lower() for k in kws)), None)
        if factor is None:
            raise ValueError(f"Cannot map index '{name}' to SMB/HML/MOM ({path})")
        if factor in series:
            raise ValueError(f"Two files map to {factor}: check the index names")
        print(f"  {factor} <- {name}  ({dates[0].date()} to {dates[-1].date()}, {len(dates)} days)")
        series[factor] = pd.Series(levels, index=pd.DatetimeIndex(dates)).sort_index()
    missing = [f for f in FACTORS if f not in series]
    if missing:
        raise ValueError(f"Missing factors: {missing}")
    levels = pd.concat(series, axis=1, join="inner")[FACTORS]
    return levels.pct_change().dropna()


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Compound daily decimal returns into weeks ending Friday; drop empty weeks."""
    grouped = (1.0 + daily).resample("W-FRI")
    weekly = grouped.prod() - 1.0
    return weekly[grouped.count().min(axis=1) > 0]


def make_observations(weekly: pd.DataFrame, vol_scale: bool, vol_window: int):
    """o_t = index of the factor with the highest (optionally vol-scaled) return.
    Vol scaling uses only past weeks (shifted rolling std) to avoid look-ahead."""
    x = weekly[FACTORS]
    if vol_scale:
        sd = x.rolling(vol_window, min_periods=vol_window).std().shift(1)
        x = (x / sd).dropna()
    obs = x.to_numpy().argmax(axis=1)
    return x.index, obs


# ---------------------------------------------------------------------------
# 2. HMM building blocks
# ---------------------------------------------------------------------------

def normalise_rows(m: np.ndarray) -> np.ndarray:
    return m / m.sum(axis=1, keepdims=True)


def base_rates(obs: np.ndarray) -> np.ndarray:
    return np.bincount(obs, minlength=N) / len(obs)


def soft_init(base: np.ndarray, stay: float, boost: float):
    """Soft naming: state j's emission row is the overall top-factor frequencies
    with factor j multiplied by `boost`. A starts with a heavy diagonal."""
    B0 = np.tile(base, (N, 1))
    B0[np.arange(N), np.arange(N)] *= boost
    B0 = normalise_rows(B0)
    A0 = np.full((N, N), (1 - stay) / (N - 1))
    np.fill_diagonal(A0, stay)
    return A0, B0


def make_model(pi, A, B, n_iter, tol, learn_pi, seed=0) -> CategoricalHMM:
    m = CategoricalHMM(n_components=N, n_features=N, n_iter=n_iter, tol=tol,
                       init_params="", params="ste" if learn_pi else "te",
                       random_state=seed)
    m.startprob_ = np.asarray(pi, float).copy()
    m.transmat_ = np.asarray(A, float).copy()
    m.emissionprob_ = np.asarray(B, float).copy()
    return m


def fit_with_restarts(obs, pi, A0, B0, restarts, jitter, n_iter, tol, learn_pi, seed):
    """Baum-Welch from the soft-named start plus jittered copies of it.
    Returns the best model and a summary of all runs."""
    rng = np.random.default_rng(seed)
    X = obs.reshape(-1, 1)
    runs = []
    for r in range(restarts):
        if r == 0:
            A, B = A0, B0
        else:
            A = normalise_rows((1 - jitter) * A0 + jitter * rng.dirichlet(np.ones(N), N))
            B = normalise_rows((1 - jitter) * B0 + jitter * rng.dirichlet(np.ones(N), N))
        m = make_model(pi, A, B, n_iter, tol, learn_pi, seed + r)
        m.fit(X)
        runs.append((m.score(X), m))
    runs.sort(key=lambda x: x[0], reverse=True)
    best_ll, best = runs[0]
    keeps_names = [bool(np.all(state_names(m.emissionprob_, base_rates(obs)) == np.arange(N)))
                   for _, m in runs]
    summary = {"best_loglik": best_ll,
               "logliks": [ll for ll, _ in runs],
               "share_of_runs_keeping_soft_names": float(np.mean(keeps_names))}
    return best, summary


def forward_filter(obs, pi, A, B):
    """Scaled forward algorithm. Row t = P(state at t | o_1..o_t). Also returns
    log P(O | lambda)."""
    T = len(obs)
    alpha = np.empty((T, N))
    a = pi * B[:, obs[0]]
    c = a.sum()
    alpha[0], loglik = a / c, np.log(c)
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[:, obs[t]]
        c = a.sum()
        alpha[t] = a / c
        loglik += np.log(c)
    return alpha, loglik


def forward_step(alpha_prev, o, A, B):
    a = (alpha_prev @ A) * B[:, o]
    return a / a.sum()


def stationary(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(A.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def mle_transition(obs: np.ndarray, smoothing: float) -> np.ndarray:
    """Observable-chain benchmark: count o_t -> o_{t+1} transitions.
    `smoothing` adds a pseudo-count per cell so no probability is exactly 0."""
    C = np.full((N, N), smoothing, float)
    np.add.at(C, (obs[:-1], obs[1:]), 1)
    return normalise_rows(C)


def lift(B: np.ndarray, base: np.ndarray) -> np.ndarray:
    """B[j,k] / overall share of weeks factor k topped. >1 = over-represented."""
    return B / base


def state_names(B: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Name each state by the factor with the highest lift."""
    return lift(B, base).argmax(axis=1)


def run_lengths(seq: np.ndarray) -> dict:
    """Average length of consecutive runs of each value."""
    out = {f: [] for f in FACTORS}
    start = 0
    for t in range(1, len(seq) + 1):
        if t == len(seq) or seq[t] != seq[start]:
            out[FACTORS[seq[start]]].append(t - start)
            start = t
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in out.items()}


# ---------------------------------------------------------------------------
# 3. Walk-forward validation
# ---------------------------------------------------------------------------

def state_means(obs, R, pi, A, B):
    """Average factor return in each hidden state, weighting each past period by
    its smoothed state probability (uses only the data passed in)."""
    m = make_model(pi, A, B, 1, 1e-6, False)
    g = m.predict_proba(obs.reshape(-1, 1))
    return (g.T @ R) / g.sum(axis=0)[:, None]


def walk_forward(obs, split, model, pi, refit_every, n_iter, tol, learn_pi, smoothing,
                 R=None):
    """For each test period t: use data up to t-1 to predict o_t, then observe o_t.
    The HMM uses filtered (forward-only) probabilities; it is refitted every
    `refit_every` periods on all data so far, warm-started from the current
    parameters so state labels stay put (0 = never refit).

    If R (T x 3 factor returns) is given, it also records the return earned by
    several holding rules, each decided with data up to t-1 only:
      hmm_expected  hold the factor with the highest HMM expected return
                    (next-period state probabilities x each state's average return)
      hmm_top       hold the factor the HMM thinks most likely to top
      hmm_tilt      weights = HMM probabilities of each factor topping
      chain_expected  hold the factor with the highest average return in periods
                    that followed the same top factor as last period (observable chain)
      last_winner   hold last period's top factor
      best_so_far   hold the factor with the highest average return to date
      equal_weight  one third in each factor (the default smart beta holding)
    """
    A, B = model.transmat_.copy(), model.emissionprob_.copy()
    pi = model.startprob_.copy() if learn_pi else pi
    alpha_last = forward_filter(obs[:split], pi, A, B)[0][-1]
    rows, refits, drifted = [], 0, 0
    if R is not None:
        mu_state = state_means(obs[:split], R[:split], pi, A, B)
        # running sums for the observable-chain conditional means: E[r_{s+1} | o_s = j]
        chain_sum = np.zeros((N, N))
        chain_n = np.zeros(N)
        np.add.at(chain_sum, obs[:split - 1], R[1:split])
        np.add.at(chain_n, obs[:split - 1], 1)
        cum_sum = R[:split].sum(axis=0)

    for t in range(split, len(obs)):
        if refit_every and t > split and (t - split) % refit_every == 0:
            m = make_model(pi, A, B, n_iter, tol, learn_pi)
            m.fit(obs[:t].reshape(-1, 1))
            A, B = m.transmat_.copy(), m.emissionprob_.copy()
            if learn_pi:
                pi = m.startprob_.copy()
            alpha_last = forward_filter(obs[:t], pi, A, B)[0][-1]
            refits += 1
            drifted += int(np.any(state_names(B, base_rates(obs[:t])) != np.arange(N)))
            if R is not None:
                mu_state = state_means(obs[:t], R[:t], pi, A, B)

        p_state = alpha_last @ A            # P(state_t | o_1..o_{t-1})
        p_obs = p_state @ B                 # P(o_t = k | o_1..o_{t-1})
        p_mle = mle_transition(obs[:t], smoothing)[obs[t - 1]]
        p_base = (np.bincount(obs[:t], minlength=N) + smoothing) / (t + N * smoothing)
        actual = obs[t]
        row_ret = {}
        if R is not None:
            prev = obs[t - 1]
            overall = cum_sum / t
            e_hmm = p_state @ mu_state
            e_chain = chain_sum[prev] / chain_n[prev] if chain_n[prev] else overall
            onehot = np.eye(N)
            weights = {
                "hmm_expected": onehot[e_hmm.argmax()],
                "hmm_top": onehot[p_obs.argmax()],
                "hmm_tilt": p_obs,
                "chain_expected": onehot[e_chain.argmax()],
                "last_winner": onehot[prev],
                "best_so_far": onehot[overall.argmax()],
                "equal_weight": np.full(N, 1 / N),
            }
            for k, w in weights.items():
                row_ret[f"ret_{k}"] = float(w @ R[t])
                for i, f in enumerate(FACTORS):
                    row_ret[f"w_{k}_{f}"] = float(w[i])
            row_ret.update({f"exp_ret_{f}": e_hmm[i] for i, f in enumerate(FACTORS)})
            chain_sum[prev] += R[t]
            chain_n[prev] += 1
            cum_sum += R[t]

        rows.append({
            "t": t, "actual": FACTORS[actual],
            "hmm_pred_state": FACTORS[p_state.argmax()],
            "hmm_pred_obs": FACTORS[p_obs.argmax()],
            "mle_pred": FACTORS[p_mle.argmax()],
            "persist_pred": FACTORS[obs[t - 1]],
            "base_pred": FACTORS[p_base.argmax()],
            "hmm_logp": np.log(p_obs[actual]),
            "mle_logp": np.log(p_mle[actual]),
            "base_logp": np.log(p_base[actual]),
            **{f"p_state_{f}": p_state[i] for i, f in enumerate(FACTORS)},
            **{f"p_top_{f}": p_obs[i] for i, f in enumerate(FACTORS)},
            **row_ret,
        })
        alpha_last = forward_step(alpha_last, actual, A, B)

    wf = pd.DataFrame(rows)
    return wf, {"refits": refits, "refits_where_names_drifted": drifted}


def summarise_walk_forward(wf: pd.DataFrame) -> pd.DataFrame:
    n = len(wf)
    rows = []
    for label, pred, logp in [
        ("HMM, most likely state (design doc rule)", "hmm_pred_state", "hmm_logp"),
        ("HMM, most likely top factor (state probs x B)", "hmm_pred_obs", "hmm_logp"),
        ("Raw MLE transition matrix (observable chain)", "mle_pred", "mle_logp"),
        ("Same as last week", "persist_pred", None),
        ("Most frequent factor so far", "base_pred", "base_logp"),
    ]:
        hit = (wf[pred] == wf["actual"]).mean()
        rows.append({"model": label, "hit_rate": hit,
                     "hit_rate_se": np.sqrt(hit * (1 - hit) / n),
                     "avg_log_score": wf[logp].mean() if logp else np.nan})
    return pd.DataFrame(rows)


STRATEGIES = {
    "hmm_expected": "HMM: hold highest expected return",
    "hmm_top": "HMM: hold most likely top factor",
    "hmm_tilt": "HMM: weights = P(factor tops)",
    "chain_expected": "Observable chain: highest conditional mean",
    "last_winner": "Hold last period's winner",
    "best_so_far": "Hold best average factor to date",
    "equal_weight": "Equal weight (benchmark)",
}


def summarise_returns(wf: pd.DataFrame, periods_per_year: int, cost_bps: float) -> pd.DataFrame:
    """Annualised performance of each holding rule over the test window.
    Factor returns are zero-cost long-short returns, so the Sharpe ratio is
    mean / volatility with no risk-free rate. Costs: `cost_bps` per unit of
    weight traded (switching fully from one factor to another trades 2 units)."""
    P = periods_per_year
    ew = wf["ret_equal_weight"]
    rows = []
    for k, label in STRATEGIES.items():
        r = wf[f"ret_{k}"]
        W = wf[[f"w_{k}_{f}" for f in FACTORS]].to_numpy()
        traded = np.abs(np.diff(W, axis=0)).sum(axis=1)
        traded = np.concatenate([[0.0], traded])
        net = r - traded * cost_bps / 1e4
        wealth = (1 + r).cumprod()
        diff = r - ew
        rows.append({
            "rule": label,
            "ann_return_%": r.mean() * P * 100,
            "ann_vol_%": r.std(ddof=1) * np.sqrt(P) * 100,
            "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(P),
            "max_drawdown_%": (wealth / wealth.cummax() - 1).min() * 100,
            "traded_per_period": traded.mean(),
            "net_ann_return_%": net.mean() * P * 100,
            "net_sharpe": net.mean() / net.std(ddof=1) * np.sqrt(P),
            "t_vs_equal_weight": (diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
                                  if k != "equal_weight" else np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Pipeline
# ---------------------------------------------------------------------------

def as_dict(values) -> dict:
    return {f: round(float(v), 3) for f, v in zip(FACTORS, values)}


def labelled(m: np.ndarray, rows=FACTORS, cols=FACTORS) -> pd.DataFrame:
    return pd.DataFrame(m, index=rows, columns=cols)


def run(dates, obs, weekly, args, true_params=None):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    T = len(obs)
    split = int(round(T * args.train_frac))
    train = obs[:split]
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    monthly = np.median(np.diff(dates.values).astype("timedelta64[D]").astype(int)) >= 25
    U = "month" if monthly else "week"
    P = 12 if monthly else 52

    print(f"\n{U.capitalize()}s: {T}  ({dates[0].date()} to {dates[-1].date()})")
    print(f"Training: {split} {U}s to {dates[split - 1].date()};  test: {T - split} {U}s")
    if split < 200:
        print(f"WARNING: fewer than 200 training {U}s. Estimates of A and B will be "
              "very imprecise; treat results as a pipeline test only.")

    base = base_rates(train)
    print(f"\nShare of training {U}s each factor topped:",
          as_dict(base))
    if base.min() == 0:
        raise ValueError("A factor never tops in the training window; cannot fit.")

    # Initial distribution (design doc: uniform, or training top-frequencies)
    pi = np.full(N, 1 / N) if args.pi == "uniform" else base.copy()

    # Problem 3: learning (Baum-Welch) from the soft-named start
    A0, B0 = soft_init(base, args.init_stay, args.init_boost)
    model, fit_info = fit_with_restarts(train, pi, A0, B0, args.restarts, args.jitter,
                                        args.n_iter, args.tol, args.learn_pi, args.seed)
    A, B = model.transmat_, model.emissionprob_
    pi_used = model.startprob_ if args.learn_pi else pi

    names = state_names(B, base)
    L = lift(B, base)
    print(f"\nBaum-Welch: best log-likelihood {fit_info['best_loglik']:.2f} over "
          f"{args.restarts} starts; share of starts keeping the soft names: "
          f"{fit_info['share_of_runs_keeping_soft_names']:.2f}")
    print(f"\nA (row = state this {U}, column = state next {U})")
    print(labelled(A))
    print("\nB (row = hidden state, column = factor that topped)")
    print(labelled(B, cols=[f"{f} tops" for f in FACTORS]))
    print("\nLift = B / overall share (state j keeps its name if row j peaks on column j, > 1)")
    print(labelled(L, cols=[f"{f} tops" for f in FACTORS]))
    ok = bool(np.all(names == np.arange(N)) and np.all(L[np.arange(N), np.arange(N)] > 1))
    print("Soft names supported by the data:", ok,
          "" if ok else f"(rows peak on {[FACTORS[i] for i in names]})")

    dur = 1 / (1 - np.diag(A))
    stat = stationary(A)
    print(f"\nExpected {U}s a state lasts, 1/(1 - a_jj):",
          as_dict(dur))
    print(f"Long-run share of {U}s in each state (stationary distribution of A):",
          as_dict(stat))

    mle_A = mle_transition(train, args.smoothing)
    print("\nBenchmark: raw MLE transition matrix of the observed top factor (training)")
    print(labelled(mle_A))

    # Problem 2: decoding the training window (hindsight; for description only)
    _, vit = model.decode(train.reshape(-1, 1), algorithm="viterbi")
    print("\nViterbi path over the training window (uses hindsight):")
    print(f"  {U}s per state:", dict(zip(FACTORS, np.bincount(vit, minlength=N).tolist())))
    print(f"  average run length ({U}s), Viterbi states:",
          {k: round(v, 2) for k, v in run_lengths(vit).items()})
    print(f"  average run length ({U}s), observed top factor:",
          {k: round(v, 2) for k, v in run_lengths(train).items()})

    # Problem 1 + validation: walk-forward over the test window
    if T - split > 1:
        print(f"\nWalk-forward test (refit every {args.refit_every} {U}(s); "
              "forward-only probabilities)...")
        R = weekly.loc[dates, FACTORS].to_numpy() if weekly is not None else None
        wf, wf_info = walk_forward(obs, split, model, pi, args.refit_every, args.n_iter,
                                   args.tol, args.learn_pi, args.smoothing, R)
        wf.insert(0, "period_ending", dates[wf["t"].to_numpy()])
        summary = summarise_walk_forward(wf)
        summary["model"] = summary["model"].str.replace("week", U)
        print("\nA. Predicting the top factor")
        print(summary.to_string(index=False))
        d = wf["hmm_logp"] - wf["mle_logp"]
        print(f"HMM minus MLE chain, average log score: {d.mean():.4f} "
              f"(standard error {d.std(ddof=1) / np.sqrt(len(d)):.4f})")
        print("Refits:", wf_info["refits"], "| refits where state names drifted:",
              wf_info["refits_where_names_drifted"])
        if R is not None:
            print(f"\nB. Returns from following each rule over the test window "
                  f"(annualised; costs {args.cost_bps:g} bp per unit of weight traded)")
            perf = summarise_returns(wf, P, args.cost_bps)
            print(perf.to_string(index=False))
            perf.to_csv(out / "returns_summary.csv", index=False)
            mu = state_means(train, R[:split], pi_used, A, B) * 100
            print(f"\nAverage {U}ly factor return (%) in each hidden state, training window:")
            print(labelled(mu, rows=[f"{f} state" for f in FACTORS]))
        wf.to_csv(out / "walk_forward.csv", index=False)
        summary.to_csv(out / "walk_forward_summary.csv", index=False)
    else:
        wf_info = {}

    # Next-week forecast using all data (refit on the full sample, warm start)
    final = make_model(pi_used, A, B, args.n_iter, args.tol, args.learn_pi)
    final.fit(obs.reshape(-1, 1))
    alpha, _ = forward_filter(obs, final.startprob_, final.transmat_, final.emissionprob_)
    p_next = alpha[-1] @ final.transmat_
    print(f"\nForecast for the {U} after {dates[-1].date()} (full-sample refit):")
    print("  P(state):     ", as_dict(p_next))
    print("  P(top factor):", as_dict(p_next @ final.emissionprob_))

    # Save outputs
    panel = pd.DataFrame({"period_ending": dates, "top_factor": [FACTORS[o] for o in obs],
                          "window": ["train"] * split + ["test"] * (T - split)})
    if weekly is not None:
        panel = panel.join(weekly.loc[dates, FACTORS].reset_index(drop=True))
    panel["viterbi_state_train"] = [FACTORS[s] for s in vit] + [""] * (T - split)
    alpha_train = forward_filter(train, pi_used, A, B)[0]
    for i, f in enumerate(FACTORS):
        panel[f"filtered_p_{f}_train"] = list(alpha_train[:, i]) + [np.nan] * (T - split)
    panel.to_csv(out / "panel.csv", index=False)

    params = {
        "factors": FACTORS, "train_weeks": split, "test_weeks": T - split,
        "pi_mode": args.pi, "pi": pi_used.tolist(), "A": A.tolist(), "B": B.tolist(),
        "lift": L.tolist(), "soft_names_supported": ok,
        "expected_duration_weeks": dur.tolist(), "stationary": stat.tolist(),
        "mle_transition_train": mle_A.tolist(), "fit": fit_info, "walk_forward": wf_info,
        "next_week_state_probs": p_next.tolist(), "settings": vars(args),
    }
    if true_params:
        params["true_params"] = true_params
    (out / "params.json").write_text(json.dumps(params, indent=2, default=str))
    print(f"\nSaved results to {out}/ (panel.csv, walk_forward.csv, "
          "walk_forward_summary.csv, returns_summary.csv, params.json)")
    return model


def simulate(weeks: int, seed: int):
    """Known model: persistent states, each showing up as its own factor 60% of weeks."""
    A = np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]])
    B = np.array([[0.60, 0.20, 0.20], [0.20, 0.60, 0.20], [0.20, 0.20, 0.60]])
    pi = np.full(N, 1 / N)
    m = make_model(pi, A, B, 1, 1e-4, False, seed)
    X, states = m.sample(weeks, random_state=seed)
    return X.ravel(), states, {"A": A.tolist(), "B": B.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="source", required=True)
    s_fr = sub.add_parser("french", help="Ken French daily CSV/zip files")
    s_fr.add_argument("--ff3", required=True,
                      help="F-F_Research_Data_Factors daily or weekly (CSV or zip)")
    s_fr.add_argument("--mom", required=True, help="F-F_Momentum_Factor_daily (CSV or zip)")
    s_sp = sub.add_parser("sp", help="S&P DJI PerformanceGraphExport .xls files")
    s_sp.add_argument("--xls", nargs="+", required=True)
    s_sim = sub.add_parser("simulate", help="fit on data simulated from a known model")
    s_sim.add_argument("--weeks", type=int, default=3000)

    for s in (s_fr, s_sp, s_sim):
        s.add_argument("--out", default="results")
        s.add_argument("--train-frac", type=float, default=0.5, help="design doc: half and half")
        s.add_argument("--pi", choices=["uniform", "train"], default="uniform",
                       help="uniform = 1/3 each; train = share of training weeks each factor topped")
        s.add_argument("--learn-pi", action="store_true",
                       help="let Baum-Welch update pi (default: keep it fixed)")
        s.add_argument("--vol-scale", action="store_true",
                       help="divide returns by trailing volatility before argmax")
        s.add_argument("--vol-window", type=int, default=None,
                       help="trailing periods for vol scaling (default 52 weekly, 36 monthly)")
        s.add_argument("--cost-bps", type=float, default=10.0,
                       help="trading cost in basis points per unit of weight traded")
        s.add_argument("--restarts", type=int, default=20)
        s.add_argument("--jitter", type=float, default=0.2,
                       help="weight on random noise in each restart's starting values")
        s.add_argument("--init-stay", type=float, default=0.8, help="starting diagonal of A")
        s.add_argument("--init-boost", type=float, default=2.0,
                       help="soft-naming boost on each state's own factor in B")
        s.add_argument("--refit-every", type=int, default=1,
                       help="refit during the test window every k weeks (0 = freeze A, B)")
        s.add_argument("--n-iter", type=int, default=500)
        s.add_argument("--tol", type=float, default=1e-6)
        s.add_argument("--smoothing", type=float, default=0.5,
                       help="pseudo-count for the MLE and base-rate benchmarks")
        s.add_argument("--seed", type=int, default=348)
    args = p.parse_args()

    weekly, true_params = None, None
    if args.source == "simulate":
        obs, _, true_params = simulate(args.weeks, args.seed)
        dates = pd.date_range("1970-01-02", periods=len(obs), freq="W-FRI")
        print("Simulated from known A and B:")
        print(labelled(np.array(true_params["A"])), "\n", labelled(np.array(true_params["B"])))
    else:
        print("Loading data...")
        weekly = (load_french(args.ff3, args.mom) if args.source == "french"
                  else to_weekly(load_sp_xls(args.xls)))
        if args.vol_window is None:
            gap = np.median(np.diff(weekly.index.values).astype("timedelta64[D]").astype(int))
            args.vol_window = 36 if gap >= 25 else 52
        dates, obs = make_observations(weekly, args.vol_scale, args.vol_window)

    run(dates, obs, weekly, args, true_params)


if __name__ == "__main__":
    main()
