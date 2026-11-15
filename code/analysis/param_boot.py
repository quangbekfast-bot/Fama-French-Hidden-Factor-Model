"""Bootstrap confidence intervals for the factor-leadership HMM (monthly_main spec).

Methods
  parametric : simulate top-factor sequences from the fitted (pi, A, B), refit each.
  block      : moving-block bootstrap of the observed training sequence, refit each.

Every refit repeats the full original procedure (soft start rebuilt from the
resampled sample, 20 restarts, best likelihood kept), then states are aligned to
the reference fit by the permutation that minimises
max|A[s,s]-A_ref| + max|B[s,:]-B_ref|.

Usage (from the repo root, or pass --repo):
  python param_boot.py --method parametric --reps 1000 --workers 4 --out boot_param.csv
  python param_boot.py --method block --block-len 12 --reps 1000 --out boot_block12.csv
  python param_boot.py --summarize boot_param.csv boot_block12.csv
Re-running with the same --out resumes: finished repetitions are skipped.
"""
import argparse, importlib.util, itertools, json, os, sys, warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

FACT = ["SMB", "HML", "MOM"]
N = 3
PERMS = list(itertools.permutations(range(N)))

_G = {}


def load_repo(repo):
    spec = importlib.util.spec_from_file_location("hmm", str(Path(repo) / "code" / "sta348_hmm_v1.py"))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    return h


def setup(repo, spec_dir="results/monthly_main"):
    h = load_repo(repo)
    p = json.load(open(Path(repo) / spec_dir / "params.json"))
    df = pd.read_csv(Path(repo) / "data/clean/sta348_factor_data_monthly.csv")
    train = df.loc[df["window"] == "train", "top_code"].to_numpy(int)
    st = p["settings"]
    _G.update(h=h, pi=np.array(p["pi"]), A_ref=np.array(p["A"]), B_ref=np.array(p["B"]),
              train=train, restarts=st["restarts"], jitter=st["jitter"], stay=st["init_stay"],
              boost=st["init_boost"], n_iter=st["n_iter"], tol=st["tol"], seed0=st["seed"],
              learn_pi=st["learn_pi"])


def best_perm(A, B, A_ref, B_ref):
    best, bp = np.inf, None
    for s in PERMS:
        s = list(s)
        d = np.abs(A[np.ix_(s, s)] - A_ref).max() + np.abs(B[s, :] - B_ref).max()
        if d < best:
            best, bp = d, s
    return bp, best


def fit_one(obs):
    h = _G["h"]
    base = h.base_rates(obs)
    if base.min() == 0:
        return None
    A0, B0 = h.soft_init(base, _G["stay"], _G["boost"])
    m, info = h.fit_with_restarts(obs, _G["pi"], A0, B0, _G["restarts"], _G["jitter"],
                                  _G["n_iter"], _G["tol"], _G["learn_pi"], _G["seed0"])
    return m, info, base


def resample(method, b, block_len, rng):
    train = _G["train"]
    n = len(train)
    if method == "parametric":
        gen = _G["h"].make_model(_G["pi"], _G["A_ref"], _G["B_ref"], 1, 1e-6, False)
        return gen.sample(n, random_state=int(rng.integers(2**31 - 1)))[0].ravel().astype(int)
    k = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=k)
    return np.concatenate([train[s:s + block_len] for s in starts])[:n]


def one_rep(args):
    method, b, block_len, seed = args
    rng = np.random.default_rng([seed, b])
    obs = resample(method, b, block_len, rng)
    out = {"rep": b, "ok": 0}
    r = fit_one(obs)
    if r is None:
        out["fail"] = "factor_never_tops"
        return out
    m, info, base = r
    h = _G["h"]
    A, B = m.transmat_, m.emissionprob_
    perm, dist = best_perm(A, B, _G["A_ref"], _G["B_ref"])
    A, B = A[np.ix_(perm, perm)], B[perm, :]
    L = h.lift(B, base)
    names_ok = bool(np.all(L.argmax(axis=1) == np.arange(N)) and np.all(np.diag(L) > 1))
    dur = 1 / (1 - np.clip(np.diag(A), 0, 1 - 1e-12))
    stat = h.stationary(A)
    out.update(ok=1, perm="".join(map(str, perm)), perm_identity=int(perm == [0, 1, 2]),
               align_dist=dist, names_ok=int(names_ok), loglik=info["best_loglik"],
               share_soft_names=info["share_of_runs_keeping_soft_names"])
    for i in range(N):
        for j in range(N):
            out[f"A_{FACT[i]}_{FACT[j]}"] = A[i, j]
            out[f"B_{FACT[i]}_{FACT[j]}"] = B[i, j]
    for i in range(N):
        out[f"dur_{FACT[i]}"] = dur[i]
        out[f"stat_{FACT[i]}"] = stat[i]
        out[f"lift_{FACT[i]}"] = L[i, i]
    return out


def run(a):
    setup(a.repo)
    out = Path(a.out)
    done = set()
    if out.exists():
        done = set(pd.read_csv(out)["rep"].astype(int))
    todo = [(a.method, b, a.block_len, a.seed) for b in range(a.reps) if b not in done]
    print(f"{a.method} L={a.block_len}: {len(done)} done, {len(todo)} to run, workers={a.workers}", flush=True)
    first = not out.exists()
    buf = []
    with Pool(a.workers, initializer=setup, initargs=(a.repo,)) as pool:
        for k, row in enumerate(pool.imap_unordered(one_rep, todo, chunksize=1), 1):
            buf.append(row)
            if len(buf) >= 10 or k == len(todo):
                pd.DataFrame(buf).to_csv(out, mode="a", header=first, index=False)
                first = False
                buf = []
                print(f"  {k}/{len(todo)}", flush=True)


def summarize(paths):
    rows = []
    for p in paths:
        d = pd.read_csv(p)
        good = d[d["ok"] == 1]
        print(f"\n{p}: {len(d)} reps, {len(good)} fitted, {int((d['ok'] == 0).sum())} skipped, "
              f"names kept {good['names_ok'].mean():.3f}, identity perm {good['perm_identity'].mean():.3f}")
        cols = [c for c in good.columns if c.split("_")[0] in ("A", "B", "dur", "stat", "lift")]
        for c in cols:
            x = good[c].to_numpy(float)
            rows.append(dict(file=Path(p).stem, param=c, mean=x.mean(), se=x.std(ddof=1),
                             lo=np.percentile(x, 2.5), hi=np.percentile(x, 97.5),
                             share_below_0p001=float((x < 1e-3).mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--method", choices=["parametric", "block"])
    ap.add_argument("--block-len", type=int, default=12)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=348)
    ap.add_argument("--out", default="boot.csv")
    ap.add_argument("--summarize", nargs="+")
    a = ap.parse_args()
    if a.summarize:
        s = summarize(a.summarize)
        pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format, "display.max_rows", 500)
        print(s.to_string(index=False))
    else:
        run(a)
