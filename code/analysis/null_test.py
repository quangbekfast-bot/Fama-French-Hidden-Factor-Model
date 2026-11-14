"""No-persistence null for the factor-leadership HMM, plus base-rate-adjusted summaries.

Needs param_boot.py in the same folder (code/analysis).

Modes
  --null      simulate i.i.d. top-factor labels with the observed training shares, fit
              (a) the HMM with the full original procedure and (b) an observed 3x3 Markov
              chain on the labels, record fit statistics. Resumes if --out exists.
  --compare   fit the same things to the REAL training sequence and compare with the null file.
  --eigen     base-rate-adjusted summaries from existing bootstrap CSVs (no new fits).
"""
import argparse, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import param_boot as pb  # noqa: E402

FACT = pb.FACT
N = 3


def lam2(A):
    """Second-largest eigenvalue modulus of a transition matrix (label-free memory measure)."""
    ev = np.sort(np.abs(np.linalg.eigvals(np.asarray(A, float))))[::-1]
    return float(ev[1])


def markov_fit(obs):
    obs = np.asarray(obs, int)
    C = np.zeros((N, N))
    for a, b in zip(obs[:-1], obs[1:]):
        C[a, b] += 1
    rs = C.sum(axis=1, keepdims=True)
    P = np.divide(C, rs, out=np.full_like(C, 1.0 / N), where=rs > 0)
    return P, C


def stats_for(obs):
    """All statistics for one label sequence. Returns dict or None."""
    obs = np.asarray(obs, int)
    n = len(obs)
    counts = np.bincount(obs, minlength=N)
    if counts.min() == 0:
        return None
    share = counts / n
    iid_ll = float((counts * np.log(share)).sum())
    P, C = markov_fit(obs)
    with np.errstate(divide="ignore", invalid="ignore"):
        mc_ll = float(np.log(share[obs[0]]) + np.nansum(np.where(C > 0, C * np.log(P), 0.0)))
    out = dict(iid_ll=iid_ll, mc_ll=mc_ll, mc_lam2=lam2(P))
    for i in range(N):
        out[f"share_{FACT[i]}"] = share[i]
        out[f"mc_diag_{FACT[i]}"] = P[i, i]
        out[f"mc_diag_minus_share_{FACT[i]}"] = P[i, i] - share[i]
    r = pb.fit_one(obs)
    if r is None:
        return None
    m, info, base = r
    A = m.transmat_
    d = np.diag(A)
    order = np.argsort(-d)  # label-free: rank states by diagonal
    st = pb._G["h"].stationary(A)
    out.update(hmm_ll=float(info["best_loglik"]), hmm_lam2=lam2(A))
    for k, s in enumerate(order, 1):
        out[f"hmm_diag_rank{k}"] = float(d[s])
        out[f"hmm_diag_minus_stat_rank{k}"] = float(d[s] - st[s])
        out[f"hmm_dur_rank{k}"] = float(1 / (1 - min(d[s], 1 - 1e-12)))
    out["hmm_diag_max"] = float(d.max())
    out["gain_hmm_iid"] = out["hmm_ll"] - iid_ll
    out["gain_mc_iid"] = mc_ll - iid_ll
    out["gain_hmm_mc"] = out["hmm_ll"] - mc_ll
    return out


def one_null(args):
    b, seed, p = args
    rng = np.random.default_rng([seed, b])
    obs = rng.choice(N, size=pb._G["n_obs"], p=np.array(p))
    row = {"rep": b, "ok": 0}
    s = stats_for(obs)
    if s is not None:
        row.update(s)
        row["ok"] = 1
    return row


def init_worker(repo):
    pb.setup(repo)
    pb._G["n_obs"] = len(pb._G["train"])


def run_null(a):
    init_worker(a.repo)
    train = pb._G["train"]
    p = (np.bincount(train, minlength=N) / len(train)).tolist()
    print(f"null: i.i.d. labels, n={len(train)}, shares={np.round(p, 4).tolist()}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out)["rep"].astype(int)) if out.exists() else set()
    todo = [(b, a.seed, p) for b in range(a.reps) if b not in done]
    print(f"{len(done)} done, {len(todo)} to run, workers={a.workers}", flush=True)
    first = not out.exists()
    buf = []
    with Pool(a.workers, initializer=init_worker, initargs=(a.repo,)) as pool:
        for k, row in enumerate(pool.imap_unordered(one_null, todo, chunksize=1), 1):
            buf.append(row)
            if len(buf) >= 10 or k == len(todo):
                pd.DataFrame(buf).to_csv(out, mode="a", header=first, index=False)
                first, buf = False, []
                print(f"  {k}/{len(todo)}", flush=True)


def compare(a):
    init_worker(a.repo)
    real = stats_for(pb._G["train"])
    null = pd.read_csv(a.compare)
    null = null[null["ok"] == 1]
    print(f"\nreal training sequence vs {len(null)} null fits ({a.compare})")
    print("p = share of null fits with a value >= the real one (one-sided)\n")
    keys = ["gain_hmm_iid", "gain_mc_iid", "gain_hmm_mc", "hmm_ll", "hmm_lam2", "mc_lam2",
            "hmm_diag_max", "hmm_diag_rank1", "hmm_diag_rank2", "hmm_diag_rank3",
            "hmm_diag_minus_stat_rank1", "hmm_diag_minus_stat_rank2", "hmm_diag_minus_stat_rank3",
            "hmm_dur_rank1", "hmm_dur_rank2", "hmm_dur_rank3"] + \
           [f"mc_diag_{f}" for f in FACT] + [f"mc_diag_minus_share_{f}" for f in FACT]
    rows = []
    for k in keys:
        x = null[k].to_numpy(float)
        rows.append(dict(stat=k, real=real[k], null_mean=x.mean(), null_p50=np.percentile(x, 50),
                         null_p95=np.percentile(x, 95), null_p99=np.percentile(x, 99),
                         p_value=float((x >= real[k]).mean())))
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nlog-likelihoods on the real sequence: iid %.2f, Markov %.2f, HMM %.2f"
          % (real["iid_ll"], real["mc_ll"], real["hmm_ll"]))


def eigen(a):
    init_worker(a.repo)
    train = pb._G["train"]
    share = np.bincount(train, minlength=N) / len(train)
    iid_dur = 1 / (1 - share)
    print(f"observed shares {dict(zip(FACT, np.round(share, 3)))}; "
          f"i.i.d. mean spell {dict(zip(FACT, np.round(iid_dur, 2)))}")
    rows = []
    for path in a.eigen:
        d = pd.read_csv(path)
        d = d[d["ok"] == 1]
        for label, sub in (("all", d), ("names_kept", d[d["names_ok"] == 1])):
            Amat = np.stack([sub[[f"A_{FACT[i]}_{FACT[j]}" for j in range(N)]].to_numpy(float)
                             for i in range(N)], axis=1)
            l2 = np.array([lam2(M) for M in Amat])
            rows.append(dict(file=Path(path).stem, subset=label, n=len(sub), stat="lam2",
                             med=np.median(l2), lo=np.percentile(l2, 2.5), hi=np.percentile(l2, 97.5)))
            for i, f in enumerate(FACT):
                diff = sub[f"A_{f}_{f}"].to_numpy(float) - share[i]
                ratio = sub[f"dur_{f}"].to_numpy(float) / iid_dur[i]
                rows.append(dict(file=Path(path).stem, subset=label, n=len(sub), stat=f"diag_minus_share_{f}",
                                 med=np.median(diff), lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5)))
                rows.append(dict(file=Path(path).stem, subset=label, n=len(sub), stat=f"dur_ratio_{f}",
                                 med=np.median(ratio), lo=np.percentile(ratio, 2.5), hi=np.percentile(ratio, 97.5)))
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format, "display.max_rows", 500)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--compare")
    ap.add_argument("--eigen", nargs="+")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3480)
    ap.add_argument("--out", default="my_boot/null_iid.csv")
    a = ap.parse_args()
    if a.null:
        run_null(a)
    elif a.compare:
        compare(a)
    elif a.eigen:
        eigen(a)
    else:
        ap.print_help()
