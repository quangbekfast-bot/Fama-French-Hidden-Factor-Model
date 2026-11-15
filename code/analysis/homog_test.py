"""Time-homogeneity checks for the factor-leadership HMM (build-log Item 2).

Needs param_boot.py in the same folder (code/analysis). Run from the repo root.

Modes
  --sliding      fit the model on the training half of a sliding fixed-length window
                 (50:50 split inside each window) and record A, B, names, durations
  --observed     known-date break test at the main split (Oct 1976) on the real data:
                 LR = 2(ll_before + ll_after - ll_full), chi-square(12) p-value, and the
                 chi-square homogeneity test on the raw label chain (6 df); saves a json
  --known        parametric-bootstrap null for the known-date LR (simulate from the
                 full-sample fit, no break by construction, refit restricted and unrestricted)
  --sup-obs      observed LR(tau) at every January 1942-2011 (15% trimming)
  --sup-boot     bootstrap null of sup_tau LR(tau); slow, use few reps and --restarts 5
  --report       print everything that has been saved

--restarts N overrides the 20 restarts for every fit in that run (use the same value for
--sup-obs and --sup-boot so observed and simulated statistics are comparable).
Every run resumes if its --out file exists.
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import param_boot as pb  # noqa: E402

FACT = pb.FACT
N3 = 3
SPLIT = 598  # months in the main training window (Jan 1927 - Oct 1976)


def init_worker(repo, restarts):
    pb.setup(repo)
    if restarts:
        pb._G["restarts"] = int(restarts)
    df = pd.read_csv(Path(repo) / "data/clean/sta348_factor_data_monthly.csv")
    full = df["top_code"].to_numpy(int)
    pb._G["full"] = full
    pb._G["n_full"] = len(full)
    r = pb.fit_one(full)
    m, info, _ = r
    pb._G["ll_full"] = float(info["best_loglik"])
    pb._G["full_model"] = (m.transmat_.copy(), m.emissionprob_.copy())


def ll_of(obs):
    r = pb.fit_one(np.asarray(obs, int))
    return None if r is None else float(r[1]["best_loglik"])


def simulate_full(rng):
    A, B = pb._G["full_model"]
    gen = pb._G["h"].make_model(pb._G["pi"], A, B, 1, 1e-6, False)
    return gen.sample(pb._G["n_full"], random_state=int(rng.integers(2**31 - 1)))[0].ravel().astype(int)


def lr_at(obs, tau, ll_full):
    a, b = ll_of(obs[:tau]), ll_of(obs[tau:])
    if a is None or b is None:
        return None
    return 2 * (a + b - ll_full)


# ---------- sliding window ----------
def sliding_task(args):
    start, L = args
    h = pb._G["h"]
    obs = pb._G["full"][start:start + L][: L // 2]
    row = {"start": start, "L": L, "ok": 0}
    r = pb.fit_one(obs)
    if r is None:
        return row
    m, info, base = r
    A, B = m.transmat_, m.emissionprob_
    perm, dist = pb.best_perm(A, B, pb._G["A_ref"], pb._G["B_ref"])
    A, B = A[np.ix_(perm, perm)], B[perm, :]
    Lf = h.lift(B, base)
    names_ok = bool(np.all(Lf.argmax(axis=1) == np.arange(N3)) and np.all(np.diag(Lf) > 1))
    row.update(ok=1, names_ok=int(names_ok), align_dist=dist, loglik=float(info["best_loglik"]))
    for i in range(N3):
        row[f"dur_{FACT[i]}"] = 1 / (1 - min(A[i, i], 1 - 1e-12))
        row[f"share_{FACT[i]}"] = base[i]
        for j in range(N3):
            row[f"A_{FACT[i]}_{FACT[j]}"] = A[i, j]
            row[f"B_{FACT[i]}_{FACT[j]}"] = B[i, j]
    return row


# ---------- bootstrap tasks ----------
def known_task(args):
    b, seed = args
    rng = np.random.default_rng([seed, b])
    obs = simulate_full(rng)
    row = {"rep": b, "ok": 0}
    llf = ll_of(obs)
    lr = lr_at(obs, SPLIT, llf) if llf is not None else None
    if lr is not None:
        row.update(ok=1, lr=lr)
    return row


def sup_task(args):
    b, seed, taus = args
    rng = np.random.default_rng([seed, b])
    obs = simulate_full(rng)
    row = {"rep": b, "ok": 0}
    llf = ll_of(obs)
    if llf is None:
        return row
    lrs = [lr_at(obs, t, llf) for t in taus]
    if any(x is None for x in lrs):
        return row
    k = int(np.argmax(lrs))
    row.update(ok=1, sup_lr=lrs[k], tau_idx=int(taus[k]))
    return row


def sup_obs_task(tau):
    full = pb._G["full"]
    lr = lr_at(full, tau, pb._G["ll_full"])
    return {"tau_idx": int(tau), "year": 1927 + tau // 12, "lr": lr}


def sup_taus():
    n = len(pd.read_csv(Path(ARGS.repo) / "data/clean/sta348_factor_data_monthly.csv"))
    lo, hi = int(np.ceil(0.15 * n)), int(np.floor(0.85 * n))
    return [t for t in range(lo, hi + 1) if t % 12 == 0]


def run_pool(task, tasks, out, a, key):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = set(map(tuple, pd.read_csv(out)[key].astype(int).to_numpy().reshape(-1, len(key)).tolist()))
    todo = [t for t in tasks if tuple(t_key(t, key)) not in done]
    print(f"{out.name}: {len(done)} done, {len(todo)} to run, workers={a.workers}", flush=True)
    first = not out.exists()
    buf = []
    with Pool(a.workers, initializer=init_worker, initargs=(a.repo, a.restarts)) as pool:
        for k, row in enumerate(pool.imap_unordered(task, todo, chunksize=1), 1):
            buf.append(row)
            if len(buf) >= 5 or k == len(todo):
                pd.DataFrame(buf).to_csv(out, mode="a", header=first, index=False)
                first, buf = False, []
                print(f"  {k}/{len(todo)}", flush=True)


def t_key(t, key):
    if key == ["start"]:
        return [t[0]]
    if key == ["rep"]:
        return [t[0]]
    if key == ["tau_idx"]:
        return [t]
    raise ValueError(key)


# ---------- observed known-date test and chain test ----------
def chain_test(full, split):
    from scipy.stats import chi2
    def counts(x):
        C = np.zeros((N3, N3))
        for a_, b_ in zip(x[:-1], x[1:]):
            C[a_, b_] += 1
        return C
    C1, C2 = counts(full[:split]), counts(full[split:])
    Cp = C1 + C2
    P = Cp / Cp.sum(axis=1, keepdims=True)
    stat = 0.0
    for C in (C1, C2):
        n_i = C.sum(axis=1)
        Pt = C / n_i[:, None]
        stat += float((n_i[:, None] * (Pt - P) ** 2 / P).sum())
    df_ = 6
    return stat, df_, float(chi2.sf(stat, df_)), C1, C2


def observed(a):
    from scipy.stats import chi2
    init_worker(a.repo, a.restarts)
    full = pb._G["full"]
    ll_full = pb._G["ll_full"]
    ll_b, ll_a = ll_of(full[:SPLIT]), ll_of(full[SPLIT:])
    lr = 2 * (ll_b + ll_a - ll_full)
    cs, cdf, cp, C1, C2 = chain_test(full, SPLIT)
    h = pb._G["h"]
    seg = {}
    for name, x in (("before", full[:SPLIT]), ("after", full[SPLIT:]), ("full", full)):
        m, info, base = pb.fit_one(x)
        A, B = m.transmat_, m.emissionprob_
        perm, _ = pb.best_perm(A, B, pb._G["A_ref"], pb._G["B_ref"])
        seg[name] = dict(A=A[np.ix_(perm, perm)].round(4).tolist(), B=B[perm, :].round(4).tolist(),
                         shares=base.round(4).tolist(), loglik=float(info["best_loglik"]))
    res = dict(split=SPLIT, ll_full=ll_full, ll_before=ll_b, ll_after=ll_a, lr=lr,
               chi2_12_p=float(chi2.sf(lr, 12)), chain_chi2=cs, chain_df=cdf, chain_p=cp,
               counts_before=C1.tolist(), counts_after=C2.tolist(), segments=seg,
               restarts=pb._G["restarts"])
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("segments",)}, indent=2))


# ---------- report ----------
def report(a):
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format, "display.max_rows", 400)
    p = Path(a.dir)
    f = p / "homog_sliding.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d["ok"] == 1].sort_values("start")
        d["start_year"] = 1927 + d["start"] // 12
        print(f"\nSLIDING WINDOW ({len(d)} positions, L={int(d['L'].iloc[0])}, training half {int(d['L'].iloc[0]) // 2} months)")
        print(f"names kept at {d['names_ok'].mean():.3f} of positions")
        cols = [f"A_{s}_{s}" for s in FACT] + [f"B_{s}_{s}" for s in FACT]
        sub = d[cols].describe().loc[["mean", "std", "min", "max"]].T
        print(sub.to_string())
        print("Spearman correlation with window start year (drift check):")
        print(d[cols].apply(lambda c: c.corr(d["start_year"], method="spearman")).to_string())
        print("\nDiagonals by window start year:")
        print(d[["start_year", "names_ok"] + [f"A_{s}_{s}" for s in FACT] + ["loglik"]].to_string(index=False))
    f = p / "homog_observed.json"
    if f.exists():
        r = json.load(open(f))
        print(f"\nKNOWN-DATE BREAK TEST at index {r['split']} (after Oct 1976), restarts={r['restarts']}")
        print(f"  ll_full={r['ll_full']:.2f}  ll_before={r['ll_before']:.2f}  ll_after={r['ll_after']:.2f}")
        print(f"  LR = {r['lr']:.2f}   chi-square(12) p = {r['chi2_12_p']:.4f}  (approximation; see bootstrap)")
        k = p / "homog_known.csv"
        if k.exists():
            kd = pd.read_csv(k)
            kd = kd[kd["ok"] == 1]
            print(f"  bootstrap p = {(kd['lr'] >= r['lr']).mean():.3f}   ({len(kd)} sims; null LR median {kd['lr'].median():.2f}, 95th {np.percentile(kd['lr'], 95):.2f})")
        print(f"  observable-chain chi-square homogeneity: stat {r['chain_chi2']:.2f}, df {r['chain_df']}, p = {r['chain_p']:.4f}")
        print("  segment fits (diagonal of A, aligned to the main fit):")
        for name, s in r["segments"].items():
            print(f"    {name:7s} shares {s['shares']}  diag A {[round(s['A'][i][i], 3) for i in range(3)]}  diag B {[round(s['B'][i][i], 3) for i in range(3)]}")
    f = p / "homog_sup_obs.csv"
    if f.exists():
        so = pd.read_csv(f).sort_values("tau_idx")
        top = so.loc[so["lr"].idxmax()]
        print(f"\nSUP-LR scan, observed: {len(so)} dates; max LR = {top['lr']:.2f} at January {int(top['year'])}")
        b = p / "homog_sup_boot.csv"
        if b.exists():
            bd = pd.read_csv(b)
            bd = bd[bd["ok"] == 1]
            print(f"  bootstrap: {len(bd)} sims; p = {(bd['sup_lr'] >= top['lr']).mean():.3f}; null sup-LR median {bd['sup_lr'].median():.2f}, 95th {np.percentile(bd['sup_lr'], 95):.2f}")
        print(so[["year", "lr"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--dir", default="my_boot")
    ap.add_argument("--sliding", action="store_true")
    ap.add_argument("--window", type=int, default=480)
    ap.add_argument("--step", type=int, default=12)
    ap.add_argument("--observed", action="store_true")
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--sup-obs", action="store_true")
    ap.add_argument("--sup-boot", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reps", type=int, default=199)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--restarts", type=int, default=0, help="override number of restarts (0 = spec value)")
    ap.add_argument("--seed", type=int, default=9348)
    ap.add_argument("--out", default=None)
    ARGS = ap.parse_args()
    a = ARGS
    d = Path(a.dir)
    if a.sliding:
        init_worker(a.repo, a.restarts)
        n = pb._G["n_full"]
        tasks = [(s, a.window) for s in range(0, n - a.window + 1, a.step)]
        run_pool(sliding_task, tasks, a.out or d / "homog_sliding.csv", a, ["start"])
    elif a.observed:
        a.out = a.out or str(d / "homog_observed.json")
        d.mkdir(parents=True, exist_ok=True)
        observed(a)
    elif a.known:
        tasks = [(b, a.seed) for b in range(a.reps)]
        run_pool(known_task, tasks, a.out or d / "homog_known.csv", a, ["rep"])
    elif a.sup_obs:
        run_pool(sup_obs_task, sup_taus(), a.out or d / "homog_sup_obs.csv", a, ["tau_idx"])
    elif a.sup_boot:
        taus = sup_taus()
        tasks = [(b, a.seed, taus) for b in range(a.reps)]
        run_pool(sup_task, tasks, a.out or d / "homog_sup_boot.csv", a, ["rep"])
    elif a.report:
        report(a)
    else:
        ap.print_help()
