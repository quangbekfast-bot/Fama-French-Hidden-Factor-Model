"""Remaining pieces of build-log Item 2 (time homogeneity), next to homog_test.py.

Put this file in code/analysis next to param_boot.py and homog_test.py. Run from the repo root.
It reuses homog_test.py (full-sample fit, bootstrap simulation, known-date and sup-LR tasks)
and the repo's own walk_forward / summarise_returns, so out-of-sample numbers are computed
exactly as in the main test.

Modes
  --slide            sliding fixed-length window (--window L, steps of 12 months). At each
                     position: fit the training half (full procedure), align labels, record A, B,
                     lift/name check, expected durations, stationary distribution; then score the
                     window's test half with the repo's walk-forward code: frozen parameters
                     (always) and refit every period (unless --no-refit).
  --variants         restricted break tests at the primary date (Oct 1976) and, if
                     homog_sup_obs.csv exists, at the sup-LR date: B shared / A free and
                     A shared / B free, each against the unrestricted model (6 df), using a
                     custom Baum-Welch that pools the shared matrix across the two segments.
  --variants-boot    parametric bootstrap p-values for the two restricted tests (primary date).
  --zscores          entry-by-entry z for A and B before/after, with standard errors from the Item-4
                     yearly block bootstrap computed separately in each segment (--z-method parametric
                     for the alternative); also the segment names check.
  --sup-obs          observed LR(tau) on a grid of dates (--tau-step 12 = every January, 70 dates).
  --sup-boot         bootstrap of sup_tau LR(tau) on the same grid (slow: see --restarts, --reps).
  --plots            figures into <dir>/figs
  --report           print the tables for everything that has been saved

Options: --dir my_boot  --workers 2  --restarts N (overrides 20 for every fit in that run;
use the same value for --sup-obs and --sup-boot)  --reps  --seed  --out  --limit N (smoke test: first N tasks)
Every mode resumes if its output file exists.
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import param_boot as pb  # noqa: E402
import homog_test as ht  # noqa: E402

FACT = pb.FACT
N = 3
SPLIT = ht.SPLIT
ENTRIES = [f"{m}_{FACT[i]}_{FACT[j]}" for m in "AB" for i in range(N) for j in range(N)]
WF_KEYS = ["hit", "hit_state", "hit_mle", "logp_hmm", "logp_mle", "logp_diff",
           "sharpe_hmm_tilt", "sharpe_hmm_top", "sharpe_hmm_expected", "sharpe_equal_weight",
           "sharpe_diff_tilt_ew", "t_tilt_vs_ew"]


def init2(repo, restarts):
    ht.init_worker(repo, restarts)
    df = pd.read_csv(Path(repo) / "data/clean/sta348_factor_data_monthly.csv")
    pb._G["R"] = df[FACT].to_numpy(float)
    pb._G["year"] = pd.to_datetime(df["month_end"]).dt.year.to_numpy()
    st = json.load(open(Path(repo) / "results/monthly_main/params.json"))["settings"]
    pb._G["smoothing"] = st["smoothing"]
    pb._G["cost_bps"] = st["cost_bps"]


def pool_run(task, tasks, out, keyfn, keycols, cols, a):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        d = pd.read_csv(out)
        done = set(map(tuple, d[keycols].astype(int).to_numpy().tolist()))
    todo = [t for t in tasks if tuple(keyfn(t)) not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{out.name}: {len(done)} done, {len(todo)} to run, workers={a.workers}", flush=True)
    first = not out.exists()
    buf = []
    with Pool(a.workers, initializer=init2, initargs=(a.repo, a.restarts)) as pool:
        for k, row in enumerate(pool.imap_unordered(task, todo, chunksize=1), 1):
            buf.append(row)
            if len(buf) >= 3 or k == len(todo):
                pd.DataFrame(buf).reindex(columns=cols).to_csv(out, mode="a", header=first, index=False)
                first, buf = False, []
                print(f"  {k}/{len(todo)}", flush=True)


def aligned(m):
    perm, dist = pb.best_perm(m.transmat_, m.emissionprob_, pb._G["A_ref"], pb._G["B_ref"])
    return m.transmat_[np.ix_(perm, perm)], m.emissionprob_[perm, :], dist


# --------------------------------------------------------------------------
# Sliding window with walk-forward scoring
# --------------------------------------------------------------------------
def wf_metrics(obs, R, split, model, refit_every):
    h = pb._G["h"]
    wf, _ = h.walk_forward(obs, split, model, pb._G["pi"], refit_every, pb._G["n_iter"], pb._G["tol"],
                           pb._G["learn_pi"], pb._G["smoothing"], R)
    perf = h.summarise_returns(wf, 12, pb._G["cost_bps"]).set_index("rule")
    lab = h.STRATEGIES
    o = {"hit": float((wf["hmm_pred_obs"] == wf["actual"]).mean()),
         "hit_state": float((wf["hmm_pred_state"] == wf["actual"]).mean()),
         "hit_mle": float((wf["mle_pred"] == wf["actual"]).mean()),
         "logp_hmm": float(wf["hmm_logp"].mean()), "logp_mle": float(wf["mle_logp"].mean())}
    o["logp_diff"] = o["logp_hmm"] - o["logp_mle"]
    for k in ("hmm_tilt", "hmm_top", "hmm_expected", "equal_weight"):
        o[f"sharpe_{k}"] = float(perf.loc[lab[k], "sharpe"])
    o["sharpe_diff_tilt_ew"] = o["sharpe_hmm_tilt"] - o["sharpe_equal_weight"]
    o["t_tilt_vs_ew"] = float(perf.loc[lab["hmm_tilt"], "t_vs_equal_weight"])
    return o


def slide_cols():
    c = ["start", "L", "ok", "names_ok", "align_dist", "loglik", "seconds"]
    for i in FACT:
        c += [f"dur_{i}", f"share_{i}", f"stat_{i}", f"lift_diag_{i}"]
    c += ENTRIES
    c += [f"frozen_{k}" for k in WF_KEYS] + [f"refit_{k}" for k in WF_KEYS]
    return c


def slide_task(args):
    start, L, refit_every, do_refit = args
    h = pb._G["h"]
    t0 = time.time()
    split = L // 2
    obs = pb._G["full"][start:start + L]
    R = pb._G["R"][start:start + L]
    row = {"start": start, "L": L, "ok": 0}
    r = pb.fit_one(obs[:split])
    if r is None:
        return row
    m, info, base = r
    A, B, dist = aligned(m)
    Lf = h.lift(B, base)
    row.update(ok=1, names_ok=int(np.all(Lf.argmax(axis=1) == np.arange(N)) and np.all(np.diag(Lf) > 1)),
               align_dist=dist, loglik=float(info["best_loglik"]))
    st = h.stationary(A)
    for i in range(N):
        row[f"dur_{FACT[i]}"] = 1 / (1 - min(A[i, i], 1 - 1e-12))
        row[f"share_{FACT[i]}"] = base[i]
        row[f"stat_{FACT[i]}"] = st[i]
        row[f"lift_diag_{FACT[i]}"] = Lf[i, i]
        for j in range(N):
            row[f"A_{FACT[i]}_{FACT[j]}"] = A[i, j]
            row[f"B_{FACT[i]}_{FACT[j]}"] = B[i, j]
    for k, v in wf_metrics(obs, R, split, m, 0).items():
        row[f"frozen_{k}"] = v
    if do_refit:
        for k, v in wf_metrics(obs, R, split, m, refit_every).items():
            row[f"refit_{k}"] = v
    row["seconds"] = time.time() - t0
    return row


# --------------------------------------------------------------------------
# Custom Baum-Welch with a matrix shared across the two segments
# --------------------------------------------------------------------------
def _norm(M):
    return M / np.maximum(M.sum(axis=1, keepdims=True), 1e-300)


def estep(A, B, x):
    """hmmlearn's own forward-backward on one segment: log-likelihood, expected
    transition counts, expected emission counts (uniform start, as in the main fits)."""
    m = pb._G["h"].make_model(pb._G["pi"], A, B, 1, 1e-6, False)
    stats, ll = m._do_estep(x.reshape(-1, 1), None)
    return float(ll), stats["trans"].copy(), stats["obs"].copy()


def tied_em(x1, x2, tie, A1, B1, A2, B2, n_iter=500, tol=1e-6):
    prev = -np.inf
    for _ in range(n_iter):
        l1, t1, o1 = estep(A1, B1, x1)
        l2, t2, o2 = estep(A2, B2, x2)
        ll = l1 + l2
        if ll - prev < tol:
            break
        prev = ll
        if tie == "B":      # B shared, A free in each segment
            A1, A2, B1 = _norm(t1), _norm(t2), _norm(o1 + o2)
            B2 = B1
        else:               # A shared, B free in each segment
            A1 = _norm(t1 + t2)
            A2 = A1
            B1, B2 = _norm(o1), _norm(o2)
    return estep(A1, B1, x1)[0] + estep(A2, B2, x2)[0], (A1, B1, A2, B2)


def tied_best(x1, x2, tie, extra_start=None, seed=0):
    """Best of `restarts` runs, same soft start + jitter recipe as the main fits."""
    h = pb._G["h"]
    rng = np.random.default_rng(seed)
    base = h.base_rates(np.concatenate([x1, x2]))
    A0, B0 = h.soft_init(base, pb._G["stay"], pb._G["boost"])
    starts = []
    if extra_start is not None:
        starts.append(extra_start)
    for r in range(pb._G["restarts"]):
        if r == 0:
            A, B = A0, B0
        else:
            j = pb._G["jitter"]
            A = _norm((1 - j) * A0 + j * rng.dirichlet(np.ones(N), N))
            B = _norm((1 - j) * B0 + j * rng.dirichlet(np.ones(N), N))
        starts.append((A, B, A, B))
    best = (-np.inf, None)
    for s in starts:
        ll, par = tied_em(x1, x2, tie, *s, n_iter=pb._G["n_iter"], tol=pb._G["tol"])
        if ll > best[0]:
            best = (ll, par)
    return best


def variant_stats(x1, x2, seed=0):
    """ll of the unrestricted pair and of the two restricted models, with starts taken from
    the unrestricted segment fits (segment 2 relabelled to segment 1)."""
    r1, r2 = pb.fit_one(x1), pb.fit_one(x2)
    if r1 is None or r2 is None:
        return None
    m1, i1, _ = r1
    m2, i2, _ = r2
    perm, _ = pb.best_perm(m2.transmat_, m2.emissionprob_, m1.transmat_, m1.emissionprob_)
    A1, B1 = m1.transmat_, m1.emissionprob_
    A2, B2 = m2.transmat_[np.ix_(perm, perm)], m2.emissionprob_[perm, :]
    ll_un = float(i1["best_loglik"] + i2["best_loglik"])
    out = {"ll_un": ll_un}
    for tie, es in (("B", (A1, (B1 + B2) / 2, A2, (B1 + B2) / 2)),
                    ("A", ((A1 + A2) / 2, B1, (A1 + A2) / 2, B2))):
        ll, par = tied_best(x1, x2, tie, extra_start=es, seed=seed)
        out[f"ll_tie{tie}"] = ll
        out[f"lr_tie{tie}"] = 2 * (ll_un - ll)
        out[f"par_tie{tie}"] = par
    return out


def variants_obs(a):
    from scipy.stats import chi2
    init2(a.repo, a.restarts)
    full = pb._G["full"]
    taus = [SPLIT]
    f = Path(a.dir) / "homog_sup_obs.csv"
    if f.exists():
        so = pd.read_csv(f)
        taus.append(int(so.loc[so["lr"].idxmax(), "tau_idx"]))
    res = {}
    for tau in dict.fromkeys(taus):
        v = variant_stats(full[:tau], full[tau:], seed=a.seed)
        d = {"tau": tau, "year": 1927 + tau // 12, "ll_unrestricted": v["ll_un"], "restarts": pb._G["restarts"]}
        for tie, lab in (("B", "B_shared_A_free"), ("A", "A_shared_B_free")):
            lr = v[f"lr_tie{tie}"]
            d[lab] = {"ll_restricted": v[f"ll_tie{tie}"], "lr": lr, "df": 6, "chi2_p": float(chi2.sf(max(lr, 0), 6)),
                      "A_before": v[f"par_tie{tie}"][0].round(4).tolist(), "A_after": v[f"par_tie{tie}"][2].round(4).tolist(),
                      "B_before": v[f"par_tie{tie}"][1].round(4).tolist(), "B_after": v[f"par_tie{tie}"][3].round(4).tolist()}
        res[str(tau)] = d
        print(json.dumps({k: (v_ if not isinstance(v_, dict) else {kk: vv for kk, vv in v_.items() if kk in ("lr", "chi2_p", "ll_restricted")})
                          for k, v_ in d.items()}, indent=1), flush=True)
    json.dump(res, open(a.out or Path(a.dir) / "homog_variants.json", "w"), indent=2)


def variants_boot_task(args):
    b, seed, tau = args
    rng = np.random.default_rng([seed, b, 7])
    x = ht.simulate_full(rng)
    row = {"rep": b, "ok": 0}
    v = variant_stats(x[:tau], x[tau:], seed=seed + b)
    if v is not None:
        row.update(ok=1, ll_un=v["ll_un"], lr_B_shared=v["lr_tieB"], lr_A_shared=v["lr_tieA"])
    return row


# --------------------------------------------------------------------------
# Entry-by-entry z with segment bootstrap standard errors
# --------------------------------------------------------------------------
def seg_fit(seg):
    key = ("segfit", seg)
    if key not in pb._G:
        full = pb._G["full"]
        x = full[:SPLIT] if seg == 0 else full[SPLIT:]
        m, info, base = pb.fit_one(x)
        A, B, _ = aligned(m)
        pb._G[key] = (A, B, len(x))
    return pb._G[key]


def z_cols():
    return ["seg", "rep", "ok", "names_ok"] + ENTRIES


def yearly_sample(seg, rng):
    """Item 4(b): cut the segment into calendar-year blocks, draw years with replacement
    until the pseudo-sample has the segment's length, months kept in order within a year."""
    full, yr = pb._G["full"], pb._G["year"]
    lo, hi = (0, SPLIT) if seg == 0 else (SPLIT, len(full))
    x, ys = full[lo:hi], yr[lo:hi]
    blocks = [np.flatnonzero(ys == y) for y in np.unique(ys)]
    out = []
    while len(out) < hi - lo:
        out.extend(x[blocks[int(rng.integers(len(blocks)))]].tolist())
    return np.array(out[: hi - lo], int)


def z_task(args):
    seg, b, seed, method = args
    h = pb._G["h"]
    A_s, B_s, n = seg_fit(seg)
    rng = np.random.default_rng([seed, seg, b])
    if method == "yearly":
        x = yearly_sample(seg, rng)
    else:
        gen = h.make_model(pb._G["pi"], A_s, B_s, 1, 1e-6, False)
        x = gen.sample(n, random_state=int(rng.integers(2**31 - 1)))[0].ravel().astype(int)
    row = {"seg": seg, "rep": b, "ok": 0}
    r = pb.fit_one(x)
    if r is None:
        return row
    m, info, base = r
    perm, _ = pb.best_perm(m.transmat_, m.emissionprob_, A_s, B_s)
    A, B = m.transmat_[np.ix_(perm, perm)], m.emissionprob_[perm, :]
    Lf = h.lift(B, base)
    row.update(ok=1, names_ok=int(np.all(Lf.argmax(axis=1) == np.arange(N)) and np.all(np.diag(Lf) > 1)))
    for i in range(N):
        for j in range(N):
            row[f"A_{FACT[i]}_{FACT[j]}"] = A[i, j]
            row[f"B_{FACT[i]}_{FACT[j]}"] = B[i, j]
    return row


# --------------------------------------------------------------------------
# Sup-LR on a chosen grid
# --------------------------------------------------------------------------
def taus_grid(a, n=1197):
    lo, hi = int(np.ceil(0.15 * n)), int(np.floor(0.85 * n))
    return [t for t in range(lo, hi + 1) if t % a.tau_step == 0]


# --------------------------------------------------------------------------
# Reports and plots
# --------------------------------------------------------------------------
def main_spec(repo):
    d = Path(repo) / "results/monthly_main"
    wf = pd.read_csv(d / "walk_forward.csv")
    rt = pd.read_csv(d / "returns_summary.csv").set_index("rule")
    tilt = rt.loc["HMM: weights = P(factor tops)", "sharpe"]
    ew = rt.loc["Equal weight (benchmark)", "sharpe"]
    return {"hit": float((wf["hmm_pred_obs"] == wf["actual"]).mean()), "hit_mle": float((wf["mle_pred"] == wf["actual"]).mean()),
            "logp_diff": float((wf["hmm_logp"] - wf["mle_logp"]).mean()), "sharpe_hmm_tilt": float(tilt),
            "sharpe_equal_weight": float(ew), "sharpe_diff_tilt_ew": float(tilt - ew)}


def slide_files(a):
    return sorted(Path(a.dir).glob("homog_slide_L*.csv"), key=lambda p: int(p.stem.split("L")[-1]))


def report(a):
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format, "display.max_rows", 400)
    p = Path(a.dir)
    ms = main_spec(a.repo)
    print("MAIN SPECIFICATION (test window Nov 1976 - Sep 2026, refit monthly): "
          + ", ".join(f"{k} {v:.3f}" for k, v in ms.items()))
    for f in slide_files(a):
        d = pd.read_csv(f)
        d = d[d["ok"] == 1].sort_values("start")
        L = int(d["L"].iloc[0])
        d["year"] = 1927 + d["start"] // 12
        print(f"\n=== SLIDING WINDOW L={L}: {len(d)} positions, training half {L // 2} months ===")
        bad = d[d["names_ok"] == 0]
        print(f"names kept in {d['names_ok'].mean():.1%} of positions")
        bad = d[d["names_ok"] == 0]
        if len(bad):
            print(f"positions where the names fail (counted separately, left out of the parameter table): "
                  f"{len(bad)}, window start years {bad['year'].tolist()}")
        g = d[d["names_ok"] == 1]
        cols = ENTRIES + [f"dur_{s}" for s in FACT] + [f"stat_{s}" for s in FACT]
        t = g[cols].describe().loc[["mean", "std", "min", "max"]].T
        t["spearman_vs_start"] = g[cols].apply(lambda c: c.corr(g["year"], method="spearman"))
        bp = p / "boot_param.csv"
        if bp.exists():
            bb = pd.read_csv(bp)
            bb = bb[bb["ok"] == 1]
            t["boot_sd"] = [bb[c].std(ddof=1) for c in cols]
            t["boot_p2.5"] = [bb[c].quantile(.025) for c in cols]
            t["boot_p97.5"] = [bb[c].quantile(.975) for c in cols]
            t["share_inside_boot_95"] = [float(((g[c] >= bb[c].quantile(.025)) & (g[c] <= bb[c].quantile(.975))).mean()) for c in cols]
        print(f"parameters over the {len(g)} positions with the names kept:")
        print(t.to_string())
        if bp.exists():
            print("(boot_* columns: Item-4 parametric bootstrap of the 598-month training fit. Window fits use only "
                  f"{L // 2} months, so their sampling error is larger than the bootstrap's; a low 'share_inside' "
                  "therefore points to drift only if it is far below what that extra noise explains.)")
        for pre in ("frozen", "refit"):
            ks = [k for k in WF_KEYS if f"{pre}_{k}" in d and d[f"{pre}_{k}"].notna().any()]
            if not ks:
                continue
            s_ = pd.DataFrame({k: [d[f"{pre}_{k}"].mean(), d[f"{pre}_{k}"].quantile(.025), d[f"{pre}_{k}"].quantile(.975),
                                   ms.get(k, np.nan), g[f"{pre}_{k}"].mean()] for k in ks},
                              index=["mean", "p2.5", "p97.5", "main_spec", "mean_names_kept_only"]).T
            print(f"\nWalk-forward on each window's test half, {pre} parameters ({d[f'{pre}_hit'].notna().sum()} windows; "
                  "mean/percentiles over all windows, last column over the names-kept windows only)")
            print(s_.to_string())
    f = p / "homog_variants.json"
    if f.exists():
        r = json.load(open(f))
        print("\n=== WHICH PART CHANGED (restricted variants vs unrestricted, 6 df) ===")
        bf = p / "homog_variants_boot.csv"
        bd = pd.read_csv(bf) if bf.exists() else None
        if bd is not None:
            bd = bd[bd["ok"] == 1]
        for tau, d in r.items():
            when = "main split, after Oct 1976" if int(tau) == SPLIT else f"January {d['year']} (sup-LR date)"
            print(f"tau index {tau} ({when}), restarts {d['restarts']}, ll unrestricted {d['ll_unrestricted']:.2f}")
            for lab, col in (("B_shared_A_free", "lr_B_shared"), ("A_shared_B_free", "lr_A_shared")):
                x = d[lab]
                s = f"  {lab:16s} LR {x['lr']:.2f}  chi2(6) p {x['chi2_p']:.3f}"
                if bd is not None and int(tau) == SPLIT:
                    s += f"  bootstrap p {(bd[col] >= x['lr']).mean():.3f} ({len(bd)} sims; null 95th {bd[col].quantile(.95):.2f})"
                print(s)
            if str(SPLIT) == tau:
                for lab in ("B_shared_A_free", "A_shared_B_free"):
                    x = d[lab]
                    print(f"  {lab}: diag of A before {np.round(np.diag(x['A_before']), 3).tolist()} after {np.round(np.diag(x['A_after']), 3).tolist()}"
                          f" | diag of B before {np.round(np.diag(x['B_before']), 3).tolist()} after {np.round(np.diag(x['B_after']), 3).tolist()}")
    f, o = next((q for q in (p / "homog_z_yearly.csv", p / "homog_z_parametric.csv") if q.exists()), p / "homog_z_yearly.csv"), p / "homog_observed.json"
    if f.exists() and o.exists():
        z = pd.read_csv(f)
        z = z[z["ok"] == 1]
        seg = json.load(open(o))["segments"]
        print(f"\n=== ENTRY BY ENTRY: before vs after Oct 1976 (SE from the {f.stem.split('_')[-1]} bootstrap within each segment) ===")
        ok_seg = {}
        for s_, nm in ((0, "before"), (1, "after")):
            zz = z[z["seg"] == s_]
            sh, Bm = np.array(seg[nm]["shares"]), np.array(seg[nm]["B"])
            Lf = Bm / sh
            ok_seg[nm] = bool(np.all(Lf.argmax(axis=1) == np.arange(N)) and np.all(np.diag(Lf) > 1))
            print(f"segment {nm}: names check on the point fit: {'passes' if ok_seg[nm] else 'FAILS'}; "
                  f"{len(zz)} bootstrap draws, names kept in {zz['names_ok'].mean():.1%}")
        if not all(ok_seg.values()):
            print("WARNING: names fail in at least one segment, so the state structure itself changed (build-log caveat). "
                  "Report that; the entry-by-entry z below compares states that are not the same thing and is shown for reference only.")
        rows = []
        for e in ENTRIES:
            m_, i_, j_ = e.split("_")
            I, J = FACT.index(i_), FACT.index(j_)
            vb, va = seg["before"][m_][I][J], seg["after"][m_][I][J]
            se = []
            for s_ in (0, 1):
                se.append(z[z["seg"] == s_][e].std(ddof=1))
            den = np.sqrt(se[0] ** 2 + se[1] ** 2)
            rows.append({"entry": e, "before": vb, "after": va, "se_before": se[0], "se_after": se[1],
                         "z": (vb - va) / den if den > 0 else np.nan})
        t = pd.DataFrame(rows)
        print("(standard errors: standard deviation over all bootstrap draws in the segment, as in Item 4)")
        print(t.to_string(index=False))
        print(f"{int((t['z'].abs() > 1.96).sum())} of {len(t)} entries have |z| > 1.96 "
              f"({len(t) * 0.05:.1f} expected by chance if nothing changed; entries are not independent)")
    for tag in ("", "_r5"):
        fo, fb = p / f"homog_sup_obs{tag}.csv", p / f"homog_sup_boot{tag}.csv"
        if tag and fo.exists() and fb.exists():
            so = pd.read_csv(fo)
            top = so.loc[so["lr"].idxmax()]
            bd = pd.read_csv(fb)
            bd = bd[bd["ok"] == 1]
            print(f"\n=== SUP-LR BOOTSTRAP (restarts-reduced run{tag}) ===")
            print(f"observed: {len(so)} dates, max LR {top['lr']:.2f} at January {int(top['year'])}")
            print(f"bootstrap: {len(bd)} sims; p = {(bd['sup_lr'] >= top['lr']).mean():.3f}; "
                  f"null sup-LR median {bd['sup_lr'].median():.2f}, 95th {bd['sup_lr'].quantile(.95):.2f}")


def plots(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"SMB": "#2a78d6", "HML": "#eb6834", "MOM": "#1baf7a"}   # categorical slots 1-3, validated all-pairs
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 1.6})
    from matplotlib.ticker import MaxNLocator
    out = Path(a.dir) / "figs"
    out.mkdir(parents=True, exist_ok=True)
    yr = MaxNLocator(integer=True)
    files = slide_files(a)
    ms = main_spec(a.repo)
    data = []
    for f in files:
        d = pd.read_csv(f)
        d = d[d["ok"] == 1].sort_values("start")
        d["year"] = 1927 + d["start"] // 12
        data.append(d)
    if data:
        n = len(data)
        for name, pre, ylabel in (("diag_A", "A", "a_jj (stay probability)"), ("diag_B", "B", "b_j(j) (own-factor emission)")):
            fig, axs = plt.subplots(1, n, figsize=(4.6 * n, 3.4), sharey=True, squeeze=False)
            for ax, d in zip(axs[0], data):
                for s in FACT:
                    ax.plot(d["year"], d[f"{pre}_{s}_{s}"], color=COL[s], label=s)
                    ax.annotate(s, (d["year"].iloc[-1], d[f"{pre}_{s}_{s}"].iloc[-1]), xytext=(3, 0), textcoords="offset points",
                                color=INK2, va="center", fontsize=8)
                bad = d[d["names_ok"] == 0]
                ax.plot(bad["year"], np.full(len(bad), ax.get_ylim()[0]), "|", color=INK2, ms=8, label="names fail")
                ax.set_title(f"window length L={int(d['L'].iloc[0])} months", fontsize=9, color=INK2)
                ax.set_xlabel("window start year")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[0][0].set_ylabel(ylabel)
            h_, l_ = axs[0][0].get_legend_handles_labels()
            fig.legend(h_, l_, frameon=False, loc="upper right", ncol=4, fontsize=8)
            fig.suptitle(f"{ylabel} by window start (training half of each window)", fontsize=10, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(out / f"slide_{name}.png", dpi=160)
            plt.close(fig)
        rows = [("hit", "hit rate, top factor", ms["hit"]), ("logp_diff", "log score, HMM minus chain", 0.0),
                ("sharpe_diff_tilt_ew", "Sharpe, tilt minus equal weight", 0.0)]
        fig, axs = plt.subplots(3, n, figsize=(4.6 * n, 7.6), sharex="col", squeeze=False)
        for c, d in enumerate(data):
            for r, (k, lab, ref) in enumerate(rows):
                ax = axs[r][c]
                ax.plot(d["year"], d[f"frozen_{k}"], color="#2a78d6", label="frozen")
                if f"refit_{k}" in d and d[f"refit_{k}"].notna().any():
                    ax.plot(d["year"], d[f"refit_{k}"], color="#eb6834", label="refit monthly")
                ax.axhline(ref, color=INK2, lw=0.9, ls="--", label="main spec" if k == "hit" else "zero")
                if r == 0:
                    ax.set_title(f"L={int(d['L'].iloc[0])}", fontsize=9, color=INK2)
                if c == 0:
                    ax.set_ylabel(lab)
                if r == 2:
                    ax.set_xlabel("window start year")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        hs = {}
        for ax in axs.ravel():
            for h_, l_ in zip(*ax.get_legend_handles_labels()):
                hs.setdefault(l_, h_)
        fig.legend(list(hs.values()), list(hs.keys()), frameon=False, loc="upper right", ncol=4, fontsize=8)
        fig.suptitle("Out-of-sample scores on each window's test half", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / "slide_walk_forward.png", dpi=160)
        plt.close(fig)
    # LR(tau) with bootstrap critical value
    tag = "_r5" if (Path(a.dir) / "homog_sup_boot_r5.csv").exists() and (Path(a.dir) / "homog_sup_obs_r5.csv").exists() else ""
    fo = Path(a.dir) / f"homog_sup_obs{tag}.csv"
    if fo.exists():
        so = pd.read_csv(fo).sort_values("tau_idx")
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(1927 + so["tau_idx"] / 12, so["lr"], color="#2a78d6", label="LR(tau)")
        top = so.loc[so["lr"].idxmax()]
        ax.plot([1927 + top["tau_idx"] / 12], [top["lr"]], "o", color="#2a78d6", ms=7, mfc="#fcfcfb", label=f"max: Jan {int(top['year'])}")
        fk = Path(a.dir) / "homog_known.csv"
        if fk.exists() and not tag:
            kd = pd.read_csv(fk)
            ax.axhline(kd[kd["ok"] == 1]["lr"].quantile(.95), color="#eb6834", lw=1.2, ls="--", label="single-date bootstrap 95th")
        fb = Path(a.dir) / f"homog_sup_boot{tag}.csv"
        if fb.exists():
            bd = pd.read_csv(fb)
            bd = bd[bd["ok"] == 1]
            ax.axhline(bd["sup_lr"].quantile(.95), color="#1baf7a", lw=1.2, ls="--",
                       label=f"sup-LR bootstrap 95th (5% critical value); sup-LR p = {(bd['sup_lr'] >= top['lr']).mean():.3f}")
        ax.set_xlabel("break date (January of)")
        ax.set_ylabel("LR statistic")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Likelihood-ratio statistic against break date" + (" (5 restarts)" if tag else ""), fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(out / "lr_by_date.png", dpi=160)
        plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--dir", default="my_boot")
    for m in ("slide", "variants", "variants-boot", "zscores", "sup-obs", "sup-boot", "plots", "report"):
        ap.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--window", type=int, default=480)
    ap.add_argument("--step", type=int, default=12)
    ap.add_argument("--no-refit", action="store_true")
    ap.add_argument("--refit-every", type=int, default=1)
    ap.add_argument("--tau-step", type=int, default=12)
    ap.add_argument("--z-method", choices=["yearly", "parametric"], default="yearly")
    ap.add_argument("--reps", type=int, default=199)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--restarts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=9348)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    if a.slide:
        init2(a.repo, a.restarts)
        n = pb._G["n_full"]
        tasks = [(s, a.window, a.refit_every, not a.no_refit) for s in range(0, n - a.window + 1, a.step)]
        pool_run(slide_task, tasks, a.out or d / f"homog_slide_L{a.window}.csv", lambda t: [t[0]], ["start"], slide_cols(), a)
    elif a.variants:
        variants_obs(a)
    elif a.variants_boot:
        tasks = [(b, a.seed, SPLIT) for b in range(a.reps)]
        pool_run(variants_boot_task, tasks, a.out or d / "homog_variants_boot.csv", lambda t: [t[0]], ["rep"],
                 ["rep", "ok", "ll_un", "lr_B_shared", "lr_A_shared"], a)
    elif a.zscores:
        tasks = [(s, b, a.seed, a.z_method) for b in range(a.reps) for s in (0, 1)]
        pool_run(z_task, tasks, a.out or d / f"homog_z_{a.z_method}.csv", lambda t: [t[0], t[1]], ["seg", "rep"], z_cols(), a)
    elif a.sup_obs:
        pool_run(ht.sup_obs_task, taus_grid(a), a.out or d / "homog_sup_obs.csv", lambda t: [t], ["tau_idx"],
                 ["tau_idx", "year", "lr"], a)
    elif a.sup_boot:
        taus = taus_grid(a)
        tasks = [(b, a.seed, taus) for b in range(a.reps)]
        pool_run(ht.sup_task, tasks, a.out or d / "homog_sup_boot.csv", lambda t: [t[0]], ["rep"],
                 ["rep", "ok", "sup_lr", "tau_idx"], a)
    elif a.plots:
        plots(a)
    elif a.report:
        report(a)
    else:
        ap.print_help()
