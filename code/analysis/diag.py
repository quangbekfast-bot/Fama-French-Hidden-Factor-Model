import sys, json
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import sta348_hmm_v1 as m
run = sys.argv[2]; ff3, mom = sys.argv[3], sys.argv[4]
P = json.load(open(f"{run}/params.json"))
A, B, pi = map(np.array, (P["A"], P["B"], P["pi"]))
w = m.load_french(ff3, mom)
dates, obs = m.make_observations(w, P["settings"]["vol_scale"], 52)
split = P["train_weeks"]; tr, te = obs[:split], obs[split:]
F = m.FACTORS
print("Top-factor shares  train:", m.as_dict(m.base_rates(tr)), " test:", m.as_dict(m.base_rates(te)))

# in-sample: HMM vs observable chain (BIC), both conditional on first obs roughly
_, ll_hmm = m.forward_filter(tr, pi, A, B)
C = m.mle_transition(tr, 0.0)
ll_chain = np.log(m.base_rates(tr)[tr[0]]) + np.log(C[tr[:-1], tr[1:]]).sum()
ll_iid = np.log(m.base_rates(tr)[tr]).sum()
n = len(tr)
for name, ll, k in [("iid (no memory)", ll_iid, 2), ("observable chain", ll_chain, 6 + 2), ("HMM", ll_hmm, 12)]:
    print(f"  {name:18s} loglik {ll:9.1f}  params {k:2d}  BIC {-2*ll + k*np.log(n):9.1f}")

# state-conditional mean weekly returns (smoothed probs, training only)
hmm = m.make_model(pi, A, B, 1, 1e-6, False)
g = hmm.predict_proba(tr.reshape(-1, 1))
R = w.loc[dates[:split], F].to_numpy() * 100
mu = (g.T @ R) / g.sum(0)[:, None]
print("\nMean weekly return (%) by hidden state, training, smoothed weights:")
print(pd.DataFrame(mu, index=[f"{f} state" for f in F], columns=F).round(3))
sd = np.sqrt((g.T @ R**2) / g.sum(0)[:, None] - mu**2)
print("Std dev (%) by state:"); print(pd.DataFrame(sd, index=[f"{f} state" for f in F], columns=F).round(3))

# posterior predictive check: simulate from fitted model, compare observed-symbol statistics
rng = np.random.default_rng(0)
sim = m.make_model(pi, A, B, 1, 1e-6, False)
def stats(o):
    sw = np.mean(o[1:] != o[:-1]); C = m.mle_transition(o, 0.0)
    return sw, np.diag(C)
obs_sw, obs_d = stats(tr)
sims = [stats(sim.sample(n, random_state=s)[0].ravel()) for s in range(300)]
sw = np.array([s[0] for s in sims]); dg = np.array([s[1] for s in sims])
print(f"\nPosterior predictive check (300 sims of {n} weeks):")
print(f"  share of weeks the top factor changes: observed {obs_sw:.3f}, simulated 95% range [{np.percentile(sw,2.5):.3f}, {np.percentile(sw,97.5):.3f}]")
for i, f in enumerate(F):
    print(f"  P({f} tops again next week): observed {obs_d[i]:.3f}, simulated [{np.percentile(dg[:,i],2.5):.3f}, {np.percentile(dg[:,i],97.5):.3f}]")

# walk-forward: confidence buckets and sub-periods
wf = pd.read_csv(f"{run}/walk_forward.csv", )
wf["week_ending"] = pd.to_datetime(wf["period_ending"] if "period_ending" in wf else wf["week_ending"])
wf["conf"] = wf[[f"p_top_{f}" for f in F]].max(axis=1)
wf["hit_hmm"] = wf.hmm_pred_obs == wf.actual; wf["hit_mle"] = wf.mle_pred == wf.actual; wf["hit_base"] = wf.base_pred == wf.actual
print("\nWalk-forward hit rate by HMM confidence (max predicted top-factor prob):")
wf["bucket"] = pd.cut(wf.conf, [0, .4, .5, .6, 1.0])
print(wf.groupby("bucket", observed=True)[["hit_hmm", "hit_mle", "hit_base"]].agg(["mean"]).round(3).assign(n=wf.groupby("bucket", observed=True).size()))
wf["decade"] = (wf.week_ending.dt.year // 10) * 10
t = wf.groupby("decade")[["hit_hmm", "hit_mle", "hit_base"]].mean().round(3)
t["logscore_hmm_minus_mle"] = wf.groupby("decade").apply(lambda d: (d.hmm_logp - d.mle_logp).mean()).round(4)
t["n"] = wf.groupby("decade").size()
print("\nBy decade:"); print(t)
