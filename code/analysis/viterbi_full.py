import sys, json
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import sta348_hmm_v1 as m
run, ff3, mom = sys.argv[2:5]
P = json.load(open(f"{run}/params.json"))
F = m.FACTORS
w = m.load_french(ff3, mom)
dates, obs = m.make_observations(w, False, 36)
pi, A, B = map(np.array, (P["pi"], P["A"], P["B"]))
# full-sample refit, warm-started from the training fit (same as the script's forecast step)
fin = m.make_model(pi, A, B, 500, 1e-6, False); fin.fit(obs.reshape(-1, 1))
A2, B2 = fin.transmat_, fin.emissionprob_
names = m.state_names(B2, m.base_rates(obs))
print("Full-sample names hold:", list(names) == [0, 1, 2])
print("A full sample:\n", pd.DataFrame(A2, F, F).round(3)); print("B full sample:\n", pd.DataFrame(B2, F, [f+' tops' for f in F]).round(3))
ll, vit = fin.decode(obs.reshape(-1, 1), algorithm="viterbi")
gam = fin.predict_proba(obs.reshape(-1, 1))
R = w.loc[dates, F].to_numpy() * 100
spells = []; s0 = 0
for t in range(1, len(vit) + 1):
    if t == len(vit) or vit[t] != vit[s0]:
        seg = slice(s0, t)
        spells.append({"state": F[vit[s0]], "start": dates[s0].strftime("%Y-%m"), "end": dates[t-1].strftime("%Y-%m"),
                       "months": t - s0, **{f"avg_{f}_%": R[seg, i].mean() for i, f in enumerate(F)},
                       "share_own_tops": np.mean(obs[seg] == vit[s0]), "avg_post": gam[seg, vit[s0]].mean()})
        s0 = t
sp = pd.DataFrame(spells)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(f"\n{len(sp)} spells; average length by state:", sp.groupby("state").months.mean().round(1).to_dict())
print("Share of months by state:", (pd.Series(vit).map(dict(enumerate(F))).value_counts(normalize=True)).round(3).to_dict())
tab = pd.DataFrame({"months": pd.Series(vit).map(dict(enumerate(F))).value_counts()})
print("\nAverage monthly factor return (%) in Viterbi-assigned months:")
print(pd.DataFrame(R).groupby(pd.Series(vit).map(dict(enumerate(F)))).mean().set_axis(F, axis=1).round(2))
print("\nShare of months where the posterior prob of the Viterbi state > 0.8:", np.mean(gam[np.arange(len(vit)), vit] > 0.8).round(3))
print(sp.round(2).to_string(index=False))
sp.to_csv(f"{run}/viterbi_spells_full.csv", index=False)
pd.DataFrame({"month": dates, "top_factor": [F[o] for o in obs], "viterbi_state": [F[v] for v in vit],
              **{f"smoothed_p_{f}": gam[:, i] for i, f in enumerate(F)}}).to_csv(f"{run}/viterbi_path_full.csv", index=False)
