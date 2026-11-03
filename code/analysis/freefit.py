import sys, itertools
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import sta348_hmm_v1 as m
F = m.FACTORS; N = 3
w = m.load_french(sys.argv[2], sys.argv[3])
dates, obs = m.make_observations(w, False, 36)
split = int(round(len(obs) * 0.5)); tr = obs[:split]; base = m.base_rates(tr)
pi = np.full(N, 1/3); rng = np.random.default_rng(348)
runs = []
for r in range(100):
    A0 = rng.dirichlet(np.ones(N), N); B0 = rng.dirichlet(np.ones(N), N)
    mod = m.make_model(pi, A0, B0, 500, 1e-6, False, r); mod.fit(tr.reshape(-1, 1))
    runs.append((mod.score(tr.reshape(-1, 1)), mod.transmat_.copy(), mod.emissionprob_.copy()))
runs.sort(key=lambda x: -x[0])
lls = np.array([r[0] for r in runs])
print("log-lik of 100 random starts: best %.2f; within 0.5 of best: %d; distinct optima (rounded to 0.1): %d"
      % (lls[0], (lls > lls[0] - 0.5).sum(), len(np.unique(lls.round(1)))))
print("top values:", np.unique(lls.round(2))[::-1][:8])
ll, A, B = runs[0]
pd.set_option("display.float_format", "{:.3f}".format)
print("\nBest free fit, states in the order EM returned them:")
print("A\n", pd.DataFrame(A, [f"state {i+1}" for i in range(N)], [f"state {i+1}" for i in range(N)]))
print("B\n", pd.DataFrame(B, [f"state {i+1}" for i in range(N)], [f"{f} tops" for f in F]))
L = B / base
print("Lift\n", pd.DataFrame(L, [f"state {i+1}" for i in range(N)], [f"{f} tops" for f in F]))
names = L.argmax(1); print("Name by highest lift:", [F[i] for i in names], " max lifts:", L.max(1).round(2))
# best assignment of states to factors (permutation maximising sum of diagonal lift)
best = max(itertools.permutations(range(N)), key=lambda p: sum(L[p[k], k] for k in range(N)))
print("Best one-to-one matching (state -> factor):", {f"state {best[k]+1}": F[k] for k in range(N)},
      " diagonal lifts:", [round(L[best[k], k], 2) for k in range(N)])
P = np.array(best); A_r = A[np.ix_(P, P)]; B_r = B[P]
print("\nReordered to SMB, HML, MOM:\nA\n", pd.DataFrame(A_r, F, F), "\nB\n", pd.DataFrame(B_r, F, [f"{f} tops" for f in F]))
print("Expected durations:", (1/(1-np.diag(A_r))).round(1), " stationary:", m.stationary(A_r).round(3))
# compare with soft-named fit
import json; Ps = json.load(open(sys.argv[4] + "/params.json"))
print("\nSoft-named fit log-lik: %.2f" % Ps["fit"]["best_loglik"])
print("Max abs diff vs soft fit:  A %.4f  B %.4f" % (np.abs(A_r - np.array(Ps["A"])).max(), np.abs(B_r - np.array(Ps["B"])).max()))
# how varied are the other local optima? check label meaning of each run
def meaning(Bx):
    Lx = Bx / base; return tuple(sorted(F[i] for i in Lx.argmax(1)))
cnt = pd.Series([meaning(r[2]) for r in runs]).value_counts()
print("\nWhat the 3 states mean across the 100 runs (factor with highest lift, per state):"); print(cnt)
near = [r for r in runs if r[0] > lls[0] - 0.5]
print("Among runs within 0.5 of best:", pd.Series([meaning(r[2]) for r in near]).value_counts().to_dict())
# second-best distinct optimum
vals = np.unique(lls.round(1))[::-1]
if len(vals) > 1:
    r2 = next(r for r in runs if round(r[0], 1) == vals[1])
    L2 = r2[2] / base
    print("\nNext-best distinct optimum (log-lik %.2f): B\n" % r2[0], pd.DataFrame(r2[2], [f"state {i+1}" for i in range(N)], [f"{f} tops" for f in F]))
    print("lift peaks:", [F[i] for i in L2.argmax(1)], L2.max(1).round(2), " A diag:", np.diag(r2[1]).round(3))
