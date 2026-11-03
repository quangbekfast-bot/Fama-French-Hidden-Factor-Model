import sys
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
import sta348_hmm_v1 as m
F = m.FACTORS; N = 3
label, ff3, mom = sys.argv[2], sys.argv[3], sys.argv[4]
w = m.load_french(ff3, mom)
dates, obs = m.make_observations(w, False, 36)
T = len(obs); split = int(round(T * 0.5)); tr = obs[:split]
base = m.base_rates(tr); pi = np.full(N, 1/3)
R = w.loc[dates, F].to_numpy()
monthly = label == "monthly"; P = 12 if monthly else 52
_, B0 = m.soft_init(base, 0.8, 2.0)
A_assumed = m.soft_init(base, 0.8, 2.0)[0]
A_data = m.mle_transition(tr, 0.5)
pd.set_option("display.float_format", "{:.3f}".format); pd.set_option("display.width", 200)
print(f"===== {label}: training {dates[0].date()} to {dates[split-1].date()} ({split} periods)")
print("A0 counted from data (top-factor transitions):\n", pd.DataFrame(A_data, F, F))
res = {}
for name, A0 in [("A0 assumed (0.8/0.1)", A_assumed), ("A0 from data", A_data)]:
    mod, info = m.fit_with_restarts(tr, pi, A0, B0, 20, 0.2, 500, 1e-6, False, 348)
    single = m.make_model(pi, A0, B0, 500, 1e-6, False); single.fit(tr.reshape(-1, 1))
    A, B = mod.transmat_, mod.emissionprob_
    L = B / base; ok = bool(np.all(L.argmax(1) == np.arange(N)) and np.all(np.diag(L) > 1))
    print(f"\n--- {name}")
    print(f"exact start alone -> log-lik {single.score(tr.reshape(-1,1)):.2f}, iterations {single.monitor_.iter}")
    print(f"best of 20 starts -> log-lik {info['best_loglik']:.2f}; starts keeping names {info['share_of_runs_keeping_soft_names']:.2f}; names supported {ok}")
    print("A\n", pd.DataFrame(A, F, F)); print("B\n", pd.DataFrame(B, F, [f+' tops' for f in F]))
    print("lift diag", np.diag(L).round(2), " durations", (1/(1-np.diag(A))).round(1), " stationary", m.stationary(A).round(3))
    res[name] = (mod, A, B)
(a1, A1, B1), (a2, A2, B2) = res.values()
print(f"\nMax |difference| between the two trained models: A {np.abs(A1-A2).max():.4f}, B {np.abs(B1-B2).max():.4f}")
if len(sys.argv) < 6 and (np.abs(A1-A2).max() > 1e-3 or np.abs(B1-B2).max() > 1e-3):
    print("Models differ -> running walk-forward for both (refit every 4 periods)")
    for name, (mod, _, _) in res.items():
        wf, _ = m.walk_forward(obs, split, mod, pi, 4, 500, 1e-6, False, 0.5, R)
        s = m.summarise_walk_forward(wf); r = m.summarise_returns(wf, P, 10.0)
        print(f"  {name}: hit {s.hit_rate[1]:.3f}, log score {s.avg_log_score[1]:.4f}; "
              f"tilt Sharpe {r.sharpe[2]:.3f}, top Sharpe {r.sharpe[1]:.3f}, expected Sharpe {r.sharpe[0]:.3f}")
# is model 2 just model 1 with the states renumbered?
import itertools
best = min(itertools.permutations(range(N)), key=lambda p: np.abs(A2[np.ix_(p, p)] - A1).max() + np.abs(B2[list(p)] - B1).max())
p = list(best)
print("Best renumbering of 'A0 from data' states onto 'A0 assumed' states:",
      {F[i]: F[k] for k, i in enumerate(p)},
      f"-> remaining max |diff| A {np.abs(A2[np.ix_(p, p)] - A1).max():.4f}, B {np.abs(B2[p] - B1).max():.4f}")
