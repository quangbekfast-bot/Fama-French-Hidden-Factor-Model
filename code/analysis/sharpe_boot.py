import sys, numpy as np, pandas as pd
for run in sys.argv[1:]:
    wf = pd.read_csv(f"{run}/walk_forward.csv"); rng = np.random.default_rng(0)
    T = len(wf); L = 12; nb = T // L + 1
    print("==", run.split('/')[-1], "Sharpe minus equal-weight Sharpe, 12-month block bootstrap (gross)")
    ew = wf.ret_equal_weight.to_numpy()
    for k in ["hmm_expected","hmm_top","hmm_tilt","chain_expected","last_winner","best_so_far"]:
        r = wf[f"ret_{k}"].to_numpy(); sr = lambda x: x.mean()/x.std(ddof=1)*np.sqrt(12)
        d0 = sr(r) - sr(ew); ds = []
        for _ in range(2000):
            idx = (rng.integers(0, T - L, nb)[:, None] + np.arange(L)).ravel()[:T]
            ds.append(sr(r[idx]) - sr(ew[idx]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"  {k:15s} {d0:+.3f}  95% range [{lo:+.3f}, {hi:+.3f}]")
