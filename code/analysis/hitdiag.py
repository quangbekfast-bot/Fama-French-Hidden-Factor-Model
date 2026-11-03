import sys, numpy as np, pandas as pd
from math import comb
F = ["SMB", "HML", "MOM"]
def mcnemar(b, c):  # exact two-sided
    n = b + c; k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
for run in sys.argv[1:]:
    wf = pd.read_csv(f"{run}/walk_forward.csv")
    dcol = "period_ending" if "period_ending" in wf else "week_ending"
    wf[dcol] = pd.to_datetime(wf[dcol])
    h = wf.hmm_pred_obs == wf.actual; m = wf.mle_pred == wf.actual
    n = len(wf)
    print(f"\n==== {run.split('/')[-1]}  ({n} test periods)")
    print(f"Hit rate: HMM {h.mean():.3f}  raw matrix {m.mean():.3f}  difference {h.mean()-m.mean():+.3f}")
    agree = wf.hmm_pred_obs == wf.mle_pred
    b = int((h & ~m).sum()); c = int((~h & m).sum())
    print(f"Same prediction in {agree.mean():.1%} of periods")
    print(f"When they disagree ({(~agree).sum()} periods): HMM right {b}, raw matrix right {c}, neither {int((~agree & ~h & ~m).sum())};  exact McNemar p = {mcnemar(b, c):.3f}")
    d = wf.hmm_logp - wf.mle_logp
    print(f"Log score difference HMM - raw: {d.mean():+.4f} (se {d.std(ddof=1)/np.sqrt(n):.4f})")
    print("How often each factor is predicted (share of periods):")
    t = pd.DataFrame({"actual top": wf.actual.value_counts(normalize=True),
                      "HMM predicts": wf.hmm_pred_obs.value_counts(normalize=True),
                      "raw predicts": wf.mle_pred.value_counts(normalize=True)}).reindex(F).fillna(0)
    print(t.round(3))
    print("Hit rate given what was predicted (precision):")
    pr = pd.DataFrame({"HMM": h.groupby(wf.hmm_pred_obs).mean(), "HMM n": h.groupby(wf.hmm_pred_obs).size(),
                       "raw": m.groupby(wf.mle_pred).mean(), "raw n": m.groupby(wf.mle_pred).size()}).reindex(F)
    print(pr.round(3))
    dis = ~agree
    print("When they disagree, what each predicted (counts) and who was right:")
    print(pd.crosstab([wf.hmm_pred_obs[dis], wf.mle_pred[dis]], wf.actual[dis]))
    print("Hit rate by last period's top factor:")
    prev = wf.actual.shift(1); prev.iloc[0] = np.nan
    g = pd.DataFrame({"prev": prev, "h": h, "m": m}).dropna().groupby("prev")
    print(pd.DataFrame({"HMM": g.h.mean(), "raw": g.m.mean(), "n": g.size()}).reindex(F).round(3))
    wf["decade"] = (wf[dcol].dt.year // 10) * 10
    g = pd.DataFrame({"decade": wf.decade, "h": h, "m": m, "d": d}).groupby("decade")
    print("By decade:")
    print(pd.DataFrame({"HMM": g.h.mean(), "raw": g.m.mean(), "logscore_diff": g.d.mean(), "n": g.size()}).round(3))
