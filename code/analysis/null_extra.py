"""Extra checks that build on null_test.py (same folder, plus param_boot.py).

  --null-markov  simulate from the OBSERVED 3x3 Markov chain (memory, no hidden layer), fit the
                 same statistics. Use with `null_test.py --compare` to test HMM vs Markov chain.
  --power        simulate from the FITTED HMM (a world where the model is true), same statistics.
  --report       --report power.csv null_iid.csv : power of the main test and a calibration check.
"""
import argparse, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import param_boot as pb  # noqa: E402
import null_test as nt  # noqa: E402

N = 3


def init_worker(repo):
    nt.init_worker(repo)
    train = pb._G["train"]
    pb._G["P_mc"], _ = nt.markov_fit(train)
    pb._G["first_p"] = np.bincount(train, minlength=N) / len(train)


def sim_markov(rng):
    n, P = pb._G["n_obs"], pb._G["P_mc"]
    x = np.empty(n, int)
    x[0] = rng.choice(N, p=pb._G["first_p"])
    for t in range(1, n):
        x[t] = rng.choice(N, p=P[x[t - 1]])
    return x


def one(args):
    mode, b, seed = args
    rng = np.random.default_rng([seed, b])
    obs = sim_markov(rng) if mode == "markov" else pb.resample("parametric", b, 0, rng)
    row = {"rep": b, "ok": 0}
    s = nt.stats_for(obs)
    if s is not None:
        row.update(s)
        row["ok"] = 1
    return row


def run(a, mode):
    init_worker(a.repo)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out)["rep"].astype(int)) if out.exists() else set()
    todo = [(mode, b, a.seed) for b in range(a.reps) if b not in done]
    print(f"{mode}: {len(done)} done, {len(todo)} to run, workers={a.workers}", flush=True)
    first = not out.exists()
    buf = []
    with Pool(a.workers, initializer=init_worker, initargs=(a.repo,)) as pool:
        for k, row in enumerate(pool.imap_unordered(one, todo, chunksize=1), 1):
            buf.append(row)
            if len(buf) >= 10 or k == len(todo):
                pd.DataFrame(buf).to_csv(out, mode="a", header=first, index=False)
                first, buf = False, []
                print(f"  {k}/{len(todo)}", flush=True)


def report(power_path, null_path):
    pw = pd.read_csv(power_path)
    pw = pw[pw["ok"] == 1]
    nl = pd.read_csv(null_path)
    nl = nl[nl["ok"] == 1]
    print(f"\nPOWER: {len(pw)} histories simulated from the fitted HMM (memory is real by construction)")
    for stat in ["gain_hmm_iid", "hmm_lam2", "hmm_diag_rank3", "mc_lam2"]:
        thr = np.percentile(nl[stat], 95)
        print(f"  {stat:16s} share above the i.i.d.-null 95th percentile ({thr:.3f}): "
              f"{(pw[stat] > thr).mean():.3f}   (median {pw[stat].median():.3f})")
    x = nl["gain_hmm_iid"].to_numpy(float)
    rng = np.random.default_rng(0)
    pv = []
    for i in rng.choice(len(x), size=min(300, len(x)), replace=False):
        rest = np.delete(x, i)
        pv.append((rest >= x[i]).mean())
    pv = np.array(pv)
    print(f"\nCALIBRATION of the main test on {len(pv)} null fits treated as 'real' data:")
    print(f"  share with p < 0.05: {(pv < 0.05).mean():.3f}  (should be about 0.05)")
    print(f"  share with p < 0.01: {(pv < 0.01).mean():.3f}  (should be about 0.01)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--null-markov", action="store_true")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--report", nargs=2)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7348)
    ap.add_argument("--out", default="my_boot/extra.csv")
    a = ap.parse_args()
    if a.null_markov:
        run(a, "markov")
    elif a.power:
        run(a, "hmm")
    elif a.report:
        report(*a.report)
    else:
        ap.print_help()
