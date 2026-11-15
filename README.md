# STA348 term project: factor-leadership hidden Markov model

Centralized workspace for everyone to push and contribute files.

Hidden states = which equity factor (SMB, HML, MOM) is currently favoured; observations = which factor had the highest return each period. Trained by Baum-Welch on 1927-1976, tested walk-forward on 1976-2026.

**Start with `docs/sta348_build_log.pdf`**: full methods, starting matrices, trained model, results, return tests and robustness checks. **`docs/sta348_data_diagnostics.pdf`** documents the data: sources, cleaning steps, observations kept and dropped, train/test split and summary statistics.

## Folder layout

```
code/
  sta348_hmm_v1.py          main script (train, Viterbi, walk-forward test, return test)
  analysis/                 one-off diagnostic scripts used for the build log (see below)
data/
  raw_ken_french/           Ken French downloads as received (monthly zips, weekly 3-factor, daily momentum)
  raw_sp_pilot/             S&P DJI index exports used in the abandoned pilot (not in the public repo)
  clean/                    model inputs: factor returns (decimals), top factor, train/test label
results/
  monthly_main/             main specification (monthly, refit every month) + full-sample Viterbi
  monthly_pitrain/          monthly main specification with pi fixed at training shares (robustness; no material change)
  monthly_volscaled/        monthly with volatility scaling (state names fail; not used)
  weekly_refit/             weekly, refit every week
  weekly_frozen/            weekly, A and B frozen after training
  weekly_volscaled/         weekly with volatility scaling
  pilot_sp_indices_10y/     S&P index pilot (states track market direction; abandoned)
  simulation_check/         fit on data simulated from a known model (estimator check)
  bootstrap_null_prelim/    Yury's bootstrap of A and B and no-memory null test (preliminary, 200 reps)
docs/
  sta348_build_log.pdf/.tex build log (LaTeX source included; compile with pdflatex, bibtex, pdflatex x2)
  sta348_references.bib     all references with verified DOIs; keys listed in the build log's Citation map appendix
  sta348_data_diagnostics.pdf/.tex   data diagnostics (LaTeX source; needs cumulative_factor_returns_monthly.pdf to compile)
  Sta348_Model_design_doc_v1/v2.pdf   your design documents
  factor-leadership-markov-project.md early project notes (4-5 Oct; superseded by the build log)
  references/               papers used (not in the public repo; full citations with DOIs in sta348_references.bib)
```

## Reproducing the main results

Requires Python 3 with `numpy`, `pandas`, `hmmlearn` (0.3.x) and, for the pilot only, `xlrd`.

```
# monthly (main specification), about 2 minutes
python code/sta348_hmm_v1.py french \
    --ff3 data/raw_ken_french/F-F_Research_Data_Factors_CSV.zip \
    --mom data/raw_ken_french/F-F_Momentum_Factor_CSV.zip \
    --out results/monthly_main

# weekly; refitting every week takes about 15 minutes, --refit-every 0 about 40 seconds
python code/sta348_hmm_v1.py french \
    --ff3 data/raw_ken_french/F-F_Research_Data_Factors_weekly.csv \
    --mom data/raw_ken_french/F-F_Momentum_Factor_daily.csv \
    --out results/weekly_refit

# estimator check on simulated data
python code/sta348_hmm_v1.py simulate --weeks 3000 --refit-every 4 --out results/simulation_check
```

Useful options: `--vol-scale`, `--refit-every k`, `--pi train`, `--cost-bps`, `--restarts`, `--train-frac`. Run with `-h` for the full list.

## Files in each results folder

| File | Contents |
|---|---|
| `params.json` | trained A, B, pi, lift, expected durations, stationary distribution, raw transition matrix, fit and refit diagnostics, settings |
| `walk_forward.csv` | one row per test period: forecasts, state and top-factor probabilities, benchmark predictions, log scores; monthly also has each rule's weights and returns |
| `walk_forward_summary.csv` | hit rates and log scores by predictor |
| `returns_summary.csv` | monthly only: return, volatility, Sharpe, drawdown, turnover, net-of-cost figures by rule |
| `panel.csv` / `weekly_panel.csv` | per-period factor returns, top factor, training Viterbi state, filtered probabilities (the weekly runs used an earlier script version that named this file `weekly_panel.csv`) |
| `viterbi_spells_monthly.csv`, `viterbi_path_monthly.csv` | monthly_main only: full-sample Viterbi spells and month-by-month path (hindsight, description only) |

## Analysis scripts (`code/analysis/`)

Written to produce specific build-log numbers. Each takes paths as arguments; the first argument is the folder containing `sta348_hmm_v1.py` (i.e. `code`). Yury's three scripts instead take `--repo` (default: current folder) and are run from the repository root; preliminary results are in `results/bootstrap_null_prelim/` (build log Section 14).

| Script | Produces | Example |
|---|---|---|
| `diag.py` | BIC comparison, state-conditional means, posterior predictive check, hit rate by confidence and decade | `python code/analysis/diag.py code results/monthly_main <ff3 file> <mom file>` |
| `hitdiag.py` | HMM vs raw transition matrix: agreement, McNemar test, what each predicts | `python code/analysis/hitdiag.py results/monthly_main` |
| `sharpe_boot.py` | 12-month block bootstrap of Sharpe difference vs equal weight | `python code/analysis/sharpe_boot.py results/monthly_main` |
| `viterbi_full.py` | full-sample refit and Viterbi spells | `python code/analysis/viterbi_full.py code results/monthly_main <ff3> <mom>` |
| `freefit.py` | free fit from 100 random starts, named afterwards | `python code/analysis/freefit.py code <ff3> <mom> results/monthly_main` |
| `init_compare.py` | assumed vs data-built starting A | `python code/analysis/init_compare.py code monthly <ff3> <mom>` |
| `param_boot.py` | parametric and moving-block bootstrap of A and B, cold-start refits (Yury) | `python code/analysis/param_boot.py --method block --block-len 12 --reps 1000 --out boot_block12.csv` |
| `null_test.py` | no-memory null test: real training sequence vs simulated memoryless sequences (Yury) | `python code/analysis/null_test.py --null --reps 1000 --out null_iid.csv`, then `--compare null_iid.csv` |
| `null_extra.py` | Markov-chain null (does the hidden layer help?) and power check (Yury); needs the two scripts above | `python code/analysis/null_extra.py --null-markov ...` |
| `datadiag.py` | data diagnostics: raw file checks, rows dropped, week alignment, summary statistics, top-factor counts | `python code/analysis/datadiag.py code <monthly ff3> <monthly mom> <weekly ff3> <daily mom>` |

## Headline result

Monthly leadership regimes are persistent (7-10 months) and interpretable, but out of sample the model predicts the top factor no better than simple counting (42.2% vs 41.9%), and no rule built on it beats an equal-weight factor portfolio on a risk-adjusted basis.

## Not included in this public repository

- `docs/references/`: the papers are copyrighted. Every one is cited, with a DOI or stable link, in `docs/sta348_references.bib` and the build log's reference list.
- `data/raw_sp_pilot/`: S&P Dow Jones Indices exports are proprietary. They were used only in the abandoned pilot; the pilot's results are in `results/pilot_sp_indices_10y/`.

The Ken French factor files are freely available from the [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html); the copies used (CRSP vintage 202609) are in `data/raw_ken_french/`.
