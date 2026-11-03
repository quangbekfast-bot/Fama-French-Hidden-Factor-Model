# Stochastic Processes Term Project: Factor Leadership as a Markov Chain

Working notes. Last updated 5 October 2026. Section 2 holds provisional design notes; nothing there is final.

## 1. Prospective topic

**Working question:** Is there a persistent underlying state in which one equity factor is favoured, and does knowing the current state tell you anything about the next one?

**Idea in one paragraph:** Treat "which factor is currently favoured" as a hidden state that moves between periods as a Markov chain. The state is not observed directly; it shows up through noisy factor returns. A hidden Markov model estimates the states and the transition matrix together. If the favoured state is persistent or rotates in a pattern, the transition matrix will show it.

**Why hidden and not observable:** The factor with the best return in a single period is partly luck. Treating that label as the true state would make leadership look like it switches more often than it does, so a plain chain on the observed leader would tend to understate persistence.

**Why it matters:** If the favoured state carries information, it could inform factor rotation in smart beta strategies. This is the motivation, not a finding. It only holds if the model predicts out of sample and survives trading costs.

**Fit with the course:** The course covers Markov chains with observable states. Hidden Markov models are not covered, so the core method goes beyond the course.

### Current lean (not final)

- Hidden Markov model as the main approach.
- Observable chain on the period's top factor kept as a baseline, since it is inside the course and shows what the hidden layer adds.

### Decisions still open

- [ ] Confirm with the instructor whether going beyond course material (hidden Markov models) is allowed or rewarded
- [ ] What the model observes: factor returns directly, or sector returns tied to factors (see section 2)
- [ ] Which factors to include, and whether to leave out the market factor
- [ ] Monthly or daily data
- [ ] How many states to fit

### Known risks

- The model does not name its states. Left unconstrained it tends to split calm from turbulent periods, not factor leadership.
- The number of states the data supports may be smaller than the number of factors.
- Persistence in factor leadership may be weak. A null result is a realistic outcome and still a valid one.
- Anything built on overlapping rolling windows creates persistence mechanically.
- More volatile factors come top more often for reasons unrelated to leadership.
- Which factors to include and how they are defined is a judgement call that can drive the result.

## 2. Provisional design notes

Notes from discussion so far. These record options and reasoning, not a settled design.

### 2.1 Structure

- **Hidden state:** which factor is currently favoured.
- **Transitions:** the state moves between periods as a Markov chain.
- **Observations:** each state produces returns with its own pattern.
- **Naming states:** done after fitting, by seeing which factor has the highest average return in each state. Two states can share a leader, and one may be a crisis state where nothing leads.

### 2.2 What the model observes

| Option | Assessment |
|---|---|
| Factor returns themselves | Leanest. Directly measured, free data. The finding is the transition matrix, not the state averages. Recommended in discussion; not yet confirmed. |
| A general market index | Does not work. One series can only separate bull, bear and turbulent regimes, which is what Wang et al. already do. |
| Sector returns, unconstrained | Possible but noisier. Sectors move for their own reasons and their factor tilts shift over time. Likely to find cyclical versus defensive regimes. |
| Sector returns tied to factors through sensitivities | Legitimate but much heavier. Sensitivities are parameters estimated inside the model. This is the structure in Ibanez and Urga and in Zhu. |
| Factor sensitivities as the observed data | Set aside. Sensitivities are built from factor returns, so they inherit any flaw in that data and add estimation error. They measure exposure, not payoff. Rolling estimates create false persistence. |

**Getting a sector breakdown without sector observations:** identify the states from factor returns, then report how the index and each sector behave in each state.

### 2.3 States versus data series

- The number of states and the number of observed series are separate choices.
- To tell apart a state where factor X is favoured, factor X's returns must be in the data. So the factor series included set the upper limit on the kinds of leadership that can be distinguished.
- Each state needs its own average for every factor plus how the factors vary and move together: roughly 20 numbers per state with five factors. Five states would be around 120 parameters on about 750 monthly observations, which is too thin.
- Plan from discussion: feed in all five or six factor series, fit two, three and four states, and let BIC choose. Daily data would support more states at the cost of noisier returns.

### 2.4 Handling the factor-choice problem

Commit upfront to a standard published set (French five plus momentum), then re-run on an alternative set as a check. A result that holds for only one set is worth reporting as such.

## 3. Data sources

Factor returns are published and free. They are realised returns of sorted portfolios, so there is nothing to estimate.

| Source | What it has | Notes |
|---|---|---|
| [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | Fama/French 5 Factors (market, size, value, profitability, investment); Momentum Factor, published separately; Industry Portfolios (5 to 49 industries). Daily and monthly. | CSV files with description lines at the top; monthly files have an annual table appended that needs trimming. Five-factor set starts in 1963. |
| [AQR Datasets](https://www.aqr.com/Insights/Datasets) | [Quality Minus Junk](https://www.aqr.com/Insights/Datasets/Quality-Minus-Junk-Factors-Daily); Betting Against Beta ([monthly](https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Monthly), [daily](https://www.aqr.com/Insights/Datasets/Betting-Against-Beta-Equity-Factors-Daily)) | Excel workbooks, one tab per factor. Last updated 31 July 2026 when checked. |
| MSCI factor indices and matching ETFs | Long-only, investable versions | Closer to smart beta in practice, but ETF history only goes back to roughly 2011 to 2013. Not yet looked up. |

**Preparation checks**

- All series on the same dates and frequency, one row per period, no gaps.
- The usable sample starts at the latest start date among the series.
- Units match across sources (percent versus decimal).

**What would still need estimating:** factor sensitivities of sectors or stocks, by regression on the factor returns, and only if the sector breakdown or the heavier sector-based model is used.

## 4. How this differs from existing work

Most papers below fit a regime model to the **market** (returns and volatility) and then look up which factor did best in each regime. The factors never enter the chain. This project puts the factors in the state itself.

In short: Wang et al. model market regimes with a factor overlay on top; this project tries to model factor rotation directly.

Of the papers found so far, none uses "which factor is favoured" as the state. The nearest in spirit is Shu and Mulvey (2024), which defines a regime per factor but uses a jump model, not a Markov chain on leadership. The papers in 5.2 fit a hidden Markov model on factor returns, which is the same input, but they read the states as market conditions.

## 5. Related literature

Status key: **PDF** = have the full paper; **Abstract** = only the abstract checked; **Unopened** = could not access, content unverified.

### 5.1 Starting points (have the PDFs)

| Paper | What it does | Relevance | Status |
|---|---|---|---|
| [Yuan (2019), "Market Regime Identification Using Hidden Markov Model"](https://ssrn.com/abstract=3406068) | Fits 2, 3 and 4 state models to daily FTSE 100 and Euro Stoxx 50 returns, 2011 to 2016. BIC picks 2 states. | Method reference for the hidden version. In sample only, no forecast test. Notation is loose in places. | PDF |
| [Wang, Lin and Mikhelson (2020), "Regime-Switching Factor Investing with Hidden Markov Models"](https://www.mdpi.com/1911-8074/13/12/311) | 3 market regimes from S&P 500 return and volatility; switches between six factor strategies by regime. Tested September 2017 to April 2020. | Closest prior work in goal. Market regime with a factor lookup on top. Result rests on few switches, short window, leverage. | PDF |

### 5.2 Hidden Markov model fitted on factor returns

| Paper | What it does | Relevance | Status |
|---|---|---|---|
| [Fons, Dawson, Yau, Zeng and Keane, "A novel dynamic asset allocation system using Feature Saliency HMMs for smart beta investing"](https://arxiv.org/abs/1902.10849) | 2 state model on factor index returns (25 S&P 500 style factors; 6 MSCI USA indices). Allocates across smart beta strategies by regime. | Closest to the smart beta angle. Performance claims (50 to 60% a year above market) need scrutiny. | Abstract |
| [Ibanez and Urga (2024), "Incorporating Market Regimes into Large-Scale Stock Portfolios"](https://mpra.ub.uni-muenchen.de/121552/1/MPRA_paper_121552.pdf) | Model on the Fama-French five factors, 1963 to 2022. Four regimes: bull, greed, fear, bear. | Same free data this project would use. Proper training and validation split. | Abstract |
| [Guidolin and Timmermann, "Size and Value Anomalies under Regime Shifts"](https://fedinprint.org/item/fedlwp/10192/original) | Joint regime model of market, size and value portfolios. Premia and correlations vary by regime. | Standard academic reference. Published in the Journal of Financial Econometrics, 2008. | Abstract |
| [Zhu (2022), "Regime-switching factor models with applications to portfolio selection and demand estimation"](https://economics.yale.edu/sites/default/files/Forms/zhu_brian_senior_essay_econ_492.pdf) | Yale senior essay. Derives the estimation steps; applied to crypto factors. | Template for a student write-up. Finds regimes split by volatility, not returns. | Abstract |

### 5.3 Market regimes with factors conditioned on them

| Paper | What it does | Relevance | Status |
|---|---|---|---|
| [Li and Piqueira (2019), "State-dependent size and value premium"](https://ideas.repec.org/a/pal/assmgt/v20y2019i3d10.1057_s41260-019-00113-9.html) | Regime model on CAPM and VIX. Size and value premia larger in low VIX state, reversed in high VIX state. | Evidence that factor premia depend on the regime. | Abstract |
| [Ammann and Verhofen (2006), "The Effect of Market Regimes on Style Allocation"](https://ideas.repec.org/a/kap/fmktpm/v20y2006i3p309-337.html) | Early regime-switching study of style allocation. | Cited by Wang et al. | Citation only |
| [Daniel, Jagannathan and Kim, "Tail Risk in Momentum Strategy Returns"](https://www.nber.org/papers/w18169) | Hidden Markov model to forecast momentum crashes. | One factor only, but the most rigorous of the set. | Abstract |
| [Lee (2026 preprint), "Regime-Dependent Predictive Structure Between Equity Factors"](https://arxiv.org/abs/2601.10732) | 3 regime model; value predicts size in crisis regimes only. Trading strategy on it loses money. | Honest example of statistical signal without economic value. | Abstract |

### 5.4 Regime per factor, not a hidden Markov model

| Paper | What it does | Relevance | Status |
|---|---|---|---|
| [Shu and Mulvey (2024), "Dynamic Factor Allocation Leveraging Regime-Switching Signals"](https://arxiv.org/abs/2410.14841) | Bull or bear regime for each factor's return relative to the market. Six style factors, 1993 to 2024. Uses a jump model. | Nearest in spirit: regimes are defined on the factors themselves. | Abstract |
| [Yu, Mulvey and Nie (2026), "Allocation-Focused Regimes and Applications to Dynamic Factor Investing"](https://collaborate.princeton.edu/en/publications/allocation-focused-regimes-and-applications-to-dynamic-factor-inv/) | Regimes defined by which strategy is outperforming; jump models plus a classifier. US factors, 1960 to 2024. | Regimes defined on relative strategy performance, which is close to the leadership idea. | Abstract |

### 5.5 Found but not opened

| Paper | Status |
|---|---|
| [Yang, "Harvesting Factor Premia Across Regimes: An Anchor-Stabilized Hidden Markov Framework for Multifactor Portfolios"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6823998) | Unopened |
| ["A Comparison of Value and Growth Investment Strategies Using the Hidden Markov Model With Switching and the Kaplan-Meier Method"](https://www.erfin.org/journal/index.php/erfin/article/view/220) | Unopened |

### 5.6 Background to track down

These came up in discussion from general knowledge and have not been checked against the sources. Verify before citing.

| Paper | Why it matters |
|---|---|
| Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition" | The standard tutorial for the three algorithms (forward-backward, Viterbi, Baum-Welch). Better to learn from than Yuan. |
| Gupta and Kelly (2019), "Factor Momentum Everywhere" | Evidence that factor returns persist, which is the premise behind leadership persistence. |
| Ehsani and Linnainmaa (2022), "Factor Momentum and the Momentum Factor" | Same theme. |
| Asness (2016), "The Siren Song of Factor Timing" | The sceptical view: factor timing adds little over a diversified factor mix. |
| Cooper and Ross (1998); Goldstein and Pauzner (2005) | Only relevant if the bank-run topic is revived. |

## 6. Topics considered and set aside

- **Diamond-Dybvig bank runs with a Markov sunspot:** the chain's probabilities would be assumed, not estimated, so most of the work is economics.
- **Kalman filter on a mean-reverting asset:** relies on hidden-state methods outside the course.
- **Credit rating migration, risk of ruin, loan roll rates, order book queues, binned mean-reverting spread:** viable alternatives if the factor topic falls through.

## 7. Notes and log

- 2026-10-04: Topic narrowed to factor leadership. Literature search run. Model design not started.
- 2026-10-05: Framing moved to a hidden Markov model, with the hidden state as the favoured factor. Discussed what the model should observe; factor returns recommended, sensitivities set aside. Data sources located. Design still provisional.
