# pynowcast

A simple, complete Python re-implementation of the **ECB nowcasting toolbox**
([Linzenich & Meunier, 2024](https://github.com/baptiste-meunier/Nowcasting_toolbox)),
rebuilt from scratch with a clean, intuitive API.

It produces nowcasts of a quarterly target (e.g. GDP growth) from a panel of
monthly indicators with ragged edges, and explains *why* the nowcast moved
between two data vintages ("news" decomposition).

## What's inside

| Model | Reference | Notes |
|---|---|---|
| `DFM` | Banbura & Modugno (2014); blocks as in Delle Chiaie, Ferrara & Giannone (2022) | Mixed-frequency dynamic factor model estimated by EM with arbitrary missing data. Faithful re-implementation: Mariano–Murasawa tent aggregation, block structure, AR(1) idiosyncratic components, exact Kalman filter/smoother. |
| `BridgeEquations` | Banbura, Belousova, Bodnar & Toth (2023) | Combination of bridge equations: all combinations of 1–2 monthly + 0–1 quarterly regressors (n + C(n,2) equations and more), lagM/lagQ/lagY settings, dummy support, AR(p)-BIC completion of monthly series, equal or inverse-MSE weights. |
| `BVAR` | Cimadomo, Giannone, Lenza, Monti & Sokol (2022) | **Blocking B-BVAR**: each monthly variable enters as three quarterly series (one per month of the quarter); Normal-Inverse-Wishart prior combining Minnesota and sum-of-coefficients priors via dummy observations (Banbura–Giannone–Reichlin 2010); ragged edge handled by Kalman filtering/smoothing with EM iterations (`bvar_thresh`/`bvar_max_iter`); exact news decomposition available. |

Plus, as in the original toolbox:

- **`news_decomposition`** — exact decomposition of a nowcast change into
  data revisions and series-by-series news (released − expected, with Kalman
  gain weights). Verified to satisfy the identity
  `new = old + revisions + Σ news` to machine precision.
- **`evaluate`** — pseudo-real-time backtesting: at each month of each
  quarter the data are cut to what would actually have been available
  (using per-series publication lags), the model is re-estimated and the
  nowcast is compared with an AR(1) benchmark.
- **`plots`** — news waterfall, nowcast-evolution, factor, heatmap,
  contributions and alternative-range charts.

## The three-step model-building approach (v1.1)

The original toolbox organizes model creation along three steps, all now
available in Python:

**Step 1 — Variable pre-selection** (`preselect`, replacing
`Variable_selection_vF.R`): ranks all candidate regressors by predictive
power with three methods — Sure Independence Screening (Fan & Lv, 2008),
the t-stat method of Bair et al. (2006) with four lags of the target, and
Least-Angle Regression (Efron et al., 2004) — combined into a weighted
score (higher weight on LARS, as in the paper's application), alongside
each variable's frequency, publication lag and group. `apply_selection`
restricts the dataset to the chosen variables (the `do_subset` option).

**Step 2 — Model selection** (`random_search` = `do_loop = 1`,
`custom_search` = `do_loop = 2`): evaluates many specifications
out-of-sample in pseudo real time, drawing the estimation start, the
variable subset and the model parameters at random within user bounds, and
reports RMSE and FDA per horizon and month of the quarter in a tidy
DataFrame (save it to Excel/CSV for inspection). Fixing the seed reproduces
the same set of models (`Loop.do_random = 0`), e.g. to isolate the effect
of a Covid correction.

**Step 3 — Covid robustness** (`covid_correct`, same numbering as
`do_Covid`): 0 = nothing; 1 = dummies for 2020 Q2 and Q3; 2 = delete
observations from Feb. to Sep. 2020; 3 = outlier correction (median ± 4
inter-quintile distances, replaced by NaN); 4 = dummies for 2020 Q1 and Q2.
`custom_search(..., covid_methods=(0, 1, 2, 3))` compares corrections on
identical specifications. A general `outlier_correct` is also available,
and `BridgeEquations(dummies=[...])` reproduces the `Par.Dum` mechanism.

## Policy outputs (v1.1)

- **`confidence_bands`** — uncertainty bands à la Reifschneider & Tulip
  (2019): rolling MAE of past pseudo-real-time predictions per horizon ×
  month of quarter, adjusted for outliers as in ECB (2009); ±1 MAE is the
  57.5% confidence interval. FDA over the same window gauges directional
  uncertainty. Reuse a saved table (`do_mae = 0`) or recompute
  (`do_mae = 1`).
- **`contributions`** — approximate contributions of input variables (or
  groups), proxied by the news from all releases over the past two years;
  satisfies mean + Σ contributions = prediction exactly.
- **`heatmap`** — z-scores of input variables (5-month smoothing for
  monthly series per Mariano–Murasawa), by variable or by group.
- **`share_of_available_data`** — fraction of the target quarter's monthly
  observations already released.
- **`alternative_range`** — predictions of alternative models obtained by
  disconnecting one or two *groups* of variables (`do_range = 1`); groups
  are set in the `group` column of the spec sheet.

## Installation

```bash
pip install -r requirements.txt   # numpy, scipy, pandas (+ matplotlib, openpyxl optional)
pip install -e .                  # or just put the repo on your PYTHONPATH
```

## Quickstart

```python
import pynowcast as pn

# 1. data — one Excel file with sheets 'monthly', 'quarterly' and 'spec'
data = pn.load_data("example_data/example_data.xlsx", target="GDP")

# 2. dynamic factor model: 1 global factor + 1 factor for the 'Soft' block
model = pn.DFM(factors={"Global": 1, "Soft": 1}, lags=2).fit(data)

# 3. nowcast with uncertainty
point, std = model.nowcast(period="2026Q2", with_uncertainty=True)

# 4. why did it move between April and May?
news = pn.news_decomposition(model,
                             data.vintage("2026-04"), data.vintage("2026-05"),
                             target="GDP", period="2026Q2")
print(news.summary())

# 5. alternative models
pn.BridgeEquations().fit(data).nowcast(period="2026Q2")
pn.BVAR().fit(data).nowcast(period="2026Q2")

# 6. backtest
res = pn.evaluate(lambda: pn.DFM(factors={"Global": 1, "Soft": 1}, lags=2),
                  data, start="2023Q1", end="2026Q1")
print(pn.evaluation_summary(res))
```

Model building and policy outputs:

```python
ranking  = pn.preselect(data)                          # step 1
data_sel = pn.apply_selection(data, ranking.head(10).index)
search   = pn.random_search(data_sel, "DFM", n_iter=100,
                            start="2015Q1")            # step 2
robust   = pn.custom_search(data_sel, [best_spec],
                            covid_methods=(0, 1, 2, 3))  # step 3

bands = pn.confidence_bands(model_factory, data, years=10)
contr, mean = pn.contributions(model, data, period="2026Q2")
pn.heatmap(data, by_group=True)
pn.alternative_range(model_factory, data, period="2026Q2")
pn.share_of_available_data(data.vintage("2026-05"), "2026Q2")
```

Run the full demo: `python examples/quickstart.py`.

**Prefer a narrated, end-to-end walkthrough?** Open
[`examples/nowcasting_walkthrough.ipynb`](examples/nowcasting_walkthrough.ipynb),
a Jupyter notebook that follows the natural order of a forecasting
exercise — data import, transformation (incl. Covid & outlier
treatment), variable pre-selection, model selection, model fit,
pseudo-real-time evaluation, and policy interpretation with all the
plots — with markdown explaining what each step does and what you can
configure.

## Input format

One Excel workbook (or three CSVs with a common prefix):

- **`monthly`** — first column dates (`2026-05` or any parseable date), one
  column per monthly indicator, in raw (untransformed) levels.
- **`quarterly`** — first column quarters (`2026Q1`), one column per
  quarterly series; the first one (or `target=`) is the nowcast target.
- **`spec`** *(optional)* — one row per series:
  - `series` — name (must match a column),
  - `transform` — `pch` (log-difference ×100, default), `pchy`
    (year-on-year), `diff`, `none`,
  - `pub_lag` — publication lag in months (e.g. `2` = the March value
    becomes available at the end of May); used to build realistic
    pseudo-real-time vintages,
  - `block_<Name>` — 0/1 columns assigning series to factor blocks. A
    "Global" block loaded by every series is enforced automatically,
  - `group` — variable group (e.g. "Surveys", "Industry") used for news
    aggregation, contributions, the heatmap and the range of alternative
    models (distinct from factor blocks).

`pynowcast.make_example_dataset()` generates a synthetic dataset (and the
shipped `example_data/example_data.xlsx`) so everything runs out of the box.

## How the DFM works (short version)

Each monthly series is `y_it = λ_i' f_t + e_it` with block-restricted
loadings; quarterly series load the factors through the Mariano–Murasawa
tent `[1 2 3 2 1]` so that quarterly growth is consistent with monthly
growth. Factors follow a VAR(p), idiosyncratic terms are AR(1). Everything
is cast in state space and estimated by the EM algorithm of Banbura &
Modugno (2014), which handles arbitrary patterns of missing data — ragged
edges, mixed frequencies and series with different start dates all come for
free. The Kalman smoother then delivers the nowcast and its variance, and
the news decomposition follows Banbura & Modugno's Section 4.

## Design choices vs. the MATLAB original

- **From scratch, pure Python** — `numpy`/`scipy`/`pandas` only; no MATLAB
  translation artifacts. The MATLAB code's `nowcasting.m` pipeline maps to:
  data template → `load_data` / `build_data`; `common_*` model code →
  `DFM` / `BridgeEquations` / `BVAR`; news → `news_decomposition`;
  out-of-sample evaluation → `evaluate`.
- **DFM is the faithful core**; bridge equations are close to the original;
  the BVAR is a deliberately simplified quarterly version (documented in
  `bvar.py`) rather than the full Cimadomo et al. (2022) mixed-frequency
  model.
- Sensible defaults everywhere; every model exposes the same three calls:
  `fit(data)`, `nowcast(period=...)`, and works with `evaluate`.

## Mapping to the original toolbox

| Original (MATLAB / R) | pynowcast |
|---|---|
| `Variable_selection_vF.R` (SIS, t-stat, LARS) | `preselect`, `sis_rank`, `tstat_rank`, `lars_rank` |
| `do_Covid = 0..4` | `covid_correct(data, method=0..4)` |
| `do_subset` / `var_keep` | `apply_selection` |
| `do_loop = 1` (random models) | `random_search` |
| `do_loop = 2` (custom list, `alter_covid`) | `custom_search(..., covid_methods=...)` |
| `do_range = 1` (disconnect groups) | `alternative_range` |
| `do_mae = 1` (MAE/FDA over 10 years) | `confidence_bands` |
| `common_heatmap` | `heatmap`, `plots.plot_heatmap` |
| Contributions (news over 2 years) | `contributions` |
| `common_eval_models` (RMSE/FDA, horizons, sub-periods) | `evaluate`, `evaluation_summary` |
| News decomposition vs previous run | `news_decomposition` between any two vintages |
| `BVAR_News_Mainfile` | `news_decomposition(bvar_model, ...)` -- same exact identity |
| `Par.Dum` (BEQ dummies) | `BridgeEquations(dummies=[...])` |

The one intentional simplification left: monthly indicators in the
*bridge equations* are extrapolated with univariate AR-BIC instead of the
auxiliary BVAR(6) (both are variants in Banbura et al., 2023). The B-BVAR
itself needs no extrapolation -- the Kalman smoother conditions directly
on the ragged edge.

## Testing

```bash
python tests/test_toolbox.py       # or: pytest tests/
```

Checks include: monotone EM log-likelihood, news identity to machine
precision, vintage construction respecting publication lags, all models
working in every month of the quarter, and the DFM beating the AR(1)
benchmark late in the quarter on factor-driven data.

## References

- Banbura, M. & Modugno, M. (2014), "Maximum likelihood estimation of factor
  models on datasets with arbitrary pattern of missing data", *Journal of
  Applied Econometrics* 29(1).
- Banbura, M., Belousova, I., Bodnar, K. & Toth, M.B. (2023), "Nowcasting
  employment in the euro area", ECB Working Paper 2815.
- Banbura, M., Giannone, D. & Reichlin, L. (2010), "Large Bayesian vector
  auto regressions", *Journal of Applied Econometrics* 25(1).
- Cimadomo, J., Giannone, D., Lenza, M., Monti, F. & Sokol, A. (2022),
  "Nowcasting with large Bayesian vector autoregressions", *Journal of
  Econometrics* 231(2).
- Delle Chiaie, S., Ferrara, L. & Giannone, D. (2022), "Common factors of
  commodity prices", *Journal of Applied Econometrics* 37(3).
- Linzenich, J. & Meunier, B. (2024), "The ECB nowcasting toolbox",
  https://github.com/baptiste-meunier/Nowcasting_toolbox.
