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
| `BridgeEquations` | Banbura, Belousova, Bodnar & Toth (2023) | Combination of per-indicator bridge equations; monthly series completed by AR(p)-BIC, equal or inverse-MSE weights. |
| `BVAR` | inspired by Cimadomo, Giannone, Lenza, Monti & Sokol (2022) | *Simplified* quarterly Bayesian VAR (Minnesota prior via dummy observations, Banbura–Giannone–Reichlin 2010) with Gaussian conditioning on indicators observed within the quarter. Not the full mixed-frequency BVAR of the original paper. |

Plus, as in the original toolbox:

- **`news_decomposition`** — exact decomposition of a nowcast change into
  data revisions and series-by-series news (released − expected, with Kalman
  gain weights). Verified to satisfy the identity
  `new = old + revisions + Σ news` to machine precision.
- **`evaluate`** — pseudo-real-time backtesting: at each month of each
  quarter the data are cut to what would actually have been available
  (using per-series publication lags), the model is re-estimated and the
  nowcast is compared with an AR(1) benchmark.
- **`plots`** — news waterfall, nowcast-evolution and factor charts.

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

Run the full demo: `python examples/quickstart.py`.

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
    "Global" block loaded by every series is enforced automatically.

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
