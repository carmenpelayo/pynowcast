"""
pynowcast quickstart (script form)
==================================

A compact, run-from-the-CLI tour of the toolbox. For a fully narrated,
step-by-step version that follows the natural order of a forecasting
exercise (import -> transform/Covid/outliers -> variable selection ->
model selection -> fit -> evaluation -> policy interpretation & plots),
open the companion notebook instead:

    examples/nowcasting_walkthrough.ipynb

Run this script from the repository root:  python examples/quickstart.py
"""
import warnings
warnings.filterwarnings("ignore")

import pynowcast as pn

# ---------------------------------------------------------------- 1. data
data = pn.load_data("example_data/example_data.xlsx", target="GDP")
print(data.summary(), "\n")

# ------------------------------------------------------- 2. fit the DFM
# one global factor loaded by everything + one factor for soft data only
model = pn.DFM(factors={"Global": 1, "Soft": 1}, lags=2)
model.fit(data)

# ------------------------------------------------------------ 3. nowcast
point, std = model.nowcast(period="2026Q2", with_uncertainty=True)
print(f"DFM nowcast 2026Q2 : {point:+.2f} (± {std:.2f})\n")

# ------------------------------------------------- 4. news decomposition
# what moved the nowcast between the April and May vintages?
old = data.vintage("2026-04")
new = data.vintage("2026-05")
news = pn.news_decomposition(model, old, new, target="GDP", period="2026Q2")
print(news.summary(), "\n")
print(news.table.round(3).to_string(index=False), "\n")

# ------------------------------------------------- 5. alternative models
bridge = pn.BridgeEquations().fit(data)
bvar = pn.BVAR().fit(data)
print(f"Bridge equations    : {bridge.nowcast(period='2026Q2'):+.2f}")
print(f"BVAR                : {bvar.nowcast(period='2026Q2'):+.2f}\n")

# --------------------------------------------- 6. pseudo-real-time backtest
results = pn.evaluate(lambda: pn.DFM(factors={"Global": 1, "Soft": 1}, lags=2),
                      data, start="2024Q1", end="2026Q1", verbose=False)
print("DFM backtest (RMSE by month within the quarter, vs AR(1) benchmark):")
print(pn.evaluation_summary(results).round(3).to_string(), "\n")

# ------------------------------------------------------------------ plots
try:
    import matplotlib
    matplotlib.use("Agg")
    from pynowcast import plots

    plots.plot_news(news).figure.savefig("news_waterfall.png", dpi=150,
                                         bbox_inches="tight")
    plots.plot_nowcast_evolution(model, data, period="2026Q2") \
         .figure.savefig("nowcast_evolution.png", dpi=150, bbox_inches="tight")
    print("Saved news_waterfall.png and nowcast_evolution.png")
except ImportError:
    print("matplotlib not installed - skipping plots")

# ======================================================================
# NEW IN 1.1 -- the toolbox's three-step model-building approach
# ======================================================================

# ------------------------------------------- step 1: variable pre-selection
print("\n--- Step 1: variable pre-selection (SIS / t-stat / LARS) ---")
ranking = pn.preselect(data)
print(ranking.round(3).head(8).to_string())
data_sel = pn.apply_selection(data, ranking.head(8).index)

# ------------------------------------------------- step 2: model selection
print("\n--- Step 2: model selection (random search, small demo run) ---")
search = pn.random_search(data_sel, "BVAR", n_iter=3, seed=1,
                          start="2025Q1", end="2025Q4",
                          months_in_quarter=(3,), verbose=False)
print(search[["spec_id", "lags", "rmse_nowcast", "fda_nowcast"]]
      .round(3).to_string(index=False))

# ----------------------------------------------- step 3: Covid robustness
print("\n--- Step 3: Covid robustness (corrections 0/2/3 on one spec) ---")
robust = pn.custom_search(data_sel, [{"model": "BVAR", "lags": 2}],
                          start="2025Q1", end="2025Q4",
                          months_in_quarter=(3,),
                          covid_methods=(0, 2, 3), verbose=False)
print(robust[["covid_method", "rmse_nowcast", "fda_nowcast"]]
      .round(3).to_string(index=False))

# ======================================================================
# NEW IN 1.1 -- policy outputs
# ======================================================================
print("\n--- Policy outputs ---")
print("Share of 2026Q2 data available in the May vintage: "
      f"{pn.share_of_available_data(data.vintage('2026-05'), '2026Q2'):.0%}")

contr, mean_ = pn.contributions(model, data, period="2026Q2")
print(f"\nApproximate contributions (mean = {mean_:+.2f}):")
print(contr.round(3).to_string())

rng = pn.alternative_range(lambda: pn.DFM(factors=1, lags=2, max_iter=30),
                           data, period="2026Q2")
print("\nRange of alternative models:")
print(rng.round(3).to_string(index=False))

try:
    from pynowcast import plots as _plots
    _plots.plot_heatmap(data, last=8).figure.savefig(
        "heatmap.png", dpi=150, bbox_inches="tight")
    _plots.plot_range(rng).figure.savefig(
        "alternative_range.png", dpi=150, bbox_inches="tight")
    _plots.plot_contributions(contr, mean_).figure.savefig(
        "contributions.png", dpi=150, bbox_inches="tight")
    print("\nSaved heatmap.png, alternative_range.png, contributions.png")
except ImportError:
    pass
