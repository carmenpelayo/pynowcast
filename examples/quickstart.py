"""
pynowcast quickstart
====================

Complete tour of the toolbox in ~60 lines:

1. load data (Excel template, same logic as the original MATLAB toolbox)
2. estimate a dynamic factor model (Banbura & Modugno, 2014, with blocks)
3. produce a nowcast with uncertainty bands
4. decompose the change between two data vintages into "news"
5. run alternative models (bridge equations, BVAR)
6. backtest everything in pseudo-real time

Run from the repository root:  python examples/quickstart.py
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
