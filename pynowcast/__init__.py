"""
pynowcast: a simple Python re-implementation of the ECB nowcasting toolbox
(Linzenich & Meunier, 2024, ECB WP 3004).

Models
------
DFM              dynamic factor model (Banbura-Modugno EM, mixed frequency,
                 optional block structure) -- the workhorse
BridgeEquations  combination of bridge equations
BVAR             Bayesian VAR with a Minnesota prior (simplified)

Workflow
--------
>>> import pynowcast as pn
>>> data = pn.load_data("my_data.xlsx")          # or pn.build_data(...)
>>> model = pn.DFM(factors=2, lags=2).fit(data)
>>> model.nowcast("GDP", "2026Q2")
>>> news = model.explain_change(old_vintage, new_vintage)
>>> print(news.summary())
>>> res = pn.evaluate(lambda: pn.DFM(2, 2), data, start="2022Q1")
>>> pn.evaluation_summary(res)
"""

from .data import (
    NowcastData,
    build_data,
    load_data,
    make_example_dataset,
    quarter_of,
)
from .dfm import DFM
from .bridge import BridgeEquations
from .bvar import BVAR
from .news import news_decomposition, NewsResult
from .evaluate import evaluate, evaluation_summary
from . import plots

__version__ = "1.0.0"
__all__ = [
    "NowcastData", "build_data", "load_data", "make_example_dataset",
    "quarter_of", "DFM", "BridgeEquations", "BVAR",
    "news_decomposition", "NewsResult", "evaluate", "evaluation_summary",
    "plots",
]
