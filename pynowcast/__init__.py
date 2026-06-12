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
>>> res = pn.evaluate(lambda: pn.DFM(2, 2), data, start="2022Q1",
...                   horizons=("backcast", "nowcast", "forecast"))
>>> pn.evaluation_summary(res)                   # RMSE + FDA

Model building (the toolbox's three-step approach)
--------------------------------------------------
>>> pn.preselect(data)                           # 1. variable pre-selection
>>> pn.random_search(data, "DFM", n_iter=100)    # 2. model selection
>>> pn.custom_search(data, specs,
...                  covid_methods=(0, 1, 2, 3)) # 3. Covid robustness

Policy outputs
--------------
>>> pn.confidence_bands(...)                     # Reifschneider-Tulip bands
>>> pn.contributions(model, data)                # approximate contributions
>>> pn.heatmap(data)                             # z-score heatmap
>>> pn.alternative_range(...)                    # range of alternative models
>>> pn.share_of_available_data(data, "2026Q2")
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
from .evaluate import evaluate, evaluation_summary, DEFAULT_SUBPERIODS
from .corrections import covid_correct, outlier_correct, detect_outliers
from .preselect import preselect, apply_selection, sis_rank, tstat_rank, lars_rank
from .model_search import random_search, custom_search
from .policy import (
    confidence_bands,
    prediction_with_bands,
    contributions,
    share_of_available_data,
    alternative_range,
    heatmap,
)
from . import plots

__version__ = "1.2.0"
__all__ = [
    # data
    "NowcastData", "build_data", "load_data", "make_example_dataset",
    "quarter_of",
    # models
    "DFM", "BridgeEquations", "BVAR",
    # news & evaluation
    "news_decomposition", "NewsResult", "evaluate", "evaluation_summary",
    "DEFAULT_SUBPERIODS",
    # corrections (Covid / outliers)
    "covid_correct", "outlier_correct", "detect_outliers",
    # variable pre-selection
    "preselect", "apply_selection", "sis_rank", "tstat_rank", "lars_rank",
    # model selection
    "random_search", "custom_search",
    # policy outputs
    "confidence_bands", "prediction_with_bands", "contributions",
    "share_of_available_data", "alternative_range", "heatmap",
    "plots",
]
