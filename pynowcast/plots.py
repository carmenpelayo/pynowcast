"""Plotting helpers (matplotlib)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NowcastData, quarter_of


def plot_news(news_result, ax=None, top: int = 10):
    """Waterfall-style bar chart of a news decomposition."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    contrib = news_result.by_series()
    if news_result.revision_impact:
        contrib = pd.concat(
            [pd.Series({"(data revisions)": news_result.revision_impact}), contrib]
        )
    if len(contrib) > top:
        other = contrib.iloc[top:].sum()
        contrib = pd.concat([contrib.iloc[:top], pd.Series({"(other)": other})])
    colors = ["tab:green" if v >= 0 else "tab:red" for v in contrib]
    ax.barh(contrib.index[::-1], contrib.values[::-1], color=colors[::-1])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(
        f"Why did the {news_result.target} nowcast for {news_result.period} "
        f"move {news_result.total_change:+.2f}?"
    )
    ax.set_xlabel("impact on nowcast")
    ax.figure.tight_layout()
    return ax


def plot_nowcast_evolution(model, data: NowcastData, period, target: str = None,
                           freq: str = "W", ax=None):
    """Track how the nowcast for a given quarter evolves as data accumulate.

    Re-runs the (already estimated) model on successively larger vintages.
    """
    import matplotlib.pyplot as plt

    target = target or data.target
    q = quarter_of(period)
    start = q.asfreq("M", how="start") - 3
    end = min(q.asfreq("M", how="end") + 2, data.index[-1])
    months = pd.period_range(start, end, freq="M")
    vals = []
    for m in months:
        vint = data.vintage(m)
        vint.quarterly.loc[vint.quarterly.index >= q.asfreq("M", "start"),
                           target] = np.nan
        try:
            vals.append(model.nowcast(target, q, data=vint))
        except Exception:
            vals.append(np.nan)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months.astype(str), vals, marker="o")
    actual = data.quarterly[target].dropna()
    actual.index = actual.index.asfreq("Q")
    if q in actual.index:
        ax.axhline(actual[q], color="tab:red", ls="--", label="realized")
        ax.legend()
    ax.set_title(f"Evolution of the {target} nowcast for {q}")
    ax.set_ylabel(target)
    ax.tick_params(axis="x", rotation=45)
    ax.figure.tight_layout()
    return ax


def plot_factors(model, data: NowcastData = None, ax=None):
    """Plot the smoothed factors."""
    import matplotlib.pyplot as plt

    f = model.extract_factors(data)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    for c in f.columns:
        ax.plot(f.index.to_timestamp(), f[c], label=c)
    ax.legend()
    ax.set_title("Smoothed factors")
    ax.figure.tight_layout()
    return ax
