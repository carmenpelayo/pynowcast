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


def plot_heatmap(data, last: int = 9, by_group: bool = False, ax=None):
    """Heatmap of input-variable z-scores (red = below mean, blue = above,
    grey = not yet released), as in the original toolbox."""
    import matplotlib.pyplot as plt
    from .policy import heatmap as _heatmap

    z = _heatmap(data, last=last, by_group=by_group)
    vals = z.drop(columns="group") if "group" in z.columns else z
    if ax is None:
        _, ax = plt.subplots(figsize=(0.9 * vals.shape[1] + 3,
                                      0.32 * len(vals) + 1.2))
    masked = vals.to_numpy(float)
    im = ax.imshow(masked, cmap="RdBu", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(vals.shape[1]), vals.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(vals)), vals.index, fontsize=8)
    for (i, j), v in __import__("numpy").ndenumerate(masked):
        txt = "" if __import__("numpy").isnan(v) else f"{v:.1f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    ax.figure.colorbar(im, ax=ax, label="z-score", shrink=0.8)
    ax.set_title("Heatmap of input variables (z-scores)")
    ax.figure.tight_layout()
    return ax


def plot_range(range_table, point_history=None, ax=None):
    """Strip plot of the range of alternative models around the main
    prediction (cf. Figure 10 of the working paper)."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    alt = range_table[range_table["excluded"] != "(none)"]
    main = range_table.loc[range_table["excluded"] == "(none)", "prediction"]
    x = np.random.default_rng(0).normal(0, 0.02, len(alt))
    ax.scatter(x, alt["prediction"], alpha=0.45, s=60, label="alternative models")
    if len(main):
        ax.scatter([0], [float(main.iloc[0])], color="goldenrod", zorder=3,
                   s=110, label="main model")
    ax.set_xticks([])
    ax.set_ylabel("prediction")
    ax.set_title("Range of alternative models\n(excluding 1-2 groups of variables)")
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    return ax


def plot_contributions(contrib, mean: float, ax=None):
    """Bar chart of (approximate) contributions plus the model mean."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    items = list(contrib.items())
    labels = ["Mean"] + [k for k, _ in items] + ["Prediction"]
    vals = [mean] + [v for _, v in items]
    colors = ["grey"] + ["#2c7fb8" if v >= 0 else "#d7301f" for v in vals[1:]]
    ax.bar(range(len(vals)), vals, color=colors)
    total = float(np.nansum(vals))
    ax.axhline(0, color="black", lw=0.8)
    ax.bar([len(vals)], [total], color="black", alpha=0.85,
           label=f"prediction = {total:+.2f}")
    ax.set_xticks(range(len(vals) + 1), labels, rotation=30, ha="right")
    ax.set_title("Approximate contributions to the prediction")
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    return ax
