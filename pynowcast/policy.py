"""
Outputs for policy analysis (section 3 of Linzenich & Meunier, 2024).

This module provides the operational outputs of the original toolbox that
go beyond the point forecast:

- :func:`confidence_bands` -- uncertainty bands a la Reifschneider & Tulip
  (2019): rolling Mean Absolute Error of past pseudo-real-time predictions,
  adjusted for outliers as in ECB (2009), computed per horizon and month of
  the quarter. The +/- 1 MAE band corresponds to the 57.5% confidence
  interval under normality. FDA over the same window quantifies directional
  uncertainty. This is the ``do_mae = 1`` mode; the resulting table can be
  stored and reused (``do_mae = 0``).
- :func:`contributions` -- approximate contributions of input variables to
  the point forecast, proxied by the news from all data releases over the
  ``lookback`` months before the target date (2 years in the original).
  The "Mean" is the model's prediction with no recent data.
- :func:`share_of_available_data` -- share of the target quarter's monthly
  observations already released, a simple gauge of how much the prediction
  rests on actual versus extrapolated data.
- :func:`alternative_range` -- predictions of alternative models obtained
  by disconnecting one or two groups of variables from the information set
  (the ``do_range = 1`` option).
- :func:`heatmap` -- z-scores of input variables (smoothed over 5 months
  for monthly series, per the Mariano-Murasawa approximation), by variable
  and by group.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from .data import NowcastData, _to_month, quarter_of
from .evaluate import evaluate
from .news import news_decomposition

SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))   # MAE = sqrt(2/pi) * sigma


# ----------------------------------------------------------------------------
# confidence bands (Reifschneider & Tulip, 2019; ECB, 2009)
# ----------------------------------------------------------------------------

def _outlier_adjust_errors(e: np.ndarray, k: float = 4.0) -> np.ndarray:
    """Drop past errors that are outliers (median +/- k inter-quintile
    distances), so the bands represent uncertainty in 'normal times' as in
    ECB (2009)."""
    e = e[np.isfinite(e)]
    if len(e) < 12:
        return e
    med = np.median(e)
    iqd = np.quantile(e, 0.8) - np.quantile(e, 0.2)
    if iqd <= 0:
        return e
    return e[np.abs(e - med) <= k * iqd]


def confidence_bands(model_factory, data: NowcastData, years: int = 10,
                     end=None, months_in_quarter=(1, 2, 3),
                     horizons=("backcast", "nowcast", "forecast"),
                     results: pd.DataFrame = None,
                     verbose: bool = False) -> pd.DataFrame:
    """MAE / FDA per horizon x month of quarter from past pseudo-real-time
    predictions over the last ``years`` years.

    The band around a point forecast is ``point +/- mae`` (57.5% CI). Pass
    a pre-computed ``results`` frame (from :func:`pynowcast.evaluate`) to
    skip the costly evaluation loop. Returns a DataFrame indexed by
    (horizon, month_in_quarter) with columns ``mae``, ``fda``, ``n`` and
    ``sigma`` (= mae / sqrt(2/pi)).
    """
    if results is None:
        y_q = data.quarterly[data.target].dropna()
        last = quarter_of(y_q.index.max()) if end is None else quarter_of(end)
        start = last - 4 * years + 1
        results = evaluate(model_factory, data, start=start, end=last,
                           months_in_quarter=months_in_quarter,
                           horizons=horizons, verbose=verbose)
    rows = []
    for (hz, mth), g in results.groupby(["horizon", "month_in_quarter"]):
        e = _outlier_adjust_errors(g["error"].to_numpy(float))
        rows.append({
            "horizon": hz, "month_in_quarter": mth,
            "mae": float(np.mean(np.abs(e))) if len(e) else np.nan,
            "fda": float(g["hit"].mean()),
            "n": int(np.isfinite(g["error"]).sum()),
        })
    out = pd.DataFrame(rows).set_index(["horizon", "month_in_quarter"])
    out["sigma"] = out["mae"] / SQRT_2_OVER_PI
    return out


def prediction_with_bands(point: float, mae_table: pd.DataFrame,
                          horizon: str, month_in_quarter: int) -> dict:
    """Attach the +/- 1 MAE band (57.5% CI) to a point forecast."""
    mae = float(mae_table.loc[(horizon, month_in_quarter), "mae"])
    return {"point": point, "lower": point - mae, "upper": point + mae,
            "mae": mae, "coverage": 0.575}


# ----------------------------------------------------------------------------
# approximate contributions
# ----------------------------------------------------------------------------

def contributions(model, data: NowcastData, target: str = None, period=None,
                  lookback: int = 24, by_group: bool = True):
    """Approximate contributions of input variables to the prediction.

    As in the original toolbox, contributions are proxied by the news from
    all data released over the ``lookback`` months before the latest data
    point: the 'old' information set removes every observation of the last
    ``lookback`` months, and the news decomposition against the current
    data attributes the difference to individual series. The base
    prediction from the emptied information set is reported as ``Mean``
    (the value the model reverts to absent recent data).

    Returns (table, mean) where ``table`` has one row per series (or per
    group) with its contribution in percentage points.
    """
    target = target or data.target
    if period is None:
        last = data.quarterly[target].dropna().index.max()
        period = quarter_of(last) + 1 if last is not None else quarter_of(data.index[-1])
    cutoff = data.index.max() - int(lookback)

    old = data.copy()
    old.monthly.loc[old.monthly.index > cutoff, :] = np.nan
    old.quarterly.loc[old.quarterly.index > cutoff, :] = np.nan

    news = news_decomposition(model, old, data, target=target, period=period)
    tab = news.table.copy()
    # B-BVAR blocked names like "x (M2)" map back to the base series "x"
    base = tab["series"].str.replace(r" \(M[123]\)$", "", regex=True)
    tab["group"] = data.groups.reindex(base).to_numpy()
    if by_group:
        out = (tab.groupby("group")["impact"].sum()
               .sort_values(key=np.abs, ascending=False))
    else:
        tab["series"] = base
        out = (tab.groupby("series")["impact"].sum()
               .sort_values(key=np.abs, ascending=False))
    return out, float(news.nowcast_old)


# ----------------------------------------------------------------------------
# share of available data
# ----------------------------------------------------------------------------

def share_of_available_data(data: NowcastData, period, as_of=None) -> float:
    """Share of the target quarter's monthly observations already released.

    A low share means the prediction relies mostly on model-extrapolated
    data; a high share, on actual releases.
    """
    q = quarter_of(period)
    months = pd.period_range(q.asfreq("M", "start"), q.asfreq("M", "end"),
                             freq="M")
    panel = data.monthly if as_of is None else data.vintage(as_of).monthly
    months = [m for m in months if m in panel.index]
    if not months:
        return 0.0
    block = panel.loc[months]
    return float(block.notna().to_numpy().mean())


# ----------------------------------------------------------------------------
# range of alternative models
# ----------------------------------------------------------------------------

def alternative_range(model_factory, data: NowcastData, target: str = None,
                      period=None, max_drop: int = 2,
                      verbose: bool = False) -> pd.DataFrame:
    """Predictions of alternative models obtained by removing one or two
    groups of variables from the information set (``do_range = 1``).

    Returns a DataFrame with one row per alternative model: the group(s)
    excluded and the resulting prediction. The first row ("(none)") is the
    main model. Useful to assess what the prediction would be if some
    variables -- e.g. surveys sending an erroneous signal -- were ignored.
    """
    from itertools import combinations
    target = target or data.target
    groups = [g for g in data.groups.unique() if g not in ("Target",)]
    drops = [()] + [(g,) for g in groups]
    if max_drop >= 2 and len(groups) >= 2:
        drops += list(combinations(groups, 2))

    rows = []
    for drop in drops:
        d = _drop_groups(data, drop)
        if d.monthly.shape[1] == 0:
            continue
        try:
            model = model_factory()
            model.fit(d)
            pred = model.nowcast(target, period, data=d)
        except Exception as exc:
            warnings.warn(f"range {drop}: {exc}")
            pred = np.nan
        rows.append({"excluded": " + ".join(drop) if drop else "(none)",
                     "n_excluded": len(drop), "prediction": pred})
        if verbose:
            print(f"  excl. {drop or '(none)'} -> {pred}")
    return pd.DataFrame(rows)


def _drop_groups(data: NowcastData, drop) -> NowcastData:
    keep = [n for n in data.series_names
            if data.groups[n] not in drop or n == data.target]
    m_keep = [c for c in data.monthly.columns if c in keep]
    q_keep = [c for c in data.quarterly.columns if c in keep]
    out = data.copy()
    return replace(
        out,
        monthly=out.monthly[m_keep],
        quarterly=out.quarterly[q_keep],
        blocks=out.blocks.loc[m_keep + q_keep],
        pub_lag=out.pub_lag.loc[m_keep + q_keep],
        transforms=out.transforms.loc[m_keep + q_keep],
        groups=out.groups.loc[m_keep + q_keep],
    )


# ----------------------------------------------------------------------------
# heatmap
# ----------------------------------------------------------------------------

def heatmap(data: NowcastData, last: int = 9, smooth: int = 5,
            by_group: bool = False) -> pd.DataFrame:
    """Z-scores of input variables over the last ``last`` months.

    Monthly series are smoothed with a ``smooth``-month moving average (5
    months, per the Mariano-Murasawa approximation of quarterly growth
    from monthly growth) before computing the distance from the long-term
    mean in standard deviations. NaN = not yet released. With
    ``by_group=True`` the unweighted average z-score per group is returned.
    """
    z_all = {}
    for name in data.monthly.columns:
        x = data.monthly[name]
        sm = x.rolling(smooth, min_periods=max(smooth - 2, 1)).mean()
        mu, sd = sm.mean(), sm.std()
        z_all[name] = (sm - mu) / (sd if sd > 0 else 1.0)
        # months with no actual release stay NaN
        z_all[name][x.isna()] = np.nan
    for name in data.quarterly.columns:
        x = data.quarterly[name]
        mu, sd = x.mean(), x.std()
        z_all[name] = (x - mu) / (sd if sd > 0 else 1.0)
    z = pd.DataFrame(z_all).iloc[-int(last):].T
    z.columns = [str(c) for c in z.columns]
    z.insert(0, "group", data.groups.reindex(z.index).to_numpy())
    z["_ord"] = (z["group"] == "Target").astype(int)   # target row last
    z = z.sort_values(["_ord", "group"]).drop(columns="_ord")
    if by_group:
        return z.groupby("group").mean(numeric_only=True)
    return z
