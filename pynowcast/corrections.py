"""
Covid-19 and outlier corrections, mirroring the ``do_Covid`` switch of the
original toolbox (Linzenich & Meunier, 2024, ECB WP 3004) and its outlier
rule.

Methods (same numbering as the MATLAB toolbox):

    0 : do nothing
    1 : add dummies (one for June 2020 and one for September 2020)
    2 : set observations to NaN from February 2020 to September 2020
        (deletion as in Schorfheide & Song, 2021)
    3 : outlier correction - outliers replaced by NaN
        (detected as exceeding the median +/- k times the inter-quintile
        distance, k = 4 by default, cf. Rousseeuw & Croux, 1993)
    4 : add dummies (one for March 2020 and one for June 2020)

Setting observations to NaN is "free" for the DFM since the EM algorithm of
Banbura & Modugno (2014) handles arbitrary missing-data patterns. Dummies
are appended as additional monthly series (as in Figure 3a of the paper);
for bridge equations they should additionally be passed to
``BridgeEquations(dummies=...)`` so that they enter each equation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NowcastData

COVID_NAN_START = pd.Period("2020-02", "M")
COVID_NAN_END = pd.Period("2020-09", "M")

#: dummy dates per method: {method: [month of dummy, ...]}
COVID_DUMMIES = {
    1: ["2020-06", "2020-09"],   # 2020 Q2 and 2020 Q3
    4: ["2020-03", "2020-06"],   # 2020 Q1 and 2020 Q2
}


# ----------------------------------------------------------------------------
# outlier detection / correction
# ----------------------------------------------------------------------------

def detect_outliers(x: pd.Series, k: float = 4.0,
                    exclude: pd.PeriodIndex = None) -> pd.Series:
    """Boolean mask of outliers in ``x``.

    An observation is an outlier if it exceeds the median +/- ``k`` times
    the inter-quintile distance (distance between the 20th and 80th
    percentiles), the rule used by the original toolbox.

    Parameters
    ----------
    exclude : optional index of observations to ignore when computing the
        median and quantiles (but still flagged if outlying).
    """
    obs = x.dropna()
    ref = obs.drop(index=[i for i in (exclude or []) if i in obs.index]) \
        if exclude is not None else obs
    if len(ref) < 20:
        return pd.Series(False, index=x.index)
    med = ref.median()
    iqd = ref.quantile(0.8) - ref.quantile(0.2)
    if iqd <= 0:
        return pd.Series(False, index=x.index)
    mask = (x - med).abs() > k * iqd
    return mask.fillna(False)


def outlier_correct(x: pd.Series, k: float = 4.0) -> pd.Series:
    """Replace outliers (see :func:`detect_outliers`) by NaN."""
    out = x.copy()
    out[detect_outliers(x, k)] = np.nan
    return out


# ----------------------------------------------------------------------------
# Covid correction
# ----------------------------------------------------------------------------

def covid_correct(data: NowcastData, method: int = 0, k: float = 4.0,
                  correct_target: bool = True) -> NowcastData:
    """Return a copy of ``data`` with the chosen Covid correction applied.

    Parameters
    ----------
    method : int
        0-4, same numbering as ``do_Covid`` in the original toolbox (see
        module docstring).
    k : float
        Threshold (in inter-quintile distances) for method 3.
    correct_target : bool
        Whether the correction also applies to the quarterly target
        (methods 2 and 3). The original deletes/corrects the full input
        matrix including the target.
    """
    if method == 0:
        return data.copy()
    if method not in (1, 2, 3, 4):
        raise ValueError("method must be one of 0, 1, 2, 3, 4")

    out = data.copy()

    if method in (1, 4):                            # ---- dummy variables
        for d in COVID_DUMMIES[method]:
            p = pd.Period(d, "M")
            name = f"dummy_{p.year}q{p.quarter}"
            col = pd.Series(0.0, index=out.monthly.index)
            if p in col.index:
                col.loc[p] = 1.0
            out.monthly[name] = col
            # dummies belong to the global block, are never transformed
            # and have no publication lag
            if out.blocks is not None:
                row = {c: 0 for c in out.blocks.columns}
                row[out.blocks.columns[0]] = 1      # "Global"
                out.blocks.loc[name] = row
            out.pub_lag[name] = 0
            out.transforms[name] = "none"
            if getattr(out, "groups", None) is not None:
                out.groups[name] = "Dummies"

    elif method == 2:                               # ---- delete Feb-Sep 2020
        win = (out.monthly.index >= COVID_NAN_START) & \
              (out.monthly.index <= COVID_NAN_END)
        out.monthly.loc[win, :] = np.nan
        if correct_target:
            winq = (out.quarterly.index >= COVID_NAN_START) & \
                   (out.quarterly.index <= COVID_NAN_END)
            out.quarterly.loc[winq, :] = np.nan

    elif method == 3:                               # ---- outlier correction
        for c in out.monthly.columns:
            out.monthly[c] = outlier_correct(out.monthly[c], k)
        if correct_target:
            for c in out.quarterly.columns:
                out.quarterly[c] = outlier_correct(out.quarterly[c], k)

    return out
