"""
Pseudo-real-time (out-of-sample) evaluation.

For every quarter in the evaluation window and for a set of 'nowcast dates'
within (or around) the quarter, the dataset is cut to what would have been
available on that date (respecting each series' publication lag), the model
is re-estimated, and a nowcast is produced. Results are compared to the
realized values and to a naive AR(1) benchmark.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import NowcastData, quarter_of


def _ar1_benchmark(y_q: pd.Series, q) -> float:
    """AR(1) forecast of the target for quarter ``q`` (data up to q-1)."""
    hist = y_q[y_q.index < q].dropna().to_numpy(float)
    if len(hist) < 8:
        return float(np.nan)
    x, y = hist[:-1], hist[1:]
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0] + beta[1] * hist[-1])


def evaluate(
    model_factory,
    data: NowcastData,
    start,
    end=None,
    months_in_quarter=(1, 2, 3),
    target: str = None,
    refit: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Pseudo-real-time evaluation of a nowcasting model.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning a *fresh, unfitted* model, e.g.
        ``lambda: DFM(factors=2, lags=2)``.
    data : NowcastData
        Full dataset, including the realized target values.
    start, end : quarter labels, e.g. ``"2022Q1"``
        Evaluation window (``end`` defaults to the last quarter with an
        observed target).
    months_in_quarter : tuple of int
        When within the quarter the nowcast is made: 1, 2 and/or 3 = end of
        the first/second/third month of the target quarter.
    refit : bool
        Re-estimate the model at every vintage (slower, more realistic).
        If False, the model is estimated once on the first vintage.

    Returns
    -------
    DataFrame with one row per (quarter, nowcast month): the nowcast, the
    realized value, the AR(1) benchmark, and errors. Use
    :func:`evaluation_summary` on it.
    """
    target = target or data.target
    y_q = data.quarterly[target].dropna()
    y_q.index = y_q.index.asfreq("Q")
    start_q = quarter_of(start)
    end_q = quarter_of(end) if end is not None else y_q.index.max()

    quarters = [q for q in y_q.index if start_q <= q <= end_q]
    if not quarters:
        raise ValueError("No realized target values in the evaluation window.")

    rows = []
    model = None
    for q in quarters:
        actual = float(y_q[q])
        bench = _ar1_benchmark(y_q, q)
        for mth in months_in_quarter:
            as_of = q.asfreq("M", how="start") + (mth - 1)
            vint = data.vintage(as_of)
            # remove the target for q and beyond, in case pub lags are short
            vint.quarterly.loc[vint.quarterly.index >= q.asfreq("M", "start"),
                               target] = np.nan
            try:
                if model is None or refit:
                    model = model_factory()
                    model.fit(vint)
                    fitted_on_vintage = True
                else:
                    fitted_on_vintage = False
                nc = model.nowcast(target, q, data=vint)
            except Exception as exc:  # keep the loop alive
                warnings.warn(f"{q} M{mth}: {exc}")
                nc = np.nan
            rows.append({
                "quarter": str(q), "month_in_quarter": mth,
                "as_of": str(as_of), "nowcast": nc, "actual": actual,
                "ar1": bench, "error": nc - actual,
                "ar1_error": bench - actual,
                "refit": refit or fitted_on_vintage,
            })
            if verbose:
                print(f"  {q} M{mth}: nowcast={nc: .3f}  actual={actual: .3f}")
    return pd.DataFrame(rows)


def evaluation_summary(results: pd.DataFrame) -> pd.DataFrame:
    """RMSE by nowcast month, with the AR(1) benchmark and relative RMSE."""
    def rmse(x):
        x = x.dropna()
        return float(np.sqrt((x ** 2).mean())) if len(x) else np.nan

    g = results.groupby("month_in_quarter")
    out = pd.DataFrame({
        "rmse_model": g["error"].apply(rmse),
        "rmse_ar1": g["ar1_error"].apply(rmse),
        "n": g["error"].apply(lambda s: int(s.notna().sum())),
    })
    out["relative_rmse"] = out["rmse_model"] / out["rmse_ar1"]
    return out
