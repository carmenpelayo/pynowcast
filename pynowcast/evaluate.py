"""
Pseudo-real-time (out-of-sample) evaluation.

For every quarter in the evaluation window and for each month within the
quarter, the dataset is cut to what would have been available at that date
(respecting each series' publication lag), the model is re-estimated, and
predictions are produced for up to three horizons, as in the original
toolbox:

- **backcast**  : previous quarter (when its target value was not yet
  published at the prediction date),
- **nowcast**   : current quarter,
- **forecast**  : next quarter.

Both accuracy metrics of the original toolbox are computed:

- **RMSE** -- root mean squared error of the point predictions,
- **FDA**  -- Forecast Directional Accuracy, the share of observations for
  which sign(y_t - y_{t-1}) = sign(yhat_t - y_{t-1}).

Summaries can be broken down by month of the quarter and by sub-period
(e.g. pre-Covid / Covid / post-Covid), again mirroring the original.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import NowcastData, quarter_of

HORIZONS = ("backcast", "nowcast", "forecast")

#: default sub-periods used by the original toolbox
DEFAULT_SUBPERIODS = {
    "pre-Covid": (None, "2019Q4"),
    "Covid": ("2020Q1", "2021Q4"),
    "post-Covid": ("2022Q1", None),
}


def _ar_benchmark(y_q: pd.Series, q) -> float:
    """AR(1) forecast of the target for quarter ``q``, iterated from the
    last observation strictly before ``q``."""
    hist = y_q[y_q.index < q].dropna()
    if len(hist) < 8:
        return float(np.nan)
    z = hist.to_numpy(float)
    x, y = z[:-1], z[1:]
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    steps = (q - hist.index.max()).n
    f = z[-1]
    for _ in range(max(steps, 1)):
        f = float(beta[0] + beta[1] * f)
    return f


def _hit(pred: float, actual: float, prev: float):
    """Directional hit: 1 if the predicted direction (vs the previous
    quarter) matches the realized one, 0 otherwise, NaN if not computable."""
    if not (np.isfinite(pred) and np.isfinite(actual) and np.isfinite(prev)):
        return np.nan
    return float((actual - prev) * (pred - prev) > 0)


def evaluate(
    model_factory,
    data: NowcastData,
    start,
    end=None,
    months_in_quarter=(1, 2, 3),
    horizons=("nowcast",),
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
        Evaluation window for the *reference* (nowcast) quarter; ``end``
        defaults to the last quarter with an observed target.
    months_in_quarter : tuple of int
        When within the quarter predictions are made: 1, 2 and/or 3 = end
        of the first/second/third month of the reference quarter.
    horizons : tuple of str
        Any of ``"backcast"`` (previous quarter, only evaluated when its
        value was not yet published at the prediction date), ``"nowcast"``
        (current quarter), ``"forecast"`` (next quarter).
    refit : bool
        Re-estimate the model at every vintage (slower, more realistic).

    Returns
    -------
    DataFrame with one row per (quarter, month, horizon): prediction,
    realized value, AR(1) benchmark, errors, and directional hits. Use
    :func:`evaluation_summary` on it.
    """
    target = target or data.target
    horizons = tuple(h for h in HORIZONS if h in horizons)  # canonical order
    y_q = data.quarterly[target].dropna()
    y_q.index = y_q.index.asfreq("Q")
    start_q = quarter_of(start)
    end_q = quarter_of(end) if end is not None else y_q.index.max()

    quarters = [q for q in y_q.index if start_q <= q <= end_q]
    if not quarters:
        raise ValueError("No realized target values in the evaluation window.")

    rows = []
    model = None
    for q in quarters:                      # q = reference (nowcast) quarter
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
            except Exception as exc:
                warnings.warn(f"{q} M{mth}: fit failed: {exc}")
                model = None
                continue

            vint_y = vint.quarterly[target].dropna()
            vint_y.index = vint_y.index.asfreq("Q")
            vint_q_avail = set(vint_y.index) if len(vint_y) else set()
            for hz in horizons:
                qq = {"backcast": q - 1, "nowcast": q, "forecast": q + 1}[hz]
                if hz == "backcast" and qq in vint_q_avail:
                    continue                # already published: not a backcast
                if qq not in y_q.index:
                    continue                # realized value not yet known
                actual = float(y_q[qq])
                prev = float(y_q[qq - 1]) if (qq - 1) in y_q.index else np.nan
                bench = _ar_benchmark(vint_y.copy() if len(vint_y) else y_q[y_q.index < qq], qq)
                try:
                    pred = model.nowcast(target, qq, data=vint)
                except Exception as exc:
                    warnings.warn(f"{q} M{mth} {hz}: {exc}")
                    pred = np.nan
                rows.append({
                    "quarter": str(qq), "ref_quarter": str(q), "horizon": hz,
                    "month_in_quarter": mth, "as_of": str(as_of),
                    "prediction": pred, "actual": actual,
                    "ar1": bench, "error": pred - actual,
                    "ar1_error": bench - actual,
                    "hit": _hit(pred, actual, prev),
                    "ar1_hit": _hit(bench, actual, prev),
                })
                if verbose:
                    print(f"  {q} M{mth} {hz:<8s}: pred={pred: .3f} "
                          f"actual={actual: .3f}")
    out = pd.DataFrame(rows)
    # backward compatibility: a 'nowcast' column as in the previous API
    if not out.empty:
        out["nowcast"] = out["prediction"]
    return out


def _rmse(x: pd.Series) -> float:
    x = x.dropna()
    return float(np.sqrt((x ** 2).mean())) if len(x) else np.nan


def evaluation_summary(results: pd.DataFrame,
                       by=("horizon", "month_in_quarter"),
                       subperiods: dict = None) -> pd.DataFrame:
    """RMSE and FDA of the model and the AR(1) benchmark.

    Parameters
    ----------
    by : sequence of str
        Grouping columns; default horizon x month of quarter. Groups with
        a single horizon collapse naturally.
    subperiods : dict, optional
        ``{label: (start_quarter, end_quarter)}`` with ``None`` for open
        ends; pass :data:`DEFAULT_SUBPERIODS` for the original toolbox's
        pre-Covid / Covid / post-Covid split. Adds a "subperiod" level.
    """
    res = results.copy()
    if "horizon" not in res.columns:
        res["horizon"] = "nowcast"
    if "prediction" not in res.columns and "nowcast" in res.columns:
        res["prediction"] = res["nowcast"]
    by = [b for b in by if b in res.columns]
    if res["horizon"].nunique() == 1 and "horizon" in by and len(by) > 1:
        by = [b for b in by if b != "horizon"]

    if subperiods:
        frames = []
        for label, (s, e) in subperiods.items():
            qs = pd.PeriodIndex(res["quarter"].astype(str), freq="Q")
            m = pd.Series(True, index=res.index)
            if s is not None:
                m &= pd.Series(qs >= quarter_of(s), index=res.index)
            if e is not None:
                m &= pd.Series(qs <= quarter_of(e), index=res.index)
            sub = res[m]
            if sub.empty:
                continue
            t = evaluation_summary(sub, by=by).reset_index()
            t.insert(0, "subperiod", label)
            frames.append(t)
        return (pd.concat(frames, ignore_index=True)
                .set_index(["subperiod"] + by)) if frames else pd.DataFrame()

    g = res.groupby(by)
    out = pd.DataFrame({
        "rmse_model": g["error"].apply(_rmse),
        "rmse_ar1": g["ar1_error"].apply(_rmse),
        "fda_model": g["hit"].mean(),
        "fda_ar1": g["ar1_hit"].mean(),
        "n": g["error"].apply(lambda s: int(s.notna().sum())),
    })
    out["relative_rmse"] = out["rmse_model"] / out["rmse_ar1"]
    return out
