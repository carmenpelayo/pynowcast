"""
Combination of bridge equations, in the spirit of

    Banbura, M., Belousova, I., Bodnar, K. and Toth, M. B. (2023),
    "Nowcasting employment in the euro area", ECB WP 2815.

For each monthly indicator x_i:

1. the monthly (transformed) series is completed through the end of the
   target quarter with AR(p) forecasts (p chosen by BIC),
2. it is aggregated to quarterly frequency (3-month average),
3. a 'bridge' regression links target growth to the aggregated indicator:

       y_q = c + b0 x_q [+ b1 x_{q-1}] [+ g y_{q-1}] + e_q

4. the individual nowcasts are combined (equal weights by default, or
   weights inversely proportional to in-sample MSE).

This is intentionally simple and transparent; it is a strong benchmark for
the DFM and often hard to beat in practice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NowcastData, _to_month, quarter_of


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def _ar_fit_forecast(x: pd.Series, n_ahead: int, max_p: int = 6):
    """Fit AR(p) by OLS with BIC selection, forecast ``n_ahead`` steps."""
    x = x.dropna()
    z = x.to_numpy(float)
    T = len(z)
    if T < 12:
        return np.full(n_ahead, z.mean() if T else 0.0)
    best = (np.inf, None, None)
    for p in range(1, min(max_p, T // 5) + 1):
        Y = z[p:]
        X = np.column_stack([np.ones(T - p)] +
                            [z[p - k - 1: T - k - 1] for k in range(p)])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        sig2 = max(res @ res / len(Y), 1e-12)
        bic = len(Y) * np.log(sig2) + (p + 1) * np.log(len(Y))
        if bic < best[0]:
            best = (bic, p, beta)
    _, p, beta = best
    hist = list(z)
    out = []
    for _ in range(n_ahead):
        xrow = np.array([1.0] + [hist[-k - 1] for k in range(p)])
        f = float(xrow @ beta)
        hist.append(f)
        out.append(f)
    return np.array(out)


def _to_quarterly(monthly: pd.Series) -> pd.Series:
    """3-month average of a monthly (transformed) series, by quarter."""
    g = monthly.groupby(monthly.index.asfreq("Q"))
    out = g.mean()
    out = out[g.count() == 3]          # only complete quarters
    return out


# ----------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------

class BridgeEquations:
    """Combination of bridge equations.

    Parameters
    ----------
    indicator_lags : int
        Lags of the quarterly-aggregated indicator in each bridge equation
        (0 = contemporaneous only).
    target_lags : int
        Autoregressive lags of the target included in each equation.
    weighting : 'equal' or 'mse'
        How individual equations are combined.
    """

    def __init__(self, indicator_lags: int = 0, target_lags: int = 1,
                 weighting: str = "equal"):
        self.indicator_lags = int(indicator_lags)
        self.target_lags = int(target_lags)
        if weighting not in ("equal", "mse"):
            raise ValueError("weighting must be 'equal' or 'mse'")
        self.weighting = weighting
        self._fitted = False

    # ------------------------------------------------------------------- fit
    def fit(self, data: NowcastData) -> "BridgeEquations":
        self.data = data.copy()
        self._fitted = True
        return self

    # --------------------------------------------------------------- nowcast
    def nowcast(self, target: str = None, period=None,
                data: NowcastData = None, details: bool = False):
        """Combined bridge-equation nowcast of ``target`` for ``period``.

        If intermediate quarters of the target are still unpublished (e.g.
        nowcasting Q2 in April when Q1 GDP is not out yet), they are
        backcast recursively with the same set of equations.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit(data) first.")
        data = data or self.data
        target = target or data.target
        if period is None:
            last = data.quarterly[target].dropna().index.max()
            period = quarter_of(last) + 1 if last is not None else quarter_of(data.index[-1])
        q_target = quarter_of(period)

        y_q = data.quarterly[target].dropna()
        y_q.index = y_q.index.asfreq("Q")
        last_y = y_q.index.max()

        # quarters to predict sequentially (backcasts feed into y lags)
        first = min(last_y + 1, q_target) if last_y is not None else q_target
        out = None
        for q in pd.period_range(first, q_target, freq="Q"):
            out = self._nowcast_one(data, target, q, y_q, details and q == q_target)
            val = out[0] if details and q == q_target else out
            y_q = pd.concat([y_q, pd.Series([val], index=[q])])
        return out

    def _nowcast_one(self, data, target, q_target, y_q, details):
        end_month = q_target.asfreq("M", how="end")

        preds, mses, rows = [], [], []
        for name in data.monthly.columns:
            x = data.monthly[name]
            # 1. complete the series through the end of the target quarter
            last_obs = x.dropna().index.max()
            if last_obs is None:
                continue
            n_ahead = max((end_month - last_obs).n, 0)
            if n_ahead:
                fc = _ar_fit_forecast(x, n_ahead)
                ext_idx = pd.period_range(last_obs + 1, end_month, freq="M")
                x = pd.concat([x.dropna(), pd.Series(fc, index=ext_idx)])
            else:
                x = x.dropna()
            # 2. aggregate to quarterly
            x_q = _to_quarterly(x)
            if q_target not in x_q.index:
                continue
            # 3. bridge regression on the common sample
            df = pd.DataFrame({"y": y_q})
            for l in range(self.indicator_lags + 1):
                df[f"x{l}"] = x_q.shift(l)
            for l in range(1, self.target_lags + 1):
                df[f"y{l}"] = y_q.shift(l)
            est = df.dropna()
            if len(est) < 12:
                continue
            Xmat = np.column_stack(
                [np.ones(len(est))] + [est[c].to_numpy() for c in est.columns[1:]]
            )
            beta, *_ = np.linalg.lstsq(Xmat, est["y"].to_numpy(), rcond=None)
            resid = est["y"].to_numpy() - Xmat @ beta
            mse = float(resid @ resid / max(len(est) - len(beta), 1))
            # 4. forecast the target quarter
            xrow = [1.0]
            ok = True
            for l in range(self.indicator_lags + 1):
                v = x_q.get(q_target - l, np.nan)
                xrow.append(v)
                ok &= np.isfinite(v)
            for l in range(1, self.target_lags + 1):
                v = y_q.get(q_target - l, np.nan)
                xrow.append(v)
                ok &= np.isfinite(v)
            if not ok:
                continue
            pred = float(np.array(xrow) @ beta)
            preds.append(pred)
            mses.append(mse)
            rows.append({"indicator": name, "nowcast": pred, "mse": mse})

        if not preds:
            raise RuntimeError(
                f"No bridge equation could be estimated for {q_target}."
            )
        preds = np.array(preds)
        if self.weighting == "mse":
            w = 1.0 / np.array(mses)
            w /= w.sum()
        else:
            w = np.full(len(preds), 1.0 / len(preds))
        combined = float(w @ preds)
        if details:
            tab = pd.DataFrame(rows)
            tab["weight"] = w
            return combined, tab.sort_values("weight", ascending=False)
        return combined
