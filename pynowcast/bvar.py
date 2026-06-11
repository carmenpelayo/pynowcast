"""
Bayesian VAR nowcasting model.

A quarterly Bayesian VAR with a Minnesota (Litterman) prior implemented via
dummy observations, as in Banbura, Giannone & Reichlin (2010). This is a
*simplified*, transparent alternative to the mixed-frequency BVAR of
Cimadomo, Giannone, Lenza, Monti & Sokol (2022) used by the original
toolbox: monthly indicators are aggregated to quarterly frequency (with the
current quarter completed by univariate AR forecasts, as in the bridge
equations) and a standard quarterly BVAR produces the nowcast.

Use it as a complementary benchmark to the DFM and the bridge equations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NowcastData, quarter_of
from .bridge import _ar_fit_forecast, _to_quarterly


class BVAR:
    """Quarterly Bayesian VAR with a Minnesota prior (dummy observations).

    Parameters
    ----------
    lags : int
        Number of VAR lags (default 2).
    shrinkage : float
        Overall tightness (lambda) of the Minnesota prior; smaller = more
        shrinkage toward univariate random walks / white noise (default 0.2).
    n_indicators : int or None
        Keep only the ``n`` monthly indicators most correlated with the
        target (to keep the quarterly VAR small). ``None`` = all.
    sum_of_coefficients : float or None
        If set, adds 'sum-of-coefficients' dummy observations with this
        tightness (mu), shrinking toward unit roots / persistence.
    """

    def __init__(self, lags: int = 2, shrinkage: float = 0.2,
                 n_indicators: int = 6, sum_of_coefficients: float = None):
        self.lags = int(lags)
        self.shrinkage = float(shrinkage)
        self.n_indicators = n_indicators
        self.mu = sum_of_coefficients
        self._fitted = False

    def fit(self, data: NowcastData) -> "BVAR":
        self.data = data.copy()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ core
    def _quarterly_panel(self, data: NowcastData, end_q) -> pd.DataFrame:
        """Aggregate monthly indicators to quarterly through ``end_q``."""
        end_month = end_q.asfreq("M", how="end")
        cols = {}
        target = data.target
        y = data.quarterly[target].dropna()
        y.index = y.index.asfreq("Q")
        cols[target] = y
        for name in data.monthly.columns:
            x = data.monthly[name]
            last_obs = x.dropna().index.max()
            if last_obs is None:
                continue
            n_ahead = max((end_month - last_obs).n, 0)
            if n_ahead:
                fc = _ar_fit_forecast(x, n_ahead)
                ext = pd.period_range(last_obs + 1, end_month, freq="M")
                x = pd.concat([x.dropna(), pd.Series(fc, index=ext)])
            else:
                x = x.dropna()
            cols[name] = _to_quarterly(x)
        return pd.DataFrame(cols)

    def nowcast(self, target: str = None, period=None,
                data: NowcastData = None) -> float:
        if not self._fitted:
            raise RuntimeError("Call .fit(data) first.")
        data = data or self.data
        target = target or data.target
        if period is None:
            last = data.quarterly[target].dropna().index.max()
            period = quarter_of(last) + 1 if last is not None else quarter_of(data.index[-1])
        q_target = quarter_of(period)

        panel = self._quarterly_panel(data, q_target)

        # ---- variable selection: most correlated indicators
        if self.n_indicators is not None:
            corr = (panel.corr()[target].drop(target)
                    .abs().sort_values(ascending=False))
            keep = [target] + list(corr.index[: self.n_indicators])
            panel = panel[keep]

        # estimation sample: complete rows up to the last published target
        y_last = data.quarterly[target].dropna().index.max()
        y_last = y_last.asfreq("Q") if y_last is not None else None
        est = panel.dropna()
        if y_last is not None:
            est = est.loc[est.index <= y_last]

        Y = est.to_numpy(float)
        names = list(est.columns)
        n, p = Y.shape[1], self.lags
        T = Y.shape[0]
        if T < n * p + 12:
            raise RuntimeError("Sample too short for the chosen BVAR size; "
                               "reduce n_indicators or lags.")

        # ---- build regression Y = X B + U with Minnesota dummies
        X = np.column_stack(
            [np.ones(T - p)] +
            [Y[p - k - 1: T - k - 1] for k in range(p)]
        )
        Yreg = Y[p:]

        sig = np.array([
            np.sqrt(np.var(np.diff(Y[:, i])) + 1e-12) for i in range(n)
        ])
        lam = self.shrinkage
        # data are transformed to stationarity -> prior mean 0 (white noise)
        Yd, Xd = [], []
        for k in range(1, p + 1):
            Yd.append(np.zeros((n, n)))
            Xk = np.zeros((n, 1 + n * p))
            Xk[:, 1 + (k - 1) * n: 1 + k * n] = np.diag(sig) * k / lam
            Xd.append(Xk)
        # prior on the covariance
        Yd.append(np.diag(sig))
        Xd.append(np.zeros((n, 1 + n * p)))
        # loose prior on the constant
        eps = 1e-3
        Yd.append(np.zeros((1, n)))
        Xc = np.zeros((1, 1 + n * p))
        Xc[0, 0] = eps
        Xd.append(Xc)
        if self.mu:
            ybar = Y[:p].mean(axis=0)
            Yd.append(np.diag(ybar) / self.mu)
            Xs = np.zeros((n, 1 + n * p))
            for k in range(p):
                Xs[:, 1 + k * n: 1 + (k + 1) * n] = np.diag(ybar) / self.mu
            Xd.append(Xs)

        Ystar = np.vstack([Yreg] + Yd)
        Xstar = np.vstack([X] + Xd)
        B, *_ = np.linalg.lstsq(Xstar, Ystar, rcond=None)   # posterior mean

        # ---- iterate forecasts, conditioning on observed indicators
        # (Gaussian conditional formula, residual covariance Sigma)
        U = Yreg - X @ B
        Sigma = U.T @ U / max(len(Yreg) - X.shape[1], 1)
        j = names.index(target)
        hist = [Y[-(k + 1)] for k in range(p)]              # newest first
        last_q = est.index[-1]
        steps = (q_target - last_q).n
        fc = Y[-1]
        for s in range(1, max(steps, 1) + 1):
            q = last_q + s
            xrow = np.concatenate([[1.0]] + hist[:p])
            fc = xrow @ B
            if q in panel.index:
                avail = panel.loc[q]
                obs = [i for i, nm in enumerate(names)
                       if nm != target and np.isfinite(avail.get(nm, np.nan))]
                if obs:
                    d = np.array([avail[names[i]] - fc[i] for i in obs])
                    S_oo = Sigma[np.ix_(obs, obs)]
                    S_jo = Sigma[j, obs]
                    fc = fc.copy()
                    fc[j] += S_jo @ np.linalg.solve(
                        S_oo + 1e-10 * np.eye(len(obs)), d
                    )
                    for i in obs:                    # actuals feed the lags
                        fc[i] = avail[names[i]]
                # if the target itself is already published, use it
                if np.isfinite(avail.get(target, np.nan)):
                    fc = fc.copy()
                    fc[j] = avail[target]
            hist = [fc] + hist
        return float(fc[j])
