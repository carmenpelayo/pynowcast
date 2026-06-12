"""
Blocking Bayesian VAR (B-BVAR) for nowcasting, following

    Cimadomo, J., Giannone, D., Lenza, M., Monti, F. and Sokol, A. (2022),
    "Nowcasting with large Bayesian vector autoregressions",
    Journal of Econometrics 231(2),

as implemented in the ECB nowcasting toolbox (Linzenich & Meunier, 2024).

The model in three pieces, all mirroring the original:

**Blocking.** Each monthly variable enters the (quarterly) VAR as *three*
separate quarterly series -- one per month of the quarter:
``x (M1), x (M2), x (M3)``. Stacked with the genuinely quarterly variables
this turns the mixed-frequency problem into a single-frequency VAR of
dimension ``3 x n_monthly + n_quarterly``.

**Prior.** Normal-Inverse-Wishart conjugate prior combining a Minnesota
(Litterman) prior and the sum-of-coefficients prior, implemented with
dummy observations as in Banbura, Giannone & Reichlin (2010). Since the
input data are transformed to stationarity, the Minnesota prior mean is
zero (white noise); the overall tightness is ``shrinkage`` (lambda) and the
sum-of-coefficients tightness is ``mu``.

**Ragged edge.** The estimated VAR is cast in state-space form and the
Kalman filter/smoother conditions on whatever entries of the blocked
vector are observed -- e.g. in the second month of the quarter, M1 values
of timely indicators are observed while M2/M3 are not. Missing in-sample
values are handled with an EM-type iteration (fill with smoothed
expectations, re-estimate, repeat until convergence -- the
``bvar_thresh`` / ``bvar_max_iter`` settings of the original toolbox), and
the nowcast is the smoothed expectation of the target in the target
quarter. This also makes the exact news decomposition of Banbura & Modugno
(2014) available for the B-BVAR via ``pynowcast.news_decomposition``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from .data import NowcastData, _to_month, quarter_of
from .kalman import kalman_filter_smoother

KAPPA = 1e-4          # tiny observation noise (numerical regularization)


class BVAR:
    """Blocking Bayesian VAR (Cimadomo et al., 2022).

    Parameters
    ----------
    lags : int
        Number of *quarterly* VAR lags (default 2; the original toolbox's
        example setting is 5 -- heavier but feasible thanks to shrinkage).
    shrinkage : float
        Overall tightness (lambda) of the Minnesota prior (default 0.2).
    sum_of_coefficients : float, optional
        Tightness (mu) of the sum-of-coefficients prior; ``None`` disables
        it (with stationarity-transformed data its effect is mild).
    n_indicators : int, optional
        Keep only the ``n`` monthly indicators most correlated with the
        target. ``None`` (default) = all variables, the spirit of a *large*
        BVAR; set a small number to speed up loops.
    max_iter, thresh : EM iterations for in-sample missing values
        (``bvar_max_iter`` / ``bvar_thresh`` in the original).
    """

    def __init__(self, lags: int = 2, shrinkage: float = 0.2,
                 sum_of_coefficients: float = None, n_indicators: int = None,
                 max_iter: int = 10, thresh: float = 1e-3,
                 verbose: bool = False):
        self.lags = int(lags)
        self.shrinkage = float(shrinkage)
        self.mu = sum_of_coefficients
        self.n_indicators = n_indicators
        self.max_iter = int(max_iter)
        self.thresh = float(thresh)
        self.verbose = verbose
        self._fitted = False

    # ------------------------------------------------------------------- fit
    def fit(self, data: NowcastData) -> "BVAR":
        self.data = data.copy()
        self.target_ = data.target

        # ---- variable selection (optional)
        self.m_names_ = list(data.monthly.columns)
        if self.n_indicators is not None:
            self.m_names_ = self._select_indicators(data)
        self.q_names_ = list(data.quarterly.columns)
        self.series_names_ = (
            [f"{m} (M{k})" for m in self.m_names_ for k in (1, 2, 3)]
            + self.q_names_
        )

        # ---- blocked quarterly panel
        Y, idx = self._blocked_panel(data)
        self.mean_ = np.nanmean(Y, axis=0)
        std = np.nanstd(Y, axis=0, ddof=1)
        std[~np.isfinite(std) | (std < 1e-10)] = 1.0
        self.std_ = std
        Ys = (Y - self.mean_) / self.std_

        N, p = Ys.shape[1], self.lags
        if Ys.shape[0] < N // 2 + p + 8:
            warnings.warn("Short sample for the chosen B-BVAR size; the "
                          "prior will dominate. Consider n_indicators or "
                          "fewer lags.")

        # ---- EM iterations: estimate <-> fill missing with smoothed values
        miss = np.isnan(Ys)
        Yfill = np.where(miss, 0.0, Ys)             # standardized mean = 0
        prev_fill = Yfill[miss].copy()
        for it in range(self.max_iter):
            B, Sigma = self._posterior(Yfill)
            self._build_state_space(B, Sigma)
            ko = kalman_filter_smoother(Ys, self.Z_, self.R_, self.A_,
                                        self.Q_, self.a0_, self.P0_)
            sm = ko.a_smooth[:, :N]                 # smoothed E[y_t]
            Yfill = np.where(miss, sm, Ys)
            delta = (np.max(np.abs(Yfill[miss] - prev_fill))
                     if miss.any() else 0.0)
            prev_fill = Yfill[miss].copy()
            if self.verbose:
                print(f"  B-BVAR EM iter {it + 1}: "
                      f"loglik={ko.loglik:,.1f}  d(fill)={delta:.5f}")
            if delta < self.thresh:
                break
        self.B_, self.Sigma_ = B, Sigma
        self.n_iter_ = it + 1
        self._fitted = True
        return self

    # ------------------------------------------------------------------- API
    def nowcast(self, target: str = None, period=None,
                data: NowcastData = None, with_uncertainty: bool = False):
        """Smoothed expectation of ``target`` in ``period`` conditional on
        all observed (possibly ragged) data, at the fitted parameters."""
        self._check_fitted()
        data = data or self.data
        target = target or data.target
        if period is None:
            last = data.quarterly[target].dropna().index.max()
            period = quarter_of(last) + 1 if last is not None \
                else quarter_of(data.index[-1])
        month = _to_month(period)

        d = data.extend_to(max(month, data.index[-1]))
        X, idx = self._standardize(d)
        ko = kalman_filter_smoother(X, self.Z_, self.R_, self.A_, self.Q_,
                                    self.a0_, self.P0_)
        t_star = idx.get_loc(month)
        j = self.series_names_.index(target)
        point = float(self.Z_[j] @ ko.a_smooth[t_star]
                      * self.std_[j] + self.mean_[j])
        if with_uncertainty:
            var = float(self.Z_[j] @ ko.P_smooth[t_star] @ self.Z_[j])
            return point, float(np.sqrt(max(var, 0.0)) * self.std_[j])
        return point

    def forecast_path(self, data: NowcastData = None, horizon: int = 4,
                      target: str = None) -> pd.Series:
        """Smoothed/forecast path of the target up to ``horizon`` quarters
        beyond the last quarter in the data."""
        self._check_fitted()
        data = data or self.data
        target = target or data.target
        end = quarter_of(data.index[-1]) + horizon
        d = data.extend_to(end)
        X, idx = self._standardize(d)
        ko = kalman_filter_smoother(X, self.Z_, self.R_, self.A_, self.Q_,
                                    self.a0_, self.P0_)
        j = self.series_names_.index(target)
        vals = ko.a_smooth[:, :len(self.series_names_)][:, j] \
            * self.std_[j] + self.mean_[j]
        qidx = idx.asfreq("Q")
        return pd.Series(vals, index=qidx, name=target)

    # ------------------------------------------------------------ internals
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit(data) first.")

    def _select_indicators(self, data: NowcastData) -> list:
        """Top-|corr| monthly indicators vs the target (quarterly means)."""
        y = data.quarterly[data.target].dropna()
        y.index = y.index.asfreq("Q")
        scores = {}
        for name in data.monthly.columns:
            x = data.monthly[name].dropna()
            if x.empty:
                continue
            q = x.copy()
            q.index = q.index.asfreq("Q")
            xq = q.groupby(level=0).mean()
            common = y.index.intersection(xq.index)
            if len(common) > 12:
                scores[name] = abs(np.corrcoef(y[common], xq[common])[0, 1])
        ranked = sorted(scores, key=scores.get, reverse=True)
        return ranked[: self.n_indicators]

    def _blocked_panel(self, data: NowcastData):
        """(T_q x N) blocked matrix and the quarter-end monthly index."""
        first_q = data.index[0].asfreq("Q")
        last_q = data.index[-1].asfreq("Q")
        quarters = pd.period_range(first_q, last_q, freq="Q")
        T = len(quarters)

        cols = []
        for m in self.m_names_:
            x = data.monthly[m]
            for k in (1, 2, 3):
                months = quarters.asfreq("M", how="start") + (k - 1)
                vals = x.reindex(months).to_numpy(float)
                cols.append(vals)
        for qn in self.q_names_:
            months = quarters.asfreq("M", how="end")
            cols.append(data.quarterly[qn].reindex(months).to_numpy(float))
        Y = np.column_stack(cols)
        idx = quarters.asfreq("M", how="end")       # monthly labels (Q ends)
        return Y, idx

    def _standardize(self, data: NowcastData):
        """Blocked, standardized observation matrix + its (monthly) index.
        Interface shared with the DFM so that
        :func:`pynowcast.news_decomposition` works for the B-BVAR too."""
        Y, idx = self._blocked_panel(data)
        return (Y - self.mean_) / self.std_, idx

    def _posterior(self, Y: np.ndarray):
        """Posterior mean of (B, Sigma) with Minnesota (+ optional
        sum-of-coefficients) dummy observations, on a balanced panel."""
        N, p = Y.shape[1], self.lags
        T = Y.shape[0]
        X = np.column_stack(
            [np.ones(T - p)] + [Y[p - k - 1: T - k - 1] for k in range(p)]
        )
        Yreg = Y[p:]

        # per-series scale: AR(1)-residual std, the BGR convention
        sig = np.empty(N)
        for i in range(N):
            z = Y[:, i]
            zl, zc = z[:-1], z[1:]
            denom = float(zl @ zl) + 1e-12
            rho = float(zl @ zc) / denom
            res = zc - rho * zl
            sig[i] = np.sqrt(res @ res / max(len(res) - 1, 1) + 1e-12)

        lam = self.shrinkage
        Yd, Xd = [], []
        # Minnesota: prior mean 0 (stationarity-transformed data),
        # tightness lam, lag decay k
        for k in range(1, p + 1):
            Yd.append(np.zeros((N, N)))
            Xk = np.zeros((N, 1 + N * p))
            Xk[:, 1 + (k - 1) * N: 1 + k * N] = np.diag(sig) * k / lam
            Xd.append(Xk)
        # prior on the residual covariance
        Yd.append(np.diag(sig))
        Xd.append(np.zeros((N, 1 + N * p)))
        # diffuse prior on the constant
        eps = 1e-3
        Yd.append(np.zeros((1, N)))
        Xc = np.zeros((1, 1 + N * p))
        Xc[0, 0] = eps
        Xd.append(Xc)
        # sum-of-coefficients (no-cointegration) dummies
        if self.mu:
            ybar = Y[:p].mean(axis=0)
            Yd.append(np.diag(ybar) / self.mu)
            Xs = np.zeros((N, 1 + N * p))
            for k in range(p):
                Xs[:, 1 + k * N: 1 + (k + 1) * N] = np.diag(ybar) / self.mu
            Xd.append(Xs)

        Ystar = np.vstack([Yreg] + Yd)
        Xstar = np.vstack([X] + Xd)
        B, *_ = np.linalg.lstsq(Xstar, Ystar, rcond=None)
        U = Ystar - Xstar @ B
        Sigma = U.T @ U / max(Ystar.shape[0] - X.shape[1], 1)
        return B, Sigma

    def _build_state_space(self, B: np.ndarray, Sigma: np.ndarray):
        """Companion form with an appended constant state.

        State a_t = [y_t, y_{t-1}, ..., y_{t-p+1}, 1]."""
        N, p = Sigma.shape[0], self.lags
        m = N * p + 1
        c = B[0]                       # intercept
        Bl = [B[1 + k * N: 1 + (k + 1) * N].T for k in range(p)]

        A = np.zeros((m, m))
        for k in range(p):
            A[:N, k * N:(k + 1) * N] = Bl[k]
        A[:N, -1] = c
        if p > 1:
            A[N: N * p, : N * (p - 1)] = np.eye(N * (p - 1))
        A[-1, -1] = 1.0

        Q = np.zeros((m, m))
        Q[:N, :N] = Sigma

        Z = np.zeros((N, m))
        Z[:, :N] = np.eye(N)
        R = KAPPA * np.eye(N)

        a0 = np.zeros(m)
        a0[-1] = 1.0
        P0 = np.zeros((m, m))
        try:
            P0[:N * p, :N * p] = solve_discrete_lyapunov(
                A[:N * p, :N * p], Q[:N * p, :N * p]
            )
        except Exception:
            P0[:N * p, :N * p] = 10.0 * np.eye(N * p)
        if not np.all(np.isfinite(P0)):
            P0 = np.zeros((m, m))
            P0[:N * p, :N * p] = 10.0 * np.eye(N * p)

        self.Z_, self.R_, self.A_, self.Q_ = Z, R, A, Q
        self.a0_, self.P0_ = a0, P0
