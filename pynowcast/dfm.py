"""
Dynamic factor model for nowcasting, estimated by EM with arbitrary patterns
of missing data, following

    Banbura, M. and Modugno, M. (2014), "Maximum likelihood estimation of
    factor models on datasets with arbitrary pattern of missing data",
    Journal of Applied Econometrics 29(11).

with an optional block structure as in

    Delle Chiaie, S., Ferrara, L. and Giannone, D. (2022), "Common factors
    of commodity prices", Journal of Applied Econometrics 37(3).

Specification
-------------
Let x_t be the standardized monthly indicators and y_t the standardized
quarterly series (recorded in the third month of each quarter):

    x_it = sum_b  L_i^b f_t^b                      + e_it
    y_jt = sum_b  L_j^b (f_t^b + 2 f_{t-1}^b + 3 f_{t-2}^b
                          + 2 f_{t-3}^b + f_{t-4}^b)
           +  (e_jt + 2 e_jt-1 + 3 e_jt-2 + 2 e_jt-3 + e_jt-4)

    f_t^b : VAR(p) factors of block b
    e_it  : AR(1) idiosyncratic components

The [1 2 3 2 1] 'tent' implements the Mariano-Murasawa (2003) approximation
linking quarterly growth rates to monthly ones. All components live in the
state vector so that the Kalman smoother handles missing data exactly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

from .data import NowcastData, _to_month, quarter_of
from .kalman import kalman_filter_smoother

TENT = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
KAPPA = 1e-4  # measurement noise (idiosyncratic terms are in the state)


# ============================================================================
# State-space layout helper
# ============================================================================

@dataclass
class _Layout:
    """Bookkeeping of where everything lives inside the state vector."""
    block_names: list
    r: dict                 # factors per block
    p: int                  # VAR lags
    pp: int                 # lags kept in the state, max(p, 5)
    n_m: int
    n_q: int
    block_offset: dict      # state offset of each block's factor segment
    idio_m_offset: int
    idio_q_offset: int
    m: int                  # total state dimension

    @classmethod
    def build(cls, block_names, r, p, n_m, n_q):
        pp = max(p, 5)
        off, block_offset = 0, {}
        for b in block_names:
            block_offset[b] = off
            off += r[b] * pp
        idio_m_offset = off
        off += n_m
        idio_q_offset = off
        off += 5 * n_q
        return cls(block_names, r, p, pp, n_m, n_q,
                   block_offset, idio_m_offset, idio_q_offset, off)

    def factor_idx(self, b, lag=0):
        """State indices of block b's factors at a given lag."""
        o = self.block_offset[b] + lag * self.r[b]
        return np.arange(o, o + self.r[b])


# ============================================================================
# The model
# ============================================================================

class DFM:
    """Mixed-frequency dynamic factor model for nowcasting.

    Parameters
    ----------
    factors : int or dict
        Number of factors. An ``int`` applies to every block; a dict maps
        block names (as in ``data.blocks``) to factor counts,
        e.g. ``{"Global": 2, "Soft": 1}``.
    lags : int
        Number of lags in the factor VAR (default 2).
    max_iter : int
        Maximum number of EM iterations (default 100).
    tol : float
        EM convergence threshold on the relative log-likelihood change.
    verbose : bool
        Print EM progress.

    Examples
    --------
    >>> model = DFM(factors=2, lags=2).fit(data)
    >>> model.nowcast("GDP", "2026Q2")
    """

    def __init__(self, factors=2, lags=2, max_iter=100, tol=1e-4, verbose=False):
        self.factors = factors
        self.lags = int(lags)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self._fitted = False

    # ------------------------------------------------------------------- fit
    def fit(self, data: NowcastData) -> "DFM":
        """Estimate the model by EM on (a copy of) ``data``."""
        self.data = data.copy()
        X = data.to_matrix()                       # (T, n), monthly cols first
        self.series_names_ = data.series_names
        self.n_m_, self.n_q_ = data.n_monthly, data.n_quarterly
        n = X.shape[1]

        # ---- standardization (on observed entries)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0, ddof=1)
        self.std_[self.std_ < 1e-10] = 1.0
        Xs = (X - self.mean_) / self.std_

        # ---- blocks / layout
        blocks = data.blocks
        block_names = list(blocks.columns)
        if isinstance(self.factors, dict):
            r = {b: int(self.factors.get(b, 1)) for b in block_names}
        else:
            r = {b: int(self.factors) for b in block_names}
        self.layout_ = _Layout.build(block_names, r, self.lags,
                                     self.n_m_, self.n_q_)
        self.block_members_ = {
            b: np.flatnonzero(blocks[b].to_numpy()) for b in block_names
        }

        # ---- selection matrices used in the loadings M-step
        self._build_selectors()

        # ---- initialize and run EM
        Z, A, Q = self._initialize(Xs)
        R = KAPPA * np.eye(n)
        ll_prev = -np.inf
        self.loglik_path_ = []
        for it in range(self.max_iter):
            Z, A, Q, ll = self._em_step(Xs, Z, A, Q, R)
            self.loglik_path_.append(ll)
            if self.verbose:
                print(f"  EM iter {it + 1:3d}  loglik = {ll:,.2f}")
            if np.isfinite(ll_prev):
                denom = (abs(ll) + abs(ll_prev)) / 2 + 1e-12
                if ll < ll_prev - 1e-6 * denom and self.verbose:
                    warnings.warn("EM log-likelihood decreased slightly "
                                  "(numerical noise).")
                if abs(ll - ll_prev) / denom < self.tol:
                    break
            ll_prev = ll
        self.n_iter_ = it + 1

        self.Z_, self.A_, self.Q_, self.R_ = Z, A, Q, R
        self.a0_, self.P0_ = self._initial_state(A, Q)
        self._fitted = True
        return self

    # --------------------------------------------------------------- predict
    def predict(self, data: NowcastData = None, horizon=None) -> pd.DataFrame:
        """Model-implied values for every series and period.

        Parameters
        ----------
        data : NowcastData, optional
            Dataset (vintage) to condition on; defaults to the training data.
        horizon : str/Period, optional
            Extend the projection through this month or quarter
            (e.g. ``"2026Q4"``).

        Returns
        -------
        DataFrame of fitted/predicted values in *original* (transformed
        series) units, same columns as the data.
        """
        self._check_fitted()
        data = (data or self.data)
        if horizon is not None:
            data = data.extend_to(horizon)
        Xs, idx = self._standardize(data)
        ko = kalman_filter_smoother(Xs, self.Z_, self.R_, self.A_, self.Q_,
                                    self.a0_, self.P0_)
        fitted = ko.a_smooth @ self.Z_.T
        out = fitted * self.std_ + self.mean_
        return pd.DataFrame(out, index=idx, columns=self.series_names_)

    def nowcast(self, target: str = None, period=None, data: NowcastData = None,
                with_uncertainty: bool = False):
        """Point nowcast of ``target`` for ``period`` (e.g. ``'2026Q2'``).

        Defaults: the dataset's target variable and the quarter following the
        last observed target value.
        """
        self._check_fitted()
        data = (data or self.data)
        target = target or data.target
        if period is None:
            last = data.quarterly[target].dropna().index.max()
            period = (quarter_of(last) + 1) if last is not None else quarter_of(data.index[-1])
        month = _to_month(period)
        ds = data.extend_to(month)
        Xs, idx = self._standardize(ds)
        ko = kalman_filter_smoother(Xs, self.Z_, self.R_, self.A_, self.Q_,
                                    self.a0_, self.P0_)
        t = idx.get_loc(month)
        j = self.series_names_.index(target)
        z = self.Z_[j]
        val = z @ ko.a_smooth[t] * self.std_[j] + self.mean_[j]
        if not with_uncertainty:
            return float(val)
        var = z @ ko.P_smooth[t] @ z + self.R_[j, j]
        return float(val), float(np.sqrt(var) * self.std_[j])

    def extract_factors(self, data: NowcastData = None, horizon=None) -> pd.DataFrame:
        """Smoothed factor estimates."""
        self._check_fitted()
        data = (data or self.data)
        if horizon is not None:
            data = data.extend_to(horizon)
        Xs, idx = self._standardize(data)
        ko = kalman_filter_smoother(Xs, self.Z_, self.R_, self.A_, self.Q_,
                                    self.a0_, self.P0_)
        cols, vals = [], []
        for b in self.layout_.block_names:
            fi = self.layout_.factor_idx(b)
            for k, i in enumerate(fi):
                cols.append(f"{b}_{k + 1}")
                vals.append(ko.a_smooth[:, i])
        return pd.DataFrame(np.column_stack(vals), index=idx, columns=cols)

    def explain_change(self, data_old: NowcastData, data_new: NowcastData,
                       target: str = None, period=None):
        """News decomposition of the nowcast revision between two vintages.

        See :func:`pynowcast.news.news_decomposition`.
        """
        from .news import news_decomposition
        return news_decomposition(self, data_old, data_new,
                                  target=target, period=period)

    # ===================================================== internal machinery
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit(data) first.")

    def _standardize(self, data: NowcastData):
        if data.series_names != self.series_names_:
            raise ValueError("Dataset series do not match the fitted model.")
        X = data.to_matrix()
        return (X - self.mean_) / self.std_, data.index

    # ----------------------------------------------------------- selectors
    def _build_selectors(self):
        """For every series i, build
        W_i : maps the state to the factor regressor x_t (k_i x m)
        u_i : maps the state to the series' idiosyncratic component (m,)
        so that  y_it = lambda_i @ (W_i a_t) + u_i @ a_t .
        """
        L, n = self.layout_, self.n_m_ + self.n_q_
        self._W, self._u, self._member_blocks = [], [], []
        for i in range(n):
            is_q = i >= self.n_m_
            mblocks = [b for b in L.block_names
                       if i in self.block_members_[b]]
            k = sum(L.r[b] for b in mblocks)
            W = np.zeros((k, L.m))
            row = 0
            for b in mblocks:
                rb = L.r[b]
                if is_q:
                    for lag in range(5):
                        W[row:row + rb, L.factor_idx(b, lag)] = (
                            TENT[lag] * np.eye(rb)
                        )
                else:
                    W[row:row + rb, L.factor_idx(b, 0)] = np.eye(rb)
                row += rb
            u = np.zeros(L.m)
            if is_q:
                j = i - self.n_m_
                u[L.idio_q_offset + 5 * j: L.idio_q_offset + 5 * j + 5] = TENT
            else:
                u[L.idio_m_offset + i] = 1.0
            self._W.append(W)
            self._u.append(u)
            self._member_blocks.append(mblocks)

    def _z_from_loadings(self, lambdas):
        """Assemble the observation matrix from per-series loading vectors."""
        L = self.layout_
        n = self.n_m_ + self.n_q_
        Z = np.zeros((n, L.m))
        for i in range(n):
            Z[i] = lambdas[i] @ self._W[i] + self._u[i]
        return Z

    # ------------------------------------------------------- initialization
    def _initialize(self, Xs):
        """PCA-based starting values (sequential by block on residuals)."""
        L = self.layout_
        T, n = Xs.shape
        Xf = (pd.DataFrame(Xs)
              .interpolate(limit_direction="both")
              .fillna(0.0)
              .to_numpy())

        resid = Xf.copy()
        F = {}
        for b in L.block_names:
            cols = self.block_members_[b]
            Xb = resid[:, cols]
            # PCA via SVD of the (T x nb) panel
            U, s, Vt = np.linalg.svd(Xb, full_matrices=False)
            rb = L.r[b]
            f = U[:, :rb] * s[:rb]            # (T, rb) principal components
            lam = Vt[:rb].T                    # (nb, rb)
            resid[:, cols] = Xb - f @ lam.T
            F[b] = f

        # loadings: OLS of each series on its (tent-aggregated) factors
        lambdas = []
        for i in range(n):
            is_q = i >= self.n_m_
            xparts = []
            for b in self._member_blocks[i]:
                f = F[b]
                if is_q:
                    agg = np.zeros_like(f)
                    for lag in range(5):
                        agg[lag:] += TENT[lag] * f[: T - lag if lag else T]
                    xparts.append(agg)
                else:
                    xparts.append(f)
            Xreg = np.hstack(xparts)
            y = Xf[:, i]
            mask = ~np.isnan(Xs[:, i])
            if is_q:
                mask &= np.arange(T) >= 4
            Xm, ym = Xreg[mask], y[mask]
            lam, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
            lambdas.append(lam)
        self._lambdas = lambdas
        Z = self._z_from_loadings(lambdas)

        # VAR(p) per block by OLS, idiosyncratic AR(1) from residuals
        A = np.zeros((L.m, L.m))
        Q = np.zeros((L.m, L.m))
        for b in L.block_names:
            f, rb, p = F[b], L.r[b], L.p
            Y = f[p:]
            Xlag = np.hstack([f[p - k - 1: T - k - 1] for k in range(p)])
            coef, *_ = np.linalg.lstsq(Xlag, Y, rcond=None)
            coef = coef.T                                   # (rb, rb*p)
            res = Y - Xlag @ coef.T
            Qb = np.cov(res.T) if rb > 1 else np.array([[np.var(res)]])
            o = L.block_offset[b]
            A[o:o + rb, o:o + rb * p] = coef
            A[o + rb:o + rb * L.pp, o:o + rb * (L.pp - 1)] = np.eye(rb * (L.pp - 1))
            Q[o:o + rb, o:o + rb] = np.atleast_2d(Qb) + 1e-6 * np.eye(rb)

        # idiosyncratic components from the residuals of the loading fit
        common = np.zeros((T, n))
        for i in range(n):
            xparts = []
            for b in self._member_blocks[i]:
                f = F[b]
                if i >= self.n_m_:
                    agg = np.zeros_like(f)
                    for lag in range(5):
                        agg[lag:] += TENT[lag] * f[: T - lag if lag else T]
                    xparts.append(agg)
                else:
                    xparts.append(f)
            common[:, i] = np.hstack(xparts) @ lambdas[i]
        E = Xf - common
        for i in range(self.n_m_):
            e = E[:, i]
            num = float(e[1:] @ e[:-1])
            den = float(e[:-1] @ e[:-1]) + 1e-12
            a = np.clip(num / den, -0.95, 0.95)
            s2 = max(np.var(e[1:] - a * e[:-1]), 1e-4)
            o = L.idio_m_offset + i
            A[o, o] = a
            Q[o, o] = s2
        for j in range(self.n_q_):
            e = E[:, self.n_m_ + j]
            s2 = max(np.var(e) / float(TENT @ TENT), 1e-4)
            o = L.idio_q_offset + 5 * j
            A[o, o] = 0.2
            A[o + 1:o + 5, o:o + 4] = np.eye(4)
            Q[o, o] = s2
        return Z, A, Q

    def _initial_state(self, A, Q):
        """Unconditional moments, computed blockwise for stability."""
        L = self.layout_
        m = L.m
        a0 = np.zeros(m)
        P0 = np.zeros((m, m))
        segs = []
        for b in L.block_names:
            o = L.block_offset[b]
            segs.append((o, o + L.r[b] * L.pp))
        for i in range(self.n_m_):
            o = L.idio_m_offset + i
            segs.append((o, o + 1))
        for j in range(self.n_q_):
            o = L.idio_q_offset + 5 * j
            segs.append((o, o + 5))
        for lo, hi in segs:
            Ab, Qb = A[lo:hi, lo:hi], Q[lo:hi, lo:hi]
            try:
                P = solve_discrete_lyapunov(Ab, Qb)
            except Exception:
                P = np.eye(hi - lo)
            P0[lo:hi, lo:hi] = 0.5 * (P + P.T)
        P0 += 1e-8 * np.eye(m)
        return a0, P0

    # --------------------------------------------------------------- EM step
    def _em_step(self, Xs, Z, A, Q, R):
        L = self.layout_
        T, n = Xs.shape
        a0, P0 = self._initial_state(A, Q)
        ko = kalman_filter_smoother(Xs, Z, R, A, Q, a0, P0)
        a, P, PP = ko.a_smooth, ko.P_smooth, ko.PP_smooth

        # within-period second moments  M_t = E[alpha_t alpha_t']
        M = P + np.einsum("ti,tj->tij", a, a)
        Msum = M.sum(axis=0)

        A_new = np.zeros_like(A)
        Q_new = np.zeros_like(Q)

        # ---- factor VAR per block (lags live inside the state)
        for b in L.block_names:
            rb, p, o = L.r[b], L.p, L.block_offset[b]
            fi = np.arange(o, o + rb)                  # f_t
            gi = np.arange(o + rb, o + rb * (p + 1))   # f_{t-1..t-p}
            Sff = Msum[np.ix_(fi, fi)]
            Sfg = Msum[np.ix_(fi, gi)]
            Sgg = Msum[np.ix_(gi, gi)]
            coef = Sfg @ np.linalg.pinv(Sgg)
            Qb = (Sff - coef @ Sfg.T) / T
            Qb = 0.5 * (Qb + Qb.T)
            w, V = np.linalg.eigh(Qb)
            Qb = (V * np.maximum(w, 1e-8)) @ V.T
            A_new[o:o + rb, o:o + rb * p] = coef
            A_new[o + rb:o + rb * L.pp, o:o + rb * (L.pp - 1)] = (
                np.eye(rb * (L.pp - 1))
            )
            Q_new[o:o + rb, o:o + rb] = Qb

        # ---- monthly idiosyncratic AR(1): needs lag-1 cross moments
        for i in range(self.n_m_):
            o = L.idio_m_offset + i
            See = M[1:, o, o].sum()
            See1 = M[:-1, o, o].sum()
            cross = (PP[1:, o, o] + a[1:, o] * a[:-1, o]).sum()
            alpha = np.clip(cross / (See1 + 1e-12), -0.98, 0.98)
            s2 = max((See - alpha * cross) / (T - 1), 1e-8)
            A_new[o, o] = alpha
            Q_new[o, o] = s2

        # ---- quarterly idiosyncratic AR(1): lag is inside the 5-state block
        for j in range(self.n_q_):
            o = L.idio_q_offset + 5 * j
            See = Msum[o, o]
            See1 = Msum[o + 1, o + 1]
            cross = Msum[o, o + 1]
            alpha = np.clip(cross / (See1 + 1e-12), -0.98, 0.98)
            s2 = max((See - alpha * cross) / T, 1e-8)
            A_new[o, o] = alpha
            A_new[o + 1:o + 5, o:o + 4] = np.eye(4)
            Q_new[o, o] = s2

        # ---- loadings, series by series (restricted GLS using selectors)
        lambdas = []
        for i in range(n):
            W, u = self._W[i], self._u[i]
            obs = ~np.isnan(Xs[:, i])
            Mt = M[obs]                                # (To, m, m)
            at = a[obs]
            y = Xs[obs, i]
            Sxx = W @ Mt.sum(axis=0) @ W.T             # E[x x']
            Syx = y @ (at @ W.T)                       # sum_t y_t E[x_t]'
            Sex = u @ Mt.sum(axis=0) @ W.T             # E[e x']
            lam = np.linalg.solve(
                Sxx + 1e-8 * np.eye(Sxx.shape[0]), (Syx - Sex)
            )
            lambdas.append(lam)
        self._lambdas = lambdas
        Z_new = self._z_from_loadings(lambdas)

        return Z_new, A_new, Q_new, ko.loglik
