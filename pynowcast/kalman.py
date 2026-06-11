"""
Kalman filter and smoother for state-space models with missing observations.

Model (time-invariant system matrices):

    y_t = Z a_t + e_t,        e_t ~ N(0, R)         (observation)
    a_t = A a_{t-1} + u_t,    u_t ~ N(0, Q)         (transition)
    a_0 ~ N(a0, P0)

Missing entries of ``y_t`` (NaN) are skipped row-wise, which is the textbook
treatment and the one used by Banbura & Modugno (2014).

The smoother also returns the lag-one smoothed covariances
``Cov(a_t, a_{t-1} | Y)`` required by the EM algorithm, and the sequence of
smoother gain matrices ``J_t`` required by the news decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanOutput:
    a_filt: np.ndarray      # (T, m)   E[a_t | y_1..t]
    P_filt: np.ndarray      # (T, m, m)
    a_pred: np.ndarray      # (T, m)   E[a_t | y_1..t-1]
    P_pred: np.ndarray      # (T, m, m)
    a_smooth: np.ndarray    # (T, m)   E[a_t | y_1..T]
    P_smooth: np.ndarray    # (T, m, m)
    PP_smooth: np.ndarray   # (T, m, m) Cov(a_t, a_{t-1} | Y); index 0 unused
    J: np.ndarray           # (T, m, m) smoother gains (J[T-1] unused)
    loglik: float


def _symmetrize(P):
    return 0.5 * (P + P.T)


def kalman_filter_smoother(y, Z, R, A, Q, a0, P0) -> KalmanOutput:
    """Run the Kalman filter and Rauch-Tung-Striebel smoother.

    Parameters
    ----------
    y : (T, n) array with NaNs for missing observations.
    """
    T, n = y.shape
    m = A.shape[0]

    a_filt = np.zeros((T, m))
    P_filt = np.zeros((T, m, m))
    a_pred = np.zeros((T, m))
    P_pred = np.zeros((T, m, m))
    loglik = 0.0

    a_prev, P_prev = np.asarray(a0, float).ravel(), np.asarray(P0, float)
    for t in range(T):
        # ---- predict
        ap = A @ a_prev
        Pp = _symmetrize(A @ P_prev @ A.T + Q)
        a_pred[t], P_pred[t] = ap, Pp

        # ---- update with observed rows only
        obs = ~np.isnan(y[t])
        if obs.any():
            Zt = Z[obs]
            yt = y[t, obs]
            Rt = R[np.ix_(obs, obs)]
            v = yt - Zt @ ap
            F = Zt @ Pp @ Zt.T + Rt
            F = _symmetrize(F)
            try:
                Fchol = np.linalg.cholesky(F)
                Finv_v = np.linalg.solve(Fchol.T, np.linalg.solve(Fchol, v))
                logdetF = 2.0 * np.log(np.diag(Fchol)).sum()
                FinvZP = np.linalg.solve(Fchol.T, np.linalg.solve(Fchol, Zt @ Pp))
            except np.linalg.LinAlgError:
                Finv = np.linalg.pinv(F)
                Finv_v = Finv @ v
                sign, logdetF = np.linalg.slogdet(F)
                FinvZP = Finv @ (Zt @ Pp)
            K = Pp @ Zt.T  # times F^{-1} implicitly
            a_new = ap + K @ Finv_v
            P_new = _symmetrize(Pp - K @ FinvZP)
            loglik += -0.5 * (obs.sum() * np.log(2 * np.pi) + logdetF + v @ Finv_v)
        else:
            a_new, P_new = ap, Pp

        a_filt[t], P_filt[t] = a_new, P_new
        a_prev, P_prev = a_new, P_new

    # ---- RTS smoother
    a_smooth = np.zeros_like(a_filt)
    P_smooth = np.zeros_like(P_filt)
    PP_smooth = np.zeros_like(P_filt)
    J = np.zeros((T, m, m))

    a_smooth[-1], P_smooth[-1] = a_filt[-1], P_filt[-1]
    for t in range(T - 2, -1, -1):
        Pp_next = P_pred[t + 1]
        Jt = P_filt[t] @ A.T @ np.linalg.pinv(Pp_next)
        J[t] = Jt
        a_smooth[t] = a_filt[t] + Jt @ (a_smooth[t + 1] - a_pred[t + 1])
        P_smooth[t] = _symmetrize(
            P_filt[t] + Jt @ (P_smooth[t + 1] - Pp_next) @ Jt.T
        )

    # lag-one smoothed covariances: Cov(a_t, a_{t-1} | Y) = P^s_t J_{t-1}'
    for t in range(1, T):
        PP_smooth[t] = P_smooth[t] @ J[t - 1].T

    return KalmanOutput(
        a_filt=a_filt, P_filt=P_filt, a_pred=a_pred, P_pred=P_pred,
        a_smooth=a_smooth, P_smooth=P_smooth, PP_smooth=PP_smooth,
        J=J, loglik=float(loglik),
    )
