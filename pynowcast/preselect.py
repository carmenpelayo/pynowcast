"""
Variable pre-selection (step 1 of the toolbox's three-step approach).

Python counterpart of ``Variable_selection_vF.R`` in the original toolbox.
Three ranking methods from the literature are implemented:

- **SIS** -- Sure Independence Screening (Fan & Lv, 2008): regressors are
  ranked by the absolute marginal correlation with the target.
- **t-stat** (Bair et al., 2006): regressors are ranked by the absolute
  t-statistic of their coefficient in a univariate regression of the target
  on the regressor, controlling for four lags of the target (as in the
  original toolbox).
- **LARS** -- Least Angle Regression (Efron et al., 2004): an iterative
  forward-selection algorithm that, unlike the two methods above, accounts
  for the presence of the other predictors. Variables are ranked by the
  order in which they enter the LARS path.

``preselect`` combines the three rankings into a weighted aggregate score
(higher weight on LARS by default, as in the paper's application) and adds
complementary information (frequency, publication lag, group) to help the
user decide which variables to keep.

All regressors are aligned with the quarterly target by averaging the
monthly observations of each quarter (Mariano-Murasawa-consistent for
growth rates), using only quarters where the target is observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NowcastData


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _quarterly_design(data: NowcastData):
    """Quarterly target y and quarterly-aggregated regressors X."""
    y = data.quarterly[data.target].dropna()
    y.index = y.index.asfreq("Q")

    X = {}
    for name in data.monthly.columns:
        x = data.monthly[name].dropna()
        if x.empty:
            continue
        q = x.copy()
        q.index = q.index.asfreq("Q")
        agg = q.groupby(level=0).agg(["mean", "size"])
        X[name] = agg["mean"][agg["size"] == 3]      # complete quarters only
    for name in data.quarterly.columns:
        if name == data.target:
            continue
        x = data.quarterly[name].dropna()
        x.index = x.index.asfreq("Q")
        X[name] = x
    X = pd.DataFrame(X)
    common = y.index.intersection(X.index)
    return y.loc[common], X.loc[common]


def _standardize(A: np.ndarray) -> np.ndarray:
    mu = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0)
    sd[sd == 0] = 1.0
    return (A - mu) / sd


# ----------------------------------------------------------------------------
# the three ranking methods
# ----------------------------------------------------------------------------

def sis_rank(data: NowcastData) -> pd.Series:
    """|marginal correlation| with the target (Fan & Lv, 2008)."""
    y, X = _quarterly_design(data)
    return X.corrwith(y).abs().sort_values(ascending=False).rename("sis")


def tstat_rank(data: NowcastData, target_lags: int = 4) -> pd.Series:
    """|t-stat| in a univariate regression with lags of the target
    (Bair et al., 2006; four target lags as in the original toolbox)."""
    y, X = _quarterly_design(data)
    ylags = pd.DataFrame({f"y_l{k}": y.shift(k) for k in range(1, target_lags + 1)})
    out = {}
    for name in X.columns:
        df = pd.concat([y.rename("y"), X[name].rename("x"), ylags], axis=1).dropna()
        if len(df) < target_lags + 8:
            continue
        Z = np.column_stack([np.ones(len(df)), df["x"],
                             df[[c for c in df.columns if c.startswith("y_l")]]])
        yy = df["y"].to_numpy()
        beta, *_ = np.linalg.lstsq(Z, yy, rcond=None)
        res = yy - Z @ beta
        dof = max(len(df) - Z.shape[1], 1)
        s2 = res @ res / dof
        XtX_inv = np.linalg.pinv(Z.T @ Z)
        se = np.sqrt(max(s2 * XtX_inv[1, 1], 1e-300))
        out[name] = abs(beta[1] / se)
    return pd.Series(out).sort_values(ascending=False).rename("tstat")


def lars_rank(data: NowcastData, max_steps: int = None) -> pd.Series:
    """Order of entry in the LARS path (Efron et al., 2004).

    The score is ``n_vars - entry_rank + 1`` so that, like the other
    methods, a higher score means a more relevant variable. Variables that
    never enter the path get score 0. Uses scikit-learn if available,
    otherwise a built-in implementation.
    """
    y, X = _quarterly_design(data)
    # LARS needs a complete design: restrict to the common sample, drop
    # variables with insufficient coverage, and mean-impute the few
    # remaining holes
    cov = X.notna().mean()
    Xc = X.loc[:, cov >= 0.6]
    mask = y.notna()
    Xc, yv = Xc.loc[mask], y.loc[mask]
    Xz = _standardize(Xc.to_numpy(float))
    Xz = np.where(np.isnan(Xz), 0.0, Xz)
    yz = (yv - yv.mean()).to_numpy(float)
    names = list(Xc.columns)
    n = len(names)
    max_steps = max_steps or n

    order = _lars_path_order(Xz, yz, max_steps)
    score = pd.Series(0.0, index=names, name="lars")
    for rank, j in enumerate(order):
        score.iloc[j] = n - rank
    return score.sort_values(ascending=False)


def _lars_path_order(X: np.ndarray, y: np.ndarray, max_steps: int) -> list:
    """Entry order of the LARS path. Tries scikit-learn, falls back to a
    minimal pure-numpy LARS."""
    try:
        from sklearn.linear_model import lars_path
        _, active, _ = lars_path(X, y, method="lar",
                                 max_iter=min(max_steps, X.shape[1]))
        return list(active)
    except Exception:
        return _lars_numpy(X, y, max_steps)


def _lars_numpy(X: np.ndarray, y: np.ndarray, max_steps: int) -> list:
    """Minimal LARS (entry order only)."""
    T, n = X.shape
    mu = np.zeros(T)
    active, inactive = [], list(range(n))
    for _ in range(min(max_steps, n, T - 1)):
        c = X.T @ (y - mu)
        if not inactive:
            break
        j = inactive[int(np.argmax(np.abs(c[inactive])))]
        active.append(j)
        inactive.remove(j)
        Xa = X[:, active] * np.sign(c[active])
        G = Xa.T @ Xa + 1e-10 * np.eye(len(active))
        Ginv1 = np.linalg.solve(G, np.ones(len(active)))
        A = 1.0 / np.sqrt(np.ones(len(active)) @ Ginv1)
        w = A * Ginv1
        u = Xa @ w                                   # equiangular direction
        a = X.T @ u
        C = np.max(np.abs(c))
        if inactive:
            gammas = []
            for k in inactive:
                for g in [(C - c[k]) / (A - a[k] + 1e-12),
                          (C + c[k]) / (A + a[k] + 1e-12)]:
                    if g > 1e-12:
                        gammas.append(g)
            gamma = min(gammas) if gammas else C / A
        else:
            gamma = C / A
        mu = mu + gamma * u
    return active


# ----------------------------------------------------------------------------
# combined pre-selection
# ----------------------------------------------------------------------------

def preselect(data: NowcastData, n_keep: int = None,
              weights: dict = None) -> pd.DataFrame:
    """Rank all candidate regressors by predictive power.

    Parameters
    ----------
    n_keep : int, optional
        If given, only the top ``n_keep`` variables are returned.
    weights : dict, optional
        Weights of each method in the aggregate score, default
        ``{"lars": 0.5, "sis": 0.25, "tstat": 0.25}`` (higher weight on
        LARS, as in the paper's application).

    Returns
    -------
    DataFrame indexed by series, sorted by aggregate ``score`` (descending),
    with the rank under each method and complementary information
    (frequency, publication lag, group) to support the final user choice.
    """
    weights = weights or {"lars": 0.5, "sis": 0.25, "tstat": 0.25}
    ranks = {
        "sis": sis_rank(data),
        "tstat": tstat_rank(data),
        "lars": lars_rank(data),
    }
    names = [n for n in data.series_names if n != data.target]
    out = pd.DataFrame(index=pd.Index(names, name="series"))
    score = pd.Series(0.0, index=out.index)
    for m, r in ranks.items():
        # convert each method's ordering into a [0, 1] rank score
        # (1 = most relevant) so that methods are comparable
        rk = r.rank(ascending=True, pct=True).reindex(out.index)
        out[f"rank_{m}"] = r.rank(ascending=False).reindex(out.index)
        score = score + weights.get(m, 0) * rk.fillna(0.0)
    out["score"] = score / sum(weights.values())
    out["frequency"] = ["M" if n in data.monthly.columns else "Q" for n in out.index]
    out["pub_lag"] = data.pub_lag.reindex(out.index)
    out["group"] = data.groups.reindex(out.index)
    out = out.sort_values("score", ascending=False)
    return out.head(n_keep) if n_keep else out


def apply_selection(data: NowcastData, keep: list) -> NowcastData:
    """Return a copy of ``data`` restricted to the series in ``keep``
    (the target is always retained). Python counterpart of the
    ``do_subset`` / ``var_keep`` option of the original toolbox."""
    keep = list(keep)
    out = data.copy()
    m_keep = [c for c in out.monthly.columns if c in keep]
    q_keep = [c for c in out.quarterly.columns if c in keep or c == out.target]
    names = m_keep + q_keep
    from dataclasses import replace
    return replace(
        out,
        monthly=out.monthly[m_keep],
        quarterly=out.quarterly[q_keep],
        blocks=out.blocks.loc[names],
        pub_lag=out.pub_lag.loc[names],
        transforms=out.transforms.loc[names],
        groups=out.groups.loc[names],
    )
