"""
News decomposition: why did the nowcast change between two data vintages?

Following Banbura & Modugno (2014, section 4), the revision of the nowcast is
split into

* a *revision* effect (previously published figures were revised), and
* the *news* effect of each new release: weight x (released value - model
  expectation of that value).

Model parameters are kept fixed at their estimated values, so the
decomposition is exact up to numerical precision:

    nowcast_new = nowcast_old + revision effect + sum of news impacts

The required covariances between the signal and the news,
Cov(y_target, news_k | old info), are computed from the Kalman smoother using
the identity  Cov(a_s, a_t | Y) = J_s J_{s+1} ... J_{t-1} P_{t|T}  for s < t,
where J are the smoother gain matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import NowcastData, _to_month, quarter_of
from .kalman import kalman_filter_smoother


@dataclass
class NewsResult:
    """Output of :func:`news_decomposition`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per news item (new data release) with the released value,
        the model's expectation, the surprise ('news'), its weight and its
        impact on the nowcast (in target units).
    nowcast_old, nowcast_new : float
        Nowcasts based on the old and new vintages.
    revision_impact : float
        Part of the change due to revisions of previously released data.
    news_impact : float
        Sum of all news impacts.
    target, period : str
        What is being nowcast.
    """
    table: pd.DataFrame
    nowcast_old: float
    nowcast_new: float
    revision_impact: float
    news_impact: float
    target: str
    period: str

    @property
    def total_change(self) -> float:
        return self.nowcast_new - self.nowcast_old

    @property
    def residual(self) -> float:
        """Decomposition check: should be ~0 (numerical precision)."""
        return self.total_change - self.revision_impact - self.news_impact

    def by_series(self) -> pd.Series:
        """Aggregate news impacts by series, sorted by absolute size."""
        if self.table.empty:
            return pd.Series(dtype=float)
        s = self.table.groupby("series")["impact"].sum()
        return s.reindex(s.abs().sort_values(ascending=False).index)

    def summary(self) -> str:
        lines = [
            f"Nowcast of {self.target} for {self.period}",
            f"  old vintage : {self.nowcast_old: .3f}",
            f"  new vintage : {self.nowcast_new: .3f}",
            f"  change      : {self.total_change:+.3f}",
            f"    of which revisions : {self.revision_impact:+.3f}",
            f"    of which news      : {self.news_impact:+.3f}",
        ]
        bys = self.by_series()
        if len(bys):
            lines.append("  main news contributions:")
            for name, imp in bys.head(8).items():
                lines.append(f"    {name:<20s} {imp:+.3f}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()


def news_decomposition(model, data_old: NowcastData, data_new: NowcastData,
                       target: str = None, period=None) -> NewsResult:
    """Decompose the nowcast revision between ``data_old`` and ``data_new``.

    Parameters
    ----------
    model : fitted :class:`pynowcast.DFM`
    data_old, data_new : NowcastData
        Two vintages of the same dataset (same series).
    target : str, optional
        Defaults to the dataset target.
    period : str, optional
        Quarter (or month) being nowcast; defaults to the first quarter with
        a missing target value in the *new* vintage.
    """
    model._check_fitted()
    target = target or data_new.target
    if period is None:
        last = data_new.quarterly[target].dropna().index.max()
        period = quarter_of(last) + 1 if last is not None else quarter_of(data_new.index[-1])
    month = _to_month(period)

    # common, extended index
    end = max(month, data_new.index[-1], data_old.index[-1])
    d_old = data_old.extend_to(end)
    d_new = data_new.extend_to(end)
    Xo, idx = model._standardize(d_old)
    Xn, _ = model._standardize(d_new)
    t_star = idx.get_loc(month)
    j_star = model.series_names_.index(target)
    Z, R, A, Q = model.Z_, model.R_, model.A_, model.Q_
    a0, P0 = model.a0_, model.P0_

    def project(X):
        ko = kalman_filter_smoother(X, Z, R, A, Q, a0, P0)
        return ko, float(Z[j_star] @ ko.a_smooth[t_star])

    # ---- old vintage
    ko_old, y_old_s = project(Xo)

    # ---- intermediate vintage: old availability pattern, revised values
    both = ~np.isnan(Xo) & ~np.isnan(Xn)
    Xm = Xo.copy()
    Xm[both] = Xn[both]
    ko_mid, y_mid_s = project(Xm)

    # ---- news items: observed in new but not old
    new_obs = np.isnan(Xo) & ~np.isnan(Xn)
    ts, js = np.nonzero(new_obs)
    order = np.argsort(ts, kind="stable")
    ts, js = ts[order], js[order]
    k = len(ts)

    rows = []
    news_impact_s = 0.0
    if k:
        aS, PS, J = ko_mid.a_smooth, ko_mid.P_smooth, ko_mid.J

        # cumulative products of smoother gains: G[t] = J_t J_{t+1} ... J_{T-2}
        # so that Cov(a_s, a_t | Y) = (G[s] @ inv(G[t])) P_t|T  for s < t.
        # We only need pairwise products over the (few) news dates + t*,
        # so compute them directly per pair instead.
        dates = sorted(set(ts.tolist()) | {t_star})

        def cov_states(s, t):
            """Cov(a_s, a_t | old info), s <= t."""
            if s == t:
                return PS[s]
            Gp = np.eye(A.shape[0])
            for u in range(s, t):
                Gp = Gp @ J[u]
            return Gp @ PS[t]

        cov_cache = {}
        for a_ in dates:
            for b_ in dates:
                if a_ <= b_ and (a_, b_) not in cov_cache:
                    cov_cache[(a_, b_)] = cov_states(a_, b_)

        def cov(s, t):
            return cov_cache[(s, t)] if s <= t else cov_cache[(t, s)].T

        # E[v v'] and E[(y - yhat) v']
        v = np.array([Xn[ts[i], js[i]] - Z[js[i]] @ aS[ts[i]] for i in range(k)])
        Evv = np.empty((k, k))
        for a_ in range(k):
            for b_ in range(a_, k):
                c = Z[js[a_]] @ cov(ts[a_], ts[b_]) @ Z[js[b_]]
                if a_ != b_ and ts[a_] == ts[b_] and js[a_] == js[b_]:
                    c += R[js[a_], js[b_]]
                if a_ == b_:
                    c += R[js[a_], js[a_]]
                Evv[a_, b_] = Evv[b_, a_] = c
        Eyv = np.array([
            Z[j_star] @ cov(t_star, ts[i]) @ Z[js[i]] for i in range(k)
        ])
        weights = Eyv @ np.linalg.pinv(Evv)
        impacts_s = weights * v
        news_impact_s = float(impacts_s.sum())

        sd_t, mu = model.std_[j_star], model.mean_
        for i in range(k):
            jj, tt = js[i], ts[i]
            name = model.series_names_[jj]
            rows.append({
                "series": name,
                "period": str(idx[tt]),
                "released": Xn[tt, jj] * model.std_[jj] + mu[jj],
                "expected": float(Z[jj] @ aS[tt]) * model.std_[jj] + mu[jj],
                "news (std.)": float(v[i]),
                "weight": float(weights[i]),
                "impact": float(impacts_s[i]) * sd_t,
            })

    table = pd.DataFrame(
        rows, columns=["series", "period", "released", "expected",
                       "news (std.)", "weight", "impact"]
    )

    sd_t, mu_t = model.std_[j_star], model.mean_[j_star]
    _, y_new_s = project(Xn)
    y_old = y_old_s * sd_t + mu_t
    y_mid = y_mid_s * sd_t + mu_t
    y_new = y_new_s * sd_t + mu_t
    return NewsResult(
        table=table,
        nowcast_old=float(y_old),
        nowcast_new=float(y_new),
        revision_impact=float(y_mid - y_old),
        news_impact=float(news_impact_s * sd_t),
        target=target,
        period=str(quarter_of(month)),
    )
