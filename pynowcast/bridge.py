"""
Combination of bridge equations, following

    Banbura, M., Belousova, I., Bodnar, K. and Toth, M. B. (2023),
    "Nowcasting employment in the euro area", ECB WP 2815,

as implemented in the ECB nowcasting toolbox (Linzenich & Meunier, 2024):
the model is a *combination of many small bridge equations*, one for every
combination of

    - one or two monthly indicators, and
    - zero or one quarterly indicator (if any are available),

i.e. with n monthly and m quarterly indicators the combination contains
(n + C(n,2)) x (1 + m) equations. Each equation is

    y_q = c + sum_{l=0..lagM} b_l x_{q-l}  [for each monthly regressor]
            + sum_{l=0..lagQ} g_l z_{q-l}  [for the quarterly regressor]
            + sum_{k=1..lagY} a_k y_{q-k}
            + d' D_q + e_q

where x_q is the monthly indicator aggregated to quarterly frequency after
completing the missing months with univariate AR(p) forecasts (BIC-selected
order), and D_q are optional dummies (e.g. Covid dummies, cf. Par.Dum in
the original toolbox). Individual nowcasts are combined with equal weights
(default, as in the original) or weights inversely proportional to the
in-sample MSE.

Note: the original toolbox extrapolates monthly indicators jointly with an
auxiliary BVAR(6); here a univariate AR(p)-BIC is used, which is one of the
extrapolation variants in Banbura et al. (2023).
"""

from __future__ import annotations

from itertools import combinations

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


def _to_quarterly(x: pd.Series) -> pd.Series:
    """3-month average; complete quarters only."""
    q = x.dropna().copy()
    q.index = q.index.asfreq("Q")
    agg = q.groupby(level=0).agg(["mean", "size"])
    out = agg["mean"][agg["size"] == 3]
    out.index.name = None
    return out


class BridgeEquations:
    """Combination of bridge equations (Banbura et al., 2023).

    Parameters
    ----------
    lagM : int
        Lags (in quarters) of the monthly regressors, beyond the
        contemporaneous term (default 0 = contemporaneous only).
    lagQ : int
        Lags of the quarterly regressor, beyond the contemporaneous term.
    lagY : int
        Lags of the endogenous (target) variable (default 1).
    weighting : {"equal", "mse"}
        Combination scheme across equations.
    max_monthly : int
        Use one *and* two monthly regressors per equation (pairs) only if
        ``max_monthly >= 2``; set 1 for single-indicator equations only.
    use_quarterly : bool
        Include quarterly regressors (non-target quarterly series) in the
        combinations.
    dummies : list, optional
        Months (e.g. ``["2020-06", "2020-09"]``) at which quarterly dummy
        variables equal 1 (the dummy is attached to the quarter containing
        the month) -- the Par.Dum mechanism of the original toolbox.
    max_combinations : int
        Safety cap on the number of equations (random thinning above it).
    """

    def __init__(self, lagM: int = 0, lagQ: int = 0, lagY: int = 1,
                 weighting: str = "equal", max_monthly: int = 2,
                 use_quarterly: bool = True, dummies: list = None,
                 max_combinations: int = 2000, indicator_lags: int = None,
                 target_lags: int = None):
        # backward-compatible aliases (previous pynowcast API)
        self.lagM = int(indicator_lags) if indicator_lags is not None else int(lagM)
        self.lagY = int(target_lags) if target_lags is not None else int(lagY)
        self.lagQ = int(lagQ)
        self.weighting = weighting
        self.max_monthly = int(max_monthly)
        self.use_quarterly = bool(use_quarterly)
        self.dummies = [pd.Period(d, "M").asfreq("Q") for d in (dummies or [])]
        self.max_combinations = int(max_combinations)
        self._fitted = False

    def fit(self, data: NowcastData) -> "BridgeEquations":
        self.data = data.copy()
        self._fitted = True
        return self

    # ------------------------------------------------------------------ API
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

        # pre-compute the quarterly panel of regressors once
        panel = self._regressor_panel(data, target, q_target)

        # quarters to predict sequentially (backcasts feed into y lags)
        first = min(last_y + 1, q_target) if last_y is not None else q_target
        out = None
        for q in pd.period_range(first, q_target, freq="Q"):
            out = self._nowcast_one(panel, q, y_q, details and q == q_target)
            val = out[0] if details and q == q_target else out
            y_q = pd.concat([y_q, pd.Series([val], index=[q])])
        return out

    # ------------------------------------------------------------- internals
    def _regressor_panel(self, data, target, q_target) -> dict:
        """AR-complete every monthly indicator through the target quarter
        and aggregate to quarterly; collect quarterly regressors."""
        end_month = q_target.asfreq("M", how="end")
        monthly, quarterly = {}, {}
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
            xq = _to_quarterly(x)
            if q_target in xq.index:
                monthly[name] = xq
        for name in data.quarterly.columns:
            if name == target:
                continue
            z = data.quarterly[name].dropna()
            if z.empty:
                continue
            z.index = z.index.asfreq("Q")
            # complete the quarterly regressor with an AR forecast if needed
            if q_target not in z.index:
                steps = (q_target - z.index.max()).n
                if 0 < steps <= 4:
                    fc = _ar_fit_forecast(z, steps)
                    z = pd.concat([z, pd.Series(
                        fc, index=pd.period_range(z.index.max() + 1,
                                                  q_target, freq="Q"))])
            if q_target in z.index:
                quarterly[name] = z
        return {"monthly": monthly, "quarterly": quarterly}

    def _equations(self, panel):
        """All (monthly combo, quarterly regressor) pairs."""
        m_names = list(panel["monthly"])
        q_names = [None] + (list(panel["quarterly"]) if self.use_quarterly else [])
        combos = [(m,) for m in m_names]
        if self.max_monthly >= 2:
            combos += list(combinations(m_names, 2))
        eqs = [(mc, qn) for mc in combos for qn in q_names]
        if len(eqs) > self.max_combinations:
            rng = np.random.default_rng(0)
            keep = rng.choice(len(eqs), self.max_combinations, replace=False)
            eqs = [eqs[i] for i in sorted(keep)]
        return eqs

    def _nowcast_one(self, panel, q_target, y_q, details):
        preds, mses, rows = [], [], []
        for m_combo, q_name in self._equations(panel):
            res = self._estimate_one(panel, m_combo, q_name, q_target, y_q)
            if res is None:
                continue
            pred, mse = res
            preds.append(pred)
            mses.append(mse)
            rows.append({
                "monthly": " + ".join(m_combo),
                "quarterly": q_name or "",
                "nowcast": pred, "mse": mse,
            })

        if not preds:
            raise RuntimeError(
                f"No bridge equation could be estimated for {q_target}."
            )
        preds = np.array(preds)
        if self.weighting == "mse":
            w = 1.0 / np.maximum(np.array(mses), 1e-12)
            w /= w.sum()
        else:
            w = np.full(len(preds), 1.0 / len(preds))
        combined = float(w @ preds)
        if details:
            tab = pd.DataFrame(rows)
            tab["weight"] = w
            return combined, tab.sort_values("weight", ascending=False)
        return combined

    def _estimate_one(self, panel, m_combo, q_name, q_target, y_q):
        """Estimate one bridge equation by OLS and forecast q_target."""
        df = pd.DataFrame({"y": y_q})
        terms = []
        for m in m_combo:
            x = panel["monthly"][m]
            for l in range(self.lagM + 1):
                col = f"{m}_l{l}"
                df[col] = x.shift(l)
                terms.append((col, x, l))
        if q_name is not None:
            z = panel["quarterly"][q_name]
            for l in range(self.lagQ + 1):
                col = f"{q_name}_l{l}"
                df[col] = z.shift(l)
                terms.append((col, z, l))
        for k in range(1, self.lagY + 1):
            df[f"y_l{k}"] = y_q.shift(k)
        for d in self.dummies:
            df[f"dum_{d}"] = (df.index == d).astype(float)

        est = df.dropna()
        if len(est) < len(df.columns) + 8:
            return None
        X = np.column_stack([np.ones(len(est))] +
                            [est[c].to_numpy() for c in est.columns[1:]])
        beta, *_ = np.linalg.lstsq(X, est["y"].to_numpy(), rcond=None)
        resid = est["y"].to_numpy() - X @ beta
        mse = float(resid @ resid / max(len(est) - len(beta), 1))

        # forecast row for q_target
        xrow = [1.0]
        for col, series, l in terms:
            v = series.get(q_target - l, np.nan)
            if not np.isfinite(v):
                return None
            xrow.append(v)
        for k in range(1, self.lagY + 1):
            v = y_q.get(q_target - k, np.nan)
            if not np.isfinite(v):
                return None
            xrow.append(v)
        for d in self.dummies:
            xrow.append(1.0 if q_target == d else 0.0)
        return float(np.array(xrow) @ beta), mse
