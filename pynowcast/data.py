"""
Data handling for the nowcasting toolbox.

The toolbox works with a single object, :class:`NowcastData`, which holds

* a *monthly* panel of indicators (rows = months, columns = series), and
* a *quarterly* panel (rows = months too! quarterly values are stored in the
  third month of each quarter, all other months are NaN) which contains the
  target variable (e.g. GDP growth) and possibly other quarterly series.

All series are stored *after* transformation to (approximate) stationarity,
e.g. month-on-month percentage changes. The original toolbox does the same.

Input format (kept deliberately simple)
---------------------------------------
One Excel file with up to three sheets, or two/three CSV files:

* ``monthly``   : first column ``date`` (anything pandas can parse, monthly),
                  remaining columns = raw monthly indicators.
* ``quarterly`` : first column ``date`` (one row per quarter),
                  remaining columns = raw quarterly series (target first).
* ``spec``      : (optional) one row per series with columns
                  ``series`` (name), ``transform`` (see below),
                  ``block_<name>`` columns with 0/1 block membership, and
                  ``pub_lag`` (publication delay in months, used only for
                  pseudo-real-time evaluation).

Supported transformations (column ``transform``):

* ``pch``   : 100 * log-difference (~ % change), default for most indicators
* ``pchy``  : 100 * 12-month (or 4-quarter) log-difference
* ``diff``  : first difference
* ``none``  : leave the series as is (already stationary, e.g. a survey index)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

VALID_TRANSFORMS = {"pch", "pchy", "diff", "none"}


# ----------------------------------------------------------------------------
# Transformations
# ----------------------------------------------------------------------------

def transform_series(x: pd.Series, how: str, periods_per_year: int) -> pd.Series:
    """Apply a stationarity transformation to a raw series."""
    how = (how or "none").strip().lower()
    if how not in VALID_TRANSFORMS:
        raise ValueError(
            f"Unknown transform '{how}' for series '{x.name}'. "
            f"Valid options: {sorted(VALID_TRANSFORMS)}"
        )
    if how == "none":
        return x.astype(float)
    if how == "diff":
        return x.astype(float).diff()
    if how == "pch":
        return 100.0 * np.log(x.astype(float)).diff()
    if how == "pchy":
        return 100.0 * np.log(x.astype(float)).diff(periods_per_year)
    raise AssertionError("unreachable")


# ----------------------------------------------------------------------------
# The central data container
# ----------------------------------------------------------------------------

@dataclass
class NowcastData:
    """Container for a (transformed) mixed-frequency nowcasting dataset.

    Attributes
    ----------
    monthly : pd.DataFrame
        Transformed monthly indicators, monthly ``PeriodIndex``.
    quarterly : pd.DataFrame
        Transformed quarterly series mapped onto the *monthly* index
        (value in the 3rd month of each quarter, NaN elsewhere).
    target : str
        Name of the target variable (must be a column of ``quarterly``).
    blocks : pd.DataFrame
        0/1 membership matrix (rows = all series, cols = block names).
        Every series belongs at least to the first ("Global") block.
    pub_lag : pd.Series
        Publication lag in months for each series (for evaluation).
    transforms : pd.Series
        Transformation applied to each series (for reporting/inversion).
    """

    monthly: pd.DataFrame
    quarterly: pd.DataFrame
    target: str
    blocks: pd.DataFrame = None
    pub_lag: pd.Series = None
    transforms: pd.Series = None

    # ------------------------------------------------------------------ setup
    def __post_init__(self):
        self.monthly = self.monthly.copy()
        self.quarterly = self.quarterly.copy()
        if not isinstance(self.monthly.index, pd.PeriodIndex):
            self.monthly.index = pd.PeriodIndex(self.monthly.index, freq="M")
        if not isinstance(self.quarterly.index, pd.PeriodIndex):
            self.quarterly.index = pd.PeriodIndex(self.quarterly.index, freq="M")

        # align on the union of indices
        idx = self.monthly.index.union(self.quarterly.index)
        idx = pd.period_range(idx.min(), idx.max(), freq="M")
        self.monthly = self.monthly.reindex(idx)
        self.quarterly = self.quarterly.reindex(idx)

        if self.target not in self.quarterly.columns:
            raise ValueError(
                f"Target '{self.target}' not found among quarterly series "
                f"{list(self.quarterly.columns)}"
            )

        names = list(self.monthly.columns) + list(self.quarterly.columns)
        if len(set(names)) != len(names):
            raise ValueError("Series names must be unique across monthly and quarterly panels.")

        if self.blocks is None:
            self.blocks = pd.DataFrame(1, index=names, columns=["Global"])
        else:
            self.blocks = self.blocks.reindex(names).fillna(0).astype(int)
            first = self.blocks.columns[0]
            if (self.blocks[first] == 0).any():
                warnings.warn(
                    f"All series should load on the first block '{first}'; forcing membership."
                )
                self.blocks[first] = 1
            if (self.blocks.sum(axis=1) == 0).any():
                raise ValueError("Every series must belong to at least one block.")

        if self.pub_lag is None:
            self.pub_lag = pd.Series(0, index=names, dtype=int)
        else:
            self.pub_lag = self.pub_lag.reindex(names).fillna(0).astype(int)

        if self.transforms is None:
            self.transforms = pd.Series("none", index=names)

    # ----------------------------------------------------------------- access
    @property
    def series_names(self) -> list:
        return list(self.monthly.columns) + list(self.quarterly.columns)

    @property
    def n_monthly(self) -> int:
        return self.monthly.shape[1]

    @property
    def n_quarterly(self) -> int:
        return self.quarterly.shape[1]

    @property
    def index(self) -> pd.PeriodIndex:
        return self.monthly.index

    def to_matrix(self) -> np.ndarray:
        """Stacked data matrix (T x n), monthly columns first."""
        return np.hstack([self.monthly.to_numpy(float), self.quarterly.to_numpy(float)])

    # ------------------------------------------------------------- operations
    def extend_to(self, period) -> "NowcastData":
        """Return a copy whose index extends (with NaNs) through ``period``.

        ``period`` may be a monthly period ('2026-08') or a quarter
        ('2026Q3'), in which case the index runs through its last month.
        """
        period = _to_month(period)
        if period <= self.index[-1]:
            return self.copy()
        idx = pd.period_range(self.index[0], period, freq="M")
        return replace(
            self,
            monthly=self.monthly.reindex(idx),
            quarterly=self.quarterly.reindex(idx),
        )

    def vintage(self, as_of, use_pub_lags: bool = True) -> "NowcastData":
        """Simulate the dataset as it would have looked at date ``as_of``.

        Every series ``i`` is cut at ``as_of - pub_lag[i]`` months. This is the
        standard 'pseudo real-time' device used for model evaluation.
        """
        as_of = _to_month(as_of)
        out = self.copy()
        for name in out.series_names:
            lag = int(out.pub_lag[name]) if use_pub_lags else 0
            cutoff = as_of - lag
            panel = out.monthly if name in out.monthly.columns else out.quarterly
            panel.loc[panel.index > cutoff, name] = np.nan
        idx = pd.period_range(self.index[0], as_of, freq="M")
        return replace(
            out,
            monthly=out.monthly.reindex(idx),
            quarterly=out.quarterly.reindex(idx),
        )

    def copy(self) -> "NowcastData":
        return replace(
            self,
            monthly=self.monthly.copy(),
            quarterly=self.quarterly.copy(),
            blocks=self.blocks.copy(),
            pub_lag=self.pub_lag.copy(),
            transforms=self.transforms.copy(),
        )

    def summary(self) -> pd.DataFrame:
        """Quick overview of the dataset (one row per series)."""
        rows = []
        for name in self.series_names:
            panel = self.monthly if name in self.monthly.columns else self.quarterly
            s = panel[name]
            obs = s.dropna()
            rows.append({
                "series": name,
                "frequency": "M" if name in self.monthly.columns else "Q",
                "transform": self.transforms.get(name, "none"),
                "first_obs": str(obs.index[0]) if len(obs) else None,
                "last_obs": str(obs.index[-1]) if len(obs) else None,
                "n_obs": len(obs),
                "pub_lag": int(self.pub_lag[name]),
                "blocks": ",".join(self.blocks.columns[self.blocks.loc[name] == 1]),
            })
        return pd.DataFrame(rows).set_index("series")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_data(
    path: str,
    target: str = None,
    monthly_sheet: str = "monthly",
    quarterly_sheet: str = "quarterly",
    spec_sheet: str = "spec",
) -> NowcastData:
    """Load a nowcasting dataset from an Excel template (or CSV pair).

    Parameters
    ----------
    path : str
        Path to an ``.xlsx`` file with sheets ``monthly``, ``quarterly`` and
        (optionally) ``spec``; *or* a path *prefix* for CSV files, e.g.
        ``"data/mydata"`` for ``data/mydata_monthly.csv`` etc.
    target : str, optional
        Name of the target column in the quarterly sheet. Defaults to the
        first quarterly column.
    """
    if str(path).lower().endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(path)
        raw_m = pd.read_excel(xls, monthly_sheet, index_col=0)
        raw_q = pd.read_excel(xls, quarterly_sheet, index_col=0)
        spec = (
            pd.read_excel(xls, spec_sheet, index_col=0)
            if spec_sheet in xls.sheet_names
            else None
        )
    else:
        raw_m = pd.read_csv(f"{path}_monthly.csv", index_col=0)
        raw_q = pd.read_csv(f"{path}_quarterly.csv", index_col=0)
        try:
            spec = pd.read_csv(f"{path}_spec.csv", index_col=0)
        except FileNotFoundError:
            spec = None

    return build_data(raw_m, raw_q, spec=spec, target=target)


def build_data(
    raw_monthly: pd.DataFrame,
    raw_quarterly: pd.DataFrame,
    spec: pd.DataFrame = None,
    target: str = None,
) -> NowcastData:
    """Build a :class:`NowcastData` from raw (untransformed) panels."""
    raw_monthly = raw_monthly.copy()
    raw_quarterly = raw_quarterly.copy()
    raw_monthly.index = pd.PeriodIndex(pd.to_datetime(raw_monthly.index.astype(str)), freq="M")
    try:
        q_idx = pd.PeriodIndex(raw_quarterly.index.astype(str), freq="Q")
    except Exception:
        q_idx = pd.PeriodIndex(pd.to_datetime(raw_quarterly.index.astype(str)), freq="Q")
    raw_quarterly.index = q_idx

    if target is None:
        target = raw_quarterly.columns[0]

    # --- spec defaults
    names = list(raw_monthly.columns) + list(raw_quarterly.columns)
    transforms = pd.Series("pch", index=names)
    pub_lag = pd.Series(0, index=names, dtype=int)
    blocks = None
    if spec is not None:
        spec = spec.copy()
        if "series" in spec.columns:
            spec = spec.set_index("series")
        spec.index = spec.index.astype(str).str.strip()
        spec = spec[~spec.index.duplicated(keep="first")]
        if "transform" in spec.columns:
            transforms.update(spec["transform"].dropna().astype(str).str.lower())
        if "pub_lag" in spec.columns:
            pub_lag.update(spec["pub_lag"].dropna().astype(int))
        block_cols = [c for c in spec.columns if c.lower().startswith("block")]
        if block_cols:
            blocks = spec[block_cols].copy()
            blocks.columns = [c.split("_", 1)[1] if "_" in c else c for c in block_cols]

    # --- transform
    tm = pd.DataFrame({
        c: transform_series(raw_monthly[c], transforms[c], 12) for c in raw_monthly.columns
    })
    tq = pd.DataFrame({
        c: transform_series(raw_quarterly[c], transforms[c], 4) for c in raw_quarterly.columns
    })

    # map quarterly values onto the 3rd month of each quarter
    tq.index = tq.index.asfreq("M", how="end")

    return NowcastData(
        monthly=tm, quarterly=tq, target=target,
        blocks=blocks, pub_lag=pub_lag, transforms=transforms,
    )


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _to_month(period) -> pd.Period:
    """'2026Q2' -> 2026-06 ; '2026-04' -> 2026-04 ; pd.Period passes through."""
    if isinstance(period, pd.Period):
        return period.asfreq("M", how="end") if period.freqstr.startswith("Q") else period
    s = str(period).strip().upper()
    if "Q" in s:
        return pd.Period(s, freq="Q").asfreq("M", how="end")
    return pd.Period(s, freq="M")


def quarter_of(period) -> pd.Period:
    """Return the quarter containing a monthly period / parse a quarter label."""
    m = _to_month(period)
    return m.asfreq("Q")


def make_example_dataset(
    n_monthly: int = 12,
    start: str = "2005-01",
    end: str = "2026-05",
    seed: int = 0,
    target: str = "GDP",
) -> tuple:
    """Generate a realistic synthetic dataset (raw levels) with a factor
    structure, ragged edges and publication lags. Returns
    ``(raw_monthly, raw_quarterly, spec, truth)`` where ``truth`` holds the
    'future' target values useful for checking results.
    """
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    T = len(months)

    # two AR(1) factors: 'real' and 'soft'
    f = np.zeros((T, 2))
    for t in range(1, T):
        f[t, 0] = 0.85 * f[t - 1, 0] + rng.normal(0, 0.6)
        f[t, 1] = 0.70 * f[t - 1, 1] + 0.3 * f[t - 1, 0] + rng.normal(0, 0.6)

    raw_m = {}
    spec_rows = []
    for i in range(n_monthly):
        lam = rng.uniform(0.4, 1.0, size=2) * (1 if i % 4 else 0.6)
        idio = np.zeros(T)
        a = rng.uniform(0.2, 0.6)
        for t in range(1, T):
            idio[t] = a * idio[t - 1] + rng.normal(0, 0.5)
        growth = lam @ f.T + idio  # ~ % m/m growth
        is_survey = i >= n_monthly - n_monthly // 3
        name = f"survey_{i}" if is_survey else f"hard_{i}"
        if is_survey:
            # survey: stationary index around 50, no transformation needed
            raw_m[name] = 50 + 5 * (lam @ f.T) + idio
            tr = "none"
        else:
            raw_m[name] = 100 * np.exp(np.cumsum(growth / 100.0))
            tr = "pch"
        spec_rows.append({
            "series": name, "transform": tr,
            "pub_lag": 1 if is_survey else 2,
            "block_Global": 1,
            "block_Soft": int(is_survey),
        })
    raw_monthly = pd.DataFrame(raw_m, index=months.astype(str))

    # quarterly GDP: loads on 3-month sums of the factors + noise
    qmask = months.month % 3 == 0
    gdp_growth_m = 0.8 * f[:, 0] + 0.4 * f[:, 1] + rng.normal(0, 0.25, T)
    gdp_growth_q = pd.Series(gdp_growth_m, index=months).rolling(3).mean()[qmask]
    gdp_level = 100 * np.exp(np.cumsum(gdp_growth_q / 100.0))
    quarters = months[qmask].asfreq("Q")
    raw_quarterly = pd.DataFrame({target: gdp_level.values}, index=quarters.astype(str))
    spec_rows.append({
        "series": target, "transform": "pch", "pub_lag": 2,
        "block_Global": 1, "block_Soft": 0,
    })

    spec = pd.DataFrame(spec_rows).set_index("series")
    truth = pd.Series(gdp_growth_q.values, index=quarters, name=f"{target}_growth_true")
    return raw_monthly, raw_quarterly, spec, truth
