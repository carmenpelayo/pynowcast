"""
Model selection (step 2 of the toolbox's three-step approach).

Python counterpart of the ``do_loop`` modes of the original toolbox:

- :func:`random_search` (= ``do_loop = 1``): evaluates many model
  specifications with settings drawn at random within user-defined bounds
  -- estimation start date, subset of input variables, and model
  parameters (factors/lags for the DFM, lags for BEQ and BVAR).
- :func:`custom_search` (= ``do_loop = 2``): evaluates a user-defined list
  of specifications.

Each specification is evaluated out-of-sample in pseudo real time (via
:func:`pynowcast.evaluate.evaluate`) and summarized with RMSE and FDA per
horizon and month of the quarter. The result is a tidy DataFrame with one
row per specification, sorted by nowcast RMSE, which can be saved to Excel
or CSV for inspection -- the same workflow as the original's loop output.

As the paper notes, the idea is an automated "trial-and-error" rather than
an exhaustive grid search: run a broad first pass, narrow the bounds, and
iterate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import NowcastData, quarter_of
from .evaluate import evaluate, evaluation_summary
from .preselect import apply_selection

#: default bounds, mirroring the Loop struct of Nowcast_Main_vF.m
DEFAULT_BOUNDS = {
    "min_startyear": None, "max_startyear": None,   # None = full sample
    "min_var": 5, "max_var": 10,                    # number of monthly vars
    # DFM
    "min_p": 1, "max_p": 4,                         # p > 5 is not supported
    "min_r": 1, "max_r": 3,
    # BEQ
    "min_lagM": 0, "max_lagM": 1,
    "min_lagQ": 0, "max_lagQ": 1,
    "min_lagY": 1, "max_lagY": 2,
    # BVAR
    "min_bvar_lags": 2, "max_bvar_lags": 4,
}


def _truncate_start(data: NowcastData, startyear: int) -> NowcastData:
    if startyear is None:
        return data
    out = data.copy()
    cut = pd.Period(f"{int(startyear)}-01", "M")
    from dataclasses import replace
    idx = out.index[out.index >= cut]
    return replace(out, monthly=out.monthly.loc[idx],
                   quarterly=out.quarterly.loc[idx])


def _draw_spec(rng, model: str, data: NowcastData, b: dict) -> dict:
    m_names = list(data.monthly.columns)
    n_var = int(rng.integers(min(b["min_var"], len(m_names)),
                             min(b["max_var"], len(m_names)) + 1))
    variables = sorted(rng.choice(m_names, size=n_var, replace=False).tolist())
    spec = {"model": model, "variables": variables}
    if b["min_startyear"] is not None:
        spec["startyear"] = int(rng.integers(b["min_startyear"],
                                             b["max_startyear"] + 1))
    if model == "DFM":
        spec["p"] = int(rng.integers(b["min_p"], b["max_p"] + 1))
        spec["r"] = int(rng.integers(b["min_r"], b["max_r"] + 1))
    elif model == "BEQ":
        spec["lagM"] = int(rng.integers(b["min_lagM"], b["max_lagM"] + 1))
        spec["lagQ"] = int(rng.integers(b["min_lagQ"], b["max_lagQ"] + 1))
        spec["lagY"] = int(rng.integers(b["min_lagY"], b["max_lagY"] + 1))
    elif model == "BVAR":
        spec["lags"] = int(rng.integers(b["min_bvar_lags"],
                                        b["max_bvar_lags"] + 1))
    return spec


def _make_factory(spec: dict):
    from .dfm import DFM
    from .bridge import BridgeEquations
    from .bvar import BVAR
    model = spec["model"].upper()
    if model == "DFM":
        return lambda: DFM(factors=spec.get("r", 2), lags=spec.get("p", 2),
                           max_iter=spec.get("max_iter", 50))
    if model in ("BEQ", "BRIDGE"):
        return lambda: BridgeEquations(lagM=spec.get("lagM", 0),
                                       lagQ=spec.get("lagQ", 0),
                                       lagY=spec.get("lagY", 1),
                                       dummies=spec.get("dummies"))
    if model == "BVAR":
        return lambda: BVAR(lags=spec.get("lags", 2),
                            shrinkage=spec.get("shrinkage", 0.2),
                            n_indicators=spec.get("n_indicators"))
    raise ValueError(f"Unknown model class '{spec['model']}'")


def _run_spec(spec: dict, data: NowcastData, start, end,
              horizons, covid_method: int = None,
              months_in_quarter=(1, 2, 3), verbose=False):
    """Evaluate one spec; return a flat result row."""
    from .corrections import covid_correct
    d = data
    if spec.get("variables"):
        d = apply_selection(d, spec["variables"])
    if spec.get("startyear"):
        d = _truncate_start(d, spec["startyear"])
    if covid_method:
        d = covid_correct(d, covid_method)
    res = evaluate(_make_factory(spec), d, start=start, end=end,
                   horizons=horizons, months_in_quarter=months_in_quarter,
                   verbose=False)
    if res.empty:
        return None
    summ = evaluation_summary(res, by=("horizon", "month_in_quarter"))
    row = dict(spec)
    row["variables"] = ", ".join(spec.get("variables", []))
    if covid_method is not None:
        row["covid_method"] = covid_method
    only_hz = horizons[0] if len(horizons) == 1 else None
    for key, r in summ.iterrows():
        hz, mth = key if isinstance(key, tuple) else (only_hz or "nowcast", key)
        tag = f"{str(hz)[:3]}_m{mth}"
        row[f"rmse_{tag}"] = r["rmse_model"]
        row[f"fda_{tag}"] = r["fda_model"]
    now = res.query("horizon=='nowcast'") if "horizon" in res.columns else res
    row["rmse_nowcast"] = float(np.sqrt((now["error"].dropna() ** 2).mean())) \
        if len(now) else np.nan
    row["fda_nowcast"] = float(now["hit"].mean()) if len(now) else np.nan
    return row


def random_search(data: NowcastData, model: str = "DFM", n_iter: int = 20,
                  start="2022Q1", end=None, horizons=("nowcast",),
                  months_in_quarter=(1, 2, 3), bounds: dict = None,
                  seed: int = None, covid_method: int = None,
                  verbose: bool = True) -> pd.DataFrame:
    """Evaluate ``n_iter`` random specifications of one model class.

    Parameters mirror the ``Loop`` struct of the original toolbox: bounds
    on the start year, the number of (randomly drawn) monthly variables,
    and the model parameters. ``seed=None`` randomizes across runs
    (``Loop.do_random = 1``); fixing the seed reproduces the same set of
    models, e.g. to test the effect of a Covid correction on identical
    specifications (``Loop.do_random = 0``).
    """
    b = dict(DEFAULT_BOUNDS)
    b.update(bounds or {})
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(n_iter)):
        spec = _draw_spec(rng, model.upper(), data, b)
        try:
            row = _run_spec(spec, data, start, end, horizons, covid_method,
                            months_in_quarter)
        except Exception as exc:
            warnings.warn(f"spec {i}: {exc}")
            row = None
        if row:
            row["spec_id"] = i
            rows.append(row)
            if verbose:
                print(f"  [{i + 1}/{n_iter}] {model} "
                      f"rmse_nowcast={row.get('rmse_nowcast', np.nan):.3f}")
    out = pd.DataFrame(rows)
    return out.sort_values("rmse_nowcast").reset_index(drop=True) \
        if not out.empty else out


def custom_search(data: NowcastData, specs: list, start="2022Q1", end=None,
                  horizons=("nowcast",), covid_methods=(None,),
                  months_in_quarter=(1, 2, 3),
                  verbose: bool = True) -> pd.DataFrame:
    """Evaluate a user-defined list of specifications (``do_loop = 2``).

    Each spec is a dict, e.g. ``{"model": "DFM", "p": 2, "r": 2,
    "variables": [...]}``. ``covid_methods`` allows testing each spec under
    several Covid corrections (``Loop.alter_covid``): pass e.g.
    ``(0, 1, 2, 3)`` to compare corrections on the same specifications.
    """
    rows = []
    for i, spec in enumerate(specs):
        for cm in covid_methods:
            try:
                row = _run_spec(dict(spec), data, start, end, horizons, cm,
                                months_in_quarter)
            except Exception as exc:
                warnings.warn(f"spec {i} (covid={cm}): {exc}")
                row = None
            if row:
                row["spec_id"] = i
                rows.append(row)
                if verbose:
                    print(f"  [{i + 1}/{len(specs)}] covid={cm} "
                          f"rmse_nowcast={row.get('rmse_nowcast', np.nan):.3f}")
    out = pd.DataFrame(rows)
    return out.sort_values("rmse_nowcast").reset_index(drop=True) \
        if not out.empty else out
