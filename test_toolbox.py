"""
Smoke tests for pynowcast. Run with:  python -m pytest tests/  (or just
``python tests/test_toolbox.py``).
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pynowcast import (BVAR, DFM, BridgeEquations, build_data, evaluate,
                       evaluation_summary, make_example_dataset,
                       news_decomposition)


def _data(seed=1):
    raw_m, raw_q, spec, truth = make_example_dataset(seed=seed)
    return build_data(raw_m, raw_q, spec, target="GDP"), truth


def test_data_construction():
    data, _ = _data()
    assert data.n_monthly == 12 and data.n_quarterly == 1
    assert data.target == "GDP"
    # quarterly values sit on the 3rd month of each quarter
    q = data.quarterly["GDP"].dropna()
    assert all(m.month % 3 == 0 for m in q.index)


def test_vintage_respects_publication_lags():
    data, _ = _data()
    v = data.vintage("2026-05")
    # pub_lag=2 hard data: last obs should be 2026-03
    assert v.monthly["hard_0"].dropna().index.max() == pd.Period("2026-03", "M")
    # pub_lag=1 surveys: last obs 2026-04
    assert v.monthly["survey_8"].dropna().index.max() == pd.Period("2026-04", "M")


def test_dfm_loglik_monotone_and_accurate():
    data, truth = _data()
    m = DFM(factors={"Global": 1, "Soft": 1}, lags=2, max_iter=50).fit(data)
    ll = np.array(m.loglik_path_)
    assert np.all(np.diff(ll) > -1e-6), "EM log-likelihood must not decrease"
    # in-sample fit of the last published quarter should be close to truth
    fit = m.nowcast(period="2026Q1")
    actual = float(data.quarterly["GDP"].dropna().iloc[-1])
    assert abs(fit - actual) < 0.25


def test_dfm_uncertainty():
    data, _ = _data()
    m = DFM(factors=1, lags=2, max_iter=30).fit(data)
    point, std = m.nowcast(period="2026Q2", with_uncertainty=True)
    assert np.isfinite(point) and std > 0


def test_news_identity():
    data, _ = _data()
    m = DFM(factors={"Global": 1, "Soft": 1}, lags=2, max_iter=50).fit(data)
    res = news_decomposition(m, data.vintage("2026-04"), data.vintage("2026-05"),
                             target="GDP", period="2026Q2")
    # old + revisions + news must equal the new nowcast (identity)
    assert abs(res.residual) < 1e-6
    assert len(res.table) > 0


def test_bridge_all_months():
    data, _ = _data()
    br = BridgeEquations().fit(data)
    for as_of in ["2026-04", "2026-05", "2026-06"]:
        v = data.vintage(as_of)
        x = br.nowcast(period="2026Q2", data=v)
        assert np.isfinite(x)


def test_bvar_all_months_and_data_flow_matters():
    data, _ = _data()
    bv = BVAR().fit(data)
    vals = [bv.nowcast(period="2026Q2", data=data.vintage(a))
            for a in ["2026-04", "2026-05", "2026-06"]]
    assert all(np.isfinite(v) for v in vals)
    assert len(set(np.round(vals, 6))) > 1, "nowcast must react to new data"


def test_backtest_beats_ar1_late_in_quarter():
    data, _ = _data()
    res = evaluate(lambda: DFM(factors={"Global": 1, "Soft": 1}, lags=2,
                               max_iter=30),
                   data, start="2024Q1", end="2025Q4", verbose=False)
    s = evaluation_summary(res)
    assert s.loc[3, "relative_rmse"] < 1.0, \
        "DFM should beat AR(1) in month 3 on factor-driven data"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\n{len(fns)} tests passed.")
