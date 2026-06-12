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




# --------------------------------------------------------------- new in 1.1
def test_covid_corrections():
    from pynowcast import covid_correct, detect_outliers
    data, _ = _data()
    d1 = covid_correct(data, 1)
    assert "dummy_2020q2" in d1.monthly.columns
    assert d1.monthly["dummy_2020q2"][pd.Period("2020-06", "M")] == 1.0
    d2 = covid_correct(data, 2)
    assert d2.monthly.loc["2020-02":"2020-09"].isna().all().all()
    x = data.monthly["hard_0"].copy()
    x.iloc[100] = 80.0
    assert bool(detect_outliers(x).iloc[100])


def test_preselection_ranks_informative_variables():
    from pynowcast import preselect, apply_selection
    data, _ = _data()
    tab = preselect(data)
    assert set(tab.columns) >= {"score", "rank_sis", "rank_tstat", "rank_lars",
                                "frequency", "pub_lag", "group"}
    assert len(tab) == data.n_monthly            # all candidates ranked
    sub = apply_selection(data, tab.head(5).index)
    assert sub.n_monthly == 5 and sub.target == "GDP"


def test_evaluate_horizons_and_fda():
    from pynowcast.evaluate import evaluate, evaluation_summary
    data, _ = _data()
    res = evaluate(lambda: DFM(factors=1, lags=2, max_iter=25), data,
                   start="2025Q1", end="2025Q4",
                   horizons=("backcast", "nowcast", "forecast"),
                   verbose=False)
    assert set(res["horizon"].unique()) == {"backcast", "nowcast", "forecast"}
    s = evaluation_summary(res)
    assert "fda_model" in s.columns
    # backcasts (re-estimating an almost-published quarter) should beat
    # forecasts (a quarter ahead)
    rb = res.query("horizon=='backcast'")["error"].abs().mean()
    rf = res.query("horizon=='forecast'")["error"].abs().mean()
    assert rb < rf


def test_bridge_combinations_and_dummies():
    data, _ = _data()
    br = BridgeEquations().fit(data)
    val, tab = br.nowcast(period="2026Q2", details=True)
    n = data.n_monthly
    assert len(tab) == n + n * (n - 1) // 2      # singles + pairs
    brd = BridgeEquations(dummies=["2020-06", "2020-09"]).fit(data)
    assert np.isfinite(brd.nowcast(period="2026Q2"))


def test_model_search():
    from pynowcast import random_search
    data, _ = _data()
    tab = random_search(data, "BVAR", n_iter=2, start="2025Q3", end="2025Q4",
                        months_in_quarter=(3,), seed=0, verbose=False)
    assert len(tab) == 2 and "rmse_nowcast" in tab.columns


def test_policy_outputs():
    from pynowcast import (confidence_bands, contributions, heatmap,
                           share_of_available_data, alternative_range,
                           prediction_with_bands)
    from pynowcast.evaluate import evaluate
    data, _ = _data()
    model = DFM(factors=1, lags=2, max_iter=25).fit(data)

    # share of available data
    v = data.vintage("2026-05")
    s2 = share_of_available_data(v, "2026Q2")
    s1 = share_of_available_data(v, "2026Q1")
    assert 0 <= s2 < s1 <= 1.0

    # heatmap
    hm = heatmap(data, last=4)
    assert hm.shape[0] == len(data.series_names)
    hg = heatmap(data, last=4, by_group=True)
    assert "Surveys" in hg.index

    # contributions: mean + sum of impacts = prediction
    contr, mean = contributions(model, data, period="2026Q2", lookback=18)
    pred = model.nowcast(period="2026Q2")
    assert abs(mean + contr.sum() - pred) < 1e-6

    # confidence bands from a small evaluation run
    res = evaluate(lambda: DFM(factors=1, lags=2, max_iter=25), data,
                   start="2025Q1", end="2025Q4", months_in_quarter=(3,),
                   verbose=False)
    cb = confidence_bands(None, data, results=res)
    band = prediction_with_bands(pred, cb, "nowcast", 3)
    assert band["lower"] < pred < band["upper"]

    # alternative range
    rng = alternative_range(lambda: DFM(factors=1, lags=2, max_iter=20),
                            data, period="2026Q2", max_drop=1)
    assert (rng["excluded"] == "(none)").any() and len(rng) >= 3




def test_bbvar_blocking_and_news():
    """The blocking B-BVAR: structure, exact smoothing, news identity."""
    from pynowcast import news_decomposition
    data, _ = _data()
    bv = BVAR(lags=2).fit(data)
    # 3 blocked series per monthly variable + quarterly series
    assert len(bv.series_names_) == 3 * data.n_monthly + data.n_quarterly
    # the smoother must reproduce published target values (tiny obs noise)
    smoothed = bv.nowcast(period="2026Q1")
    actual = float(data.quarterly["GDP"].dropna().iloc[-1])
    assert abs(smoothed - actual) < 0.02
    # exact news identity, as for the DFM
    res = news_decomposition(bv, data.vintage("2026-04"),
                             data.vintage("2026-05"),
                             target="GDP", period="2026Q2")
    assert abs(res.residual) < 1e-8
    # uncertainty should shrink as data accumulate within the quarter
    _, s1 = bv.nowcast(period="2026Q2", data=data.vintage("2026-04"),
                       with_uncertainty=True)
    _, s3 = bv.nowcast(period="2026Q2", data=data.vintage("2026-06"),
                       with_uncertainty=True)
    assert s3 < s1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"PASS  {f.__name__}")
    print(f"\n{len(fns)} tests passed.")
