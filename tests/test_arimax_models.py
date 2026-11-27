"""
Tests for aliciadata_finance_backend.forecasting.arimax_models: the
exogenous transforms, how build_arimax_window() lines the lagged regressors
up with a training window (and leaves no room for look-ahead), and
ArimaxModels.fit_arimax(), which fits every candidate order including
(0, 0), selects the lowest BIC, and forecasts with the next session's
regressor.

Most model tests use the real statsmodels ARIMA on small synthetic windows;
one replaces it with a fake whose fits fail. In-memory only.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import LinAlgError

from aliciadata_finance_backend.config import EXOGENOUS_TRANSFORMS, ExogenousVariable
from aliciadata_finance_backend.data.date_manager import DateManager
from aliciadata_finance_backend.forecasting import arimax_models
from aliciadata_finance_backend.forecasting.arimax_models import (
    EXOGENOUS_TRANSFORM_FUNCTIONS,
    ArimaxModels,
    ArimaxWindow,
    build_arimax_window,
    transform_exogenous,
)

SPX = ExogenousVariable(symbol="I:SPX", source="massive", transform="log_return", lag=1)
BETA = 0.8  # Synthetic tickers' loading on the previous day's index return.


def session_index(n: int) -> pd.DatetimeIndex:
    """n consecutive NYSE sessions at 17:00, with the NYSE frequency set."""
    return pd.date_range("2025-03-03 17:00", periods=n, freq=DateManager().cbd)


def synthetic_window(
    n: int = 253, seed: int = 0, beta: float = BETA
) -> tuple[pd.Series, pd.DataFrame]:
    """
    A ticker whose log return on session t is beta times the index's log
    return on t - 1 plus noise, and the index's levels, on n sessions.
    """
    index = session_index(n)
    rng = np.random.default_rng(seed)
    index_returns = rng.normal(0, 0.01, n)
    ticker_returns = np.empty(n)
    ticker_returns[0] = 0.0
    ticker_returns[1:] = beta * index_returns[:-1] + rng.normal(0, 0.002, n - 1)
    prices = pd.Series(100 * np.exp(np.cumsum(ticker_returns)), index=index, name="AAA")
    levels = pd.DataFrame(
        {"I:SPX": 5000 * np.exp(np.cumsum(index_returns))}, index=index
    )
    return prices, levels


# transform_exogenous


def test_log_return_transform_is_one_difference_of_log_levels():
    levels = pd.Series([100.0, 110.0, 99.0], index=session_index(3))
    out = transform_exogenous(levels, "log_return")
    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1:], [np.log(1.1), np.log(99 / 110)])


def test_every_configurable_transform_is_implemented():
    assert set(EXOGENOUS_TRANSFORMS) <= set(EXOGENOUS_TRANSFORM_FUNCTIONS)


def test_unknown_transform_is_rejected():
    with pytest.raises(ValueError, match="Unsupported exogenous transform"):
        transform_exogenous(pd.Series([1.0, 2.0]), "level")


# build_arimax_window


def test_regressors_are_the_previous_sessions_index_return():
    prices, levels = synthetic_window(n=30)
    window = build_arimax_window(prices, levels, (SPX,))
    spx_returns = np.log(levels["I:SPX"]).diff()
    # One row per log return except the first: the first return's regressor
    # would need the index's close before the window.
    assert window.regressors.index.equals(prices.index[2:])
    assert list(window.regressors.columns) == ["I:SPX"]
    for date in window.regressors.index:
        previous = prices.index[prices.index.get_loc(date) - 1]
        assert window.regressors.loc[date, "I:SPX"] == pytest.approx(
            spx_returns.loc[previous]
        )
    assert window.regressors.notna().all().all()


def test_forecast_regressor_is_the_last_sessions_index_return():
    prices, levels = synthetic_window(n=30)
    window = build_arimax_window(prices, levels, (SPX,))
    spx = levels["I:SPX"]
    assert window.forecast_regressors["I:SPX"] == pytest.approx(
        np.log(spx.iloc[-1] / spx.iloc[-2])
    )


def test_values_after_the_window_are_never_used():
    prices, levels = synthetic_window(n=40)
    window_prices = prices.iloc[:30]
    within = build_arimax_window(window_prices, levels.iloc[:30], (SPX,))
    later = levels.copy()
    later.iloc[30:] *= 3.0  # A large move after the window's last session.
    with_later = build_arimax_window(window_prices, later, (SPX,))
    pd.testing.assert_frame_equal(with_later.regressors, within.regressors)
    pd.testing.assert_series_equal(
        with_later.forecast_regressors, within.forecast_regressors
    )


@pytest.mark.parametrize("position", [-1, -2])
def test_no_window_when_the_forecast_regressor_is_missing(position):
    # The forecast's regressor needs the index's last two closes.
    prices, levels = synthetic_window(n=30)
    levels.iloc[position] = np.nan
    assert build_arimax_window(prices, levels, (SPX,)) is None


def test_no_window_when_the_variable_is_absent():
    prices, levels = synthetic_window(n=30)
    other = levels.rename(columns={"I:SPX": "X"})
    assert build_arimax_window(prices, other, (SPX,)) is None


def test_a_missing_level_inside_the_window_leaves_two_regressors_missing():
    prices, levels = synthetic_window(n=30)
    gap = prices.index[10]
    window = build_arimax_window(prices, levels.drop(index=gap), (SPX,))
    missing = window.regressors["I:SPX"].isna()
    # The index's returns on the gap and the session after it are missing,
    # and each is the regressor of the session after that.
    assert list(window.regressors.index[missing]) == list(prices.index[11:13])


def test_window_needs_a_variable():
    prices, levels = synthetic_window(n=30)
    with pytest.raises(ValueError, match="at least one exogenous variable"):
        build_arimax_window(prices, levels, ())


def test_window_too_short_to_fit_is_rejected():
    prices, levels = synthetic_window(n=2)
    with pytest.raises(ValueError, match="no log return to fit on"):
        build_arimax_window(prices, levels, (SPX,))


# ArimaxModels.fit_arimax (real statsmodels)


def test_fit_includes_the_regression_only_order_and_forecasts_the_next_session():
    prices, levels = synthetic_window()
    model = ArimaxModels()
    window = build_arimax_window(prices, levels, (SPX,))
    pred_date, pred, order = model.fit_arimax(window)
    assert set(model.train_result.index) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert order in set(model.train_result.index)
    assert all(isinstance(value, int) for value in order)
    assert pred_date == DateManager().future_business_day(prices.index[-1], 1)
    assert np.isfinite(pred) and pred > 0


def test_fit_recovers_the_index_loading():
    prices, levels = synthetic_window()
    model = ArimaxModels()
    model.fit_arimax(build_arimax_window(prices, levels, (SPX,)))
    assert model.final_model.params["I:SPX"] == pytest.approx(BETA, abs=0.1)


def test_every_candidate_is_fit_on_the_same_rows():
    # BICs are comparable only on the same sample: the window's log returns
    # without the first, for every candidate.
    prices, levels = synthetic_window(n=60)
    window = build_arimax_window(prices, levels, (SPX,))
    model = ArimaxModels()
    endog_lengths = []
    original = model.fit_arimax_order

    def recording(endog, exog, p, q):
        endog_lengths.append((len(endog), len(exog)))
        assert endog.index.equals(window.regressors.index)
        return original(endog, exog, p, q)

    model.fit_arimax_order = recording
    model.fit_arimax(window)
    assert endog_lengths == [(len(prices) - 2, len(prices) - 2)] * 4


def test_forecast_moves_with_the_last_index_return():
    # With a positive loading, a rise of the index on the last session must
    # give a higher forecast than a fall, everything else equal.
    prices, levels = synthetic_window()
    up, down = levels.copy(), levels.copy()
    up.iloc[-1] = levels.iloc[-2] * 1.02
    down.iloc[-1] = levels.iloc[-2] * 0.98
    _, pred_up, _ = ArimaxModels().fit_arimax(
        build_arimax_window(prices, up, (SPX,))
    )
    _, pred_down, _ = ArimaxModels().fit_arimax(
        build_arimax_window(prices, down, (SPX,))
    )
    assert pred_up > prices.iloc[-1] > pred_down


def test_fit_skips_observations_whose_regressor_is_missing():
    prices, levels = synthetic_window(n=120)
    levels.iloc[50] = np.nan  # Inside the window, not at its end.
    window = build_arimax_window(prices, levels, (SPX,))
    assert window.regressors.isna().any().any()
    model = ArimaxModels()
    _, pred, _ = model.fit_arimax(window)
    assert np.isfinite(pred)
    assert np.isfinite(model.final_model.params).all()


def test_fit_rejects_regressors_that_do_not_line_up():
    prices, levels = synthetic_window(n=60)
    window = build_arimax_window(prices, levels, (SPX,))
    moved = ArimaxWindow(
        prices=window.prices,
        regressors=window.regressors.set_axis(
            window.regressors.index - pd.Timedelta(days=1)
        ),
        forecast_regressors=window.forecast_regressors,
    )
    with pytest.raises(ValueError, match="do not line up"):
        ArimaxModels().fit_arimax(moved)


def test_every_failed_fit_fails_the_window(monkeypatch):
    class FailingArima:
        def __init__(self, **kwargs):
            pass

        def fit(self):
            raise LinAlgError("singular")

    monkeypatch.setattr(arimax_models.tsa, "ARIMA", FailingArima)
    prices, levels = synthetic_window(n=30)
    model = ArimaxModels()
    with pytest.raises(ValueError, match="Every candidate ARIMAX fit failed"):
        model.fit_arimax(build_arimax_window(prices, levels, (SPX,)))
    assert (model.train_result["bic"] == np.inf).all()
    assert set(model.train_result["fit_error"]) == {"LinAlgError"}
