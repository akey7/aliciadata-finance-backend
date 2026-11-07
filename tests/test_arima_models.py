"""
Tests for aliciadata_finance_backend.forecasting.arima_models: the single
differencing step (to_log_returns), fitting each (p, q) candidate once on
the whole training window (fit_order, train_arima), and fit(), which selects
the lowest-BIC candidate and forecasts with its already-fitted model,
including how failed fits are handled.

Most tests replace statsmodels' ARIMA with a fake whose fits return chosen
BICs or fail; two tests use the real model. In-memory only.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import LinAlgError

from aliciadata_finance_backend.data.date_manager import DateManager
from aliciadata_finance_backend.forecasting import arima_models
from aliciadata_finance_backend.forecasting.arima_models import ArimaModels


def session_prices(n: int = 21, seed: int = 0) -> pd.Series:
    """Prices on n consecutive NYSE sessions, with the NYSE frequency set."""
    index = pd.date_range("2026-03-02 17:00", periods=n, freq=DateManager().cbd)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=index)


# to_log_returns: the only differencing step


def test_to_log_returns_is_one_difference_of_log_prices():
    prices = session_prices()
    returns = ArimaModels().to_log_returns(prices)
    expected = np.log(prices.to_numpy()[1:] / prices.to_numpy()[:-1])
    np.testing.assert_allclose(returns.to_numpy(), expected)


def test_to_log_returns_drops_the_leading_nan():
    prices = session_prices()
    returns = ArimaModels().to_log_returns(prices)
    assert len(returns) == len(prices) - 1
    assert returns.index[0] == prices.index[1]
    assert returns.notna().all()


def test_to_log_returns_keeps_index_and_frequency():
    prices = session_prices()
    returns = ArimaModels().to_log_returns(prices)
    assert returns.index.equals(prices.index[1:])
    assert returns.index.freq == prices.index.freq


def test_to_log_returns_keeps_a_missing_session_visible():
    prices = session_prices()
    prices.iloc[5] = np.nan  # A session missing from the data.
    returns = ArimaModels().to_log_returns(prices)
    # Both returns touching the missing price are missing; nothing else is.
    assert returns.isna().sum() == 2
    assert np.isnan(returns.loc[prices.index[5]])
    assert np.isnan(returns.loc[prices.index[6]])


# Fitting candidates with a fake ARIMA


class FakeFitted:
    def __init__(self, order: tuple[int, int], bic: float, next_return: float):
        self.order = order
        self.aic = bic - 1.0
        self.bic = bic
        self._next = next_return

    def forecast(self, steps: int = 1) -> pd.Series:
        return pd.Series([self._next])


def install_fake_arima(monkeypatch, outcomes: dict):
    """
    Replace statsmodels' ARIMA. outcomes maps (p, q) to either a BIC (the
    fit succeeds with that BIC and forecasts a log return of 0.01 * (p + 2q),
    so each order's forecast is distinguishable) or an exception class (the
    fit raises it). Returns a record of every fit: (order, length of endog).
    """
    fits = []

    class FakeARIMA:
        def __init__(self, endog, order):
            self.endog = endog
            self.order = (order[0], order[2])
            assert order[1] == 0  # d = 0: the data are already returns.

        def fit(self):
            fits.append((self.order, len(self.endog)))
            outcome = outcomes[self.order]
            if isinstance(outcome, type) and issubclass(outcome, Exception):
                raise outcome("forced failure")
            p, q = self.order
            return FakeFitted(self.order, outcome, 0.01 * (p + 2 * q))

    monkeypatch.setattr(arima_models.tsa, "ARIMA", FakeARIMA)
    return fits


def test_train_arima_fits_each_candidate_once_on_the_whole_window(monkeypatch):
    returns = ArimaModels().to_log_returns(session_prices(n=252))  # 251 returns
    fits = install_fake_arima(monkeypatch, {(0, 1): 5.0, (1, 0): 3.0, (1, 1): 4.0})
    train_result, models = ArimaModels().train_arima(returns, max_p=2, max_q=2)
    assert fits == [((0, 1), 251), ((1, 0), 251), ((1, 1), 251)]
    assert list(train_result.index) == [(0, 1), (1, 0), (1, 1)]
    assert train_result.loc[(1, 0), "bic"] == 3.0
    assert train_result["fit_error"].isna().all()
    assert set(models) == {(0, 1), (1, 0), (1, 1)}


@pytest.mark.parametrize("error", [ValueError, LinAlgError])
def test_failed_fit_scores_infinite_and_records_the_error(monkeypatch, error):
    returns = ArimaModels().to_log_returns(session_prices())
    install_fake_arima(monkeypatch, {(0, 1): error, (1, 0): 3.0, (1, 1): 4.0})
    train_result, models = ArimaModels().train_arima(returns, max_p=2, max_q=2)
    assert models[(0, 1)] is None
    assert train_result.loc[(0, 1), "bic"] == np.inf
    assert train_result.loc[(0, 1), "aic"] == np.inf
    assert train_result.loc[(0, 1), "fit_error"] == error.__name__
    # The failed candidate is never selected next to working ones.
    assert ArimaModels().select_order(train_result) == (1, 0)


def test_fit_forecasts_with_the_lowest_bic_model_without_refitting(monkeypatch):
    prices = session_prices(n=252)
    fits = install_fake_arima(monkeypatch, {(0, 1): 5.0, (1, 0): 3.0, (1, 1): 4.0})
    am = ArimaModels()
    pred_date, pred, order = am.fit(prices, max_p=2, max_q=2)
    assert order == (1, 0)
    assert len(fits) == 3  # One fit per candidate; the winner is not refit.
    assert am.final_model.order == (1, 0)
    # (1, 0)'s fake forecast is a log return of 0.01.
    assert pred == pytest.approx(prices.iloc[-1] * np.exp(0.01))
    assert pred_date == DateManager().future_business_day(prices.index[-1], 1)
    assert am.order_tie_broken is False


def test_fit_reports_a_tie_break(monkeypatch):
    install_fake_arima(monkeypatch, {(0, 1): 3.0, (1, 0): 3.0, (1, 1): 4.0})
    am = ArimaModels()
    _, _, order = am.fit(session_prices(), max_p=2, max_q=2)
    assert order == (0, 1)  # Same BIC and p + q; lower p wins.
    assert am.order_tie_broken is True


def test_fit_raises_when_every_candidate_fails(monkeypatch):
    install_fake_arima(
        monkeypatch, {(0, 1): ValueError, (1, 0): LinAlgError, (1, 1): ValueError}
    )
    with pytest.raises(ValueError, match="Every candidate ARIMA fit failed"):
        ArimaModels().fit(session_prices(), max_p=2, max_q=2)


# The real statsmodels ARIMA


def test_fit_with_real_arima_on_a_trading_year():
    prices = session_prices(n=252)  # One trading year: 251 returns.
    am = ArimaModels()
    pred_date, pred, order = am.fit(prices, max_p=2, max_q=2)
    assert order in {(0, 1), (1, 0), (1, 1)}
    assert np.isfinite(pred) and pred > 0
    assert np.isfinite(am.train_result.loc[order, "bic"])
    assert am.train_result.loc[order, "bic"] == am.train_result["bic"].min()


def test_real_arima_bic_matches_a_direct_fit():
    import warnings

    import statsmodels.tsa.api as tsa

    returns = ArimaModels().to_log_returns(session_prices(n=252))
    model, row = ArimaModels().fit_order(returns, 1, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = tsa.ARIMA(endog=returns, order=(1, 0, 0)).fit()
    assert row["bic"] == pytest.approx(direct.bic)
    assert row["fit_error"] is None
    assert model is not None
