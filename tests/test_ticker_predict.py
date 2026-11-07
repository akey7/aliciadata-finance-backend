"""
Tests for ArimaWalkForward.train_arima_models() date handling.

ARIMA fitting is replaced by a stub that keeps the real date logic
(ArimaModels.next_trading_day, i.e. the NYSE calendar) and
returns a dummy forecast, so these tests run fast and check only how forecast
rows are dated and assembled. The stub replaces ArimaModels only in this
process, so the default worker count is forced to 1 (no worker processes).
In-memory only.
"""

import numpy as np
import pandas as pd
import pytest

from aliciadata_finance_backend.config import TimeSeriesConfig
from aliciadata_finance_backend.data.date_manager import DateManager, nyse_holidays
from aliciadata_finance_backend.forecasting import ticker_predict
from aliciadata_finance_backend.forecasting.arima_models import ArimaModels
from aliciadata_finance_backend.forecasting.ticker_predict import ArimaWalkForward


STUB_UPLIFT = 1.01  # Stub ARIMA forecast = 1 % above the last observed close.


def stub_order(last_date: pd.Timestamp) -> tuple[int, int]:
    """A (p, q) that varies from window to window, never (0, 0)."""
    return (1, 0) if last_date.day % 2 else (0, 1)


class StubArimaModels:
    """
    Real pred_date logic, dummy forecast (STUB_UPLIFT x last observed price,
    deliberately different from the naive forecast), and an order
    that depends on the window (stub_order). Records every series passed to
    fit() in StubArimaModels.fitted, and (ticker, pred_date, order) for every
    forecast in StubArimaModels.orders.
    """

    fitted: list[pd.Series] = []
    orders: list[tuple[str, pd.Timestamp, tuple[int, int]]] = []

    def __init__(self):
        self._dates = ArimaModels()
        self.order_tie_broken = False

    def fit(self, original_ts, max_p=5, max_q=5):
        StubArimaModels.fitted.append(original_ts)
        pred_date = self._dates.next_trading_day(original_ts.index[-1])
        order = stub_order(original_ts.index[-1])
        StubArimaModels.orders.append((original_ts.name, pred_date, order))
        return pred_date, STUB_UPLIFT * float(original_ts.dropna().iloc[-1]), order


@pytest.fixture(autouse=True)
def stub_arima(monkeypatch):
    StubArimaModels.fitted = []
    StubArimaModels.orders = []
    monkeypatch.setattr(ticker_predict, "ArimaModels", StubArimaModels)
    # Worker processes would import the real ArimaModels, not the stub.
    monkeypatch.setattr(ticker_predict, "default_worker_count", lambda: 1)


# Most tests check dates and row assembly, not the window length, so they use
# 40 sessions of prices and a 20-return window. The configured one-year window
# is tested separately below.
SHORT_WINDOW_RETURNS = 20


class ShortWindowWalkForward(ArimaWalkForward):
    def __init__(self, n_workers: int | None = None):
        super().__init__(training_window_returns=SHORT_WINDOW_RETURNS, n_workers=n_workers)


def nyse_prices(last_session: str, n_sessions: int = 40) -> pd.DataFrame:
    """Wide prices on the n_sessions NYSE sessions ending at last_session (17:00)."""
    sessions = pd.date_range(
        end=pd.Timestamp(last_session), periods=n_sessions, freq=DateManager().cbd
    )
    index = pd.DatetimeIndex([d.replace(hour=17) for d in sessions])
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "AAA": 100 + rng.normal(0, 1, n_sessions).cumsum(),
            "BBB": 50 + rng.normal(0, 1, n_sessions).cumsum(),
        },
        index=index,
    )


@pytest.mark.parametrize(
    "last_session, expected_final",
    [
        ("2026-04-02", "2026-04-06 17:00"),  # Skips Good Friday (NYSE closed).
        ("2026-10-09", "2026-10-12 17:00"),  # Columbus Day: NYSE open.
        ("2026-11-25", "2026-11-27 17:00"),  # Skips Thanksgiving.
        ("2026-10-14", "2026-10-15 17:00"),  # Ordinary weekday.
    ],
)
def test_final_row_is_next_nyse_trading_day(last_session, expected_final):
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices(last_session))
    assert forecasts.index[-1] == pd.Timestamp(expected_final)


def test_final_row_has_forecasts_but_no_actuals():
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    last = forecasts.iloc[-1]
    assert last[["AAA", "BBB"]].isna().all()
    assert last[["AAA_arima", "BBB_arima"]].notna().all()
    # Every earlier row has an actual price to compare against.
    assert forecasts.iloc[:-1][["AAA", "BBB"]].notna().all().all()


def test_every_pred_date_is_a_unique_nyse_session_after_the_window():
    prices = nyse_prices("2026-04-02")
    forecasts = ShortWindowWalkForward().train_arima_models(prices)
    dates = forecasts.index.normalize()
    assert forecasts.index.is_unique
    assert forecasts.index.is_monotonic_increasing
    assert (dates.dayofweek < 5).all()
    assert not dates.isin(nyse_holidays()).any()
    # The first forecast comes after the first training window, not inside it.
    assert forecasts.index[0] > prices.index[0]


# The production window: one trading year per forecast


def test_one_trading_year_window_with_twenty_backtest_days():
    # A representative configuration (252 closes, 20 scored days), independent
    # of the values in config/modeling_and_optimization.yml.
    ts_config = TimeSeriesConfig(training_window_length=252, evaluation_window_length=20)
    returns = ts_config.training_window_returns
    eval_days = ts_config.evaluation_window_length
    assert returns == 251
    assert ts_config.arima_window_sessions == 272
    prices = nyse_prices("2026-10-15", n_sessions=ts_config.arima_window_sessions)
    forecasts = ArimaWalkForward(training_window_returns=returns).train_arima_models(prices)
    # 20 backtest rows with actuals plus the live row.
    assert len(forecasts) == eval_days + 1
    assert forecasts.iloc[:-1]["AAA"].notna().all()
    # Every model saw 252 closes (251 returns), ending on its forecast origin.
    assert len(StubArimaModels.fitted) == 2 * (eval_days + 1)
    assert all(len(ts) == ts_config.training_window_length for ts in StubArimaModels.fitted)
    # The live forecast's window is the last 252 sessions of the data.
    live = [ts for ts in StubArimaModels.fitted if ts.name == "AAA"][-1]
    assert live.index[-1] == prices.index[-1]
    assert live.index[0] == prices.index[-ts_config.training_window_length]


# Training windows are resampled to NYSE sessions, not pandas "B" weekdays.
# Data ending 2026-04-02 with 40 sessions starts in early February, so the
# windows span Presidents Day (2026-02-16), an NYSE holiday on a Monday.

PRESIDENTS_DAY = pd.Timestamp("2026-02-16")


def test_training_windows_skip_nyse_holidays():
    prices = nyse_prices("2026-04-02")
    assert prices.index[0].normalize() < PRESIDENTS_DAY  # The data spans it.
    ShortWindowWalkForward().train_arima_models(prices)
    assert StubArimaModels.fitted  # fit() was called.
    spans_holiday = 0
    for ts in StubArimaModels.fitted:
        dates = ts.index.normalize()
        assert not dates.isin(nyse_holidays()).any()
        assert ts.notna().all()  # Complete data: no inserted missing rows.
        if dates[0] < PRESIDENTS_DAY < dates[-1]:
            spans_holiday += 1
    assert spans_holiday > 0  # At least one window actually crossed it.


def test_training_windows_use_the_nyse_session_frequency():
    ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    for ts in StubArimaModels.fitted:
        assert isinstance(ts.index.freq, pd.offsets.CustomBusinessDay)
        # Consecutive entries are consecutive NYSE sessions.
        expected = pd.date_range(ts.index[0], periods=len(ts), freq=DateManager().cbd)
        assert ts.index.equals(expected)


def test_missing_trading_session_still_shows_as_missing():
    # A session absent from the data (e.g. a download gap) must remain
    # visible to the model as a missing value rather than silently vanish.
    prices = nyse_prices("2026-04-02")
    gap = pd.Timestamp("2026-03-10 17:00")  # An ordinary trading Tuesday.
    ShortWindowWalkForward().train_arima_models(prices.drop(index=gap))
    windows_with_gap = [ts for ts in StubArimaModels.fitted if gap in ts.index]
    assert windows_with_gap
    assert all(np.isnan(ts.loc[gap]) for ts in windows_with_gap)


def test_statsmodels_arima_fits_and_forecasts_on_nyse_frequency():
    # The real model: an ARIMA on log returns indexed by NYSE
    # sessions must fit and produce a finite one-step forecast.
    import warnings

    import statsmodels.tsa.api as tsa

    prices = nyse_prices("2026-04-02", n_sessions=30)["AAA"]
    window = prices.asfreq(DateManager().cbd)
    log_returns = ArimaModels().to_log_returns(window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tsa.ARIMA(endog=log_returns, order=(1, 0, 0)).fit()
        forecast = model.forecast(steps=1)
    assert len(forecast) == 1
    assert np.isfinite(forecast.iloc[0])


# ARIMA order columns: T_arima_p, T_arima_d, and T_arima_q


def test_order_columns_follow_each_forecast_column():
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    assert list(forecasts.columns) == [
        "AAA_arima", "AAA_arima_p", "AAA_arima_d", "AAA_arima_q", "AAA_naive", "AAA",
        "BBB_arima", "BBB_arima_p", "BBB_arima_d", "BBB_arima_q", "BBB_naive", "BBB",
    ]


def test_order_columns_hold_the_order_of_each_rows_model():
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    assert StubArimaModels.orders
    for ticker, pred_date, (p, q) in StubArimaModels.orders:
        assert forecasts.loc[pred_date, f"{ticker}_arima_p"] == p
        assert forecasts.loc[pred_date, f"{ticker}_arima_q"] == q
    # Both stub orders occur, so a constant column would not pass.
    assert set(forecasts["AAA_arima_p"]) == {0, 1}


def test_order_columns_are_integers_on_every_row_including_the_last():
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    order_columns = [c for c in forecasts.columns if c.endswith(("_arima_p", "_arima_d", "_arima_q"))]
    assert len(order_columns) == 6
    for column in order_columns:
        assert pd.api.types.is_integer_dtype(forecasts[column])
        assert forecasts[column].notna().all()


def test_order_columns_round_trip_through_csv_as_integers(tmp_path):
    # The uploaded CSV is written by load_or_train_forecasts().
    path = str(tmp_path / "all_forecasts.csv")
    ticker_predict.load_or_train_forecasts(
        nyse_prices("2026-04-02"), path, trainer=ShortWindowWalkForward()
    )
    loaded = pd.read_csv(path, index_col="pred_date", parse_dates=True)
    assert pd.api.types.is_integer_dtype(loaded["AAA_arima_p"])
    assert pd.api.types.is_integer_dtype(loaded["BBB_arima_q"])
    # Forecast columns are still found by the "_arima" suffix alone.
    assert [c for c in loaded.columns if c.endswith("_arima")] == ["AAA_arima", "BBB_arima"]


# ArimaModels.select_order (pure; no model fitting)


def order_results(bics: dict[tuple[int, int], float]) -> pd.DataFrame:
    """train_arima()-shaped results: (p, q) index and a bic column."""
    index = pd.MultiIndex.from_tuples(list(bics), names=["p", "q"])
    return pd.DataFrame({"bic": list(bics.values())}, index=index)


def test_select_order_picks_lowest_bic_and_returns_ints():
    results = order_results({(0, 1): -50.0, (1, 0): -40.0, (1, 1): -60.0})
    assert ArimaModels().select_order(results) == (1, 1)
    p, q = ArimaModels().select_order(results)
    assert type(p) is int and type(q) is int
    assert not ArimaModels().order_was_tie_broken(results)


def test_select_order_lower_bic_wins_even_with_more_parameters():
    # Parsimony only breaks exact ties; BIC already penalizes parameters.
    results = order_results({(1, 1): -60.0, (0, 1): -59.9, (1, 0): -59.9})
    assert ArimaModels().select_order(results) == (1, 1)


def test_select_order_equal_bic_goes_to_fewer_parameters():
    results = order_results({(1, 1): -50.0, (1, 0): -50.0, (0, 1): -40.0})
    # (1, 1) comes first, but (1, 0) has fewer parameters.
    assert ArimaModels().select_order(results) == (1, 0)
    assert ArimaModels().order_was_tie_broken(results)


def test_select_order_equal_bic_and_parameters_goes_to_lower_p():
    results = order_results({(1, 0): -50.0, (0, 1): -50.0, (1, 1): -40.0})
    assert ArimaModels().select_order(results) == (0, 1)
    assert ArimaModels().order_was_tie_broken(results)


def test_select_order_skips_failed_fits():
    inf = float("inf")
    results = order_results({(0, 1): inf, (1, 1): inf, (1, 0): -10.0})
    assert ArimaModels().select_order(results) == (1, 0)
    assert not ArimaModels().order_was_tie_broken(results)
    # If only failed candidates remain, they tie and the parsimony rule
    # picks one; ArimaModels.fit() then raises because it has no model.
    failed_only = results.drop(index=(1, 0))
    assert ArimaModels().select_order(failed_only) == (0, 1)
    assert ArimaModels().order_was_tie_broken(failed_only)


@pytest.mark.parametrize("seed", range(5))
def test_select_order_does_not_depend_on_row_order(seed):
    results = order_results({(1, 1): -50.0, (1, 0): -50.0, (0, 1): -50.0})
    shuffled = results.sample(frac=1, random_state=seed)
    # Three-way tie: fewer parameters, then lower p.
    assert ArimaModels().select_order(shuffled) == (0, 1)


def test_d_column_is_one_on_every_row_and_survives_csv(tmp_path):
    # ARIMA(p, 0, q) on log returns is ARIMA(p, 1, q) on log prices.
    from aliciadata_finance_backend.forecasting.arima_models import (
        PRICE_DIFFERENCING_ORDER,
    )

    assert PRICE_DIFFERENCING_ORDER == 1
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    for ticker in ("AAA", "BBB"):
        assert (forecasts[f"{ticker}_arima_d"] == 1).all()  # Including the last row.
    path = str(tmp_path / "all_forecasts.csv")
    forecasts.to_csv(path, index=True)
    loaded = pd.read_csv(path, index_col="pred_date", parse_dates=True)
    assert pd.api.types.is_integer_dtype(loaded["AAA_arima_d"])
    assert (loaded["AAA_arima_d"] == 1).all()
    # The suffix rule still finds exactly the forecast columns.
    assert [c for c in loaded.columns if c.endswith("_arima")] == ["AAA_arima", "BBB_arima"]


# Naive baseline: T_naive


def previous_observed_close(prices: pd.Series, pred_date: pd.Timestamp) -> float:
    """The most recent observed close strictly before pred_date."""
    return float(prices[prices.index < pred_date].dropna().iloc[-1])


def test_naive_is_previous_trading_days_close():
    prices = nyse_prices("2026-04-02")
    forecasts = ShortWindowWalkForward().train_arima_models(prices)
    for pred_date, row in forecasts.iterrows():
        previous_session = DateManager().past_business_day(pred_date, 1)
        assert row["AAA_naive"] == pytest.approx(prices.loc[previous_session, "AAA"])
        assert row["BBB_naive"] == pytest.approx(prices.loc[previous_session, "BBB"])


def test_naive_on_final_row_is_last_available_close():
    prices = nyse_prices("2026-04-02")
    forecasts = ShortWindowWalkForward().train_arima_models(prices)
    assert forecasts["AAA_naive"].iloc[-1] == pytest.approx(prices["AAA"].iloc[-1])
    assert forecasts["BBB_naive"].iloc[-1] == pytest.approx(prices["BBB"].iloc[-1])


def test_naive_equals_previous_rows_actual():
    # With complete data, today's naive forecast is yesterday's actual close.
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    np.testing.assert_allclose(
        forecasts["AAA_naive"].iloc[1:].to_numpy(),
        forecasts["AAA"].iloc[:-1].to_numpy(),
    )


def test_naive_uses_most_recent_observed_close_when_a_session_is_missing():
    prices = nyse_prices("2026-04-02")
    gap = pd.Timestamp("2026-03-10 17:00")
    gapped = prices.drop(index=gap)
    forecasts = ShortWindowWalkForward().train_arima_models(gapped)
    for pred_date, row in forecasts.iterrows():
        assert row["AAA_naive"] == pytest.approx(
            previous_observed_close(gapped["AAA"], pred_date)
        )
    # The row forecasting the missing day uses the close before it.
    assert forecasts.loc[gap, "AAA_naive"] == pytest.approx(
        prices.loc[pd.Timestamp("2026-03-09 17:00"), "AAA"]
    )


def test_naive_never_uses_the_forecast_days_own_close():
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    backtest = forecasts.iloc[:-1]
    assert (backtest["AAA_naive"] != backtest["AAA"]).all()


def test_naive_is_independent_of_the_arima_forecast():
    # The stub's ARIMA forecast is STUB_UPLIFT x the last close; naive is the
    # last close itself, so the two columns must differ by exactly that factor.
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    np.testing.assert_allclose(
        forecasts["AAA_arima"].to_numpy(), STUB_UPLIFT * forecasts["AAA_naive"].to_numpy()
    )


def test_naive_is_float_on_every_row_and_survives_csv(tmp_path):
    forecasts = ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    for ticker in ("AAA", "BBB"):
        assert pd.api.types.is_float_dtype(forecasts[f"{ticker}_naive"])
        assert forecasts[f"{ticker}_naive"].notna().all()  # Including the last row.
    path = str(tmp_path / "all_forecasts.csv")
    forecasts.to_csv(path, index=True)
    loaded = pd.read_csv(path, index_col="pred_date", parse_dates=True)
    np.testing.assert_allclose(loaded["AAA_naive"], forecasts["AAA_naive"])
    # Suffix rules still separate the column kinds.
    assert [c for c in loaded.columns if c.endswith("_arima")] == ["AAA_arima", "BBB_arima"]
    assert [c for c in loaded.columns if c.endswith("_naive")] == ["AAA_naive", "BBB_naive"]


# Missing data. A whole session absent from the data, or one ticker's close
# missing on a session that exists for the others.

GAP = pd.Timestamp("2026-03-10 17:00")  # An ordinary trading Tuesday.
AFTER_GAP = pd.Timestamp("2026-03-11 17:00")


def test_training_windows_end_on_observed_dates_only():
    index = nyse_prices("2026-04-02").drop(index=GAP).index
    windows = ArimaWalkForward.training_window_start_end(index, num_days=20)
    ends = [end for _, end in windows]
    starts = [start for start, _ in windows]
    assert ends == list(index[20:])
    assert starts == list(index[:-20])
    assert GAP not in ends
    assert len(set(ends)) == len(ends)


def test_training_windows_on_complete_data_are_one_per_session():
    index = nyse_prices("2026-04-02").index
    windows = ArimaWalkForward.training_window_start_end(index, num_days=20)
    assert len(windows) == len(index) - 20
    for start, end in windows:
        assert len(index[(index >= start) & (index <= end)]) == 21  # 20 returns


def test_training_windows_need_more_dates_than_the_window():
    index = nyse_prices("2026-04-02", n_sessions=20).index
    with pytest.raises(ValueError):
        ArimaWalkForward.training_window_start_end(index, num_days=20)


def test_whole_missing_session_gives_unique_dates_and_no_two_step_row():
    forecasts = ShortWindowWalkForward().train_arima_models(
        nyse_prices("2026-04-02").drop(index=GAP)
    )
    assert forecasts.index.is_unique
    assert GAP in forecasts.index  # Forecast from the close before the gap.
    assert np.isnan(forecasts.loc[GAP, "AAA"])  # No actual for the missing day.
    assert AFTER_GAP not in forecasts.index  # Would need the missing close.


def test_one_tickers_missing_close_leaves_its_arima_and_orders_empty():
    prices = nyse_prices("2026-04-02")
    prices.loc[GAP, "AAA"] = np.nan  # Only AAA is missing on GAP.
    forecasts = ShortWindowWalkForward().train_arima_models(prices)

    # Read single cells, not the whole row: a row mixing float and nullable
    # Int64 columns is upcast to a nullable dtype, turning NaN into pd.NA.
    def cell(column: str):
        return forecasts.loc[AFTER_GAP, column]  # The window ending on GAP.

    assert pd.isna(cell("AAA_arima"))
    assert pd.isna(cell("AAA_arima_p"))
    assert pd.isna(cell("AAA_arima_d"))
    assert pd.isna(cell("AAA_arima_q"))
    # The naive forecast still uses the most recent observed close.
    assert cell("AAA_naive") == pytest.approx(
        prices.loc[pd.Timestamp("2026-03-09 17:00"), "AAA"]
    )
    # BBB is unaffected on the same row.
    assert np.isfinite(cell("BBB_arima"))
    assert cell("BBB_arima_d") == 1
    # Every other AAA row still has a forecast and its order.
    others = forecasts.drop(index=AFTER_GAP)
    assert others["AAA_arima"].notna().all()
    assert others["AAA_arima_d"].notna().all()
    # No model was fit on a window whose last close is missing.
    assert all(pd.notna(ts.iloc[-1]) for ts in StubArimaModels.fitted)


def test_empty_orders_round_trip_through_csv(tmp_path):
    prices = nyse_prices("2026-04-02")
    prices.loc[GAP, "AAA"] = np.nan
    forecasts = ShortWindowWalkForward().train_arima_models(prices)
    assert pd.api.types.is_integer_dtype(forecasts["AAA_arima_p"])  # Nullable Int64.
    path = str(tmp_path / "all_forecasts.csv")
    forecasts.to_csv(path, index=True)
    loaded = pd.read_csv(path, index_col="pred_date", parse_dates=True)
    assert pd.isna(loaded.loc[AFTER_GAP, "AAA_arima_p"])
    filled = loaded["AAA_arima_p"].dropna()
    assert (filled == filled.round()).all()  # Whole numbers everywhere else.
    assert loaded["BBB_arima_p"].notna().all()


# Worker count


@pytest.fixture
def recorded_workers(monkeypatch):
    """Record the n_workers passed to fit_windows, but fit in this process."""
    recorded = []
    in_process_fit_windows = ticker_predict.fit_windows

    def recording_fit_windows(windows, max_p, max_q, n_workers):
        recorded.append(n_workers)
        return in_process_fit_windows(windows, max_p, max_q, n_workers=1)

    monkeypatch.setattr(ticker_predict, "fit_windows", recording_fit_windows)
    return recorded


def test_explicit_worker_count_is_passed_to_the_pool(recorded_workers):
    ShortWindowWalkForward(n_workers=3).train_arima_models(nyse_prices("2026-04-02"))
    assert recorded_workers == [3]


def test_default_worker_count_is_used_when_none_is_given(monkeypatch, recorded_workers):
    monkeypatch.setattr(ticker_predict, "default_worker_count", lambda: 5)
    ShortWindowWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    assert recorded_workers == [5]


def test_all_windows_are_fit_in_one_call_in_output_order(recorded_workers):
    prices = nyse_prices("2026-04-02")
    prices.loc[GAP, "AAA"] = np.nan  # This window is skipped, not fit.
    forecasts = ShortWindowWalkForward().train_arima_models(prices)
    assert len(recorded_workers) == 1  # One pool for the whole run.
    # Each forecast is matched with its own window: the stub's forecast is
    # STUB_UPLIFT x that window's last close, which is the row's naive value.
    for ticker in ("AAA", "BBB"):
        fitted = forecasts[f"{ticker}_arima"].dropna()
        naive = forecasts.loc[fitted.index, f"{ticker}_naive"]
        np.testing.assert_allclose(fitted, STUB_UPLIFT * naive)
