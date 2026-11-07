"""
Tests for aliciadata_finance_backend.data.prices. In-memory only.
"""

import pandas as pd
import pytest

from conftest import SYNTHETIC_SESSIONS, SYNTHETIC_TICKERS
from aliciadata_finance_backend.data.prices import last_n_sessions, pivot_close_wide


def test_pivot_close_wide_shape_columns_and_order(long_prices):
    wide = pivot_close_wide(long_prices)
    assert wide.shape == (SYNTHETIC_SESSIONS, len(SYNTHETIC_TICKERS))
    assert list(wide.columns) == sorted(SYNTHETIC_TICKERS)
    assert wide.index.is_monotonic_increasing
    assert isinstance(wide.index, pd.DatetimeIndex)


def test_pivot_close_wide_sets_fixed_17h_label(long_prices):
    wide = pivot_close_wide(long_prices)
    assert set(wide.index.hour) == {17}
    assert set(wide.index.minute) == {0}


def test_pivot_close_wide_values_match_long(long_prices):
    wide = pivot_close_wide(long_prices)
    first = long_prices.iloc[0]
    date = pd.Timestamp(first["datetime"]).replace(hour=17, minute=0, second=0)
    assert wide.loc[date, first["ticker"]] == pytest.approx(first["close"])


def test_pivot_close_wide_accepts_datetime_index(long_prices):
    # A fresh download has datetime as the index rather than a column.
    wide = pivot_close_wide(long_prices.set_index("datetime"))
    assert wide.shape == (SYNTHETIC_SESSIONS, len(SYNTHETIC_TICKERS))


def test_last_n_sessions_keeps_most_recent_rows(long_prices):
    wide = pivot_close_wide(long_prices)
    window = last_n_sessions(wide, 40)
    assert len(window) == 40
    assert window.index[-1] == wide.index[-1]
    assert window.index[0] == wide.index[-40]
    pd.testing.assert_frame_equal(window, wide.iloc[-40:])


def test_last_n_sessions_with_fewer_rows_returns_all(long_prices):
    wide = pivot_close_wide(long_prices)
    assert len(last_n_sessions(wide, 1000)) == len(wide)


def test_last_n_sessions_sorts_unsorted_input(long_prices):
    wide = pivot_close_wide(long_prices)
    shuffled = wide.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(last_n_sessions(shuffled, 5), wide.iloc[-5:])


def test_last_n_sessions_returns_a_copy(long_prices):
    wide = pivot_close_wide(long_prices)
    window = last_n_sessions(wide, 5)
    window.iloc[0, 0] = -1.0
    assert (wide.to_numpy() != -1.0).all()


@pytest.mark.parametrize("bad_n", [0, -3])
def test_last_n_sessions_rejects_non_positive(long_prices, bad_n):
    with pytest.raises(ValueError):
        last_n_sessions(pivot_close_wide(long_prices), bad_n)
