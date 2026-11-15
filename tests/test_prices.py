"""
Tests for aliciadata_finance_backend.data.prices. In-memory only.
"""

import pandas as pd
import pytest

from conftest import SYNTHETIC_SESSIONS, SYNTHETIC_TICKERS
from aliciadata_finance_backend.data.prices import (
    last_n_sessions,
    pivot_close_wide,
    sessions_through,
    weekdays_after,
)


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


def test_sessions_through_keeps_dates_up_to_and_including_as_of(long_prices):
    wide = pivot_close_wide(long_prices)
    as_of = wide.index[9]  # Labeled 17:00; the time of day is ignored.
    kept = sessions_through(wide, as_of.normalize())
    assert list(kept.index) == list(wide.index[:10])
    assert kept.equals(wide.iloc[:10])


def test_sessions_through_ignores_the_time_of_day_of_as_of(long_prices):
    wide = pivot_close_wide(long_prices)
    as_of = wide.index[9].normalize() + pd.Timedelta(hours=23, minutes=59)
    assert len(sessions_through(wide, as_of)) == 10


def test_sessions_through_a_weekend_ends_on_the_friday_before(long_prices):
    wide = pivot_close_wide(long_prices)
    days = pd.date_range(wide.index[0].normalize(), periods=14)
    saturday = next(d for d in days if d.dayofweek == 5)
    kept = sessions_through(wide, saturday)
    assert kept.index[-1].normalize() == saturday.normalize() - pd.Timedelta(days=1)


def test_sessions_through_before_the_data_is_empty(long_prices):
    wide = pivot_close_wide(long_prices)
    assert sessions_through(wide, wide.index[0] - pd.Timedelta(days=1)).empty


def test_sessions_through_returns_a_sorted_copy(long_prices):
    wide = pivot_close_wide(long_prices)
    kept = sessions_through(wide.iloc[::-1], wide.index[-1])
    assert kept.index.is_monotonic_increasing
    kept.iloc[0, 0] = -1.0
    assert wide.iloc[0, 0] != -1.0


@pytest.mark.parametrize(
    "as_of, last, expected",
    [
        ("2026-11-06", "2026-11-13", 5),  # Friday to Friday: Mon-Fri.
        ("2026-11-13", "2026-11-13", 0),  # Same day.
        ("2026-11-13", "2026-11-06", 0),  # Last before as-of.
        ("2026-11-13", "2026-11-16", 1),  # Over a weekend.
        ("2026-11-25", "2026-11-27", 2),  # Counts Thanksgiving, an NYSE holiday.
        ("2026-11-06 17:00", "2026-11-09 23:59", 1),  # Times of day ignored.
    ],
)
def test_weekdays_after(as_of, last, expected):
    assert weekdays_after(pd.Timestamp(as_of), pd.Timestamp(last)) == expected
