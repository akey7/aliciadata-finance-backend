"""
Tests for the NYSE business-day calendar in DateManager and ArimaModels.

Uses the exchange_calendars NYSE calendar bundled with the package: no
network, no S3, and no reads or writes to input/ or output/.

Reference dates (2026):
- Good Friday, 2026-04-03: NYSE closed, not a federal holiday.
- Juneteenth, 2026-06-19 (Friday): NYSE closed.
- Columbus Day, 2026-10-12 (Monday): NYSE open, federal holiday.
- Veterans Day, 2026-11-11 (Wednesday): NYSE open, federal holiday.
- Thanksgiving, 2026-11-26 (Thursday): NYSE closed.
"""

import pandas as pd
import pytest

from fuzzy_system_research.data.date_manager import (
    DateManager,
    nyse_holidays,
    nyse_sessions_ending,
)
from fuzzy_system_research.forecasting.arima_models import ArimaModels


@pytest.fixture(scope="module")
def dm() -> DateManager:
    return DateManager()


@pytest.mark.parametrize(
    "closed_day", ["2026-04-03", "2026-06-19", "2026-11-26", "2026-12-25"]
)
def test_nyse_closures_are_holidays(closed_day):
    assert pd.Timestamp(closed_day) in nyse_holidays()


@pytest.mark.parametrize("open_day", ["2026-10-12", "2026-11-11"])
def test_federal_only_holidays_are_trading_days(open_day):
    assert pd.Timestamp(open_day) not in nyse_holidays()


def test_nyse_holidays_contain_no_weekends():
    assert (nyse_holidays().dayofweek < 5).all()


def test_nyse_sessions_ending_skips_closures_and_drops_the_time():
    sessions = nyse_sessions_ending(pd.Timestamp("2026-04-06 17:00:00"), 3)
    # Good Friday, 2026-04-03, and the weekend are skipped.
    assert list(sessions) == [
        pd.Timestamp("2026-04-01"),
        pd.Timestamp("2026-04-02"),
        pd.Timestamp("2026-04-06"),
    ]


def test_nyse_sessions_ending_includes_federal_only_holidays():
    sessions = nyse_sessions_ending(pd.Timestamp("2026-10-13"), 2)
    assert list(sessions) == [pd.Timestamp("2026-10-12"), pd.Timestamp("2026-10-13")]


def test_past_business_day_includes_columbus_day(dm):
    # The day after Columbus Day: the previous trading day is Columbus Day
    # itself, not the Friday before (the old federal-calendar behavior).
    assert dm.past_business_day(pd.Timestamp("2026-10-13"), 1) == pd.Timestamp(
        "2026-10-12"
    )


def test_past_business_day_skips_good_friday(dm):
    assert dm.past_business_day(pd.Timestamp("2026-04-06"), 1) == pd.Timestamp(
        "2026-04-02"
    )


def test_past_business_day_from_weekend_is_friday(dm):
    assert dm.past_business_day(pd.Timestamp("2026-10-11"), 1) == pd.Timestamp(
        "2026-10-09"
    )


def test_future_business_day_skips_thanksgiving(dm):
    assert dm.future_business_day(pd.Timestamp("2026-11-25"), 1) == pd.Timestamp(
        "2026-11-27"
    )


def test_create_business_day_range_skips_good_friday(dm):
    days = dm.create_business_day_range(pd.Timestamp("2026-04-01"), 4)
    expected = pd.to_datetime(["2026-04-01", "2026-04-02", "2026-04-06", "2026-04-07"])
    assert list(days) == list(expected)


def test_arima_next_trading_day_is_nyse_and_keeps_time():
    # Forecast dates carry a fixed 17:00 time label that must be preserved.
    am = ArimaModels()
    friday_before = pd.Timestamp("2026-10-09 17:00:00")
    assert am.next_trading_day(friday_before) == pd.Timestamp(
        "2026-10-12 17:00:00"
    )
    thursday_before_good_friday = pd.Timestamp("2026-04-02 17:00:00")
    assert am.next_trading_day(
        thursday_before_good_friday
    ) == pd.Timestamp("2026-04-06 17:00:00")
