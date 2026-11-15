"""
Tests for aliciadata_finance_backend.backfill: the session counts, the dates
to backfill, the early check of the price data, the one download and the
risk-free cache warming (with doubles for the download, the NYSE calendar,
and FRED), the completeness checks, the log, and the loop. Files are written
only under pytest's tmp_path; nothing touches the network, a bucket, or the
real input/ and output/ folders.
"""

import csv
import os

import numpy as np
import pandas as pd
import pytest

from conftest import SYNTHETIC_TICKERS, make_long_prices
from aliciadata_finance_backend.backfill import (
    BACKFILL_LOG_COLUMNS,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    DayResult,
    append_backfill_log,
    backfill_dates,
    backfill_download_sessions,
    backfill_log_row,
    backfill_sessions,
    bucket_date_is_complete,
    check_backfill_prices,
    format_backfill_summary,
    local_date_is_complete,
    prepare_backfill,
    run_backfill,
)
from aliciadata_finance_backend.config import PortfolioBacktestConfig, TimeSeriesConfig
from aliciadata_finance_backend.data.prices import pivot_close_wide
from aliciadata_finance_backend.memory_usage import BYTES_PER_MIB, PeakMemory
from aliciadata_finance_backend.pipeline import (
    first_rebalance_date,
    price_download_sessions,
    required_sessions,
)
from aliciadata_finance_backend.storage.layout import (
    ALL_FORECASTS_FILENAME,
    OPTIMIZATION_METADATA_FILENAME,
    PORTFOLIO_OPTIMIZATION,
    TIME_SERIES,
    local_output_path,
)

# Small windows keep the synthetic data short: ARIMA needs 40 closes + 10
# scored days (50 sessions); the backtest 31 closes + 10 days (41 sessions).
TS_CONFIG = TimeSeriesConfig(training_window_length=40, evaluation_window_length=10)
BT_CONFIG = PortfolioBacktestConfig(
    training_window_returns=30,
    evaluation_window_length=10,
    rebalance_interval_sessions=5,
)
PAST_DAYS = 4
REQUIRED = required_sessions(TS_CONFIG, BT_CONFIG)  # 50
NEEDED = backfill_sessions(TS_CONFIG, BT_CONFIG, PAST_DAYS)  # 53
DOWNLOAD = backfill_download_sessions(TS_CONFIG, BT_CONFIG, PAST_DAYS)  # 58


def fake_sessions(last_session: pd.Timestamp, n_sessions: int) -> pd.DatetimeIndex:
    # The synthetic prices are on weekdays, without NYSE holidays.
    return pd.bdate_range(end=last_session.normalize(), periods=n_sessions)


class FakeDateManager:
    def get_today_date(self) -> str:
        return "2026-11-14"


class FakeTickerDownloadManager:
    """Returns cached_df when the cache is allowed (if given), else long_df."""

    def __init__(self, long_df: pd.DataFrame, cached_df: pd.DataFrame | None = None):
        self.long_df = long_df
        self.cached_df = cached_df
        self.calls: list[dict] = []

    def get_latest_tickers(self, days_in_past=40, use_cache=True):
        self.calls.append({"days_in_past": days_in_past, "use_cache": use_cache})
        if use_cache and self.cached_df is not None:
            return self.cached_df, "start", "end"
        return self.long_df, "start", "end"


class RecordingRiskFree:
    def __init__(self):
        self.calls: list[pd.Timestamp] = []

    def __call__(self, dm, first_date):
        self.calls.append(first_date)
        return pd.DataFrame({"DTB3": [4.0]}, index=[first_date - pd.Timedelta(days=1)])


def prepare(tdm, risk_free=None, refresh_data=False):
    return prepare_backfill(
        PAST_DAYS,
        TS_CONFIG,
        BT_CONFIG,
        tdm=tdm,
        dm=FakeDateManager(),
        refresh_data=refresh_data,
        sessions_getter=fake_sessions,
        risk_free_history_getter=risk_free or RecordingRiskFree(),
    )


# Session counts and dates


def test_backfill_sessions_add_one_session_per_later_date():
    assert REQUIRED == 50
    assert NEEDED == REQUIRED + PAST_DAYS - 1
    assert DOWNLOAD == price_download_sessions(TS_CONFIG, BT_CONFIG) + PAST_DAYS - 1


def test_one_past_day_needs_what_a_normal_run_needs():
    assert backfill_sessions(TS_CONFIG, BT_CONFIG, 1) == REQUIRED
    normal_download = price_download_sessions(TS_CONFIG, BT_CONFIG)
    assert backfill_download_sessions(TS_CONFIG, BT_CONFIG, 1) == normal_download


def test_backfill_dates_are_the_latest_sessions_oldest_first():
    wide = pivot_close_wide(make_long_prices(n_sessions=20))
    dates = backfill_dates(wide.index, 3)
    assert dates == list(wide.index[-3:].normalize())
    assert dates == sorted(dates)


def test_one_backfill_date_is_the_latest_session():
    wide = pivot_close_wide(make_long_prices(n_sessions=20))
    assert backfill_dates(wide.index, 1) == [wide.index[-1].normalize()]


@pytest.mark.parametrize("past_days", [0, 21])
def test_backfill_dates_reject_impossible_counts(past_days):
    wide = pivot_close_wide(make_long_prices(n_sessions=20))
    with pytest.raises(ValueError):
        backfill_dates(wide.index, past_days)


# The early check of the price data


def complete_prices(n_sessions: int = NEEDED) -> pd.DataFrame:
    return pivot_close_wide(make_long_prices(n_sessions=n_sessions))


def check(wide: pd.DataFrame) -> None:
    check_backfill_prices(wide, NEEDED, fake_sessions(wide.index.max(), NEEDED))


def test_complete_prices_pass_the_check():
    check(complete_prices())


def test_too_few_sessions_fail_the_check():
    with pytest.raises(ValueError, match=f"fewer than the {NEEDED}"):
        check(complete_prices(NEEDED - 1))


def test_a_missing_session_fails_the_check_naming_it():
    wide = complete_prices(NEEDED + 1)
    missing = wide.index[10]
    with pytest.raises(ValueError, match=f"first on {missing:%Y-%m-%d}"):
        check(wide.drop(index=missing))


def test_a_missing_close_fails_the_check_naming_the_ticker_and_date():
    wide = complete_prices()
    wide.loc[wide.index[5], "QQQ"] = np.nan
    with pytest.raises(ValueError, match=f"QQQ has no close.*{wide.index[5]:%Y-%m-%d}"):
        check(wide)


def test_gaps_before_the_needed_sessions_are_ignored():
    wide = complete_prices(NEEDED + 5)
    wide.loc[wide.index[0], "GLD"] = np.nan  # Older than any run needs.
    check(wide)


# prepare_backfill(): one download, the check, and the risk-free cache


def test_prepare_downloads_once_and_warms_the_risk_free_cache_for_the_oldest():
    long_df = make_long_prices(n_sessions=DOWNLOAD)
    tdm = FakeTickerDownloadManager(long_df)
    risk_free = RecordingRiskFree()
    dates = prepare(tdm, risk_free)
    wide = pivot_close_wide(long_df)
    assert tdm.calls == [{"days_in_past": DOWNLOAD, "use_cache": True}]
    assert dates == list(wide.index[-PAST_DAYS:].normalize())
    oldest = wide[wide.index.normalize() <= dates[0]]
    assert risk_free.calls == [first_rebalance_date(oldest, BT_CONFIG)]


def test_prepare_downloads_again_once_when_the_cache_is_too_short():
    # Today's cache, from a normal run, is too short for the oldest date.
    long_df = make_long_prices(n_sessions=DOWNLOAD)
    cached = make_long_prices(n_sessions=REQUIRED)
    tdm = FakeTickerDownloadManager(long_df, cached_df=cached)
    prepare(tdm)
    assert tdm.calls == [
        {"days_in_past": DOWNLOAD, "use_cache": True},
        {"days_in_past": DOWNLOAD, "use_cache": False},
    ]


def test_prepare_with_refresh_downloads_once_without_the_cache():
    tdm = FakeTickerDownloadManager(make_long_prices(n_sessions=DOWNLOAD))
    prepare(tdm, refresh_data=True)
    assert tdm.calls == [{"days_in_past": DOWNLOAD, "use_cache": False}]


def test_prepare_fails_early_on_a_gap_before_touching_the_risk_free_rates():
    long_df = make_long_prices(n_sessions=DOWNLOAD)
    long_df = long_df[
        ~((long_df["ticker"] == SYNTHETIC_TICKERS[0]) & (long_df.index == 20))
    ]
    risk_free = RecordingRiskFree()
    with pytest.raises(ValueError, match="has no close"):
        prepare(FakeTickerDownloadManager(long_df), risk_free)
    assert risk_free.calls == []


# Completeness checks


def test_bucket_date_needs_both_marker_objects():
    asked = []
    present = {
        "po/2026-11-13/optimization_metadata.yml",
        "ts/2026-11-13/all_forecasts.csv",
        "po/2026-11-12/optimization_metadata.yml",
    }

    def exists(key):
        asked.append(key)
        return key in present

    assert bucket_date_is_complete("2026-11-13", exists, "ts", "po")
    assert not bucket_date_is_complete("2026-11-12", exists, "ts", "po")
    assert not bucket_date_is_complete("2026-11-11", exists, "ts", "po")
    assert asked[:2] == [
        "po/2026-11-13/optimization_metadata.yml",
        "ts/2026-11-13/all_forecasts.csv",
    ]


def test_local_date_needs_both_marker_files(tmp_path):
    root = str(tmp_path / "output")
    as_of = "2026-11-13"
    assert not local_date_is_complete(as_of, root)
    for kind, filename in (
        (PORTFOLIO_OPTIMIZATION, OPTIMIZATION_METADATA_FILENAME),
        (TIME_SERIES, ALL_FORECASTS_FILENAME),
    ):
        path = local_output_path(kind, as_of, filename, root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        assert not local_date_is_complete(as_of, root)
        with open(path, "w") as f:
            f.write("x")
    assert local_date_is_complete(as_of, root)


# The log


PEAK = PeakMemory(
    main_bytes=300 * BYTES_PER_MIB, largest_worker_bytes=200 * BYTES_PER_MIB, workers=2
)


def test_log_row_in_mib():
    row = backfill_log_row(DayResult("2026-11-13", STATUS_COMPLETED, 12.345, PEAK))
    assert row == {
        "as_of": "2026-11-13",
        "status": STATUS_COMPLETED,
        "seconds": "12.3",
        "main_mib": "300.0",
        "largest_worker_mib": "200.0",
        "workers": "2",
        "upper_bound_mib": "700.0",
        "error": "",
    }
    assert tuple(row) == BACKFILL_LOG_COLUMNS


def test_log_row_without_memory_leaves_it_empty():
    row = backfill_log_row(
        DayResult(as_of="2026-11-13", status=STATUS_SKIPPED, seconds=0.0, peak=None)
    )
    assert row["main_mib"] == row["upper_bound_mib"] == row["workers"] == ""


def test_log_has_one_header_and_a_row_per_append(tmp_path):
    path = str(tmp_path / "output" / "backfill" / "log.csv")
    append_backfill_log(DayResult("2026-11-12", STATUS_COMPLETED, 1.0, PEAK), path)
    append_backfill_log(DayResult("2026-11-13", STATUS_FAILED, 2.0, None, "boom"), path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["as_of"] for row in rows] == ["2026-11-12", "2026-11-13"]
    assert rows[1]["error"] == "boom"


# The loop


DATES = [pd.Timestamp(day) for day in ("2026-11-11", "2026-11-12", "2026-11-13")]
COMPLETED, SKIPPED, FAILED = STATUS_COMPLETED, STATUS_SKIPPED, STATUS_FAILED


class FakeRunDay:
    def __init__(self, failing: set[str] = frozenset()):
        self.failing = failing
        self.calls: list[str] = []

    def __call__(self, date):
        as_of = date.strftime("%Y-%m-%d")
        self.calls.append(as_of)
        if as_of in self.failing:
            return DayResult(as_of, STATUS_FAILED, 1.0, None, "exit status 1")
        return DayResult(as_of, STATUS_COMPLETED, 1.0, PEAK)


def read_log(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_loop_runs_oldest_first_and_skips_complete_dates(tmp_path):
    log = str(tmp_path / "log.csv")
    run_day = FakeRunDay()
    def is_complete(as_of):
        return as_of == "2026-11-12"

    results = run_backfill(DATES, run_day, is_complete, False, log)
    assert run_day.calls == ["2026-11-11", "2026-11-13"]
    assert [r.status for r in results] == [COMPLETED, SKIPPED, COMPLETED]
    assert [row["status"] for row in read_log(log)] == [r.status for r in results]


def test_force_runs_complete_dates_too(tmp_path):
    run_day = FakeRunDay()
    run_backfill(DATES, run_day, lambda as_of: True, True, str(tmp_path / "log.csv"))
    assert run_day.calls == ["2026-11-11", "2026-11-12", "2026-11-13"]


def test_a_failed_date_does_not_stop_the_others(tmp_path):
    log = str(tmp_path / "log.csv")
    run_day = FakeRunDay(failing={"2026-11-12"})
    results = run_backfill(DATES, run_day, lambda as_of: False, False, log)
    assert run_day.calls == ["2026-11-11", "2026-11-12", "2026-11-13"]
    assert [r.status for r in results] == [COMPLETED, FAILED, COMPLETED]
    summary = format_backfill_summary(results, log)
    assert "2 completed, 0 skipped, 1 failed, of 3 date(s)" in summary
    assert "2026-11-12: exit status 1" in summary
    assert log in summary


def test_the_log_is_written_as_each_date_finishes(tmp_path):
    # An interrupted backfill still leaves the rows of the dates it finished.
    log = str(tmp_path / "log.csv")

    def run_day(date):
        if date == DATES[-1]:
            raise KeyboardInterrupt
        return DayResult(date.strftime("%Y-%m-%d"), STATUS_COMPLETED, 1.0, PEAK)

    with pytest.raises(KeyboardInterrupt):
        run_backfill(DATES, run_day, lambda as_of: False, False, log)
    assert [row["as_of"] for row in read_log(log)] == ["2026-11-11", "2026-11-12"]
