"""
A one-time backfill of past dates (scripts/run_optimization_and_time_series.py
--past-days N): one full pipeline run for each of the N most recent NYSE
sessions, oldest first, each as its own run for that as-of date (see
pipeline.run_pipeline(as_of_date=...)), uploading each date as soon as it
finishes. With N = 1 the one date is the latest session, as in a normal run.

How it avoids repeated downloads and look-ahead:

1. prepare_backfill() downloads prices once, long enough for the oldest
   date, and checks before any run that every ticker has every session and
   close that the runs need (check_backfill_prices()), so a gap fails the
   backfill early instead of one day at a time. It then fills the risk-free
   rate cache back to before the oldest date's first rebalance. Both caches
   in input/ are named by the day the backfill started; every run reads
   them with that day pinned (pipeline.run_pipeline(cache_date=...)), even
   after midnight, so nothing is downloaded again.
2. Every run cuts the prices off at its own date before using them, so a
   past date sees no later price.
3. run_backfill() runs the dates in order. A date that is already complete
   where the backfill writes (in the bucket's folders when uploading, so dev
   and production are judged separately, else in output/) is skipped unless
   force is set. A failed date does not stop the others;
   it is reported in the results. Each result is appended to a CSV log as
   it finishes (see backfill_log_row()), so an interrupted backfill still
   leaves a table.

Each run is a separate process (the script starts it), because the peak
memory counters cannot be reset within one process (see memory_usage.py).
Everything here is pure except prepare_backfill(), which downloads through
the objects passed to it, and append_backfill_log(), which writes the log.
"""

import csv
import os
from dataclasses import dataclass
from typing import Callable
import pandas as pd
from aliciadata_finance_backend.config import (
    PortfolioBacktestConfig,
    TimeSeriesConfig,
)
from aliciadata_finance_backend.data.prices import pivot_close_wide, sessions_through
from aliciadata_finance_backend.memory_usage import BYTES_PER_MIB, PeakMemory
from aliciadata_finance_backend.pipeline import (
    first_rebalance_date,
    price_download_sessions,
    required_sessions,
)
from aliciadata_finance_backend.storage.layout import (
    ALL_FORECASTS_FILENAME,
    OPTIMIZATION_METADATA_FILENAME,
    OUTPUT_ROOT,
    PORTFOLIO_OPTIMIZATION,
    TIME_SERIES,
    ensure_parent_dir,
    local_output_path,
    object_key,
)

STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"

# Columns of the backfill log, in order. Memory is in MiB, as printed by a
# run (see memory_usage.format_peak_memory()), and empty for a skipped date
# or one whose run reported no memory.
BACKFILL_LOG_COLUMNS = (
    "as_of",
    "status",
    "seconds",
    "main_mib",
    "largest_worker_mib",
    "workers",
    "upper_bound_mib",
    "error",
)


def backfill_sessions(
    ts_config: TimeSeriesConfig, bt_config: PortfolioBacktestConfig, past_days: int
) -> int:
    """
    Trading sessions every backfill run together needs, up to the latest
    date: the oldest date's required_sessions() plus the past_days - 1
    sessions after it. Pure function.
    """
    return required_sessions(ts_config, bt_config) + past_days - 1


def backfill_download_sessions(
    ts_config: TimeSeriesConfig, bt_config: PortfolioBacktestConfig, past_days: int
) -> int:
    """
    Trading sessions the backfill's one download requests: a normal run's
    price_download_sessions() plus past_days - 1, so it is long enough for
    the oldest date. Pure function.
    """
    return price_download_sessions(ts_config, bt_config) + past_days - 1


def backfill_dates(index: pd.DatetimeIndex, past_days: int) -> list[pd.Timestamp]:
    """
    The backfill's as-of dates: the past_days most recent sessions in the
    price data, oldest first, as dates (time of day dropped). The last is
    the latest session, a normal run's date. Pure function.

    Raises
    ------
    ValueError
        If past_days is below 1 or the data has fewer sessions.
    """
    if past_days < 1:
        raise ValueError(f"past_days must be at least 1, got {past_days}.")
    sessions = pd.DatetimeIndex(index).normalize().sort_values()
    if len(sessions) < past_days:
        raise ValueError(
            f"The price data has {len(sessions)} sessions, fewer than the "
            f"{past_days} dates to backfill."
        )
    return list(sessions[-past_days:])


def check_backfill_prices(
    wide_df: pd.DataFrame, n_sessions: int, expected_sessions: pd.DatetimeIndex
) -> None:
    """
    Check that the price data has everything the backfill's runs need, so a
    gap fails the backfill before any run instead of failing runs one by
    one: at least n_sessions sessions (backfill_sessions()), every expected
    NYSE session among the last n_sessions, and a close for every ticker on
    each of them. Pure function.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide prices, as returned by data.prices.pivot_close_wide().

    n_sessions : int
        Sessions needed, up to the latest one.

    expected_sessions : pd.DatetimeIndex
        The n_sessions NYSE sessions ending on the latest session in the
        data (data.date_manager.nyse_sessions_ending()).

    Raises
    ------
    ValueError
        Naming the shortfall: the number of sessions, the first missing
        session, or the first ticker with a missing close and its date.
    """
    if len(wide_df) < n_sessions:
        raise ValueError(
            f"The price data has {len(wide_df)} sessions, fewer than the "
            f"{n_sessions} the backfill needs (the oldest date's history plus one "
            "session per later date)."
        )
    window = wide_df.sort_index().iloc[-n_sessions:]
    present = set(window.index.normalize())
    expected = pd.DatetimeIndex(expected_sessions).normalize()
    missing = [day for day in expected if day not in present]
    if missing:
        raise ValueError(
            f"The price data is missing {len(missing)} NYSE session(s) the backfill "
            f"needs, the first on {missing[0]:%Y-%m-%d}."
        )
    for ticker in window.columns:
        gaps = window.index[window[ticker].isna()]
        if len(gaps):
            raise ValueError(
                f"{ticker} has no close on {len(gaps)} session(s) the backfill "
                f"needs, the first on {gaps[0]:%Y-%m-%d}."
            )


def prepare_backfill(
    past_days: int,
    ts_config: TimeSeriesConfig,
    bt_config: PortfolioBacktestConfig,
    *,
    tdm,
    dm,
    refresh_data: bool,
    sessions_getter: Callable[[pd.Timestamp, int], pd.DatetimeIndex],
    risk_free_history_getter: Callable[..., pd.DataFrame],
) -> list[pd.Timestamp]:
    """
    Get everything the backfill's runs read into the caches, once, and
    return the dates to run.

    1. Prices: one download of backfill_download_sessions() sessions (or
       today's cached download if it is long enough). A cached download that
       is too short is downloaded again once.
    2. check_backfill_prices() on them.
    3. The risk-free rate history, back to before the oldest date's first
       rebalance, the earliest optimization date of the whole backfill.

    tdm and dm must name their caches by the same pinned day that the runs
    will use (see DateManager(today=...)).

    Parameters
    ----------
    past_days : int
        Number of dates to backfill, at least 1.

    ts_config, bt_config : TimeSeriesConfig, PortfolioBacktestConfig
        The settings every run will use.

    tdm : TickerDownloadManager
        Downloads and caches the prices.

    dm : DateManager
        Passed to risk_free_history_getter for its cache.

    refresh_data : bool
        If True, download prices even if today's download is cached.

    sessions_getter, risk_free_history_getter : callables
        data.date_manager.nyse_sessions_ending() and
        optimization.portfolio.get_risk_free_rate_history(), or test doubles.

    Returns
    -------
    list[pd.Timestamp]
        The as-of dates, oldest first (backfill_dates()).

    Raises
    ------
    ValueError
        If the prices are not enough for every date (check_backfill_prices()),
        or as raised by risk_free_history_getter.
    """
    n_sessions = backfill_sessions(ts_config, bt_config, past_days)
    n_download = backfill_download_sessions(ts_config, bt_config, past_days)
    long_df, _, _ = tdm.get_latest_tickers(
        days_in_past=n_download, use_cache=not refresh_data
    )
    wide_df = pivot_close_wide(long_df)
    if len(wide_df) < n_sessions and not refresh_data:
        print(
            f"Cached prices cover {len(wide_df)} sessions, fewer than the "
            f"{n_sessions} the backfill needs; downloading again."
        )
        long_df, _, _ = tdm.get_latest_tickers(days_in_past=n_download, use_cache=False)
        wide_df = pivot_close_wide(long_df)
    check_backfill_prices(
        wide_df, n_sessions, sessions_getter(wide_df.index.max(), n_sessions)
    )
    dates = backfill_dates(wide_df.index, past_days)
    oldest_prices = sessions_through(wide_df, dates[0])
    risk_free_history_getter(dm, first_rebalance_date(oldest_prices, bt_config))
    print(
        f"Backfilling {len(dates)} date(s), {dates[0]:%Y-%m-%d} to "
        f"{dates[-1]:%Y-%m-%d}, from prices and risk-free rates cached "
        f"{dm.get_today_date()}."
    )
    return dates


def bucket_date_is_complete(
    as_of: str,
    object_exists: Callable[[str], bool],
    time_series_folder: str,
    portfolio_folder: str,
) -> bool:
    """
    Whether a date is available in the bucket's folders by the contract's
    rule (docs/INTERFACE_CONTRACT.md, "Which dates are available"): its
    optimization_metadata.yml, uploaded last, and its all_forecasts.csv both
    exist. Checked in the configured folders, so dev and production are
    judged separately.

    Parameters
    ----------
    as_of : str
        The date, YYYY-MM-DD.

    object_exists : callable
        Whether a full object key exists (S3Uploader.object_exists()).

    time_series_folder, portfolio_folder : str
        The two top-level folders (pipeline.upload_folders()).
    """
    return object_exists(
        object_key(portfolio_folder, as_of, OPTIMIZATION_METADATA_FILENAME)
    ) and object_exists(object_key(time_series_folder, as_of, ALL_FORECASTS_FILENAME))


def local_date_is_complete(as_of: str, output_root: str = OUTPUT_ROOT) -> bool:
    """
    Whether a date's local output has the same two files that make it
    available in the bucket (see bucket_date_is_complete()). Used when the
    backfill does not upload.
    """
    return os.path.isfile(
        local_output_path(
            PORTFOLIO_OPTIMIZATION, as_of, OPTIMIZATION_METADATA_FILENAME, output_root
        )
    ) and os.path.isfile(
        local_output_path(TIME_SERIES, as_of, ALL_FORECASTS_FILENAME, output_root)
    )


@dataclass(frozen=True)
class DayResult:
    """
    The outcome of one backfill date.

    Attributes
    ----------
    as_of : str
        The date, YYYY-MM-DD.

    status : str
        STATUS_COMPLETED, STATUS_SKIPPED, or STATUS_FAILED.

    seconds : float
        Wall-clock time of the run (0 for a skipped date).

    peak : PeakMemory or None
        The run's peak memory, or None if it reported none (a skipped date,
        or a run that failed before reporting).

    error : str
        A short description of the failure, empty otherwise.
    """

    as_of: str
    status: str
    seconds: float
    peak: PeakMemory | None
    error: str = ""


def _mib(n_bytes: int) -> str:
    return f"{n_bytes / BYTES_PER_MIB:.1f}"


def backfill_log_row(result: DayResult) -> dict[str, str]:
    """
    One row of the backfill log for a result, keyed by BACKFILL_LOG_COLUMNS,
    with memory in MiB to one decimal. Pure function.
    """
    peak = result.peak
    return {
        "as_of": result.as_of,
        "status": result.status,
        "seconds": f"{result.seconds:.1f}",
        "main_mib": _mib(peak.main_bytes) if peak else "",
        "largest_worker_mib": _mib(peak.largest_worker_bytes) if peak else "",
        "workers": str(peak.workers) if peak else "",
        "upper_bound_mib": _mib(peak.upper_bound_bytes) if peak else "",
        "error": result.error,
    }


def append_backfill_log(result: DayResult, path: str) -> None:
    """
    Append one result to the backfill log at path (see
    storage.layout.backfill_log_path()), writing the header first when the
    file does not exist yet. Missing folders are created.
    """
    is_new = not os.path.exists(path)
    with open(ensure_parent_dir(path), "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BACKFILL_LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(backfill_log_row(result))


def run_backfill(
    dates: list[pd.Timestamp],
    run_day: Callable[[pd.Timestamp], DayResult],
    is_complete: Callable[[str], bool],
    force: bool,
    log_path: str,
) -> list[DayResult]:
    """
    Run every date in order, logging each result as it finishes.

    Parameters
    ----------
    dates : list[pd.Timestamp]
        The as-of dates, oldest first (prepare_backfill()).

    run_day : callable
        Runs the whole pipeline for one date and returns its DayResult. It
        reports a failure in the result instead of raising.

    is_complete : callable
        Whether a date (YYYY-MM-DD) is already complete where this backfill
        writes: in the bucket's folders when uploading, else in output/.

    force : bool
        If True, run every date, even complete ones (they are recomputed and
        overwritten). If False, skip complete dates.

    log_path : str
        The backfill log to append to (append_backfill_log()).

    Returns
    -------
    list[DayResult]
        One result per date, in order.
    """
    results = []
    for date in dates:
        as_of = date.strftime("%Y-%m-%d")
        if not force and is_complete(as_of):
            print(f"Skipping {as_of}: already complete (use --force to recompute).")
            result = DayResult(
                as_of=as_of, status=STATUS_SKIPPED, seconds=0.0, peak=None
            )
        else:
            print(f"Running {as_of}...")
            result = run_day(date)
        append_backfill_log(result, log_path)
        results.append(result)
    return results


def format_backfill_summary(results: list[DayResult], log_path: str) -> str:
    """
    Lines summarizing a backfill: the count of each status, the failed
    dates with their errors, and where the log is. Pure function.
    """
    counts = {
        status: sum(result.status == status for result in results)
        for status in (STATUS_COMPLETED, STATUS_SKIPPED, STATUS_FAILED)
    }
    lines = [
        f"Backfill finished: {counts[STATUS_COMPLETED]} completed, "
        f"{counts[STATUS_SKIPPED]} skipped, {counts[STATUS_FAILED]} failed, "
        f"of {len(results)} date(s).",
    ]
    failed = [result for result in results if result.status == STATUS_FAILED]
    if failed:
        lines.append("Failed dates:")
        lines += [f"  {result.as_of}: {result.error}" for result in failed]
    lines.append(f"Per-date log: {log_path}")
    return "\n".join(lines)
