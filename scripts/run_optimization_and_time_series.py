"""
Run the whole backend in one command: download prices (or use today's
cached download), fit walk-forward ARIMA forecasts (or reuse the cached
forecasts for the data's as-of date), optimize the portfolio, backtest it
against equal weights, and upload all results for the frontend.

Run from the repository root:

    uv run python scripts/run_optimization_and_time_series.py [--no-upload]
        [--retrain-arima] [--refresh-data] [--workers N] [--config PATH]
        [--as-of YYYY-MM-DD | --past-days N [--force]]

Model settings are read from config/modeling_and_optimization.yml (or
--config PATH). With --as-of, the run uses only prices up to that NYSE
session, as if the data ended that day, and writes (and uploads) that date's
folder.

With --past-days N, the script is a one-time backfill: it downloads prices
and risk-free rates once, then runs the whole pipeline for each of the N most
recent sessions, oldest first, each in its own process (this script with
--as-of), uploading each date as it finishes. Dates already complete are
skipped unless --force is given; a failed date does not stop the others, and
the script exits with status 1 if any failed. A table of the dates, their
run times, and their peak memory is written to output/backfill/ (see
backfill.py).

After a successful run, the script prints the peak memory of the main
process and of the largest ARIMA worker (see memory_usage.py), to help size
the machine it runs on.
"""

import argparse
import datetime
import os
import subprocess
import sys
import tempfile
import time
import pandas as pd
from dotenv import load_dotenv
from aliciadata_finance_backend.backfill import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    DayResult,
    bucket_date_is_complete,
    format_backfill_summary,
    local_date_is_complete,
    prepare_backfill,
    run_backfill,
)
from aliciadata_finance_backend.config import (
    DEFAULT_CONFIG_PATH,
    ModelingConfig,
    load_config,
)
from aliciadata_finance_backend.data.date_manager import (
    DateManager,
    nyse_sessions_ending,
)
from aliciadata_finance_backend.data.ticker_download_manager import (
    TickerDownloadManager,
)
from aliciadata_finance_backend.forecasting.ticker_predict import default_worker_count
from aliciadata_finance_backend.memory_usage import (
    format_peak_memory,
    measure_peak_memory,
    read_peak_memory_json,
    write_peak_memory_json,
)
from aliciadata_finance_backend.optimization.portfolio import (
    get_risk_free_rate_history,
)
from aliciadata_finance_backend.pipeline import (
    PRICE_DOWNLOAD_FOLDER,
    run_pipeline,
    upload_folders,
)
from aliciadata_finance_backend.storage.layout import backfill_log_path
from aliciadata_finance_backend.storage.s3_uploader import S3Uploader


def positive_int(value: str) -> int:
    """
    argparse type for an integer of at least 1.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def iso_date(value: str) -> pd.Timestamp:
    """
    argparse type for a date written as YYYY-MM-DD.
    """
    try:
        return pd.Timestamp(datetime.date.fromisoformat(value))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be a date as YYYY-MM-DD, got {value!r}"
        ) from None


def parse_args() -> argparse.Namespace:
    """
    Parse the command line options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download prices, fit ARIMA forecasts, optimize and backtest the "
            "portfolio, and upload the results under a YYYY-MM-DD/ prefix (the "
            "data as-of date)."
        )
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Write the output/ files but do not upload anything.",
    )
    parser.add_argument(
        "--retrain-arima",
        action="store_true",
        help="Retrain ARIMA models even if forecasts for the as-of date are cached.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help=(
            "Download prices again even if today's download is cached. With "
            "--past-days, applies to the backfill's one download."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        metavar="N",
        help=(
            "Worker processes for ARIMA training. Defaults to half the logical "
            "CPUs; 1 trains everything in this process."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help=(
            "Model settings file, relative to the working directory. Defaults to "
            f"{DEFAULT_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--as-of",
        type=iso_date,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Run as if the price data ended on this NYSE session: later prices "
            "are not used, and results go to (and are uploaded under) this "
            "date's folder. Defaults to the latest session."
        ),
    )
    parser.add_argument(
        "--past-days",
        type=positive_int,
        default=None,
        metavar="N",
        help=(
            "One-time backfill: run the whole pipeline for each of the N most "
            "recent sessions, oldest first, uploading each date as it finishes. "
            "1 is the latest session only, like a normal run."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "With --past-days, recompute (and overwrite) dates that are already "
            "complete instead of skipping them."
        ),
    )
    # Set by a backfill for each of its runs; not for use by hand.
    parser.add_argument("--cache-date", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--memory-report", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.past_days is not None and args.as_of is not None:
        parser.error("--past-days and --as-of cannot be combined.")
    if args.force and args.past_days is None:
        parser.error("--force only applies with --past-days.")
    return args


def run_one(args: argparse.Namespace, config: ModelingConfig) -> None:
    """
    Run the pipeline once, for the latest session or --as-of, then print the
    uploaded keys and the run's peak memory (also written to
    --memory-report, if given, for a backfill to collect).
    """
    result = run_pipeline(
        upload=not args.no_upload,
        retrain_arima=args.retrain_arima,
        refresh_data=args.refresh_data,
        arima_workers=args.workers,
        ts_config=config.time_series_models,
        po_config=config.portfolio_optimization,
        bt_config=config.portfolio_backtest,
        as_of_date=args.as_of,
        cache_date=args.cache_date,
    )
    print(f"Done: {result.as_of} (ARIMA {'trained' if result.trained_arima else 'cached'})")
    for key in result.uploaded:
        print(f"  uploaded {key}")
    # ARIMA uses worker processes only when it trains with more than one
    # worker (see forecasting/ticker_predict.py, fit_windows()).
    n_workers = args.workers if args.workers is not None else default_worker_count()
    workers_ran = n_workers if result.trained_arima and n_workers > 1 else 0
    peak = measure_peak_memory(workers_ran)
    print(format_peak_memory(peak))
    if args.memory_report:
        write_peak_memory_json(peak, args.memory_report)


def run_day_in_subprocess(
    date: pd.Timestamp, args: argparse.Namespace, cache_date: str
) -> DayResult:
    """
    Run this script for one backfill date, in a new process, with the
    backfill's options and its pinned cache date, and collect the outcome.
    Each run's output (progress, uploads, peak memory) is shown as it runs.
    --refresh-data is not passed on: the backfill already downloaded.
    """
    as_of = date.strftime("%Y-%m-%d")
    with tempfile.TemporaryDirectory() as folder:
        report_path = os.path.join(folder, "peak_memory.json")
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--as-of",
            as_of,
            "--cache-date",
            cache_date,
            "--memory-report",
            report_path,
            "--config",
            args.config,
        ]
        if args.no_upload:
            command.append("--no-upload")
        if args.retrain_arima:
            command.append("--retrain-arima")
        if args.workers is not None:
            command += ["--workers", str(args.workers)]
        start = time.monotonic()
        returncode = subprocess.run(command).returncode
        seconds = time.monotonic() - start
        # A run that failed before measuring has written no report.
        peak = None
        if os.path.exists(report_path):
            peak = read_peak_memory_json(report_path)
    if returncode == 0:
        return DayResult(
            as_of=as_of, status=STATUS_COMPLETED, seconds=seconds, peak=peak
        )
    return DayResult(
        as_of=as_of,
        status=STATUS_FAILED,
        seconds=seconds,
        peak=peak,
        error=f"exit status {returncode}; see that run's output above",
    )


def run_backfill_mode(args: argparse.Namespace, config: ModelingConfig) -> int:
    """
    Run the --past-days backfill (see backfill.py) and return the exit
    status: 0 if every date completed or was skipped, 1 if any failed.
    """
    started = datetime.datetime.now(datetime.timezone.utc)
    # Pin "today" for the whole backfill, so the caches it writes now are the
    # ones every run reads, even after midnight.
    cache_date = DateManager().get_today_date()
    dm = DateManager(today=cache_date)
    if args.no_upload:
        is_complete = local_date_is_complete
    else:
        # Check the upload settings before the download, as a normal run does.
        time_series_folder, portfolio_folder = upload_folders()
        uploader = S3Uploader()

        def is_complete(as_of: str) -> bool:
            return bucket_date_is_complete(
                as_of, uploader.object_exists, time_series_folder, portfolio_folder
            )

    dates = prepare_backfill(
        args.past_days,
        config.time_series_models,
        config.portfolio_backtest,
        tdm=TickerDownloadManager(PRICE_DOWNLOAD_FOLDER, dm=dm),
        dm=dm,
        refresh_data=args.refresh_data,
        sessions_getter=nyse_sessions_ending,
        risk_free_history_getter=get_risk_free_rate_history,
    )
    log_path = backfill_log_path(started)
    results = run_backfill(
        dates,
        lambda date: run_day_in_subprocess(date, args, cache_date),
        is_complete,
        args.force,
        log_path,
    )
    print(format_backfill_summary(results, log_path))
    return 1 if any(result.status == STATUS_FAILED for result in results) else 0


def main() -> None:
    """
    Load the configuration file and .env, then run the pipeline once, or the
    --past-days backfill, with the command line options. The configuration is
    checked first, so a bad file fails before any download.
    """
    args = parse_args()
    config = load_config(args.config)
    print(f"Using settings from {args.config}: {config.time_series_models}")
    print(f"Using settings from {args.config}: {config.portfolio_optimization}")
    print(f"Using settings from {args.config}: {config.portfolio_backtest}")
    load_dotenv(override=True)
    if args.past_days is not None:
        sys.exit(run_backfill_mode(args, config))
    run_one(args, config)


if __name__ == "__main__":
    main()
