"""
One backend run: download (or load cached) prices, ARIMA forecasts (or load
cached), forecast metrics, portfolio optimization, a walk-forward portfolio
backtest, then upload all six frontend files.

The time series settings (training window, scored days), the portfolio
optimization settings (weight cap, frontier points, random portfolios, seed),
and the backtest settings (training window, rebalance interval) come from the
configuration file, passed in as a TimeSeriesConfig, a
PortfolioOptimizationConfig, and a PortfolioBacktestConfig (see config.py).

Order of operations:
1. Price data: one download of price_download_sessions() trading days,
   cached per run date in input/annual/. A cached download too short for the
   steps below is downloaded again once.
2. ARIMA: trained on the last TimeSeriesConfig.arima_window_sessions of that data,
   cached as output/time_series/<as-of date>/all_forecasts.csv. Metrics on
   those forecasts are written next to it as all_forecasts_metadata.yml.
3. Optimization: on the last PortfolioBacktestConfig.training_window_closes
   sessions, with the last DTB3 rate dated before the as-of date, written to
   output/portfolio_optimization/<as-of date>/. Its invariants are checked
   before the files are written, and the written .h5 and .yml files are read
   back and checked against each other before any upload.
4. Backtest: the max Sharpe ratio portfolio against equal weights over the
   same scored days as the forecasts, on the last
   PortfolioBacktestConfig.sessions_needed() sessions, which must be complete
   consecutive NYSE sessions. Written to the same folder as the optimization
   and checked the same way. Its metadata also records step 3's result as
   final_optimization, and that record is checked against step 3's two files.
5. Upload, only after every step above succeeded, to the one frontend Space
   under <folder>/<as-of date>/: all_forecasts.csv, all_forecasts_metadata.yml,
   portfolio_optimization_plot_data.h5, portfolio_backtest_daily.csv,
   portfolio_backtest_metadata.yml, then optimization_metadata.yml last, so
   the metadata file marks a complete date (see docs/INTERFACE_CONTRACT.md).

Network, S3, and model training are passed in as dependencies so the
orchestration can be tested without them.
"""

import os
from dataclasses import dataclass
from typing import Callable, Protocol
import pandas as pd
from aliciadata_finance_backend.config import (
    PortfolioBacktestConfig,
    PortfolioOptimizationConfig,
    TimeSeriesConfig,
)
from aliciadata_finance_backend.data.date_manager import (
    DateManager,
    nyse_sessions_ending,
)
from aliciadata_finance_backend.data.prices import last_n_sessions, pivot_close_wide
from aliciadata_finance_backend.data.ticker_download_manager import (
    TickerDownloadManager,
)
from aliciadata_finance_backend.forecasting.ticker_predict import (
    ArimaWalkForward,
    ForecastTrainer,
    load_or_train_forecasts,
)
from aliciadata_finance_backend.forecasting.metrics import (
    build_forecast_metadata,
    write_forecast_metadata,
)
from aliciadata_finance_backend.optimization.backtest import (
    build_backtest_metadata,
    build_final_optimization,
    run_portfolio_backtest,
    verify_backtest_files,
    verify_final_optimization,
    write_backtest_daily,
    write_backtest_metadata,
)
from aliciadata_finance_backend.optimization.portfolio import (
    get_risk_free_rate_history,
    risk_free_rate_before,
)
from aliciadata_finance_backend.optimization.workflow import (
    build_optimization_metadata,
    run_portfolio_optimization,
    verify_optimization_files,
    write_optimization_metadata,
    write_plot_data_h5,
)
from aliciadata_finance_backend.storage.layout import (
    ALL_FORECASTS_FILENAME,
    ALL_FORECASTS_METADATA_FILENAME,
    OPTIMIZATION_METADATA_FILENAME,
    OPTIMIZATION_PLOT_DATA_FILENAME,
    OUTPUT_ROOT,
    PORTFOLIO_BACKTEST_DAILY_FILENAME,
    PORTFOLIO_BACKTEST_METADATA_FILENAME,
    PORTFOLIO_OPTIMIZATION,
    TIME_SERIES,
    as_of_date_str,
    ensure_parent_dir,
    local_output_path,
    object_key,
)
from aliciadata_finance_backend.storage.s3_uploader import BUCKET_NAME_ENV, S3Uploader

# Extra sessions requested from the API beyond required_sessions(), so that an
# off-by-one in counting sessions, or a session missing from the data, does
# not leave the run short.
PRICE_DOWNLOAD_MARGIN_SESSIONS = 5


def required_sessions(
    ts_config: TimeSeriesConfig, bt_config: PortfolioBacktestConfig
) -> int:
    """
    Trading sessions every run needs: enough for the walk-forward ARIMA
    (TimeSeriesConfig.arima_window_sessions) and the backtest
    (PortfolioBacktestConfig.sessions_needed()), whichever is larger. The
    optimization's window is the last part of the backtest's. Pure function.
    """
    return max(
        ts_config.arima_window_sessions,
        bt_config.sessions_needed(ts_config.evaluation_window_length),
    )


def price_download_sessions(
    ts_config: TimeSeriesConfig, bt_config: PortfolioBacktestConfig
) -> int:
    """
    Trading sessions requested from the API: required_sessions() plus
    PRICE_DOWNLOAD_MARGIN_SESSIONS. Pure function.
    """
    return required_sessions(ts_config, bt_config) + PRICE_DOWNLOAD_MARGIN_SESSIONS


PRICE_DOWNLOAD_FOLDER = os.path.join("input", "annual")

# Environment variables holding the names of the two top-level folders inside
# the one frontend Space. Dev and production share the Space and differ only
# in these folder names.
PORTFOLIO_FOLDER_ENV = "S3_PORTFOLIO_OPTIMIZATION_FOLDER"
TIME_SERIES_FOLDER_ENV = "S3_TIME_SERIES_FOLDER"


class Uploader(Protocol):
    """
    Anything that uploads a local file to a full object key in the frontend
    Space, e.g. S3Uploader.
    """

    def upload_file(self, local_filename: str, object_key: str) -> None: ...


@dataclass(frozen=True)
class PipelineResult:
    """
    What a run produced: the as-of date, the local files written, whether
    ARIMA models were trained (False when the cached forecasts were reused),
    and the object keys uploaded, in upload order (empty when uploads were
    skipped).
    """

    as_of: str
    forecasts_path: str
    forecasts_metadata_path: str
    plot_data_path: str
    metadata_path: str
    backtest_daily_path: str
    backtest_metadata_path: str
    trained_arima: bool
    uploaded: tuple[str, ...]


def _upload_folders() -> tuple[str, str]:
    """
    Check the environment for everything an upload needs, and return the
    time series and portfolio optimization folder names. Raises RuntimeError
    naming every variable that is not set: the Space name and both folders.
    """
    envs = (BUCKET_NAME_ENV, TIME_SERIES_FOLDER_ENV, PORTFOLIO_FOLDER_ENV)
    values = {env: os.getenv(env) for env in envs}
    missing = [env for env, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s) {', '.join(missing)}; set them in .env "
            "or run with --no-upload."
        )
    return values[TIME_SERIES_FOLDER_ENV], values[PORTFOLIO_FOLDER_ENV]


def run_pipeline(
    upload: bool = True,
    retrain_arima: bool = False,
    refresh_data: bool = False,
    arima_workers: int | None = None,
    *,
    ts_config: TimeSeriesConfig,
    po_config: PortfolioOptimizationConfig,
    bt_config: PortfolioBacktestConfig,
    tdm: TickerDownloadManager | None = None,
    trainer: ForecastTrainer | None = None,
    uploader: Uploader | None = None,
    risk_free_history_getter: Callable[
        [DateManager], pd.DataFrame
    ] = get_risk_free_rate_history,
    sessions_getter: Callable[
        [pd.Timestamp, int], pd.DatetimeIndex
    ] = nyse_sessions_ending,
    dm: DateManager | None = None,
    output_root: str = OUTPUT_ROOT,
) -> PipelineResult:
    """
    Run the whole backend: prices, ARIMA forecasts, portfolio optimization,
    and uploads. Nothing is uploaded unless every computation succeeded. If
    an upload fails, the error propagates immediately; because
    optimization_metadata.yml is uploaded last, a partially uploaded date is
    never marked complete, and re-running uploads it again.

    The Space name (S3_BUCKET_NAME) and the two top-level folder names are
    read from the environment (load .env before calling). When upload is
    True they, and the endpoint, region, and credentials (via S3Uploader),
    are checked before any work starts, so a configuration error fails fast
    instead of after a long ARIMA run.

    Parameters
    ----------
    upload : bool, optional
        If False, write the local output files but upload nothing.

    retrain_arima : bool, optional
        If True, retrain ARIMA models even if the as-of date's forecasts CSV
        already exists, and overwrite it.

    refresh_data : bool, optional
        If True, download prices again even if today's download is cached.
        Even when False, a cached download with fewer than
        required_sessions() sessions (e.g. one written before the ARIMA
        window grew) is downloaded again once.

    arima_workers : int, optional
        Number of worker processes for ARIMA training. Defaults to None,
        which means half the logical CPUs (see default_worker_count() in
        forecasting/ticker_predict.py). Ignored when trainer is given.

    ts_config : TimeSeriesConfig
        Time series settings from the configuration file: the training
        window length and the number of scored backtest days. Required; it
        has no default, so the configuration file is the one source of
        truth.

    po_config : PortfolioOptimizationConfig
        Portfolio optimization settings from the configuration file: the
        weight cap, the number of efficient-frontier points and random
        portfolios, and the random seed. Required, like ts_config.

    bt_config : PortfolioBacktestConfig
        Portfolio backtest settings from the configuration file: the
        training window, which the optimization uses too, and the rebalance
        interval. Required, like ts_config. The backtest scores ts_config's
        evaluation days and uses po_config's weight cap.

    tdm, trainer, uploader, risk_free_history_getter, sessions_getter, dm : optional
        Dependencies, defaulting to the real implementations. Tests pass
        doubles here to avoid network, S3, and model training.

    output_root : str, optional
        Root of the local output folder. Defaults to "output".

    Returns
    -------
    PipelineResult
        Summary of the run.
    """
    if upload:
        time_series_folder, portfolio_folder = _upload_folders()
        uploader = uploader if uploader is not None else S3Uploader()

    # 1. Prices
    n_required = required_sessions(ts_config, bt_config)
    n_download = price_download_sessions(ts_config, bt_config)
    arima_sessions = ts_config.arima_window_sessions
    evaluation_days = ts_config.evaluation_window_length
    backtest_sessions = bt_config.sessions_needed(evaluation_days)
    tdm = tdm if tdm is not None else TickerDownloadManager(PRICE_DOWNLOAD_FOLDER)
    dm = dm if dm is not None else DateManager()
    long_df, start_date, end_date = tdm.get_latest_tickers(
        days_in_past=n_download, use_cache=not refresh_data
    )
    wide_df = pivot_close_wide(long_df)
    if len(wide_df) < n_required and not refresh_data:
        print(
            f"Cached prices cover {len(wide_df)} sessions, fewer than the "
            f"{n_required} needed; downloading again."
        )
        long_df, start_date, end_date = tdm.get_latest_tickers(
            days_in_past=n_download, use_cache=False
        )
        wide_df = pivot_close_wide(long_df)
    if len(wide_df) < n_required:
        raise ValueError(
            f"Downloaded prices cover {len(wide_df)} sessions, fewer than the "
            f"{n_required} needed for ARIMA ({arima_sessions}) and the backtest "
            f"({backtest_sessions})."
        )
    print(f"Loaded data from {start_date} to {end_date}")
    as_of = as_of_date_str(wide_df.index)
    print(f"Results will be stored under {as_of}/")

    # 2. ARIMA forecasts (cached per as-of date)
    forecasts_path = local_output_path(
        TIME_SERIES, as_of, ALL_FORECASTS_FILENAME, output_root
    )
    arima_window_df = last_n_sessions(wide_df, arima_sessions)
    if trainer is None:
        trainer = ArimaWalkForward(
            training_window_returns=ts_config.training_window_returns,
            n_workers=arima_workers,
        )
    trained_arima = load_or_train_forecasts(
        arima_window_df,
        forecasts_path,
        retrain=retrain_arima,
        trainer=trainer,
    )

    # 2b. Forecast metrics, from the CSV exactly as written (also for cached
    # forecasts). The live forecast's training window is the last window.
    forecasts_df = pd.read_csv(forecasts_path, index_col="pred_date", parse_dates=True)
    window_start, window_end = ArimaWalkForward.training_window_start_end(
        arima_window_df.index, ts_config.training_window_returns
    )[-1]
    forecasts_metadata_path = ensure_parent_dir(
        local_output_path(
            TIME_SERIES, as_of, ALL_FORECASTS_METADATA_FILENAME, output_root
        )
    )
    write_forecast_metadata(
        build_forecast_metadata(
            forecasts_df,
            tdm.tickers,
            window_start,
            window_end,
            eval_days=evaluation_days,
            training_window_returns=ts_config.training_window_returns,
        ),
        forecasts_metadata_path,
    )
    print(f"Saved {forecasts_metadata_path}")

    # 3. Portfolio optimization
    tb3m_df = risk_free_history_getter(dm)
    risk_free_rate_data = risk_free_rate_before(tb3m_df, wide_df.index[-1])
    print(risk_free_rate_data)
    optimization_prices = last_n_sessions(wide_df, bt_config.training_window_closes)
    result = run_portfolio_optimization(
        optimization_prices,
        risk_free_rate_data["daily_risk_free_rate"],
        po_config,
    )
    plot_data_path = ensure_parent_dir(
        local_output_path(
            PORTFOLIO_OPTIMIZATION, as_of, OPTIMIZATION_PLOT_DATA_FILENAME, output_root
        )
    )
    write_plot_data_h5(result, plot_data_path)
    print(f"Saved {plot_data_path}")
    metadata_path = ensure_parent_dir(
        local_output_path(
            PORTFOLIO_OPTIMIZATION, as_of, OPTIMIZATION_METADATA_FILENAME, output_root
        )
    )
    write_optimization_metadata(
        build_optimization_metadata(result, tdm.tickers, risk_free_rate_data),
        metadata_path,
    )
    print(f"Saved {metadata_path}")
    # The two optimization files must agree before either is uploaded.
    verify_optimization_files(plot_data_path, metadata_path)

    # 4. Portfolio backtest
    backtest_prices = last_n_sessions(wide_df, backtest_sessions)
    backtest_result = run_portfolio_backtest(
        backtest_prices,
        tb3m_df,
        training_window_returns=bt_config.training_window_returns,
        evaluation_days=evaluation_days,
        rebalance_interval=bt_config.rebalance_interval_sessions,
        weight_cap_multiple=po_config.weight_cap_multiple,
        expected_sessions=sessions_getter(backtest_prices.index[-1], backtest_sessions),
    )
    backtest_daily_path = ensure_parent_dir(
        local_output_path(
            PORTFOLIO_OPTIMIZATION, as_of, PORTFOLIO_BACKTEST_DAILY_FILENAME, output_root
        )
    )
    write_backtest_daily(backtest_result, backtest_daily_path)
    print(f"Saved {backtest_daily_path}")
    backtest_metadata_path = ensure_parent_dir(
        local_output_path(
            PORTFOLIO_OPTIMIZATION,
            as_of,
            PORTFOLIO_BACKTEST_METADATA_FILENAME,
            output_root,
        )
    )
    write_backtest_metadata(
        build_backtest_metadata(
            backtest_result,
            tdm.tickers,
            build_final_optimization(
                result, risk_free_rate_data, optimization_prices.index
            ),
        ),
        backtest_metadata_path,
    )
    print(f"Saved {backtest_metadata_path}")
    verify_backtest_files(backtest_daily_path, backtest_metadata_path)
    # The backtest's final optimization must be the optimization just written.
    verify_final_optimization(backtest_metadata_path, metadata_path, plot_data_path)

    # 5. Uploads, optimization metadata last (it marks the date as complete)
    uploaded: list[str] = []
    if not upload:
        print("Skipping uploads (--no-upload).")
    else:
        for local_path, folder, filename in (
            (forecasts_path, time_series_folder, ALL_FORECASTS_FILENAME),
            (
                forecasts_metadata_path,
                time_series_folder,
                ALL_FORECASTS_METADATA_FILENAME,
            ),
            (plot_data_path, portfolio_folder, OPTIMIZATION_PLOT_DATA_FILENAME),
            (
                backtest_daily_path,
                portfolio_folder,
                PORTFOLIO_BACKTEST_DAILY_FILENAME,
            ),
            (
                backtest_metadata_path,
                portfolio_folder,
                PORTFOLIO_BACKTEST_METADATA_FILENAME,
            ),
            (metadata_path, portfolio_folder, OPTIMIZATION_METADATA_FILENAME),
        ):
            key = object_key(folder, as_of, filename)
            uploader.upload_file(local_path, key)
            uploaded.append(key)

    return PipelineResult(
        as_of=as_of,
        forecasts_path=forecasts_path,
        forecasts_metadata_path=forecasts_metadata_path,
        plot_data_path=plot_data_path,
        metadata_path=metadata_path,
        backtest_daily_path=backtest_daily_path,
        backtest_metadata_path=backtest_metadata_path,
        trained_arima=trained_arima,
        uploaded=tuple(uploaded),
    )
