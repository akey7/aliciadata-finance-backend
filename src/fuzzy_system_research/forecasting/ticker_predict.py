import os
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from typing import Protocol
import numpy as np
import pandas as pd
from tqdm import tqdm
from fuzzy_system_research.data.date_manager import DateManager
from fuzzy_system_research.forecasting.arima_models import (
    PRICE_DIFFERENCING_ORDER,
    ArimaModels,
)
from fuzzy_system_research.storage.layout import ensure_parent_dir

# The training window length and the number of scored backtest days come from
# the configuration file (see fuzzy_system_research.config.TimeSeriesConfig).
# ARIMA order search: p in range(ARIMA_MAX_P), q in range(ARIMA_MAX_Q),
# excluding (0, 0). Each candidate is fit once on the whole training window
# and the lowest BIC wins (see ArimaModels.select_order()).
ARIMA_MAX_P = 2
ARIMA_MAX_Q = 2
# Environment variables that set how many threads the numeric libraries under
# NumPy and SciPy may start (OpenMP, OpenBLAS, MKL, Apple Accelerate,
# numexpr). Worker processes get one thread each, so N workers use about N
# CPUs; the matrices in these small ARIMA fits gain nothing from more.
SINGLE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# ProcessPoolExecutor refuses more than 61 workers on Windows.
WINDOWS_MAX_WORKERS = 61


def default_worker_count(cpu_count: int | None = None) -> int:
    """
    Default number of worker processes for ARIMA training: half the logical
    CPUs available to this process, and at least 1.

    Logical CPUs count every hardware thread: on an Intel CPU with
    hyper-threading that is two per physical core, and on Apple silicon it is
    every core, performance and efficiency alike (10 on a base M4, so 5
    workers).

    Parameters
    ----------
    cpu_count : int, optional
        Logical CPUs to assume. Defaults to os.process_cpu_count(), which
        respects any limit on the CPUs this process may use, falling back to
        os.cpu_count(), and to 1 if neither is known.

    Returns
    -------
    int
        max(1, cpu_count // 2).
    """
    if cpu_count is None:
        cpu_count = os.process_cpu_count() or os.cpu_count() or 1
    return max(1, cpu_count // 2)


@contextmanager
def _single_threaded_numeric_libraries() -> Iterator[None]:
    """
    Set every variable in SINGLE_THREAD_ENV_VARS to "1" inside the block and
    restore the previous values (or absence) afterwards. Worker processes
    started inside the block inherit the setting and read it when they import
    NumPy. The current process is not affected, because its numeric libraries
    are already loaded.
    """
    saved = {name: os.environ.get(name) for name in SINGLE_THREAD_ENV_VARS}
    os.environ.update({name: "1" for name in SINGLE_THREAD_ENV_VARS})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _fit_window(
    ticker_ts: pd.Series, max_p: int, max_q: int
) -> tuple[pd.Timestamp, float, tuple[int, int], bool]:
    """
    Fit and forecast one walk-forward training window with ArimaModels.fit(),
    and report whether its order was decided by a tie-break. Defined at
    module level so that worker processes can unpickle it.
    """
    am = ArimaModels()
    pred_date, pred, order = am.fit(ticker_ts, max_p=max_p, max_q=max_q)
    return pred_date, pred, order, bool(am.order_tie_broken)


def fit_windows(
    windows: list[pd.Series], max_p: int, max_q: int, n_workers: int
) -> list[tuple[pd.Timestamp, float, tuple[int, int], bool]]:
    """
    Fit every training window and return the results in the order of
    windows. The windows are independent (each uses only its own prices), so
    they can be fit in any order or at the same time; the results do not
    depend on n_workers.

    With n_workers == 1 (or at most one window) everything runs in this
    process. Otherwise one ProcessPoolExecutor with n_workers processes is
    started for the whole call (capped at the number of windows, and at
    WINDOWS_MAX_WORKERS on Windows), with each worker's numeric libraries
    limited to one thread (see SINGLE_THREAD_ENV_VARS).

    Parameters
    ----------
    windows : list[pd.Series]
        Prices of each training window, one entry per NYSE trading session,
        each ending on a present (non-missing) close.

    max_p, max_q : int
        Exclusive upper bounds on p and q (see ArimaModels.fit()).

    n_workers : int
        Number of worker processes, at least 1.

    Returns
    -------
    list[tuple[pd.Timestamp, float, tuple[int, int], bool]]
        ArimaModels.fit()'s (pred_date, forecast price, (p, q)) for each
        window, followed by whether that order was decided by a tie-break
        (ArimaModels.order_tie_broken).

    Raises
    ------
    ValueError
        If n_workers is less than 1.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}.")
    fit = partial(_fit_window, max_p=max_p, max_q=max_q)
    progress = {"total": len(windows), "desc": "ARIMA windows", "unit": "window"}
    if n_workers == 1 or len(windows) <= 1:
        return [fit(ticker_ts) for ticker_ts in tqdm(windows, **progress)]
    max_workers = min(n_workers, len(windows))
    if sys.platform == "win32":
        max_workers = min(max_workers, WINDOWS_MAX_WORKERS)
    with (
        _single_threaded_numeric_libraries(),
        ProcessPoolExecutor(max_workers=max_workers) as pool,
    ):
        return list(tqdm(pool.map(fit, windows), **progress))


class ArimaWalkForward:
    """
    Walk-forward, one-step-ahead ARIMA forecasts of adjusted close prices
    over sliding windows of trading days. Does no downloading or uploading.
    """

    def __init__(self, training_window_returns: int, n_workers: int | None = None):
        """
        Instantiate with the DateManager used to build the training windows.

        Parameters
        ----------
        training_window_returns : int
            Daily log returns in each training window (one fewer than its
            closes), normally TimeSeriesConfig.training_window_returns.

        n_workers : int, optional
            Number of worker processes for train_arima_models(). Defaults to
            None, which means default_worker_count() (half the logical CPUs),
            worked out when the models are trained. 1 fits everything in this
            process.
        """
        self.dm = DateManager()
        self.training_window_returns = training_window_returns
        self.n_workers = n_workers

    @staticmethod
    def training_window_start_end(
        index: pd.DatetimeIndex, num_days: int
    ) -> list[list[pd.Timestamp]]:
        """
        Return the walk-forward training windows as [start, end] timestamp
        pairs. Each window spans num_days + 1 consecutive dates of index (so
        num_days returns), and consecutive windows slide by one date.

        Windows are built from the dates actually present in the data, not
        from the calendar, so every window ends on an observed date and no two
        windows end on the same one. With complete data (every NYSE session
        present) this is one window per session from the (num_days + 1)-th
        onward. If a whole session is missing from the data, no window ends on
        it, so there is no one-step-ahead forecast for the session after it
        (that would need the missing close).

        Parameters
        ----------
        index : pd.DatetimeIndex
            Sorted dates of the price data, one per observed trading session.

        num_days : int
            Number of returns in each window.

        Returns
        -------
        List[List[pd.Timestamp, pd.Timestamp]]
            Returns timestamp ranges, in date order.

        Raises
        ------
        ValueError
            If index has num_days dates or fewer.
        """
        if len(index) <= num_days:
            raise ValueError(
                f"Need more than {num_days} dates for a {num_days}-day window, "
                f"got {len(index)}."
            )
        return [[index[end - num_days], index[end]] for end in range(num_days, len(index))]

    def train_arima_models(
        self,
        df,
        max_p=ARIMA_MAX_P,
        max_q=ARIMA_MAX_Q,
    ):
        """
        Trains ARIMA models along each training window implementing a
        walk-forward backtesting scheme of model evaluation. In each window,
        every candidate (p, q) is fit once on the whole window and the
        lowest BIC wins (see ArimaModels.fit()).

        All (ticker, window) fits are independent, so they run together in
        one pool of self.n_workers processes (see fit_windows()). The output
        does not depend on the number of workers.

        This function passes prices, not returns: the single differencing
        step (prices to daily log returns) happens inside ArimaModels.fit(),
        via ArimaModels.to_log_returns().

        Each window holds self.training_window_returns returns (one more
        close), and is resampled to the NYSE trading calendar, so one model
        time step is one trading session: holidays are skipped rather than
        treated as missing observations.

        Missing data (see training_window_start_end()): windows end only on
        dates present in df, so pred_date values are unique. If one ticker's
        close is missing on a window's last date, that ticker's T_arima and
        order columns are left empty on that row (no two-step-ahead forecast
        is substituted), while T_naive still uses the most recent observed
        close.

        Parameters
        ----------
        df : pd.DataFrame
            Wide format DataFrame of adjusted close data.

        max_p : int, optional
            Exclusive upper bound on p. Defaults to ARIMA_MAX_P (2), so p is 0 or 1.

        max_q : int, optional
            Exclusive upper bound on q. Defaults to ARIMA_MAX_Q (2), so q is 0 or 1.

        Returns
        -------
        pd.DataFrame
            The forecasts after each model training, indexed by pred_date.
            For every ticker T there are six columns, in this order:
            T_arima (forecast price); T_arima_p, T_arima_d, and T_arima_q
            (nullable integer order of the model that made the row's
            forecast, stated on log prices: ARIMA(p, 0, q) on log returns is
            ARIMA(p, 1, q) on log prices, so T_arima_d is always
            PRICE_DIFFERENCING_ORDER = 1; all three are empty where T_arima
            is);
            T_naive (naive baseline forecast: the most recent observed close
            before pred_date, i.e. the previous trading day's close when no
            data is missing); and T (the actual price, missing on the final
            row).
        """
        timestamp_ranges = self.training_window_start_end(
            df.index, self.training_window_returns
        )
        # Every ticker's training windows, in output order.
        windows_by_ticker: dict[str, list[pd.Series]] = {}
        for ticker in df.columns:
            windows_by_ticker[ticker] = [
                # One row per NYSE trading session. NYSE holidays get no row
                # (pandas "B" would insert them as missing values), while a
                # session missing from the data still appears as NaN.
                df[ticker].loc[start_timestamp:end_timestamp].asfreq(self.dm.cbd)
                for start_timestamp, end_timestamp in timestamp_ranges
            ]
        # Fit every window whose last close is present, all at once. The
        # results come back in the same order, so they are consumed below in
        # the order the windows are visited.
        n_workers = (
            self.n_workers if self.n_workers is not None else default_worker_count()
        )
        fit_results = fit_windows(
            [
                ticker_ts
                for windows in windows_by_ticker.values()
                for ticker_ts in windows
                if not pd.isna(ticker_ts.iloc[-1])
            ],
            max_p=max_p,
            max_q=max_q,
            n_workers=n_workers,
        )
        n_tie_broken = sum(tie_broken for *_, tie_broken in fit_results)
        print(
            f"ARIMA order decided by a tie-break in {n_tie_broken} of "
            f"{len(fit_results)} windows."
        )
        fits = iter(fit_results)
        all_forecast_dfs = []
        for ticker, windows in windows_by_ticker.items():
            forecast_rows = []
            pred_key = f"{ticker}_arima"
            order_keys = [f"{pred_key}_p", f"{pred_key}_d", f"{pred_key}_q"]
            for ticker_ts in windows:
                if pd.isna(ticker_ts.iloc[-1]):
                    # This ticker's close at the forecast origin is missing, so
                    # there is no one-step-ahead ARIMA forecast: leave the
                    # forecast and its order empty rather than forecast two
                    # steps ahead from an older close.
                    pred_date = self.dm.future_business_day(ticker_ts.index[-1], 1)
                    pred, p, d, q = np.nan, pd.NA, pd.NA, pd.NA
                else:
                    pred_date, pred, (p, q), _ = next(fits)
                    d = PRICE_DIFFERENCING_ORDER
                # Naive (random walk) baseline: the most recent observed close
                # in the same window the ARIMA model saw, so both forecasts
                # use exactly the same information.
                naive = float(ticker_ts.dropna().iloc[-1])
                pred_dict = {
                    "pred_date": pred_date,
                    pred_key: pred,
                    order_keys[0]: p,
                    order_keys[1]: d,
                    order_keys[2]: q,
                    f"{ticker}_naive": naive,
                }
                forecast_rows.append(pred_dict)
            forecast_df = (
                pd.DataFrame(forecast_rows).set_index("pred_date").sort_index()
            )
            # Nullable integers: whole numbers, empty where there is no forecast.
            forecast_df[order_keys] = forecast_df[order_keys].astype("Int64")
            forecast_df[pred_key] = forecast_df[pred_key].astype(float)
            forecast_start_timestamp = forecast_df.index[0]
            forecast_end_timestamp = forecast_df.index[-1]
            forecast_df[ticker] = df.loc[
                forecast_start_timestamp:forecast_end_timestamp, ticker
            ].copy()
            all_forecast_dfs.append(forecast_df)
        all_forecast_df = pd.concat(all_forecast_dfs, axis=1).sort_index()
        return all_forecast_df


class ForecastTrainer(Protocol):
    """
    Anything that can produce the forecasts DataFrame, e.g. ArimaWalkForward
    (or a test double).
    """

    def train_arima_models(self, df: pd.DataFrame) -> pd.DataFrame: ...


def load_or_train_forecasts(
    window_df: pd.DataFrame,
    path: str,
    retrain: bool = False,
    *,
    trainer: ForecastTrainer,
) -> bool:
    """
    Make sure the forecasts CSV for a run exists at path, training ARIMA
    models only when needed. The CSV at path is the cache: if it exists and
    retrain is False, nothing is trained and the file is left as is.
    Otherwise the models are trained on window_df and the CSV is (over)written
    in the format of all_forecasts.csv in docs/INTERFACE_CONTRACT.md.

    Parameters
    ----------
    window_df : pd.DataFrame
        Wide adjusted close prices to train on, normally the last
        TimeSeriesConfig.arima_window_sessions trading sessions.

    path : str
        Local path of the forecasts CSV. Missing folders are created.

    retrain : bool, optional
        If True, train and overwrite the CSV even if it exists. Defaults to
        False.

    trainer : ForecastTrainer
        Object that trains the models, e.g. an ArimaWalkForward built with
        the configured training window.

    Returns
    -------
    bool
        True if models were trained and the CSV written, False if the cached
        CSV was reused.
    """
    if os.path.exists(path) and not retrain:
        print(f"Using cached forecasts {path}")
        return False
    forecasts_df = trainer.train_arima_models(window_df)
    forecasts_df.to_csv(ensure_parent_dir(path), index=True)
    print(f"Saved {path}")
    return True
