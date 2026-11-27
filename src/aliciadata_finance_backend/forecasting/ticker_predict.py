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
from aliciadata_finance_backend.config import ExogenousVariable
from aliciadata_finance_backend.data.date_manager import DateManager
from aliciadata_finance_backend.forecasting.arima_models import (
    PRICE_DIFFERENCING_ORDER,
    ArimaModels,
)
from aliciadata_finance_backend.forecasting.arimax_models import (
    ArimaxModels,
    ArimaxWindow,
    build_arimax_window,
)
from aliciadata_finance_backend.storage.layout import ensure_parent_dir

# The training window length and the number of scored backtest days come from
# the configuration file (see aliciadata_finance_backend.config.TimeSeriesConfig).
# ARIMA order search: p in range(ARIMA_MAX_P), q in range(ARIMA_MAX_Q),
# excluding (0, 0). Each candidate is fit once on the whole training window
# and the lowest BIC wins (see ArimaModels.select_order()). The ARIMAX search
# uses the same ranges but includes (0, 0), a regression on the exogenous
# variables alone (see ArimaxModels.fit_arimax()).
ARIMA_MAX_P = 2
ARIMA_MAX_Q = 2
# Column suffixes of the forecasts table: f"{ticker}_{suffix}" is a forecast
# column, followed by its _p, _d, and _q order columns.
ARIMA_SUFFIX = "arima"
ARIMAX_SUFFIX = "arimax"
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
    window: pd.Series | ArimaxWindow, max_p: int, max_q: int
) -> tuple[pd.Timestamp, float, tuple[int, int], bool]:
    """
    Fit and forecast one walk-forward training window, and report whether
    its order was decided by a tie-break: an ARIMA model
    (ArimaModels.fit()) for a price series, an ARIMAX model
    (ArimaxModels.fit_arimax()) for an ArimaxWindow. Defined at module level
    so that worker processes can unpickle it.
    """
    if isinstance(window, ArimaxWindow):
        model = ArimaxModels()
        pred_date, pred, order = model.fit_arimax(window, max_p=max_p, max_q=max_q)
    else:
        model = ArimaModels()
        pred_date, pred, order = model.fit(window, max_p=max_p, max_q=max_q)
    return pred_date, pred, order, bool(model.order_tie_broken)


def fit_windows(
    windows: list[pd.Series | ArimaxWindow], max_p: int, max_q: int, n_workers: int
) -> list[tuple[pd.Timestamp, float, tuple[int, int], bool]]:
    """
    Fit every training window and return the results in the order of
    windows. The windows are independent (each uses only its own data), so
    they can be fit in any order or at the same time; the results do not
    depend on n_workers. ARIMA windows (price series) and ARIMAX windows
    (ArimaxWindow) can be mixed in one call, and so share one pool.

    With n_workers == 1 (or at most one window) everything runs in this
    process. Otherwise one ProcessPoolExecutor with n_workers processes is
    started for the whole call (capped at the number of windows, and at
    WINDOWS_MAX_WORKERS on Windows), with each worker's numeric libraries
    limited to one thread (see SINGLE_THREAD_ENV_VARS).

    Parameters
    ----------
    windows : list[pd.Series | ArimaxWindow]
        Each training window: for ARIMA, its prices, one entry per NYSE
        trading session, ending on a present (non-missing) close; for
        ARIMAX, an ArimaxWindow (build_arimax_window()).

    max_p, max_q : int
        Exclusive upper bounds on p and q (see ArimaModels.fit() and
        ArimaxModels.fit_arimax()).

    n_workers : int
        Number of worker processes, at least 1.

    Returns
    -------
    list[tuple[pd.Timestamp, float, tuple[int, int], bool]]
        The model's (pred_date, forecast price, (p, q)) for each window,
        followed by whether that order was decided by a tie-break
        (order_tie_broken).

    Raises
    ------
    ValueError
        If n_workers is less than 1.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}.")
    fit = partial(_fit_window, max_p=max_p, max_q=max_q)
    progress = {"total": len(windows), "desc": "Model windows", "unit": "window"}
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
    over sliding windows of trading days, and, when exogenous variables are
    configured, ARIMAX forecasts on the same windows. Does no downloading or
    uploading.
    """

    def __init__(
        self,
        training_window_returns: int,
        n_workers: int | None = None,
        exogenous_variables: tuple[ExogenousVariable, ...] = (),
    ):
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

        exogenous_variables : tuple[ExogenousVariable, ...], optional
            The ARIMAX regressors (ModelingConfig.exogenous_variables). If
            empty (the default), only ARIMA is trained and the forecasts have
            no ARIMAX columns; otherwise train_arima_models() needs their
            values.
        """
        self.dm = DateManager()
        self.training_window_returns = training_window_returns
        self.n_workers = n_workers
        self.exogenous_variables = tuple(exogenous_variables)

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
        df: pd.DataFrame,
        exogenous_df: pd.DataFrame | None = None,
        max_p: int = ARIMA_MAX_P,
        max_q: int = ARIMA_MAX_Q,
    ) -> pd.DataFrame:
        """
        Trains ARIMA models, and ARIMAX models when exogenous variables are
        configured, along each training window implementing a walk-forward
        backtesting scheme of model evaluation. In each window, every
        candidate (p, q) is fit once on the whole window and the lowest BIC
        wins (see ArimaModels.fit() and ArimaxModels.fit_arimax()).

        All (ticker, window, model) fits are independent, so they run
        together in one pool of self.n_workers processes (see fit_windows()).
        The output does not depend on the number of workers.

        This function passes prices, not returns: the single differencing
        step (prices to daily log returns) happens inside the models, via
        ArimaModels.to_log_returns().

        Each window holds self.training_window_returns returns (one more
        close), and is resampled to the NYSE trading calendar, so one model
        time step is one trading session: holidays are skipped rather than
        treated as missing observations. The ARIMAX model of a window sees
        the same closes, and the exogenous variables' values on the same
        sessions, lagged (see build_arimax_window()); it is fit on one log
        return fewer with a lag of 1.

        Missing data (see training_window_start_end()): windows end only on
        dates present in df, so pred_date values are unique. If one ticker's
        close is missing on a window's last date, that ticker's T_arima,
        T_arimax, and order columns are left empty on that row (no
        two-step-ahead forecast is substituted), while T_naive still uses the
        most recent observed close. If an exogenous variable's regressor for
        the forecast day is missing, that row's T_arimax and its orders are
        empty for every ticker; a regressor missing inside the window only
        drops that observation from the ARIMAX fit.

        Parameters
        ----------
        df : pd.DataFrame
            Wide format DataFrame of adjusted close data.

        exogenous_df : pd.DataFrame, optional
            The exogenous variables' values, one column per symbol, indexed
            like df (load_exogenous() in pipeline.py), with no date after
            df's last. Required when self.exogenous_variables is not empty,
            ignored otherwise.

        max_p : int, optional
            Exclusive upper bound on p. Defaults to ARIMA_MAX_P (2), so p is 0 or 1.

        max_q : int, optional
            Exclusive upper bound on q. Defaults to ARIMA_MAX_Q (2), so q is 0 or 1.

        Returns
        -------
        pd.DataFrame
            The forecasts after each model training, indexed by pred_date.
            For every ticker T there are six columns, in this order, or ten
            with ARIMAX:
            T_arima (forecast price); T_arima_p, T_arima_d, and T_arima_q
            (nullable integer order of the model that made the row's
            forecast, stated on log prices: ARIMA(p, 0, q) on log returns is
            ARIMA(p, 1, q) on log prices, so T_arima_d is always
            PRICE_DIFFERENCING_ORDER = 1; all three are empty where T_arima
            is);
            with ARIMAX, T_arimax and T_arimax_p, T_arimax_d, T_arimax_q, in
            the same form;
            T_naive (naive baseline forecast: the most recent observed close
            before pred_date, i.e. the previous trading day's close when no
            data is missing); and T (the actual price, missing on the final
            row).

        Raises
        ------
        ValueError
            If exogenous variables are configured but exogenous_df is None.
        """
        with_arimax = bool(self.exogenous_variables)
        if with_arimax and exogenous_df is None:
            raise ValueError("ARIMAX needs the exogenous variables' values.")
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
        # The ARIMAX input of every window whose last close is present; None
        # where there is no ARIMAX forecast.
        arimax_by_ticker: dict[str, list[ArimaxWindow | None]] = {}
        if with_arimax:
            for ticker, windows in windows_by_ticker.items():
                arimax_by_ticker[ticker] = [
                    None
                    if pd.isna(ticker_ts.iloc[-1])
                    else build_arimax_window(
                        ticker_ts, exogenous_df, self.exogenous_variables
                    )
                    for ticker_ts in windows
                ]
        # Fit every window whose last close is present, ARIMA and ARIMAX, all
        # at once. The results come back in the same order, so they are
        # consumed below in the order the windows are visited.
        arima_tasks = [
            ticker_ts
            for windows in windows_by_ticker.values()
            for ticker_ts in windows
            if not pd.isna(ticker_ts.iloc[-1])
        ]
        arimax_tasks = [
            window
            for windows in arimax_by_ticker.values()
            for window in windows
            if window is not None
        ]
        n_workers = (
            self.n_workers if self.n_workers is not None else default_worker_count()
        )
        fit_results = fit_windows(
            arima_tasks + arimax_tasks,
            max_p=max_p,
            max_q=max_q,
            n_workers=n_workers,
        )
        arima_results = fit_results[: len(arima_tasks)]
        arimax_results = fit_results[len(arima_tasks) :]
        for name, results in (("ARIMA", arima_results), ("ARIMAX", arimax_results)):
            if name == "ARIMAX" and not with_arimax:
                continue
            n_tie_broken = sum(tie_broken for *_, tie_broken in results)
            print(
                f"{name} order decided by a tie-break in {n_tie_broken} of "
                f"{len(results)} windows."
            )
        arima_fits = iter(arima_results)
        arimax_fits = iter(arimax_results)
        suffixes = [ARIMA_SUFFIX] + ([ARIMAX_SUFFIX] if with_arimax else [])
        all_forecast_dfs = []
        for ticker, windows in windows_by_ticker.items():
            forecast_rows = []
            for i, ticker_ts in enumerate(windows):
                pred_date = self.dm.future_business_day(ticker_ts.index[-1], 1)
                if pd.isna(ticker_ts.iloc[-1]):
                    # This ticker's close at the forecast origin is missing, so
                    # there is no one-step-ahead forecast: leave the forecasts
                    # and their orders empty rather than forecast two steps
                    # ahead from an older close.
                    arima_fit = None
                else:
                    arima_fit = next(arima_fits)
                    pred_date = arima_fit[0]
                arimax_fit = None
                if with_arimax and arimax_by_ticker[ticker][i] is not None:
                    arimax_fit = next(arimax_fits)
                # Naive (random walk) baseline: the most recent observed close
                # in the same window the models saw, so every forecast uses
                # exactly the same prices.
                naive = float(ticker_ts.dropna().iloc[-1])
                pred_dict = {"pred_date": pred_date}
                pred_dict.update(_forecast_cells(ticker, ARIMA_SUFFIX, arima_fit))
                if with_arimax:
                    pred_dict.update(_forecast_cells(ticker, ARIMAX_SUFFIX, arimax_fit))
                pred_dict[f"{ticker}_naive"] = naive
                forecast_rows.append(pred_dict)
            forecast_df = (
                pd.DataFrame(forecast_rows).set_index("pred_date").sort_index()
            )
            for suffix in suffixes:
                pred_key = f"{ticker}_{suffix}"
                order_keys = [f"{pred_key}_p", f"{pred_key}_d", f"{pred_key}_q"]
                # Nullable integers: whole numbers, empty where there is no
                # forecast.
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


def _forecast_cells(
    ticker: str,
    suffix: str,
    fit: tuple[pd.Timestamp, float, tuple[int, int], bool] | None,
) -> dict[str, object]:
    """
    One row's forecast and order cells for one ticker and model: the
    forecast column f"{ticker}_{suffix}" and its _p, _d, and _q columns.
    With no fit (None), the forecast is NaN and the orders are pd.NA. The
    order is stated on log prices, so _d is PRICE_DIFFERENCING_ORDER.
    """
    key = f"{ticker}_{suffix}"
    if fit is None:
        pred, p, d, q = np.nan, pd.NA, pd.NA, pd.NA
    else:
        _, pred, (p, q), _ = fit
        d = PRICE_DIFFERENCING_ORDER
    return {key: pred, f"{key}_p": p, f"{key}_d": d, f"{key}_q": q}


class ForecastTrainer(Protocol):
    """
    Anything that can produce the forecasts DataFrame, e.g. ArimaWalkForward
    (or a test double).
    """

    def train_arima_models(
        self, df: pd.DataFrame, exogenous_df: pd.DataFrame | None = None
    ) -> pd.DataFrame: ...


def cached_forecasts_match(
    forecasts_df: pd.DataFrame,
    window_df: pd.DataFrame,
    training_window_returns: int,
    model_suffixes: tuple[str, ...] = (ARIMA_SUFFIX,),
) -> bool:
    """
    Whether a cached forecasts table has the shape that training on window_df
    would give, so it can be reused. Pure: compares dates and columns only,
    never forecast values.

    Training gives one row per window end, window_df.index[r:] with
    r = training_window_returns, each dated the next NYSE session after its
    window end. So a matching table has:

    - exactly len(window_df) - r rows, which catches a change in the number
      of scored days (TimeSeriesConfig.evaluation_window_length);
    - a first pred_date after the first window end, window_df.index[r], and
      no later than the next date in window_df (the next session is at most
      the next date present);
    - a last pred_date, the live forecast, after the last close in window_df;
    - a forecast column for every ticker in window_df and every model in
      model_suffixes (T_arima, and T_arimax when ARIMAX is trained), which
      catches a table written before ARIMAX was added.

    Dates are compared without their time of day. A change in the training
    window length alone cannot be detected this way, because the scored days
    and so the rows stay the same; it still needs --retrain-arima. Neither
    can a change in the exogenous variables (their symbols, transforms, or
    lags), which also needs --retrain-arima.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        The cached table, indexed by pred_date (as read from the CSV).

    window_df : pd.DataFrame
        Wide adjusted close prices the run would train on.

    training_window_returns : int
        Daily log returns in each training window
        (TimeSeriesConfig.training_window_returns).

    model_suffixes : tuple[str, ...], optional
        The models whose forecast columns must be present, as column
        suffixes. Defaults to (ARIMA_SUFFIX,), ARIMA only.

    Returns
    -------
    bool
        True if the table matches, False otherwise.
    """
    r = training_window_returns
    n_windows = len(window_df) - r
    if n_windows < 1 or len(forecasts_df) != n_windows:
        return False
    if not isinstance(forecasts_df.index, pd.DatetimeIndex):
        return False
    pred_dates = forecasts_df.index.normalize()
    data_dates = pd.DatetimeIndex(window_df.index).normalize()
    if not pred_dates[0] > data_dates[r]:
        return False
    if r + 1 < len(data_dates) and not pred_dates[0] <= data_dates[r + 1]:
        return False
    if not pred_dates[-1] > data_dates[-1]:
        return False
    return all(
        f"{ticker}_{suffix}" in forecasts_df.columns
        for ticker in window_df.columns
        for suffix in model_suffixes
    )


def load_or_train_forecasts(
    window_df: pd.DataFrame,
    path: str,
    retrain: bool = False,
    *,
    trainer: ForecastTrainer,
    training_window_returns: int,
    exogenous_df: pd.DataFrame | None = None,
    model_suffixes: tuple[str, ...] = (ARIMA_SUFFIX,),
) -> bool:
    """
    Make sure the forecasts CSV for a run exists at path, training the
    models (ARIMA, and ARIMAX if the trainer has exogenous variables) only
    when needed. The CSV at path is the cache: if it exists,
    retrain is False, and it matches window_df (cached_forecasts_match()),
    nothing is trained and the file is left as is. Otherwise the models are
    trained on window_df and the CSV is (over)written in the format of
    all_forecasts.csv in docs/INTERFACE_CONTRACT.md. A cached CSV that does
    not match, for example one written with a different number of scored
    days, or that cannot be read, is retrained instead of reused, with a
    message saying so.

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

    training_window_returns : int
        Daily log returns in each training window
        (TimeSeriesConfig.training_window_returns), used to check the cache.

    exogenous_df : pd.DataFrame, optional
        The exogenous variables' values, passed to
        trainer.train_arima_models(); needed when it trains ARIMAX.

    model_suffixes : tuple[str, ...], optional
        The models the cached CSV must have forecasts of (see
        cached_forecasts_match()). Defaults to ARIMA only; pass
        (ARIMA_SUFFIX, ARIMAX_SUFFIX) when the trainer trains ARIMAX, so a
        CSV written without ARIMAX is retrained.

    Returns
    -------
    bool
        True if models were trained and the CSV written, False if the cached
        CSV was reused.
    """
    if os.path.exists(path) and not retrain:
        try:
            cached = pd.read_csv(path, index_col="pred_date", parse_dates=True)
            matches = cached_forecasts_match(
                cached, window_df, training_window_returns, model_suffixes
            )
        except (ValueError, KeyError, pd.errors.ParserError):
            matches = False
        if matches:
            print(f"Using cached forecasts {path}")
            return False
        print(
            f"Cached forecasts {path} do not match the current data and "
            "settings; retraining."
        )
    forecasts_df = trainer.train_arima_models(window_df, exogenous_df=exogenous_df)
    forecasts_df.to_csv(ensure_parent_dir(path), index=True)
    print(f"Saved {path}")
    return True
