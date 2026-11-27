"""
Forecast accuracy metrics for all_forecasts_metadata.yml (see
docs/INTERFACE_CONTRACT.md): how well the ARIMA, ARIMAX, and naive
forecasts in all_forecasts.csv did over the most recent days with an actual
close.

Everything here is pure except write_forecast_metadata(). Metrics are
computed from the forecasts table exactly as written to the CSV, so they
describe the same numbers the frontend sees.

Definitions, per ticker T, over the evaluated rows:
- error = forecast - actual close, in US dollars per share.
- RMSE = sqrt(mean(error^2)); MAE = mean(|error|).
- MAPE = mean(|error| / actual) x 100, in percent (scale-free, so it can be
  compared across tickers; RMSE and MAE cannot).
- relative_mae = ARIMA MAE / naive MAE: below 1 means ARIMA was more
  accurate than the naive forecast. skill = 1 - relative_mae: above 0 means
  ARIMA beat naive.
- directional_accuracy (ARIMA only): the fraction of days with a nonzero
  actual move on which ARIMA predicted the direction of the move correctly.
  The move is measured from the naive forecast, the previous close. The naive
  forecast predicts no move, so it has no direction to score.
- ARIMAX has the same error metrics and directional accuracy, scored on the
  rows where ARIMAX, ARIMA, naive, and the actual are all present, and two
  comparisons on those same rows: vs_naive (ARIMAX MAE / naive MAE) and
  vs_arima (ARIMAX MAE / ARIMA MAE), each with its skill, 1 - relative_mae.
  vs_arima below 1 means the exogenous variables helped over ARIMA on those
  days.
"""

import math
import numpy as np
import pandas as pd
import yaml
from aliciadata_finance_backend.config import ExogenousVariable
from aliciadata_finance_backend.forecasting.arima_models import PRICE_DIFFERENCING_ORDER
from aliciadata_finance_backend.forecasting.ticker_predict import (
    ARIMA_MAX_P,
    ARIMA_MAX_Q,
)

FORECAST_HORIZON_TRADING_DAYS = 1


def _date_str(timestamp: pd.Timestamp) -> str:
    """YYYY-MM-DD, dropping the fixed 17:00 time label."""
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def _float_or_none(value: float) -> float | None:
    """A plain float for YAML, or None for a missing / non-finite value."""
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def error_metrics(forecast: pd.Series, actual: pd.Series) -> dict:
    """
    RMSE, MAE (US dollars per share) and MAPE (percent) of forecast against
    actual. Both series must already be restricted to the evaluated rows
    (no missing values). Pure function.

    Returns
    -------
    dict
        {"rmse", "mae", "mape"}, each a float, or None when there are no rows.
    """
    if len(actual) == 0:
        return {"rmse": None, "mae": None, "mape": None}
    error = forecast.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
    abs_error = np.abs(error)
    return {
        "rmse": _float_or_none(np.sqrt(np.mean(error**2))),
        "mae": _float_or_none(np.mean(abs_error)),
        "mape": _float_or_none(np.mean(abs_error / actual.to_numpy(dtype=float)) * 100),
    }


def directional_accuracy(
    arima: pd.Series, naive: pd.Series, actual: pd.Series
) -> tuple[float | None, int]:
    """
    Fraction of days on which ARIMA predicted the direction of the actual
    move from the previous close (the naive forecast). Days with no actual
    move are skipped. Pure function.

    Returns
    -------
    tuple[float | None, int]
        The fraction correct (None when no day had a move) and the number of
        days scored.
    """
    actual_move = np.sign(actual.to_numpy(dtype=float) - naive.to_numpy(dtype=float))
    predicted_move = np.sign(arima.to_numpy(dtype=float) - naive.to_numpy(dtype=float))
    moved = actual_move != 0
    n_scored = int(moved.sum())
    if n_scored == 0:
        return None, 0
    return float(np.mean(predicted_move[moved] == actual_move[moved])), n_scored


def relative_mae(model_mae: float | None, baseline_mae: float | None) -> dict:
    """
    {"relative_mae": model MAE / baseline MAE, "skill": 1 - relative_mae}
    for two MAEs over the same days. Both None when either MAE is missing or
    the baseline's is 0. Pure function.
    """
    if model_mae is not None and baseline_mae:
        ratio = model_mae / baseline_mae
        return {
            "relative_mae": _float_or_none(ratio),
            "skill": _float_or_none(1 - ratio),
        }
    return {"relative_mae": None, "skill": None}


def arimax_metrics(with_actual: pd.DataFrame, ticker: str) -> dict:
    """
    ARIMAX metrics for one ticker over the same candidate rows as
    ticker_metrics() (the last eval_days rows with an actual close), scored
    only where ARIMAX, ARIMA, the naive forecast, and the actual are all
    present, so the three models are compared on exactly the same days. Pure
    function.

    Those days are normally the same as ARIMA's. They are fewer when an
    exogenous value that a row's ARIMAX forecast needed was missing; then the
    comparisons here use the naive and ARIMA errors on ARIMAX's days, which
    can differ slightly from the ticker's own naive and arima blocks.

    Parameters
    ----------
    with_actual : pd.DataFrame
        The forecasts table restricted to the rows ticker_metrics()
        considers, with T_arimax, T_arima, T_naive, and T columns for this
        ticker.

    ticker : str
        The ticker T, as in the column names.

    Returns
    -------
    dict
        rmse, mae, mape, directional_accuracy, and n_directional, as for
        ARIMA; n_evaluated, the days scored; and vs_naive and vs_arima, each
        with relative_mae (ARIMAX MAE / the other model's MAE on the same
        days) and skill (1 - relative_mae).
    """
    arimax = with_actual[f"{ticker}_arimax"]
    arima = with_actual[f"{ticker}_arima"]
    naive = with_actual[f"{ticker}_naive"]
    actual = with_actual[ticker]
    complete = arimax.notna() & arima.notna() & naive.notna() & actual.notna()
    arimax, arima = arimax[complete], arima[complete]
    naive, actual = naive[complete], actual[complete]
    arimax_errors = error_metrics(arimax, actual)
    naive_mae = error_metrics(naive, actual)["mae"]
    arima_mae = error_metrics(arima, actual)["mae"]
    direction, n_directional = directional_accuracy(arimax, naive, actual)
    return {
        **arimax_errors,
        "directional_accuracy": direction,
        "n_directional": n_directional,
        "n_evaluated": int(complete.sum()),
        "vs_naive": relative_mae(arimax_errors["mae"], naive_mae),
        "vs_arima": relative_mae(arimax_errors["mae"], arima_mae),
    }


def ticker_metrics(forecasts_df: pd.DataFrame, ticker: str, eval_days: int) -> dict:
    """
    Metrics for one ticker over the last eval_days rows that have an actual
    close. Of those, only rows that also have both an ARIMA and a naive
    forecast are scored, so both models are compared on exactly the same
    days. The live forecast (no actual yet) is never scored. Pure function.

    If the table has a T_arimax column, an arimax block is added
    (arimax_metrics()), scored on its own complete rows among the same
    candidate rows. The ARIMA and naive metrics do not depend on it.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        The forecasts table as written to all_forecasts.csv, indexed by
        pred_date, with columns T_arima, T_naive and T for this ticker, and
        optionally T_arimax.

    ticker : str
        The ticker T, as in the column names.

    eval_days : int
        Number of most recent days with an actual close to consider
        (TimeSeriesConfig.evaluation_window_length).

    Returns
    -------
    dict
        Nested metrics for the YAML file (see docs/INTERFACE_CONTRACT.md).
    """
    actual_all = forecasts_df[ticker]
    with_actual = forecasts_df.loc[actual_all.notna()].tail(eval_days)
    arima = with_actual[f"{ticker}_arima"]
    naive = with_actual[f"{ticker}_naive"]
    actual = with_actual[ticker]
    complete = arima.notna() & naive.notna() & actual.notna()
    arima, naive, actual = arima[complete], naive[complete], actual[complete]

    arima_errors = error_metrics(arima, actual)
    naive_errors = error_metrics(naive, actual)
    direction, n_directional = directional_accuracy(arima, naive, actual)

    metrics = {
        "n_evaluated": int(complete.sum()),
        "evaluation_start_date": (
            _date_str(with_actual.index[0]) if len(with_actual) else None
        ),
        "evaluation_end_date": (
            _date_str(with_actual.index[-1]) if len(with_actual) else None
        ),
        **relative_mae(arima_errors["mae"], naive_errors["mae"]),
        "arima": {
            **arima_errors,
            "directional_accuracy": direction,
            "n_directional": n_directional,
        },
        "naive": naive_errors,
    }
    if f"{ticker}_arimax" in with_actual.columns:
        metrics["arimax"] = arimax_metrics(with_actual, ticker)
    return metrics


def arimax_model_block(
    training_window_returns: int, exogenous_variables: tuple[ExogenousVariable, ...]
) -> dict:
    """
    How the ARIMAX forecasts are made (see forecasting/arimax_models.py), for
    models.arimax. Pure function.

    Parameters
    ----------
    training_window_returns : int
        Daily log returns in each training window.

    exogenous_variables : tuple[ExogenousVariable, ...]
        The regressors, in the order they are used. At least one.
    """
    max_lag = max(v.lag for v in exogenous_variables)
    return {
        "description": (
            "regression of daily log returns on lagged exogenous variables "
            "with ARMA(p, q) errors, which is ARIMAX(p, "
            f"{PRICE_DIFFERENCING_ORDER}, q) on log prices"
        ),
        "d": PRICE_DIFFERENCING_ORDER,
        "p_values": list(range(ARIMA_MAX_P)),
        "q_values": list(range(ARIMA_MAX_Q)),
        "excluded_orders": [],
        "order_selection": (
            "lowest BIC, each candidate order fit once on the same rows of "
            "the training window"
        ),
        "order_selection_tie_break": "fewer parameters (p + q), then lower p",
        # The first max_lag returns of a window have no lagged regressor
        # built from data inside the window.
        "fit_returns": training_window_returns - max_lag,
        "exogenous_variables": [
            {
                "symbol": v.symbol,
                "source": v.source,
                "transform": v.transform,
                "lag_sessions": v.lag,
            }
            for v in exogenous_variables
        ],
        "missing_exogenous_values": (
            "a session whose regressor is missing is skipped in the fit; a "
            "missing regressor for the forecast day leaves that row without "
            "an ARIMAX forecast"
        ),
    }


def models_block(
    training_window_returns: int,
    exogenous_variables: tuple[ExogenousVariable, ...] = (),
) -> dict:
    """
    How the forecasts are made, from the constants and settings the code
    uses, so the metadata file explains itself. Pure function.

    Parameters
    ----------
    training_window_returns : int
        Daily log returns in each training window, on which every candidate
        order is scored.

    exogenous_variables : tuple[ExogenousVariable, ...], optional
        The ARIMAX regressors. If any are given, an arimax block is added
        (arimax_model_block()); otherwise only arima and naive are
        described.
    """
    blocks = {
        "arima": {
            "description": (
                f"ARIMA(p, 0, q) on daily log returns, which is "
                f"ARIMA(p, {PRICE_DIFFERENCING_ORDER}, q) on log prices"
            ),
            "d": PRICE_DIFFERENCING_ORDER,
            "p_values": list(range(ARIMA_MAX_P)),
            "q_values": list(range(ARIMA_MAX_Q)),
            "excluded_orders": [[0, 0]],
            "order_selection": (
                "lowest BIC, each candidate order fit once on the whole "
                "training window"
            ),
            # Kept so readers of this key still find an integer: candidates
            # are now scored on one window, the whole training window.
            "order_selection_subwindow_returns": training_window_returns,
            "order_selection_tie_break": "fewer parameters (p + q), then lower p",
        },
        "naive": {
            "description": "the most recent observed close (random walk)",
        },
    }
    if exogenous_variables:
        blocks["arimax"] = arimax_model_block(
            training_window_returns, tuple(exogenous_variables)
        )
    return blocks


def build_forecast_metadata(
    forecasts_df: pd.DataFrame,
    tickers: list[str],
    training_window_start: pd.Timestamp,
    training_window_end: pd.Timestamp,
    *,
    eval_days: int,
    training_window_returns: int,
    exogenous_variables: tuple[ExogenousVariable, ...] = (),
) -> dict:
    """
    Build the contents of all_forecasts_metadata.yml. Pure function.

    Tickers whose forecasts include T_arimax get an arimax block of metrics
    (see ticker_metrics()); models.arimax describes the model when
    exogenous_variables is not empty.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        The forecasts table as written to all_forecasts.csv, indexed by
        pred_date. Its last row is the live forecast.

    tickers : list[str]
        Tickers to report. Tickers without T_arima, T_naive and T columns
        in forecasts_df are skipped.

    training_window_start, training_window_end : pd.Timestamp
        First and last session of the live forecast's training window.

    eval_days : int
        Number of most recent days with an actual close to score
        (TimeSeriesConfig.evaluation_window_length).

    training_window_returns : int
        Daily log returns in each training window
        (TimeSeriesConfig.training_window_returns).

    exogenous_variables : tuple[ExogenousVariable, ...], optional
        The ARIMAX regressors (ModelingConfig.exogenous_variables). Defaults
        to none, which describes no ARIMAX model.

    Returns
    -------
    dict
        Nested dictionary ready for yaml.dump(). Ticker keys are lowercase.
    """
    present = [
        t
        for t in tickers
        if {t, f"{t}_arima", f"{t}_naive"}.issubset(forecasts_df.columns)
    ]
    per_ticker = {t.lower(): ticker_metrics(forecasts_df, t, eval_days) for t in present}
    starts = [m["evaluation_start_date"] for m in per_ticker.values() if m["evaluation_start_date"]]
    ends = [m["evaluation_end_date"] for m in per_ticker.values() if m["evaluation_end_date"]]
    return {
        "forecast_window": {
            "training_window_start_date": _date_str(training_window_start),
            "training_window_end_date": _date_str(training_window_end),
            "forecast_date": _date_str(forecasts_df.index[-1]),
            "training_window_returns": training_window_returns,
            "horizon_trading_days": FORECAST_HORIZON_TRADING_DAYS,
        },
        "evaluation_window": {
            "target_trading_days": eval_days,
            "start_date": min(starts) if starts else None,
            "end_date": max(ends) if ends else None,
        },
        "units": {
            "rmse": "usd_per_share",
            "mae": "usd_per_share",
            "mape": "percent",
            "relative_mae": "ratio",
            "skill": "ratio",
            "directional_accuracy": "fraction",
        },
        "models": models_block(training_window_returns, exogenous_variables),
        "ticker_metrics": per_ticker,
    }


def write_forecast_metadata(metadata: dict, path: str) -> None:
    """
    Write all_forecasts_metadata.yml (block style, keys sorted). Overwrites
    the file. Its folder must exist.
    """
    with open(path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
