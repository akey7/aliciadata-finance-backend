"""
Forecast accuracy metrics for all_forecasts_metadata.yml (see
docs/INTERFACE_CONTRACT.md): how well the ARIMA and naive forecasts in
all_forecasts.csv did over the most recent days with an actual close.

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
"""

import math
import numpy as np
import pandas as pd
import yaml
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


def ticker_metrics(forecasts_df: pd.DataFrame, ticker: str, eval_days: int) -> dict:
    """
    Metrics for one ticker over the last eval_days rows that have an actual
    close. Of those, only rows that also have both an ARIMA and a naive
    forecast are scored, so both models are compared on exactly the same
    days. The live forecast (no actual yet) is never scored. Pure function.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        The forecasts table as written to all_forecasts.csv, indexed by
        pred_date, with columns T_arima, T_naive and T for this ticker.

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
    if arima_errors["mae"] is not None and naive_errors["mae"]:
        relative_mae = arima_errors["mae"] / naive_errors["mae"]
        skill = 1 - relative_mae
    else:
        relative_mae = skill = None

    return {
        "n_evaluated": int(complete.sum()),
        "evaluation_start_date": (
            _date_str(with_actual.index[0]) if len(with_actual) else None
        ),
        "evaluation_end_date": (
            _date_str(with_actual.index[-1]) if len(with_actual) else None
        ),
        "relative_mae": _float_or_none(relative_mae),
        "skill": _float_or_none(skill),
        "arima": {
            **arima_errors,
            "directional_accuracy": direction,
            "n_directional": n_directional,
        },
        "naive": naive_errors,
    }


def models_block(training_window_returns: int) -> dict:
    """
    How the two forecasts are made, from the constants and settings the code
    uses, so the metadata file explains itself. Pure function.

    Parameters
    ----------
    training_window_returns : int
        Daily log returns in each training window, on which every candidate
        order is scored.
    """
    return {
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


def build_forecast_metadata(
    forecasts_df: pd.DataFrame,
    tickers: list[str],
    training_window_start: pd.Timestamp,
    training_window_end: pd.Timestamp,
    *,
    eval_days: int,
    training_window_returns: int,
) -> dict:
    """
    Build the contents of all_forecasts_metadata.yml. Pure function.

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
        "models": models_block(training_window_returns),
        "ticker_metrics": per_ticker,
    }


def write_forecast_metadata(metadata: dict, path: str) -> None:
    """
    Write all_forecasts_metadata.yml (block style, keys sorted). Overwrites
    the file. Its folder must exist.
    """
    with open(path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
