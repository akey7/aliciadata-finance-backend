"""
Tests for aliciadata_finance_backend.forecasting.metrics on hand-made forecast
tables whose metrics can be computed by hand. In-memory, except the YAML
round trip, which writes under pytest's tmp_path.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from aliciadata_finance_backend.forecasting.metrics import (
    build_forecast_metadata,
    directional_accuracy,
    error_metrics,
    ticker_metrics,
    write_forecast_metadata,
)


def forecast_table(
    arima: list[float], naive: list[float], actual: list[float], ticker: str = "AAA"
) -> pd.DataFrame:
    """A forecasts table like all_forecasts.csv for one ticker."""
    dates = pd.bdate_range("2026-09-01 17:00", periods=len(actual), name="pred_date")
    return pd.DataFrame(
        {f"{ticker}_arima": arima, f"{ticker}_naive": naive, ticker: actual},
        index=dates,
    )


def standard_table() -> pd.DataFrame:
    """
    23 rows: 2 old rows with huge ARIMA errors (outside the last 20), 20
    scored rows, and the live row (no actual). On scored rows the actual is
    100, naive 101 (error +1, predicts no move from 101) and ARIMA 100.5
    (error +0.5, predicts a fall, as happened).
    """
    arima = [200.0, 200.0] + [100.5] * 20 + [100.7]
    naive = [101.0] * 22 + [100.0]
    actual = [100.0] * 22 + [np.nan]
    return forecast_table(arima, naive, actual)


# error_metrics and directional_accuracy


def test_error_metrics_by_hand():
    actual = pd.Series([100.0, 200.0])
    forecast = pd.Series([101.0, 196.0])  # Errors +1 and -4.
    m = error_metrics(forecast, actual)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 16) / 2))
    assert m["mae"] == pytest.approx(2.5)
    assert m["mape"] == pytest.approx((1 / 100 + 4 / 200) / 2 * 100)  # 1.5 %


def test_error_metrics_with_no_rows_are_none():
    assert error_metrics(pd.Series([], dtype=float), pd.Series([], dtype=float)) == {
        "rmse": None,
        "mae": None,
        "mape": None,
    }


def test_directional_accuracy_skips_days_without_a_move():
    naive = pd.Series([100.0, 100.0, 100.0, 100.0])
    actual = pd.Series([101.0, 99.0, 100.0, 102.0])  # Up, down, flat, up.
    arima = pd.Series([100.5, 100.5, 99.0, 99.0])  # Up, up, down, down.
    accuracy, n = directional_accuracy(arima, naive, actual)
    assert n == 3  # The flat day is skipped.
    assert accuracy == pytest.approx(1 / 3)  # Only the first call is right.


def test_directional_accuracy_with_no_moves_is_none():
    flat = pd.Series([100.0, 100.0])
    assert directional_accuracy(flat + 1, flat, flat) == (None, 0)


# ticker_metrics


def test_ticker_metrics_score_only_the_last_20_days_with_actuals():
    m = ticker_metrics(standard_table(), "AAA", eval_days=20)
    assert m["n_evaluated"] == 20
    assert m["arima"]["mae"] == pytest.approx(0.5)  # The 200.0 rows are excluded.
    assert m["arima"]["rmse"] == pytest.approx(0.5)
    assert m["naive"]["mae"] == pytest.approx(1.0)
    assert m["naive"]["rmse"] == pytest.approx(1.0)
    assert m["arima"]["mape"] == pytest.approx(0.5)
    assert m["naive"]["mape"] == pytest.approx(1.0)


def test_ticker_metrics_relative_mae_and_skill():
    m = ticker_metrics(standard_table(), "AAA", eval_days=20)
    assert m["relative_mae"] == pytest.approx(0.5)
    assert m["skill"] == pytest.approx(0.5)


def test_ticker_metrics_directional_accuracy():
    m = ticker_metrics(standard_table(), "AAA", eval_days=20)
    assert m["arima"]["directional_accuracy"] == pytest.approx(1.0)
    assert m["arima"]["n_directional"] == 20
    assert "directional_accuracy" not in m["naive"]


def test_ticker_metrics_evaluation_dates_exclude_the_live_row():
    table = standard_table()
    m = ticker_metrics(table, "AAA", eval_days=20)
    assert m["evaluation_start_date"] == table.index[2].strftime("%Y-%m-%d")
    assert m["evaluation_end_date"] == table.index[-2].strftime("%Y-%m-%d")


def test_ticker_metrics_skip_rows_missing_a_forecast():
    table = standard_table()
    table.iloc[10, table.columns.get_loc("AAA_arima")] = np.nan  # e.g. a missing close
    m = ticker_metrics(table, "AAA", eval_days=20)
    assert m["n_evaluated"] == 19  # Both models scored on the same 19 days.
    assert m["naive"]["mae"] == pytest.approx(1.0)
    assert m["arima"]["mae"] == pytest.approx(0.5)


def test_ticker_metrics_with_perfect_naive_have_no_ratio():
    actual = [100.0] * 20 + [np.nan]
    table = forecast_table([100.5] * 21, [100.0] * 21, actual)
    m = ticker_metrics(table, "AAA", eval_days=20)
    assert m["naive"]["mae"] == pytest.approx(0.0)
    assert m["relative_mae"] is None
    assert m["skill"] is None


def test_ticker_metrics_with_fewer_days_than_requested():
    table = standard_table().iloc[-6:]  # 5 rows with actuals + live row.
    assert ticker_metrics(table, "AAA", eval_days=20)["n_evaluated"] == 5


# build_forecast_metadata and the YAML file


def two_ticker_table() -> pd.DataFrame:
    table = standard_table()
    other = standard_table().rename(
        columns={"AAA_arima": "BBB_arima", "AAA_naive": "BBB_naive", "AAA": "BBB"}
    )
    return pd.concat([table, other], axis=1)


def test_build_forecast_metadata_structure():
    table = two_ticker_table()
    start, end = table.index[1], table.index[-2]
    meta = build_forecast_metadata(
        table,
        ["BBB", "AAA", "ZZZ"],
        start,
        end,
        eval_days=20,
        training_window_returns=251,
    )
    assert set(meta["ticker_metrics"]) == {"aaa", "bbb"}  # Lowercase; ZZZ skipped.
    assert meta["evaluation_window"]["target_trading_days"] == 20
    fw = meta["forecast_window"]
    assert fw["training_window_start_date"] == start.strftime("%Y-%m-%d")
    assert fw["training_window_end_date"] == end.strftime("%Y-%m-%d")
    assert fw["forecast_date"] == table.index[-1].strftime("%Y-%m-%d")
    assert meta["evaluation_window"]["end_date"] == end.strftime("%Y-%m-%d")
    assert meta["units"]["mape"] == "percent"
    assert meta["units"]["rmse"] == "usd_per_share"
    assert meta["models"]["arima"]["d"] == 1
    assert meta["models"]["arima"]["p_values"] == [0, 1]
    assert meta["models"]["arima"]["q_values"] == [0, 1]
    assert meta["models"]["arima"]["excluded_orders"] == [[0, 0]]
    assert meta["models"]["arima"]["order_selection_tie_break"] == (
        "fewer parameters (p + q), then lower p"
    )
    # Candidates are scored on the whole training window (251 returns).
    assert meta["models"]["arima"]["order_selection_subwindow_returns"] == 251
    assert meta["forecast_window"]["training_window_returns"] == 251


def test_build_forecast_metadata_reports_the_settings_it_is_given():
    table = two_ticker_table()
    meta = build_forecast_metadata(
        table,
        ["AAA"],
        table.index[1],
        table.index[-2],
        eval_days=5,
        training_window_returns=99,
    )
    assert meta["forecast_window"]["training_window_returns"] == 99
    assert meta["models"]["arima"]["order_selection_subwindow_returns"] == 99
    assert meta["evaluation_window"]["target_trading_days"] == 5
    assert meta["ticker_metrics"]["aaa"]["n_evaluated"] == 5


def test_forecast_metadata_yaml_round_trip(tmp_path):
    table = two_ticker_table()
    meta = build_forecast_metadata(
        table,
        ["AAA", "BBB"],
        table.index[1],
        table.index[-2],
        eval_days=20,
        training_window_returns=251,
    )
    path = tmp_path / "all_forecasts_metadata.yml"
    write_forecast_metadata(meta, str(path))
    loaded = yaml.safe_load(path.read_text())
    assert loaded == meta
    # Dates load as strings and numbers as plain floats, not numpy tags.
    assert isinstance(loaded["forecast_window"]["forecast_date"], str)
    assert isinstance(loaded["ticker_metrics"]["aaa"]["arima"]["rmse"], float)
    assert "!!python" not in path.read_text()
