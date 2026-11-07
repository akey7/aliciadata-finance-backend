"""
Tests for how ARIMA training is spread over worker processes in
aliciadata_finance_backend.forecasting.ticker_predict: the default worker count,
the single-thread environment for workers, and fit_windows().

One test starts a real pool of 2 worker processes and fits the real
statsmodels ARIMA on one-year synthetic windows, so it takes a few seconds
while the workers import their libraries. In-memory only.
"""

import os

import numpy as np
import pandas as pd
import pytest

from aliciadata_finance_backend.data.date_manager import DateManager
from aliciadata_finance_backend.forecasting import ticker_predict
from aliciadata_finance_backend.forecasting.ticker_predict import (
    SINGLE_THREAD_ENV_VARS,
    default_worker_count,
    fit_windows,
)


# default_worker_count


@pytest.mark.parametrize(
    "cpu_count, expected",
    [(16, 8), (10, 5), (14, 7), (3, 1), (2, 1), (1, 1)],
)
def test_default_worker_count_is_half_the_cpus_and_at_least_one(cpu_count, expected):
    assert default_worker_count(cpu_count) == expected


def test_default_worker_count_uses_process_cpu_count(monkeypatch):
    monkeypatch.setattr(ticker_predict.os, "process_cpu_count", lambda: 12)
    monkeypatch.setattr(ticker_predict.os, "cpu_count", lambda: 64)
    assert default_worker_count() == 6


def test_default_worker_count_falls_back_to_cpu_count(monkeypatch):
    monkeypatch.setattr(ticker_predict.os, "process_cpu_count", lambda: None)
    monkeypatch.setattr(ticker_predict.os, "cpu_count", lambda: 8)
    assert default_worker_count() == 4


def test_default_worker_count_is_one_when_cpus_are_unknown(monkeypatch):
    monkeypatch.setattr(ticker_predict.os, "process_cpu_count", lambda: None)
    monkeypatch.setattr(ticker_predict.os, "cpu_count", lambda: None)
    assert default_worker_count() == 1


# _single_threaded_numeric_libraries


def test_single_thread_env_is_set_inside_and_restored_after(monkeypatch):
    preset, *unset = SINGLE_THREAD_ENV_VARS
    monkeypatch.setenv(preset, "4")
    for name in unset:
        monkeypatch.delenv(name, raising=False)
    with ticker_predict._single_threaded_numeric_libraries():
        assert all(os.environ[name] == "1" for name in SINGLE_THREAD_ENV_VARS)
    assert os.environ[preset] == "4"
    assert all(name not in os.environ for name in unset)


# fit_windows


@pytest.mark.parametrize("n_workers", [0, -1])
def test_fit_windows_rejects_fewer_than_one_worker(n_workers):
    with pytest.raises(ValueError):
        fit_windows([], max_p=2, max_q=2, n_workers=n_workers)


def synthetic_windows(n_windows: int = 3, n_sessions: int = 252) -> list[pd.Series]:
    """Price windows on consecutive NYSE sessions, as train_arima_models builds them."""
    cbd = DateManager().cbd
    index = pd.date_range(end="2026-04-02 17:00", periods=n_sessions, freq=cbd)
    windows = []
    for seed in range(n_windows):
        rng = np.random.default_rng(seed)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_sessions)))
        windows.append(pd.Series(prices, index=index, name=f"T{seed}"))
    return windows


def test_pool_gives_the_same_results_in_the_same_order_as_one_process():
    windows = synthetic_windows()
    in_process = fit_windows(windows, max_p=2, max_q=2, n_workers=1)
    pooled = fit_windows(windows, max_p=2, max_q=2, n_workers=2)
    assert len(pooled) == len(windows)
    for (date_1, pred_1, order_1, tie_1), (date_2, pred_2, order_2, tie_2) in zip(
        in_process, pooled
    ):
        assert date_2 == date_1
        assert order_2 == order_1
        assert tie_2 == tie_1
        assert pred_2 == pytest.approx(pred_1, rel=1e-9)
