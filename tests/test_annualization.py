"""
Tests for the annualization helpers in aliciadata_finance_backend.optimization.problem.

These are pure functions, so the tests use in-memory values only: no network,
no S3, and no reads or writes to input/ or output/.
"""

import numpy as np
import pytest

from aliciadata_finance_backend.optimization.problem import (
    TRADING_DAYS_PER_YEAR,
    annualize_return,
    annualize_risk,
)


def test_trading_days_per_year_is_252():
    assert TRADING_DAYS_PER_YEAR == 252


# annualize_return


def test_annualize_return_scalar_is_arithmetic():
    # 0.1 % per day -> 25.2 % per year (arithmetic, not compounded).
    result = annualize_return(0.1)
    assert np.ndim(result) == 0
    assert float(result) == pytest.approx(25.2)


def test_annualize_return_is_not_compounded():
    compounded = ((1 + 0.1 / 100) ** 252 - 1) * 100
    assert float(annualize_return(0.1)) != pytest.approx(compounded)


def test_annualize_return_array_preserves_shape_and_scales():
    daily = np.array([-0.05, 0.0, 0.02, 0.1])
    result = annualize_return(daily)
    assert result.shape == daily.shape
    np.testing.assert_allclose(result, daily * 252)


def test_annualize_return_accepts_list():
    np.testing.assert_allclose(annualize_return([0.01, 0.02]), [2.52, 5.04])


def test_annualize_return_passes_nan_through():
    result = annualize_return([0.1, np.nan])
    assert result[0] == pytest.approx(25.2)
    assert np.isnan(result[1])


def test_annualize_return_custom_periods():
    assert float(annualize_return(1.0, periods_per_year=12)) == pytest.approx(12.0)


def test_annualize_return_does_not_mutate_input():
    daily = np.array([0.1, 0.2])
    annualize_return(daily)
    np.testing.assert_array_equal(daily, [0.1, 0.2])


# annualize_risk


def test_annualize_risk_scalar_scales_by_sqrt_periods():
    result = annualize_risk(1.0)
    assert np.ndim(result) == 0
    assert float(result) == pytest.approx(np.sqrt(252))


def test_annualize_risk_array_preserves_shape_and_scales():
    daily = np.array([0.0, 0.5, 1.2])
    result = annualize_risk(daily)
    assert result.shape == daily.shape
    np.testing.assert_allclose(result, daily * np.sqrt(252))


def test_annualize_risk_passes_nan_through():
    # Infeasible efficient-frontier points are NaN and must stay NaN.
    result = annualize_risk([np.nan, 1.0])
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(np.sqrt(252))


def test_annualize_risk_custom_periods():
    assert float(annualize_risk(1.0, periods_per_year=4)) == pytest.approx(2.0)


def test_annualize_risk_variance_scales_linearly():
    # Annualized variance should be daily variance × periods.
    daily_sigma = 0.8
    annual_sigma = float(annualize_risk(daily_sigma))
    assert annual_sigma**2 == pytest.approx(daily_sigma**2 * 252)


# Sharpe ratio under arithmetic annualization (used by
# PortfolioProblem.portfolio_stats(); see tests/test_problem.py)


def test_annual_sharpe_from_annualized_inputs_is_daily_sharpe_times_sqrt_periods():
    # Contract identity: annual Sharpe == (annual return - annual rf) / annual
    # risk == daily Sharpe × √252.
    daily_mean, daily_rf, daily_sigma = 0.12, 4.05 / 252, 0.9
    daily_sharpe = (daily_mean - daily_rf) / daily_sigma
    annual_sharpe = (
        float(annualize_return(daily_mean)) - float(annualize_return(daily_rf))
    ) / float(annualize_risk(daily_sigma))
    assert annual_sharpe == pytest.approx(daily_sharpe * np.sqrt(252))


def test_annualized_daily_rf_recovers_quoted_annual_rate():
    # daily_risk_free_rate = annual / 252, so annualizing it gives the quote back.
    assert float(annualize_return(4.05 / 252)) == pytest.approx(4.05)


# Combined property relied on by the .h5 capital allocation line


def test_annualized_line_is_straight_with_annual_intercept_and_slope():
    daily_rf, daily_sharpe = 4.05 / 252, 0.15
    daily_xs = np.linspace(0, 1.5, 100)
    daily_ys = daily_rf + daily_sharpe * daily_xs

    annual_xs = annualize_risk(daily_xs)
    annual_ys = annualize_return(daily_ys)

    expected_ys = float(annualize_return(daily_rf)) + daily_sharpe * np.sqrt(
        252
    ) * annual_xs
    np.testing.assert_allclose(annual_ys, expected_ys)
    assert annual_ys[0] == pytest.approx(4.05)
