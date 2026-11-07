"""
Unit tests for aliciadata_finance_backend.optimization.problem.PortfolioProblem:
estimation from prices, the shared constraint set, portfolio_stats(), and
the feasible-set sampler. In-memory only.
"""

import numpy as np
import pytest

from conftest import make_long_prices
from aliciadata_finance_backend.data.prices import pivot_close_wide
from aliciadata_finance_backend.optimization.problem import (
    PortfolioProblem,
    PortfolioStats,
)

DAILY_RF = 4.05 / 252


def problem(n: int = 3, cap_multiple: float = 4, **kwargs) -> PortfolioProblem:
    mean = np.linspace(0.02, 0.10, n)
    cov = np.diag(np.linspace(0.5, 2.0, n)) + 0.1
    return PortfolioProblem(
        tickers=tuple(f"T{i}" for i in range(n)),
        daily_mean_returns=mean,
        daily_cov=cov,
        daily_risk_free_rate=DAILY_RF,
        weight_cap_multiple=cap_multiple,
        **kwargs,
    )


# Construction and estimation


def test_from_prices_uses_simple_percent_returns_and_sample_covariance():
    wide = pivot_close_wide(make_long_prices())
    p = PortfolioProblem.from_prices(wide, DAILY_RF, 4)
    returns = wide.pct_change().iloc[1:] * 100
    assert p.tickers == tuple(wide.columns)
    np.testing.assert_allclose(p.daily_mean_returns, returns.mean().to_numpy())
    np.testing.assert_allclose(p.daily_cov, returns.cov().to_numpy())  # ddof = 1
    assert p.daily_risk_free_rate == DAILY_RF


def test_arrays_are_read_only_copies():
    mean = np.array([0.1, 0.2])
    p = PortfolioProblem(("A", "B"), mean, np.eye(2), DAILY_RF, 2)
    mean[0] = 99.0
    assert p.daily_mean_returns[0] == 0.1
    with pytest.raises(ValueError):
        p.daily_mean_returns[0] = 1.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(daily_mean_returns=np.zeros(2)), "shape"),
        (dict(daily_cov=np.eye(2)), "shape"),
        (dict(daily_mean_returns=np.array([0.1, np.nan, 0.2])), "NaN"),
        (dict(daily_risk_free_rate=np.nan), "finite"),
        (dict(weight_cap_multiple=0.5), "at least 1"),
    ],
)
def test_invalid_problems_are_rejected(kwargs, match):
    base = dict(
        tickers=("A", "B", "C"),
        daily_mean_returns=np.array([0.1, 0.2, 0.3]),
        daily_cov=np.eye(3),
        daily_risk_free_rate=DAILY_RF,
        weight_cap_multiple=4,
    )
    with pytest.raises(ValueError, match=match):
        PortfolioProblem(**{**base, **kwargs})


# Constraint set


@pytest.mark.parametrize("n, multiple", [(3, 4), (8, 4), (10, 4), (10, 2.5)])
def test_bounds_are_long_only_and_capped_at_multiple_over_n(n, multiple):
    p = problem(n, multiple)
    assert p.weight_cap == multiple / n
    assert p.bounds() == [(0.0, multiple / n)] * n


def test_constraints_are_the_budget_and_a_fresh_list_each_call():
    p = problem()
    (budget,) = p.constraints()
    assert budget["type"] == "eq"
    assert budget["fun"](np.array([0.2, 0.3, 0.5])) == pytest.approx(0.0)
    assert budget["fun"](np.array([0.2, 0.3, 0.6])) == pytest.approx(0.1)
    p.constraints().append("extra")
    assert len(p.constraints()) == 1


def test_equal_weights_are_feasible():
    p = problem(10, 1.5)
    assert p.max_constraint_violation(p.equal_weights()) == pytest.approx(0.0, abs=1e-15)


def test_max_constraint_violation_reports_each_constraint():
    p = problem(4, 2)  # cap 0.5
    w = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],  # feasible
            [-0.1, 0.4, 0.4, 0.3],  # short
            [0.7, 0.1, 0.1, 0.1],  # above the cap
            [0.3, 0.3, 0.3, 0.3],  # sums to 1.2
            [np.nan, 0.5, 0.25, 0.25],  # unreached frontier row
        ]
    )
    violation = p.max_constraint_violation(w)
    np.testing.assert_allclose(violation[:4], [0.0, 0.1, 0.2, 0.2], atol=1e-12)
    assert np.isnan(violation[4])
    assert isinstance(p.max_constraint_violation(w[0]), float)


# portfolio_stats


def test_portfolio_stats_annualizes_arithmetically():
    p = problem()
    w = np.array([0.2, 0.3, 0.5])
    stats = p.portfolio_stats(w)
    daily_mean = w @ p.daily_mean_returns
    daily_sigma = np.sqrt(w @ p.daily_cov @ w)
    assert isinstance(stats, PortfolioStats)
    assert stats.annualized_return == pytest.approx(daily_mean * 252)
    assert stats.risk == pytest.approx(daily_sigma * np.sqrt(252))
    # (annual return − annual rf) / annual risk == daily Sharpe × √252.
    assert stats.sharpe_ratio == pytest.approx(
        (daily_mean - DAILY_RF) / daily_sigma * np.sqrt(252)
    )
    assert stats.sharpe_ratio == pytest.approx(
        (stats.annualized_return - 4.05) / stats.risk
    )


def test_portfolio_stats_returns_floats_for_one_portfolio():
    stats = problem().portfolio_stats(np.array([0.2, 0.3, 0.5]))
    for value in (stats.annualized_return, stats.risk, stats.sharpe_ratio):
        assert type(value) is float


def test_portfolio_stats_rows_match_one_at_a_time_and_nan_rows_stay_nan():
    p = problem()
    w = np.array([[0.2, 0.3, 0.5], [1 / 3, 1 / 3, 1 / 3], [np.nan] * 3])
    many = p.portfolio_stats(w)
    for i in range(2):
        one = p.portfolio_stats(w[i])
        assert many.annualized_return[i] == pytest.approx(one.annualized_return, rel=1e-14)
        assert many.risk[i] == pytest.approx(one.risk, rel=1e-14)
    assert np.isnan(many.risk[2]) and np.isnan(many.annualized_return[2])


def test_annual_risk_free_rate_recovers_the_quote():
    assert problem().annual_risk_free_rate == pytest.approx(4.05)


# sample_feasible_weights


@pytest.mark.parametrize("n, multiple", [(3, 4), (10, 4), (10, 2), (5, 1.5)])
def test_samples_satisfy_the_shared_constraints(n, multiple):
    p = problem(n, multiple)
    w = p.sample_feasible_weights(2000, np.random.default_rng(1))
    assert w.shape == (2000, n)
    assert (w >= 0).all()
    assert (w <= p.weight_cap).all()
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert p.max_constraint_violation(w).max() <= 1e-12


def test_samples_are_reproducible_with_the_same_seed():
    p = problem(10, 2)
    a = p.sample_feasible_weights(500, np.random.default_rng(42))
    b = p.sample_feasible_weights(500, np.random.default_rng(42))
    c = p.sample_feasible_weights(500, np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_samples_are_uniform_on_average():
    # Uniform on a symmetric set: every asset's mean weight is 1 / n.
    w = problem(5, 2).sample_feasible_weights(20_000, np.random.default_rng(0))
    np.testing.assert_allclose(w.mean(axis=0), 0.2, atol=0.01)


def test_binding_cap_rejects_draws_that_exceed_it():
    # With 10 assets and a cap of 0.2, many uniform simplex draws have a
    # weight above 0.2; none may be returned.
    p = problem(10, 2)
    w = p.sample_feasible_weights(1000, np.random.default_rng(5))
    assert w.max() <= 0.2


def test_sampler_fails_loudly_when_almost_nothing_is_feasible():
    # A cap of exactly 1 / n leaves only the equal-weight point.
    p = problem(4, 1)
    with pytest.raises(ValueError, match="weight cap"):
        p.sample_feasible_weights(10, np.random.default_rng(0))


def test_sampler_rejects_a_non_positive_count():
    with pytest.raises(ValueError):
        problem().sample_feasible_weights(0, np.random.default_rng(0))


# max_return_weights_closed_form and single_asset_weights


def test_closed_form_max_return_fills_the_best_assets_up_to_the_cap():
    # Cap 0.4 with 5 assets (multiple 2): 40 % best, 40 % second, 20 % third.
    p = PortfolioProblem(
        tickers=("A", "B", "C", "D", "E"),
        daily_mean_returns=np.array([0.05, 0.12, 0.08, 0.20, 0.01]),
        daily_cov=np.eye(5),
        daily_risk_free_rate=DAILY_RF,
        weight_cap_multiple=2,
    )
    np.testing.assert_allclose(
        p.max_return_weights_closed_form(), [0.0, 0.4, 0.2, 0.4, 0.0], atol=1e-15
    )


def test_closed_form_max_return_without_a_binding_cap_holds_only_the_best_asset():
    p = problem(3, 4)  # cap 4/3: never binds
    np.testing.assert_allclose(p.max_return_weights_closed_form(), [0.0, 0.0, 1.0])


@pytest.mark.parametrize("n, multiple", [(3, 4), (10, 4), (10, 2), (5, 1.5), (7, 3.3)])
def test_closed_form_max_return_is_feasible_and_beats_every_sample(n, multiple):
    p = problem(n, multiple)
    best = p.max_return_weights_closed_form()
    assert p.max_constraint_violation(best) <= 1e-12
    samples = p.sample_feasible_weights(5000, np.random.default_rng(2))
    assert p.daily_mean_return(samples).max() <= p.daily_mean_return(best) + 1e-12


def test_single_asset_weights_give_each_assets_own_statistics():
    p = problem(4)
    stats = p.portfolio_stats(p.single_asset_weights())
    np.testing.assert_allclose(stats.annualized_return, p.daily_mean_returns * 252)
    np.testing.assert_allclose(stats.risk, np.sqrt(np.diag(p.daily_cov) * 252))
