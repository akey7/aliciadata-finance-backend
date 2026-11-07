"""
Tests for aliciadata_finance_backend.optimization.workflow on seeded synthetic
prices: one shared PortfolioProblem behind every coordinate, the capital
allocation line, the efficient frontier's range, the random feasible
portfolios, the invariant checks, and the writers. In-memory, except the
writers, which write under pytest's tmp_path.
"""

import dataclasses
import inspect
import re

import h5py
import numpy as np
import pytest
import yaml

from conftest import make_long_prices
from aliciadata_finance_backend.config import PortfolioOptimizationConfig
from aliciadata_finance_backend.data.prices import pivot_close_wide
from aliciadata_finance_backend.optimization import portfolio, workflow
from aliciadata_finance_backend.optimization.portfolio import (
    OptimizationError,
    max_return_weights,
)
from aliciadata_finance_backend.optimization.problem import (
    PortfolioProblem,
    annualize_return,
)
from aliciadata_finance_backend.optimization.workflow import (
    CAPITAL_ALLOCATION_LINE_POINTS,
    OPTIMUM_PORTFOLIO_COMMENT,
    InvariantError,
    OptimizationResult,
    build_optimization_metadata,
    capital_allocation_line,
    check_optimization_invariants,
    optimization_metadata_yaml,
    run_portfolio_optimization,
    verify_optimization_files,
    write_optimization_metadata,
    write_plot_data_h5,
)

ANNUAL_RF = 4.05
DAILY_RF = ANNUAL_RF / 252
RF_DATA = {
    "risk_free_rate": ANNUAL_RF,
    "daily_risk_free_rate": DAILY_RF,
    "risk_free_rate_date": "2026-10-08 00:00:00",
}
# Deliberately different from the production settings, so the tests show
# that the array lengths and the seed come from the settings.
SETTINGS = PortfolioOptimizationConfig(
    weight_cap_multiple=4,
    efficient_frontier_points=60,
    simulated_portfolios=4000,
    random_seed=0,
    near_frontier_tolerance=1.0,
)
# Three assets with a multiple of 1.5: each weight is capped at 50 %, which
# binds, unlike 4 / 3.
CAPPED_SETTINGS = dataclasses.replace(SETTINGS, weight_cap_multiple=1.5)
TICKERS = ["XBI", "QQQ", "GLD"]


@pytest.fixture(scope="module")
def wide():
    return pivot_close_wide(make_long_prices())


@pytest.fixture(scope="module", params=["uncapped", "capped"])
def case(request, wide):
    """(settings, problem, result) for a cap that does not bind and one that does."""
    settings = SETTINGS if request.param == "uncapped" else CAPPED_SETTINGS
    problem = PortfolioProblem.from_prices(wide, DAILY_RF, settings.weight_cap_multiple)
    return settings, problem, run_portfolio_optimization(wide, DAILY_RF, settings)


@pytest.fixture(scope="module")
def result(wide) -> OptimizationResult:
    return run_portfolio_optimization(wide, DAILY_RF, SETTINGS)


# Shapes, dates, weights


def test_result_shapes_follow_the_settings(case):
    settings, _, res = case
    n_frontier = settings.efficient_frontier_points
    n_sim = settings.simulated_portfolios
    assert res.frontier_risks.shape == res.frontier_returns.shape == (n_frontier,)
    assert res.frontier_weights.shape == (n_frontier, 3)
    assert res.simulated_risks.shape == res.simulated_returns.shape == (n_sim,)
    assert res.simulated_weights.shape == (n_sim, 3)
    assert res.tangency_risks.shape == (CAPITAL_ALLOCATION_LINE_POINTS,)
    assert res.tangency_returns.shape == (CAPITAL_ALLOCATION_LINE_POINTS,)


def test_result_dates_span_the_data(result, wide):
    assert result.date_from == wide.index[0]
    assert result.date_to == wide.index[-1]


@pytest.mark.parametrize("which", ["max_sharpe_weights_pct", "min_var_weights_pct"])
def test_weights_are_long_only_capped_and_sum_to_100(case, wide, which):
    _, problem, res = case
    weights = getattr(res, which)
    assert list(weights) == list(wide.columns)
    assert all(isinstance(w, float) for w in weights.values())
    assert all(w >= -1e-6 for w in weights.values())
    assert all(w <= problem.weight_cap * 100 + 1e-4 for w in weights.values())
    # SLSQP satisfies the sum-to-1 constraint only to its tolerance.
    assert sum(weights.values()) == pytest.approx(100, abs=1e-3)


def test_binding_cap_is_respected_and_binds(wide):
    res = run_portfolio_optimization(wide, DAILY_RF, CAPPED_SETTINGS)
    problem = PortfolioProblem.from_prices(wide, DAILY_RF, 1.5)
    top = max_return_weights(problem)
    assert top.max() == pytest.approx(0.5, abs=1e-6)  # The cap binds.
    assert res.simulated_weights.max() <= 0.5


# One implementation: every coordinate comes from portfolio_stats()


def test_every_h5_coordinate_is_portfolio_stats_of_its_weights(case):
    _, problem, res = case
    sims = problem.portfolio_stats(res.simulated_weights)
    np.testing.assert_array_equal(res.simulated_risks, sims.risk)
    np.testing.assert_array_equal(res.simulated_returns, sims.annualized_return)
    np.testing.assert_array_equal(
        res.frontier_risks, problem.portfolio_stats(res.frontier_weights).risk
    )
    ms = problem.portfolio_stats(res.max_sharpe_weights)
    assert (res.max_sharpe_risk, res.max_sharpe_return, res.sharpe_ratio) == (
        ms.risk,
        ms.annualized_return,
        ms.sharpe_ratio,
    )
    mv = problem.portfolio_stats(res.min_var_weights)
    assert (res.min_var_risk, res.min_var_return) == (mv.risk, mv.annualized_return)
    # The capital allocation line is built from the max Sharpe ratio point.
    xs, ys = capital_allocation_line(problem.annual_risk_free_rate, ms.risk, ms.annualized_return)
    np.testing.assert_array_equal(res.tangency_risks, xs)
    np.testing.assert_array_equal(res.tangency_returns, ys)


def test_reached_frontier_returns_match_portfolio_stats(case):
    _, problem, res = case
    reached = ~np.isnan(res.frontier_risks)
    achieved = problem.portfolio_stats(res.frontier_weights[reached]).annualized_return
    np.testing.assert_allclose(achieved, res.frontier_returns[reached], atol=1e-3)


@pytest.mark.parametrize("module", [workflow, portfolio])
def test_no_second_implementation_of_return_risk_or_estimation(module):
    # Return (w·μ), variance (wᵀΣw), and the mean/covariance estimates live
    # only in problem.py; the optimizers and the workflow must call it.
    source = inspect.getsource(module)
    assert "einsum" not in source
    assert ".dot(" not in source
    assert not re.search(r"[\w\)\]][ \t]*@[ \t]*[\w\(]", source)  # matmul
    assert "pct_change" not in source
    assert ".cov(" not in source
    assert "annualize_risk" not in source


# Efficient frontier range


def test_frontier_runs_from_min_var_to_max_feasible_return(case):
    settings, problem, res = case
    top = problem.portfolio_stats(max_return_weights(problem)).annualized_return
    expected = np.linspace(res.min_var_return, top, settings.efficient_frontier_points)
    np.testing.assert_allclose(res.frontier_returns, expected, rtol=1e-12, atol=1e-12)
    # The old upper end, the highest random return, falls short of it.
    assert res.simulated_returns.max() <= top + 1e-9


def test_frontier_returns_are_the_annualized_target_grid(case):
    _, _, res = case
    steps = np.diff(res.frontier_returns)
    np.testing.assert_allclose(steps, steps[0], rtol=1e-9)
    assert not np.isnan(res.frontier_returns).any()


def test_min_var_risk_is_the_smallest_frontier_risk(case):
    _, _, res = case
    finite = res.frontier_risks[~np.isnan(res.frontier_risks)]
    assert finite.min() == pytest.approx(res.min_var_risk, rel=1e-3)
    assert res.frontier_risks[0] == pytest.approx(res.min_var_risk, rel=1e-3)


# Random portfolios


def test_simulated_weights_satisfy_the_shared_constraints(case):
    _, problem, res = case
    w = res.simulated_weights
    assert (w >= 0).all()
    assert (w <= problem.weight_cap).all()
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)


def test_no_simulated_portfolio_beats_the_max_sharpe_ratio(case):
    _, _, res = case
    # SLSQP converges to ~1e-6 on the daily objective (~1e-5 annualized), and
    # the synthetic Sharpe ratio may be negative, so use an absolute margin.
    tol = 1e-4 * (1 + abs(res.sharpe_ratio))
    sim_sharpe = (res.simulated_returns - res.annual_risk_free_rate) / res.simulated_risks
    assert sim_sharpe.max() <= res.sharpe_ratio + tol


def test_no_simulated_portfolio_lies_left_of_the_frontier(case):
    # Each frontier point is the least volatile portfolio with that target
    # return, and the frontier is the upper (rising) branch, so no portfolio
    # with at least that return may have less volatility. Tolerance allows for SLSQP
    # convergence.
    _, _, res = case
    finite = ~np.isnan(res.frontier_risks)
    targets = res.frontier_returns[finite]
    risks = res.frontier_risks[finite]
    sim_r = res.simulated_returns[:, None]
    sim_s = res.simulated_risks[:, None]
    assert not ((sim_r >= targets[None, :]) & (sim_s < risks[None, :] * (1 - 1e-3))).any()
    assert res.simulated_risks.min() >= res.min_var_risk * (1 - 1e-3)


def test_seeded_simulation_is_reproducible_and_seed_dependent(wide, result):
    again = run_portfolio_optimization(wide, DAILY_RF, SETTINGS)
    np.testing.assert_array_equal(again.simulated_weights, result.simulated_weights)
    other = run_portfolio_optimization(
        wide, DAILY_RF, dataclasses.replace(SETTINGS, random_seed=1)
    )
    assert not np.array_equal(other.simulated_weights, result.simulated_weights)
    # The seed changes only the random cloud, not the optimization.
    assert other.max_sharpe_risk == result.max_sharpe_risk


# Capital allocation line


def test_cal_starts_at_risk_free_rate_and_ends_exactly_at_max_sharpe(case):
    _, _, res = case
    xs, ys = res.tangency_risks, res.tangency_returns
    assert xs[0] == 0.0
    assert ys[0] == res.annual_risk_free_rate
    assert res.annual_risk_free_rate == pytest.approx(ANNUAL_RF)
    assert xs[-1] == res.max_sharpe_risk
    assert ys[-1] == res.max_sharpe_return
    assert xs.max() == res.max_sharpe_risk  # Nothing beyond the max Sharpe point.


def test_cal_is_straight_evenly_spaced_with_sharpe_slope(case):
    _, _, res = case
    xs, ys = res.tangency_risks, res.tangency_returns
    np.testing.assert_allclose(np.diff(xs), xs[-1] / (len(xs) - 1), rtol=1e-9)
    np.testing.assert_allclose(ys, ANNUAL_RF + res.sharpe_ratio * xs, rtol=1e-9)
    assert np.isfinite(xs).all() and np.isfinite(ys).all()


def test_sharpe_ratio_identity_in_annual_units(case):
    _, _, res = case
    expected = (res.max_sharpe_return - ANNUAL_RF) / res.max_sharpe_risk
    assert res.sharpe_ratio == pytest.approx(expected, rel=1e-9)


# Invariant checks reject each kind of disagreement


@pytest.fixture(scope="module")
def checked(wide, result):
    # Each test below tampers with a copy (dataclasses.replace, .copy()).
    problem = PortfolioProblem.from_prices(wide, DAILY_RF, SETTINGS.weight_cap_multiple)
    check_optimization_invariants(problem, result)  # The real result passes.
    return problem, result


def test_invariants_reject_an_infeasible_random_portfolio(checked):
    problem, res = checked
    weights = res.simulated_weights.copy()
    weights[0] = [0.7, 0.7, -0.4]  # Sums to 1 but is short one asset.
    with pytest.raises(InvariantError, match="violates the weight constraints"):
        check_optimization_invariants(
            problem, dataclasses.replace(res, simulated_weights=weights)
        )


def test_invariants_reject_a_point_above_the_max_sharpe_ratio(checked):
    problem, res = checked
    returns = res.simulated_returns.copy()
    returns[0] = 1e6
    with pytest.raises(InvariantError, match="Sharpe ratio"):
        check_optimization_invariants(
            problem, dataclasses.replace(res, simulated_returns=returns)
        )


def test_invariants_reject_a_point_left_of_the_frontier(checked):
    problem, res = checked
    reached = np.flatnonzero(~np.isnan(res.frontier_risks))[-1]
    returns, risks = res.simulated_returns.copy(), res.simulated_risks.copy()
    returns[0] = res.frontier_returns[reached]
    risks[0] = res.frontier_risks[reached] * 0.9
    # A huge Sharpe ratio, so the Sharpe check does not fire first.
    tampered = dataclasses.replace(
        res, simulated_returns=returns, simulated_risks=risks, sharpe_ratio=1e6
    )
    with pytest.raises(InvariantError, match="left of the frontier"):
        check_optimization_invariants(problem, tampered)


def test_invariants_reject_a_point_riskier_than_min_var_from_below(checked):
    problem, res = checked
    returns, risks = res.simulated_returns.copy(), res.simulated_risks.copy()
    returns[0], risks[0] = -1e6, res.min_var_risk * 0.5
    with pytest.raises(InvariantError, match="less volatility than the minimum variance"):
        check_optimization_invariants(
            problem, dataclasses.replace(res, simulated_returns=returns, simulated_risks=risks)
        )


def test_invariants_reject_a_min_var_risk_off_the_frontier(checked):
    problem, res = checked
    with pytest.raises(InvariantError, match="minimum variance volatility"):
        check_optimization_invariants(
            problem, dataclasses.replace(res, min_var_risk=res.min_var_risk * 0.9)
        )


def test_invariants_reject_a_frontier_point_off_its_target(checked):
    problem, res = checked
    with pytest.raises(InvariantError, match="misses its target return"):
        check_optimization_invariants(
            problem, dataclasses.replace(res, frontier_returns=res.frontier_returns - 0.002)
        )


def test_invariants_reject_a_cal_extended_past_max_sharpe(checked):
    # The pre-v31 line ran past the max Sharpe point; it must now fail.
    problem, res = checked
    xs, ys = capital_allocation_line(
        res.annual_risk_free_rate,
        np.nanmax(res.frontier_risks),
        res.annual_risk_free_rate + res.sharpe_ratio * np.nanmax(res.frontier_risks),
    )
    with pytest.raises(InvariantError, match="capital allocation line"):
        check_optimization_invariants(
            problem, dataclasses.replace(res, tangency_risks=xs, tangency_returns=ys)
        )


def test_invariant_error_is_an_optimization_error():
    # The pipeline already stops before uploading on OptimizationError (v29).
    assert issubclass(InvariantError, OptimizationError)


# Writers and the written files


def write_both(res, tmp_path):
    h5_path = tmp_path / "portfolio_optimization_plot_data.h5"
    yml_path = tmp_path / "optimization_metadata.yml"
    write_plot_data_h5(res, str(h5_path))
    write_optimization_metadata(build_optimization_metadata(res, TICKERS, RF_DATA), str(yml_path))
    return h5_path, yml_path


def test_write_plot_data_h5_matches_contract(result, tmp_path):
    h5_path, _ = write_both(result, tmp_path)
    with h5py.File(h5_path, "r") as hf:
        assert hf.attrs["annualization"] == "arithmetic"
        assert hf.attrs["periods_per_year"] == 252
        assert hf.attrs["return_units"] == "percent_per_year"
        assert hf.attrs["risk_units"] == "percent_per_year"
        assert hf.attrs["return_type"] == "simple"
        assert set(hf.keys()) == {
            "efficient_frontier",
            "tangency_line",
            "simulated_portfolios",
            "max_sharpe_ratio",
            "min_var_portfolio",
        }
        for group in hf.values():
            assert set(group.keys()) == {"xs", "ys"}
            assert group["xs"].dtype == np.float64
            assert group["ys"].dtype == np.float64
            assert group["xs"].shape == group["ys"].shape
        assert hf["tangency_line/xs"].shape == (100,)
        assert hf["simulated_portfolios/xs"].shape == (SETTINGS.simulated_portfolios,)
        assert hf["efficient_frontier/xs"].shape == (SETTINGS.efficient_frontier_points,)
        np.testing.assert_array_equal(hf["simulated_portfolios/xs"][()], result.simulated_risks)
        np.testing.assert_array_equal(hf["tangency_line/ys"][()], result.tangency_returns)


def test_h5_points_equal_yaml_values(result, tmp_path):
    h5_path, yml_path = write_both(result, tmp_path)
    loaded = yaml.safe_load(yml_path.read_text())
    with h5py.File(h5_path, "r") as hf:
        assert hf["max_sharpe_ratio/xs"][0] == loaded["optimum_portfolio"]["risk"]
        assert hf["max_sharpe_ratio/ys"][0] == loaded["optimum_portfolio"]["annualized_return"]
        assert hf["min_var_portfolio/xs"][0] == loaded["min_var_portfolio"]["risk"]
        assert hf["min_var_portfolio/ys"][0] == loaded["min_var_portfolio"]["annualized_return"]
        assert hf["tangency_line/xs"][-1] == hf["max_sharpe_ratio/xs"][0]
        assert hf["tangency_line/ys"][-1] == hf["max_sharpe_ratio/ys"][0]
    verify_optimization_files(str(h5_path), str(yml_path))  # Passes.


def test_verify_optimization_files_detects_a_yaml_mismatch(result, tmp_path):
    h5_path, yml_path = write_both(result, tmp_path)
    metadata = build_optimization_metadata(result, TICKERS, RF_DATA)
    metadata["min_var_portfolio"]["risk"] += 0.01
    write_optimization_metadata(metadata, str(yml_path))
    with pytest.raises(InvariantError, match="min_var_portfolio"):
        verify_optimization_files(str(h5_path), str(yml_path))


def test_verify_optimization_files_detects_an_extended_cal(result, tmp_path):
    h5_path, yml_path = write_both(result, tmp_path)
    with h5py.File(h5_path, "r+") as hf:
        xs = hf["tangency_line/xs"]
        xs[-1] = xs[-1] * 2
    with pytest.raises(InvariantError, match="capital allocation line"):
        verify_optimization_files(str(h5_path), str(yml_path))


def test_build_optimization_metadata(result):
    metadata = build_optimization_metadata(result, TICKERS, RF_DATA)
    assert metadata["date_updated"] == {
        "date_from": str(result.date_from.date()),
        "date_to": str(result.date_to.date()),
    }
    assert metadata["tickers"] == TICKERS
    assert metadata["risk_free_rate"] == RF_DATA
    assert metadata["annualization"] == {"method": "arithmetic", "periods_per_year": 252}
    assert metadata["optimum_portfolio"]["sharpe_ratio"] == result.sharpe_ratio
    assert metadata["optimum_portfolio"]["weights"] == result.max_sharpe_weights_pct
    assert metadata["min_var_portfolio"] == {
        "annualized_return": result.min_var_return,
        "risk": result.min_var_risk,
        "weights": result.min_var_weights_pct,
    }


def test_metadata_yaml_round_trip_keeps_dates_as_strings(result, tmp_path):
    _, yml_path = write_both(result, tmp_path)
    loaded = yaml.safe_load(yml_path.read_text())
    assert isinstance(loaded["date_updated"]["date_to"], str)
    assert isinstance(loaded["risk_free_rate"]["risk_free_rate_date"], str)
    assert loaded["optimum_portfolio"]["risk"] == pytest.approx(result.max_sharpe_risk)
    # Both weight maps load as plain floats keyed by ticker, not numpy tags.
    for section in ("optimum_portfolio", "min_var_portfolio"):
        weights = loaded[section]["weights"]
        assert set(weights) == set(TICKERS)
        assert all(isinstance(w, float) for w in weights.values())
    assert loaded["min_var_portfolio"]["weights"] == pytest.approx(result.min_var_weights_pct)


def test_comment_sits_directly_above_optimum_portfolio(result, tmp_path):
    _, yml_path = write_both(result, tmp_path)
    lines = yml_path.read_text().splitlines()
    i = lines.index("optimum_portfolio:")
    assert lines[i - 1] == OPTIMUM_PORTFOLIO_COMMENT
    assert "maximum Sharpe ratio" in OPTIMUM_PORTFOLIO_COMMENT
    assert OPTIMUM_PORTFOLIO_COMMENT.startswith("# ")
    assert sum(line == OPTIMUM_PORTFOLIO_COMMENT for line in lines) == 1


def test_comment_does_not_change_the_parsed_metadata(result):
    metadata = build_optimization_metadata(result, TICKERS, RF_DATA)
    text = optimization_metadata_yaml(metadata)
    assert yaml.safe_load(text) == yaml.safe_load(
        yaml.dump(metadata, default_flow_style=False)
    )
    # Apart from the comment line, the text is exactly what yaml.dump writes.
    assert text.replace(OPTIMUM_PORTFOLIO_COMMENT + "\n", "") == yaml.dump(
        metadata, default_flow_style=False
    )


# Convergence


@pytest.mark.parametrize(
    "optimizer, name",
    [
        ("min_variance_weights", "minimum variance"),
        ("max_sharpe_weights", "max Sharpe ratio"),
    ],
)
def test_unconverged_optimization_stops_the_run(wide, monkeypatch, optimizer, name):
    def not_converged(*args, **kwargs):
        raise OptimizationError(f"The {name} optimization did not converge.")

    monkeypatch.setattr(workflow, optimizer, not_converged)
    with pytest.raises(OptimizationError, match=name):
        run_portfolio_optimization(wide, DAILY_RF, SETTINGS)


def test_annualized_frontier_targets_use_the_shared_helper(result, wide):
    # frontier_returns are the daily target grid annualized by the same
    # helper as everything else (× 252).
    problem = PortfolioProblem.from_prices(wide, DAILY_RF, SETTINGS.weight_cap_multiple)
    daily_low = problem.daily_mean_return(result.min_var_weights)
    assert result.frontier_returns[0] == float(annualize_return(daily_low))


# The top of the frontier is the maximum feasible return


def test_max_return_portfolio_is_the_closed_form_maximum(case):
    _, problem, res = case
    closed_form = problem.portfolio_stats(problem.max_return_weights_closed_form())
    assert res.max_return_return == closed_form.annualized_return
    assert res.frontier_returns[-1] == pytest.approx(res.max_return_return, rel=1e-12)
    assert problem.max_constraint_violation(res.max_return_weights) <= 1e-5


def test_invariants_reject_a_frontier_that_stops_short(checked):
    problem, res = checked
    with pytest.raises(InvariantError, match="closed-form maximum feasible return"):
        check_optimization_invariants(
            problem, dataclasses.replace(res, max_return_return=res.max_return_return - 0.5)
        )


def test_verify_optimization_files_detects_a_wrong_frontier_end(result, tmp_path):
    h5_path, yml_path = write_both(result, tmp_path)
    metadata = build_optimization_metadata(result, TICKERS, RF_DATA)
    metadata["max_return_portfolio"]["annualized_return"] += 0.5
    write_optimization_metadata(metadata, str(yml_path))
    with pytest.raises(InvariantError, match="max_return_portfolio"):
        verify_optimization_files(str(h5_path), str(yml_path))


# New optimization_metadata.yml sections


def test_metadata_records_assets_constraints_max_return_and_random_summary(result, tmp_path):
    _, yml_path = write_both(result, tmp_path)
    loaded = yaml.safe_load(yml_path.read_text())
    assert set(loaded["assets"]) == set(TICKERS)
    for i, ticker in enumerate(result.tickers):
        assert loaded["assets"][ticker] == {
            "annualized_return": pytest.approx(result.asset_returns[i]),
            "risk": pytest.approx(result.asset_risks[i]),
        }
    assert loaded["constraints"] == {
        "long_only": True,
        "fully_invested": True,
        "weight_cap": pytest.approx(4 / 3 * 100),
        "weight_cap_multiple": 4,
    }
    mr = loaded["max_return_portfolio"]
    assert mr["annualized_return"] == result.max_return_return
    assert mr["risk"] == result.max_return_risk
    assert mr["weights"] == pytest.approx(result.max_return_weights_pct)
    rp = loaded["random_portfolios"]
    assert rp["sampling"] == "uniform on the constraint set"
    assert rp["count"] == SETTINGS.simulated_portfolios
    assert rp["random_seed"] == SETTINGS.random_seed
    assert rp["near_frontier_tolerance"] == SETTINGS.near_frontier_tolerance
    assert rp["max_annualized_return"] == pytest.approx(result.simulated_returns.max())
    assert 0.0 <= rp["near_frontier_share"] <= 1.0
    for value in (*rp.values(), *mr.values()):
        assert not isinstance(value, np.generic)  # Plain YAML, no numpy tags.


def test_asset_statistics_are_portfolio_stats_of_single_asset_portfolios(case):
    _, problem, res = case
    stats = problem.portfolio_stats(problem.single_asset_weights())
    np.testing.assert_array_equal(res.asset_returns, stats.annualized_return)
    np.testing.assert_array_equal(res.asset_risks, stats.risk)


# Why few random portfolios come near the frontier


def test_random_cloud_is_centered_on_the_equal_weight_portfolio(case):
    # The constraint set treats every asset alike, so uniform draws have an
    # average weight of 1 / n and an average return equal to the equal-weight
    # portfolio's, up to sampling error (5 standard errors here).
    _, _, res = case
    rp = res.random_portfolios
    standard_error = rp.return_std / np.sqrt(rp.count)
    assert abs(rp.mean_annualized_return - rp.equal_weight_annualized_return) < 5 * standard_error


def test_uncapped_spread_matches_the_uniform_theory(result):
    # SETTINGS' cap (4/3) does not bind with 3 assets, so the measured spread
    # must match sqrt(Σ(Rᵢ − R̄)² / (n (n + 1))) up to sampling error (the
    # relative standard error of a standard deviation from 4,000 draws is
    # about 1 %).
    rp = result.random_portfolios
    assert rp.return_std == pytest.approx(rp.return_std_uniform_without_cap, rel=0.05)


def test_binding_cap_narrows_the_spread(wide):
    # With 3 assets and a 50 % cap the feasible set is the middle triangle of
    # the simplex, half the size in every direction, so the spread halves.
    res = run_portfolio_optimization(wide, DAILY_RF, CAPPED_SETTINGS)
    rp = res.random_portfolios
    assert rp.return_std == pytest.approx(0.5 * rp.return_std_uniform_without_cap, rel=0.05)


def test_uniform_theory_formula():
    # R = (0, 1, 2): Σ(Rᵢ − R̄)² = 2, n (n + 1) = 12.
    assert workflow.uniform_return_std_without_cap(np.array([0.0, 1.0, 2.0])) == pytest.approx(
        np.sqrt(2 / 12)
    )


def corner_share(n: int, draws: int = 20_000) -> float:
    """
    Share of uniform random portfolios (no cap) of n assets with returns
    0, 1, ..., n − 1 whose return is within an eighth of (n − 1) of the
    maximum: return ≥ 0.875 (n − 1), i.e. near the best asset's corner.
    """
    problem = PortfolioProblem(
        tickers=tuple(f"T{i}" for i in range(n)),
        daily_mean_returns=np.arange(n, dtype=float),
        daily_cov=np.eye(n),
        daily_risk_free_rate=0.0,
        weight_cap_multiple=n,  # cap 1: does not bind
    )
    w = problem.sample_feasible_weights(draws, np.random.default_rng(0))
    return float(np.mean(problem.daily_mean_return(w) >= 0.875 * (n - 1)))


def test_portfolios_near_the_best_corner_become_vanishingly_rare_as_n_grows():
    # With 3 assets the corner region (vertices (0, 0, 1), (1/8, 0, 7/8),
    # (0, 1/4, 3/4) in weights) is 1/32 of the simplex. Its share shrinks
    # roughly like a power n − 1 of its width, so with 10 assets uniform
    # draws essentially never land there: the reason so few random
    # portfolios reach the top of the frontier.
    assert corner_share(3) == pytest.approx(1 / 32, abs=0.005)
    assert corner_share(10) < corner_share(3) / 30


def test_frontier_return_at_risk_interpolates_and_clamps():
    risks = np.array([1.0, 2.0, np.nan, 4.0])
    returns = np.array([1.0, 3.0, 3.5, 4.0])
    at = workflow.frontier_return_at_risk(risks, returns, np.array([0.5, 1.5, 3.0, 10.0]))
    np.testing.assert_allclose(at, [1.0, 2.0, 3.5, 4.0])


def test_frontier_return_at_risk_is_non_decreasing_in_risk():
    risks = np.array([1.0, 1.5, 2.0])
    returns = np.array([2.0, 1.9, 3.0])  # A tiny numerical dip is smoothed out.
    at = workflow.frontier_return_at_risk(risks, returns, np.array([1.5]))
    assert at[0] == pytest.approx(2.0)


def test_near_frontier_share_matches_a_direct_count(result):
    gap = (
        workflow.frontier_return_at_risk(
            result.frontier_risks, result.frontier_returns, result.simulated_risks
        )
        - result.simulated_returns
    )
    rp = result.random_portfolios
    assert rp.near_frontier_share == np.mean(gap <= rp.near_frontier_tolerance)
