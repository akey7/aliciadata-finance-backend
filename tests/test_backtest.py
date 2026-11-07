"""
Tests for aliciadata_finance_backend.optimization.backtest: the rebalance schedule,
temporal leakage, missing data, weight drift, wealth compounding, drawdown, the
optimization fallback, the invariants, the two written files, and the match
between final_optimization and the live optimization's files.

In-memory only, on synthetic prices, except the file tests, which write under
pytest's tmp_path. Nothing here touches the network, input/, or output/.
"""

import dataclasses

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from aliciadata_finance_backend.config import PortfolioOptimizationConfig
from aliciadata_finance_backend.optimization import backtest
from aliciadata_finance_backend.optimization.backtest import (
    STATUS_FALLBACK_EQUAL,
    STATUS_FALLBACK_PREVIOUS,
    STATUS_OPTIMIZED,
    backtest_daily_frame,
    build_backtest_metadata,
    build_final_optimization,
    check_backtest_invariants,
    drift_through_holding,
    performance_metrics,
    run_portfolio_backtest,
    simulate_portfolio,
    verify_backtest_files,
    verify_final_optimization,
    wealth_index,
    write_backtest_daily,
    write_backtest_metadata,
)
from aliciadata_finance_backend.optimization.portfolio import (
    OptimizationError,
    risk_free_rate_before,
)
from aliciadata_finance_backend.optimization.problem import PortfolioProblem
from aliciadata_finance_backend.optimization.workflow import (
    InvariantError,
    build_optimization_metadata,
    run_portfolio_optimization,
    write_optimization_metadata,
    write_plot_data_h5,
)

TICKERS = ("AAA", "BBB", "CCC")
# Small windows keep the real optimizer fast; the first test uses full-size ones.
TRAINING_RETURNS = 40
EVALUATION_DAYS = 20
INTERVAL = 5
N_REBALANCES = EVALUATION_DAYS // INTERVAL
N_SESSIONS = TRAINING_RETURNS + 1 + EVALUATION_DAYS
CAP_MULTIPLE = 2  # A cap of 2/3 with three tickers.
FIXED_WEIGHTS = np.array([0.6, 0.3, 0.1])


def make_prices(n_sessions: int = N_SESSIONS, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_sessions) + pd.Timedelta(hours=17)
    log_returns = rng.normal([0.0004, 0.0006, 0.0003], [0.01, 0.015, 0.008], (n_sessions, 3))
    return pd.DataFrame(
        100 * np.exp(np.cumsum(log_returns, axis=0)), index=dates, columns=list(TICKERS)
    )


def make_rates(n_sessions: int = N_SESSIONS) -> pd.DataFrame:
    # Starts before the prices and ends after them, like a real FRED download.
    dates = pd.bdate_range("2024-12-02", periods=n_sessions + 40)
    return pd.DataFrame({"DTB3": np.linspace(3.0, 5.0, len(dates))}, index=dates)


def fixed_optimizer(problem: PortfolioProblem) -> np.ndarray:
    return FIXED_WEIGHTS.copy()


def run(prices=None, rates=None, **overrides):
    prices = make_prices() if prices is None else prices
    kwargs = dict(
        training_window_returns=TRAINING_RETURNS,
        evaluation_days=EVALUATION_DAYS,
        rebalance_interval=INTERVAL,
        weight_cap_multiple=CAP_MULTIPLE,
        expected_sessions=prices.index.normalize(),
    )
    kwargs.update(overrides)
    return run_portfolio_backtest(
        prices, make_rates(len(prices)) if rates is None else rates, **kwargs
    )


# The schedule (off-by-one errors)


def test_schedule_of_a_full_size_backtest():
    # t is the last session. The positions below are t-60 = 252 and so on.
    n_sessions = 313
    prices = make_prices(n_sessions)
    result = run(
        prices,
        training_window_returns=252,
        evaluation_days=60,
        optimizer=fixed_optimizer,
    )
    dates = prices.index
    assert len(result.rebalances) == 12
    assert len(result.max_sharpe.daily_returns) == 60
    assert len(result.equal_weight.daily_returns) == 60
    assert len(result.max_sharpe.wealth) == 61
    assert list(result.dates) == list(dates[252:])

    first, second, last = result.rebalances[0], result.rebalances[1], result.rebalances[-1]
    assert first.date == dates[252]
    assert first.first_close_date == dates[0]  # t-312
    assert first.first_return_date == dates[1]  # t-311
    assert first.last_close_date == dates[252]  # t-60
    assert first.n_training_returns == 252
    assert first.holding_start_date == dates[253]  # t-59
    assert first.holding_end_date == dates[257]  # t-55
    assert second.date == dates[257]  # t-55
    assert second.first_return_date == dates[6]  # The window slides by 5.
    assert second.holding_start_date == dates[258]  # t-54
    assert last.date == dates[307]  # t-5
    assert last.holding_end_date == dates[312]  # t, with no rebalance there.


def test_every_optimization_gets_exactly_its_training_closes(monkeypatch):
    seen = []
    from_prices = PortfolioProblem.from_prices

    def spy(wide_df, daily_risk_free_rate, weight_cap_multiple):
        seen.append(wide_df.copy())
        return from_prices(wide_df, daily_risk_free_rate, weight_cap_multiple)

    monkeypatch.setattr(backtest.PortfolioProblem, "from_prices", spy)
    result = run(optimizer=fixed_optimizer)
    assert len(seen) == N_REBALANCES
    for window, record in zip(seen, result.rebalances):
        assert len(window) == TRAINING_RETURNS + 1
        assert window.index[-1] == record.date
        assert window.index[0] == record.first_close_date
        assert len(window.pct_change().iloc[1:]) == TRAINING_RETURNS


def test_interval_must_divide_the_scored_days():
    with pytest.raises(ValueError, match="whole holding periods"):
        run(rebalance_interval=7)


def test_one_scored_day_is_too_few_for_a_volatility():
    with pytest.raises(ValueError, match="at least 2 scored days"):
        run(make_prices(TRAINING_RETURNS + 2), evaluation_days=1, rebalance_interval=1)


def test_real_optimizer_gives_feasible_target_weights():
    result = run()
    weights = result.max_sharpe.target_weights
    assert weights.shape == (N_REBALANCES, len(TICKERS))
    assert (weights >= 0).all()
    assert (weights <= result.weight_cap + 1e-5).all()
    assert weights.sum(axis=1) == pytest.approx(np.ones(N_REBALANCES))
    assert {record.status for record in result.rebalances} == {STATUS_OPTIMIZED}


# Temporal leakage


def test_changing_future_prices_leaves_earlier_weights_unchanged():
    prices = make_prices()
    original = run(prices)
    # Rewrite everything after the second rebalance date.
    second_position = TRAINING_RETURNS + INTERVAL
    changed = prices.copy()
    n_changed = len(prices) - (second_position + 1)
    changed.iloc[second_position + 1 :, 0] *= np.exp(np.linspace(0.05, 1.0, n_changed))
    changed.iloc[second_position + 1 :, 1] *= np.exp(np.linspace(-0.05, -1.0, n_changed))
    result = run(changed)
    np.testing.assert_array_equal(
        result.max_sharpe.target_weights[:2], original.max_sharpe.target_weights[:2]
    )
    # The change does reach the later optimizations, so the test can fail.
    assert not np.array_equal(
        result.max_sharpe.target_weights[2:], original.max_sharpe.target_weights[2:]
    )


def test_changing_rates_dated_on_or_after_a_rebalance_leaves_its_weights_unchanged():
    prices = make_prices()
    rates = make_rates()
    original = run(prices, rates)
    second_date = original.rebalances[1].date.normalize()
    changed = rates.copy()
    changed.loc[changed.index >= second_date, "DTB3"] = 50.0
    result = run(prices, changed)
    np.testing.assert_array_equal(
        result.max_sharpe.target_weights[:2], original.max_sharpe.target_weights[:2]
    )
    assert result.rebalances[1].annual_risk_free_rate == (
        original.rebalances[1].annual_risk_free_rate
    )
    assert result.rebalances[2].annual_risk_free_rate == 50.0


def test_each_rebalance_uses_the_rate_of_the_day_before():
    prices = make_prices()
    rates = make_rates()
    result = run(prices, rates, optimizer=fixed_optimizer)
    for i, record in enumerate(result.rebalances):
        previous_day = prices.index[TRAINING_RETURNS + i * INTERVAL - 1].normalize()
        assert record.risk_free_rate_date == previous_day
        assert record.annual_risk_free_rate == pytest.approx(rates.loc[previous_day, "DTB3"])
    assert len({record.annual_risk_free_rate for record in result.rebalances}) == N_REBALANCES


def test_a_missing_rate_uses_the_last_published_one():
    prices = make_prices()
    rates = make_rates()
    day_before = prices.index[TRAINING_RETURNS - 1].normalize()
    two_days_before = prices.index[TRAINING_RETURNS - 2].normalize()
    rates.loc[day_before, "DTB3"] = np.nan
    result = run(prices, rates, optimizer=fixed_optimizer)
    assert result.rebalances[0].risk_free_rate_date == two_days_before


def test_no_rate_before_a_rebalance_raises():
    prices = make_prices()
    rates = make_rates().loc[prices.index[TRAINING_RETURNS].normalize() :]
    with pytest.raises(ValueError, match="DTB3"):
        run(prices, rates, optimizer=fixed_optimizer)


# Missing observations


def test_a_missing_close_raises_and_names_the_ticker():
    prices = make_prices()
    prices.iloc[10, 1] = np.nan
    with pytest.raises(ValueError, match="BBB"):
        run(prices, optimizer=fixed_optimizer)


def test_a_missing_close_in_the_evaluation_window_raises():
    prices = make_prices()
    prices.iloc[-3, 2] = np.nan
    with pytest.raises(ValueError, match="CCC"):
        run(prices, optimizer=fixed_optimizer)


def test_a_missing_session_raises_and_names_the_date():
    # One session is absent, so the right number of rows reaches further back.
    prices = make_prices(N_SESSIONS + 1)
    missing_day = prices.index[20]
    with pytest.raises(ValueError, match=str(missing_day.date())):
        run(
            prices.drop(missing_day),
            expected_sessions=prices.index.normalize()[1:],
            optimizer=fixed_optimizer,
        )


@pytest.mark.parametrize("n_sessions", [N_SESSIONS - 1, N_SESSIONS + 1])
def test_the_wrong_number_of_sessions_raises(n_sessions):
    with pytest.raises(ValueError, match="exactly"):
        run(make_prices(n_sessions), optimizer=fixed_optimizer)


# Weight drift


def test_weights_drift_with_returns_during_a_holding_period():
    # Doubling one of two equal positions leaves weights of 2/3 and 1/3.
    weights = np.array([0.5, 0.5])
    returns, drifted = drift_through_holding(weights, np.array([[1.0, 0.0]]))
    assert returns == pytest.approx([0.5])
    assert drifted == pytest.approx([2 / 3, 1 / 3])


def test_the_second_day_uses_drifted_weights_not_the_targets():
    weights = np.array([0.5, 0.5])
    asset_returns = np.array([[1.0, 0.0], [-0.5, 0.0]])
    returns, drifted = drift_through_holding(weights, asset_returns)
    # Positions go 0.5, 0.5 -> 1.0, 0.5 -> 0.5, 0.5, so the portfolio goes
    # 1.0 -> 1.5 -> 1.0. Constant weights would have lost 25 %, not a third.
    assert returns == pytest.approx([0.5, -1 / 3])
    assert drifted == pytest.approx([0.5, 0.5])


def test_a_rebalance_resets_the_drifted_weights():
    targets = np.array([[0.5, 0.5], [0.5, 0.5]])
    holdings = [np.array([[1.0, 0.0]]), np.array([[-0.5, 0.0]])]
    path = simulate_portfolio(targets, holdings)
    assert np.isnan(path.weights_before_rebalance[0]).all()
    assert path.weights_before_rebalance[1] == pytest.approx([2 / 3, 1 / 3])
    # Back at 50/50, the -50 % day costs 25 %, unlike in the test above.
    assert path.daily_returns == pytest.approx([0.5, -0.25])
    assert path.wealth == pytest.approx([1.0, 1.5, 1.125])


def test_equal_weight_rebalances_on_the_same_dates():
    prices = make_prices()
    result = run(prices, optimizer=fixed_optimizer)
    equal = result.equal_weight
    assert np.allclose(equal.target_weights, 1 / 3)
    asset_returns = prices.pct_change().iloc[TRAINING_RETURNS + 1 :].to_numpy()
    # On the first day after each rebalance the weights are exactly equal.
    for i in range(N_REBALANCES):
        assert equal.daily_returns[i * INTERVAL] == pytest.approx(
            asset_returns[i * INTERVAL].mean()
        )
    # Later in a holding period they have drifted.
    assert equal.daily_returns[1] != pytest.approx(asset_returns[1].mean(), abs=1e-12)
    assert not np.allclose(equal.weights_before_rebalance[1], 1 / 3)


def test_fixed_weights_compound_position_by_position():
    prices = make_prices()
    result = run(prices, optimizer=fixed_optimizer)
    start = TRAINING_RETURNS
    growth = prices.iloc[start + INTERVAL].to_numpy() / prices.iloc[start].to_numpy()
    assert result.max_sharpe.wealth[INTERVAL] == pytest.approx(FIXED_WEIGHTS @ growth)
    assert result.max_sharpe.weights_before_rebalance[1] == pytest.approx(
        FIXED_WEIGHTS * growth / (FIXED_WEIGHTS @ growth)
    )


# Wealth and metrics


def test_wealth_index_starts_at_one_and_compounds():
    assert wealth_index(np.array([0.1, -0.5])) == pytest.approx([1.0, 1.1, 0.55])


def test_cumulative_return_and_cagr():
    wealth = np.concatenate([np.ones(60), [1.1]])
    metrics = performance_metrics(wealth)
    assert metrics.cumulative_return == pytest.approx(0.1)
    assert metrics.cagr == pytest.approx(1.1 ** (252 / 60) - 1)


def test_max_drawdown_is_measured_from_the_running_peak():
    wealth = np.array([1.0, 0.9, 0.95, 1.2, 0.6, 0.9])
    assert performance_metrics(wealth).max_drawdown == pytest.approx(-0.5)


def test_max_drawdown_counts_the_initial_wealth_as_a_peak():
    wealth = np.array([1.0, 0.8, 0.9])
    assert performance_metrics(wealth).max_drawdown == pytest.approx(-0.2)


def test_max_drawdown_is_zero_when_wealth_only_rises():
    assert performance_metrics(np.array([1.0, 1.1, 1.2])).max_drawdown == 0.0


def test_annualized_volatility_is_the_sample_std_times_root_252():
    # Returns of +10 % and -10 %: the sample variance is 0.02 (ddof = 1).
    wealth = np.array([1.0, 1.1, 0.99])
    assert performance_metrics(wealth).annualized_volatility == pytest.approx(
        np.sqrt(0.02 * 252)
    )


def test_annualized_volatility_is_zero_for_constant_returns():
    wealth = wealth_index(np.full(10, 0.001))
    assert performance_metrics(wealth).annualized_volatility == pytest.approx(0.0, abs=1e-12)


def test_result_metrics_come_from_the_wealth_series():
    result = run()
    for path in (result.max_sharpe, result.equal_weight):
        assert path.wealth[0] == 1.0
        assert path.wealth == pytest.approx(wealth_index(path.daily_returns))
        assert path.metrics == performance_metrics(path.wealth)
        assert path.metrics.annualized_volatility == pytest.approx(
            np.std(path.daily_returns, ddof=1) * np.sqrt(252)
        )


# The fallback


def flaky_optimizer(failing_calls: set[int]):
    """Fails on the given calls (from 0) and otherwise returns distinct weights."""
    weights = iter(
        [
            np.array([0.6, 0.3, 0.1]),
            np.array([0.3, 0.6, 0.1]),
            np.array([0.1, 0.3, 0.6]),
            np.array([0.2, 0.4, 0.4]),
        ]
    )

    def optimizer(problem: PortfolioProblem) -> np.ndarray:
        call = optimizer.calls
        optimizer.calls += 1
        target = next(weights)
        if call in failing_calls:
            raise OptimizationError("did not converge")
        return target

    optimizer.calls = 0
    return optimizer


def test_failed_first_optimization_uses_equal_weights():
    result = run(optimizer=flaky_optimizer({0}))
    assert result.rebalances[0].status == STATUS_FALLBACK_EQUAL
    assert result.max_sharpe.target_weights[0] == pytest.approx(1 / 3)
    assert result.rebalances[1].status == STATUS_OPTIMIZED


def test_failed_later_optimization_rebalances_to_the_previous_target():
    result = run(optimizer=flaky_optimizer({2}))
    statuses = [record.status for record in result.rebalances]
    assert statuses == [
        STATUS_OPTIMIZED,
        STATUS_OPTIMIZED,
        STATUS_FALLBACK_PREVIOUS,
        STATUS_OPTIMIZED,
    ]
    targets = result.max_sharpe.target_weights
    np.testing.assert_array_equal(targets[2], targets[1])


def test_consecutive_failures_keep_the_last_good_target():
    result = run(optimizer=flaky_optimizer({1, 2}))
    targets = result.max_sharpe.target_weights
    np.testing.assert_array_equal(targets[1], targets[0])
    np.testing.assert_array_equal(targets[2], targets[0])


@pytest.mark.parametrize(
    "bad_weights",
    [
        [0.9, 0.05, 0.05],  # Above the cap of 2/3.
        [0.5, 0.2, 0.2],  # Does not sum to 1.
        [0.8, 0.4, -0.2],  # Short.
        [np.nan, 0.5, 0.5],
    ],
)
def test_weights_that_violate_the_constraints_use_the_fallback(bad_weights):
    result = run(optimizer=lambda problem: np.array(bad_weights))
    assert result.rebalances[0].status == STATUS_FALLBACK_EQUAL
    assert result.rebalances[1].status == STATUS_FALLBACK_PREVIOUS
    assert result.max_sharpe.target_weights == pytest.approx(1 / 3)


def test_rounding_noise_is_tidied_not_treated_as_a_failure():
    noisy = np.array([0.6 + 1e-9, 0.4, -1e-9])
    result = run(optimizer=lambda problem: noisy.copy())
    assert result.rebalances[0].status == STATUS_OPTIMIZED
    weights = result.max_sharpe.target_weights
    assert (weights >= 0).all()
    assert weights.sum(axis=1) == pytest.approx(np.ones(N_REBALANCES), abs=1e-15)


def test_other_errors_are_not_swallowed():
    def broken(problem):
        raise KeyError("a bug, not a failed optimization")

    with pytest.raises(KeyError):
        run(optimizer=broken)


# Invariants


def replace_record(result, index: int, **changes):
    records = list(result.rebalances)
    records[index] = dataclasses.replace(records[index], **changes)
    return dataclasses.replace(result, rebalances=tuple(records))


def test_a_valid_result_passes_the_invariants():
    check_backtest_invariants(run(optimizer=fixed_optimizer))


def test_invariants_catch_a_missing_rebalance():
    result = run(optimizer=fixed_optimizer)
    with pytest.raises(InvariantError, match="rebalances"):
        check_backtest_invariants(
            dataclasses.replace(result, rebalances=result.rebalances[:-1])
        )


def test_invariants_catch_a_wrong_training_length():
    result = run(optimizer=fixed_optimizer)
    with pytest.raises(InvariantError, match="returns"):
        check_backtest_invariants(
            replace_record(result, 1, n_training_returns=TRAINING_RETURNS - 1)
        )


def test_invariants_catch_a_training_window_reaching_into_its_holding_period():
    result = run(optimizer=fixed_optimizer)
    record = result.rebalances[1]
    with pytest.raises(InvariantError, match="training window"):
        check_backtest_invariants(
            replace_record(result, 1, last_close_date=record.holding_start_date)
        )


def test_invariants_catch_overlapping_holding_periods():
    result = run(optimizer=fixed_optimizer)
    with pytest.raises(InvariantError, match="holding period"):
        check_backtest_invariants(
            replace_record(result, 1, holding_start_date=result.rebalances[0].holding_end_date)
        )


def test_invariants_catch_a_risk_free_rate_from_the_cutoff():
    result = run(optimizer=fixed_optimizer)
    record = result.rebalances[2]
    with pytest.raises(InvariantError, match="risk-free"):
        check_backtest_invariants(
            replace_record(result, 2, risk_free_rate_date=record.date.normalize())
        )


def test_invariants_catch_weights_above_the_cap():
    result = run(optimizer=fixed_optimizer)
    targets = result.max_sharpe.target_weights.copy()
    targets[0] = [0.8, 0.1, 0.1]
    path = dataclasses.replace(result.max_sharpe, target_weights=targets)
    with pytest.raises(InvariantError, match="constraints"):
        check_backtest_invariants(dataclasses.replace(result, max_sharpe=path))


def test_invariants_catch_wealth_that_does_not_compound():
    result = run(optimizer=fixed_optimizer)
    wealth = result.equal_weight.wealth.copy()
    wealth[10] *= 1.01
    path = dataclasses.replace(result.equal_weight, wealth=wealth)
    with pytest.raises(InvariantError, match="compound"):
        check_backtest_invariants(dataclasses.replace(result, equal_weight=path))


def test_invariants_catch_metrics_that_differ_from_the_wealth():
    result = run(optimizer=fixed_optimizer)
    metrics = dataclasses.replace(result.max_sharpe.metrics, cagr=0.0)
    path = dataclasses.replace(result.max_sharpe, metrics=metrics)
    with pytest.raises(InvariantError, match="metrics"):
        check_backtest_invariants(dataclasses.replace(result, max_sharpe=path))


def test_invariants_catch_a_volatility_that_differs_from_the_wealth():
    result = run(optimizer=fixed_optimizer)
    metrics = dataclasses.replace(result.equal_weight.metrics, annualized_volatility=0.0)
    path = dataclasses.replace(result.equal_weight, metrics=metrics)
    with pytest.raises(InvariantError, match="metrics"):
        check_backtest_invariants(dataclasses.replace(result, equal_weight=path))


# The written files


PO_CONFIG = PortfolioOptimizationConfig(
    weight_cap_multiple=CAP_MULTIPLE,
    efficient_frontier_points=20,
    simulated_portfolios=200,
    random_seed=0,
    near_frontier_tolerance=1.0,
)


def live_optimization(prices: pd.DataFrame, rates: pd.DataFrame):
    """The live optimization at the last session, and its rate and closes."""
    training_prices = prices.iloc[-(TRAINING_RETURNS + 1) :]
    rate = risk_free_rate_before(rates, prices.index[-1])
    result = run_portfolio_optimization(
        training_prices, rate["daily_risk_free_rate"], PO_CONFIG
    )
    return result, rate, training_prices


@pytest.fixture
def written(tmp_path):
    """A backtest with its two files written; returns result and both paths."""
    prices = make_prices()
    rates = make_rates()
    result = run(prices, rates)
    optimization, rate, training_prices = live_optimization(prices, rates)
    daily_path = str(tmp_path / "portfolio_backtest_daily.csv")
    metadata_path = str(tmp_path / "portfolio_backtest_metadata.yml")
    write_backtest_daily(result, daily_path)
    write_backtest_metadata(
        build_backtest_metadata(
            result,
            list(TICKERS),
            build_final_optimization(optimization, rate, training_prices.index),
        ),
        metadata_path,
    )
    return result, daily_path, metadata_path


@pytest.fixture
def live_files(tmp_path, written):
    """The backtest metadata and the live optimization's two files."""
    prices = make_prices()
    optimization, rate, _ = live_optimization(prices, make_rates())
    optimization_path = str(tmp_path / "optimization_metadata.yml")
    plot_data_path = str(tmp_path / "portfolio_optimization_plot_data.h5")
    write_optimization_metadata(
        build_optimization_metadata(optimization, list(TICKERS), rate), optimization_path
    )
    write_plot_data_h5(optimization, plot_data_path)
    return written[2], optimization_path, plot_data_path


def edit_yaml(path: str, change) -> None:
    with open(path, encoding="utf-8") as f:
        document = yaml.safe_load(f)
    change(document)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(document, f)


def test_daily_frame_has_percent_returns_and_dollar_wealth():
    result = run()
    frame = backtest_daily_frame(result)
    assert frame.index.name == "date"
    assert len(frame) == EVALUATION_DAYS + 1
    assert frame.index[0] == result.dates[0].strftime("%Y-%m-%d")
    assert frame["max_sharpe_daily_return"].isna().sum() == 1
    assert np.isnan(frame["equal_weight_daily_return"].iloc[0])
    assert frame["max_sharpe_wealth"].iloc[0] == 1.0
    assert frame["equal_weight_daily_return"].iloc[1:].to_numpy() == pytest.approx(
        result.equal_weight.daily_returns * 100
    )


def test_daily_file_reads_back(written):
    result, daily_path, _ = written
    daily = pd.read_csv(daily_path, index_col="date", parse_dates=True)
    assert list(daily.columns) == [
        "max_sharpe_daily_return",
        "max_sharpe_wealth",
        "equal_weight_daily_return",
        "equal_weight_wealth",
    ]
    assert daily["max_sharpe_wealth"].to_numpy() == pytest.approx(result.max_sharpe.wealth)


def test_metadata_records_the_settings_and_windows(written):
    result, _, metadata_path = written
    with open(metadata_path, encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    assert metadata["tickers"] == list(TICKERS)
    assert metadata["schedule"] == {
        "training_window_returns": TRAINING_RETURNS,
        "rebalance_interval_sessions": INTERVAL,
        "rebalance_count": N_REBALANCES,
        "fallback_count": 0,
    }
    window = metadata["evaluation_window"]
    assert window["trading_days"] == EVALUATION_DAYS
    assert window["initial_wealth_date"] == result.dates[0].strftime("%Y-%m-%d")
    assert window["start_date"] == result.dates[1].strftime("%Y-%m-%d")
    assert window["end_date"] == result.dates[-1].strftime("%Y-%m-%d")
    assert metadata["constraints"]["weight_cap"] == pytest.approx(200 / 3)
    assert metadata["annualization"]["method"] == "compounded"
    assert "not an execution-realistic" in metadata["assumptions"]["execution"]


def test_metadata_metrics_are_in_percent(written):
    result, _, metadata_path = written
    with open(metadata_path, encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    for name, path in (
        ("max_sharpe", result.max_sharpe),
        ("equal_weight", result.equal_weight),
    ):
        written_metrics = metadata["metrics"][name]
        assert written_metrics["cumulative_return"] == pytest.approx(
            path.metrics.cumulative_return * 100
        )
        assert written_metrics["cagr"] == pytest.approx(path.metrics.cagr * 100)
        assert written_metrics["max_drawdown"] == pytest.approx(path.metrics.max_drawdown * 100)
        assert written_metrics["max_drawdown"] <= 0
        assert written_metrics["annualized_volatility"] == pytest.approx(
            path.metrics.annualized_volatility * 100
        )
        assert written_metrics["annualized_volatility"] > 0
    assert metadata["units"]["annualized_volatility"] == "percent_per_year"


def test_metadata_has_every_rebalance_for_both_portfolios(written):
    result, _, metadata_path = written
    with open(metadata_path, encoding="utf-8") as f:
        rebalances = yaml.safe_load(f)["rebalances"]
    assert len(rebalances) == N_REBALANCES
    for i, (entry, record) in enumerate(zip(rebalances, result.rebalances)):
        assert entry["date"] == record.date.strftime("%Y-%m-%d")
        assert entry["status"] == STATUS_OPTIMIZED
        training = entry["training_window"]
        assert training["returns"] == TRAINING_RETURNS
        assert training["first_close_date"] == record.first_close_date.strftime("%Y-%m-%d")
        assert training["first_return_date"] == record.first_return_date.strftime("%Y-%m-%d")
        assert training["last_close_date"] == entry["date"]
        assert entry["holding_period"]["end_date"] == record.holding_end_date.strftime("%Y-%m-%d")
        assert entry["risk_free_rate"]["risk_free_rate_date"] < entry["date"]
        for name in ("max_sharpe", "equal_weight"):
            assert set(entry[name]["target_weights"]) == set(TICKERS)
            assert sum(entry[name]["target_weights"].values()) == pytest.approx(100)
            before = entry[name]["weights_before_rebalance"]
            if i == 0:
                assert before is None
            else:
                assert sum(before.values()) == pytest.approx(100)
            stats = entry[name]["in_sample"]
            assert stats["sharpe_ratio"] == pytest.approx(
                (stats["annualized_return"] - entry["risk_free_rate"]["risk_free_rate"])
                / stats["risk"]
            )


def test_in_sample_stats_favor_max_sharpe_over_equal_weight():
    # On each training window, the optimized portfolio is the better one.
    result = run()
    for record in result.rebalances:
        assert record.max_sharpe_stats.sharpe_ratio >= (
            record.equal_weight_stats.sharpe_ratio - 1e-6
        )


def test_metadata_counts_fallbacks():
    result = run(optimizer=flaky_optimizer({1, 3}))
    metadata = build_backtest_metadata(result, list(TICKERS), {})
    assert metadata["schedule"]["fallback_count"] == 2


def test_written_files_pass_verification(written):
    _, daily_path, metadata_path = written
    verify_backtest_files(daily_path, metadata_path)


def test_verification_catches_changed_metrics(written):
    _, daily_path, metadata_path = written
    with open(metadata_path, encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    metadata["metrics"]["max_sharpe"]["cagr"] += 1.0
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f)
    with pytest.raises(InvariantError, match="metrics"):
        verify_backtest_files(daily_path, metadata_path)


def test_verification_catches_a_changed_volatility(written):
    _, daily_path, metadata_path = written
    with open(metadata_path, encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    metadata["metrics"]["equal_weight"]["annualized_volatility"] += 1.0
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f)
    with pytest.raises(InvariantError, match="metrics"):
        verify_backtest_files(daily_path, metadata_path)


def test_verification_catches_a_missing_row(written):
    _, daily_path, metadata_path = written
    daily = pd.read_csv(daily_path, index_col="date")
    daily.iloc[:-1].to_csv(daily_path)
    with pytest.raises(InvariantError, match="row count"):
        verify_backtest_files(daily_path, metadata_path)


def test_verification_catches_wealth_that_does_not_compound(written):
    _, daily_path, metadata_path = written
    daily = pd.read_csv(daily_path, index_col="date")
    daily.iloc[5, daily.columns.get_loc("equal_weight_wealth")] *= 1.01
    daily.to_csv(daily_path)
    with pytest.raises(InvariantError, match="compound"):
        verify_backtest_files(daily_path, metadata_path)


# final_optimization


def test_final_optimization_holds_the_live_values_and_window(written):
    result, _, metadata_path = written
    prices = make_prices()
    optimization, rate, _ = live_optimization(prices, make_rates())
    with open(metadata_path, encoding="utf-8") as f:
        final = yaml.safe_load(f)["final_optimization"]
    assert final["date"] == prices.index[-1].strftime("%Y-%m-%d")
    assert final["training_window"] == {
        "first_close_date": prices.index[-(TRAINING_RETURNS + 1)].strftime("%Y-%m-%d"),
        "first_return_date": prices.index[-TRAINING_RETURNS].strftime("%Y-%m-%d"),
        "last_close_date": final["date"],
        "returns": TRAINING_RETURNS,
    }
    assert final["annualized_return"] == optimization.max_sharpe_return
    assert final["risk"] == optimization.max_sharpe_risk
    assert final["sharpe_ratio"] == optimization.sharpe_ratio
    assert final["weights"] == optimization.max_sharpe_weights_pct
    assert final["risk_free_rate"]["risk_free_rate"] == rate["risk_free_rate"]
    assert final["risk_free_rate"]["risk_free_rate_date"] == rate["risk_free_rate_date"][:10]
    # It is an extra optimization after the rebalances, not one of them.
    assert result.rebalances[-1].date == prices.index[-(INTERVAL + 1)]
    assert final["date"] > result.rebalances[-1].date.strftime("%Y-%m-%d")


def test_the_live_optimization_follows_the_rebalance_rule():
    # Solving the last session the way the backtest solves a rebalance gives
    # the live portfolio, so final_optimization continues the rebalance series.
    prices = make_prices()
    rates = make_rates(N_SESSIONS + INTERVAL)
    optimization, rate, _ = live_optimization(prices, rates)
    # One more holding period makes the last session of prices a rebalance.
    extended = make_prices(N_SESSIONS + INTERVAL)
    extended.iloc[:N_SESSIONS] = prices.to_numpy()
    result = run(extended.iloc[INTERVAL:], rates)
    last = result.rebalances[-1]
    assert last.date == prices.index[-1]
    assert last.annual_risk_free_rate == pytest.approx(rate["risk_free_rate"])
    assert result.max_sharpe.target_weights[-1] == pytest.approx(
        optimization.max_sharpe_weights, abs=1e-9
    )


def test_matching_files_pass_verification(live_files):
    verify_final_optimization(*live_files)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["final_optimization"].update(date="2020-01-01"), "last date"),
        (
            lambda d: d["final_optimization"]["training_window"].update(
                first_close_date="2020-01-01"
            ),
            "first date",
        ),
        (
            lambda d: d["final_optimization"]["training_window"].update(returns=1),
            "training window",
        ),
        (
            lambda d: d["final_optimization"]["risk_free_rate"].update(risk_free_rate=9.0),
            "risk-free",
        ),
        (
            lambda d: d["final_optimization"]["risk_free_rate"].update(
                risk_free_rate_date="2020-01-01"
            ),
            "risk-free",
        ),
        (lambda d: d["final_optimization"].update(sharpe_ratio=9.0), "statistics"),
        (lambda d: d["final_optimization"].update(risk=9.0), "statistics"),
        (lambda d: d["final_optimization"]["weights"].update(AAA=1.0), "weights"),
        (lambda d: d["final_optimization"]["weights"].pop("AAA"), "weights"),
    ],
)
def test_verification_catches_a_final_optimization_that_differs(live_files, change, message):
    edit_yaml(live_files[0], change)
    with pytest.raises(InvariantError, match=message):
        verify_final_optimization(*live_files)


def test_verification_catches_a_different_h5_point(live_files):
    with h5py.File(live_files[2], "r+") as hf:
        hf["max_sharpe_ratio/ys"][0] += 1.0
    with pytest.raises(InvariantError, match=".h5"):
        verify_final_optimization(*live_files)
