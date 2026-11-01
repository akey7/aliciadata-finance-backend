"""
Tests for fuzzy_system_research.config: strict parsing and validation of the
time series, portfolio optimization, and portfolio backtest settings, loading
from a YAML file, and the repository's own
config/modeling_and_optimization.yml.

Files are written only under pytest's tmp_path. The repository's config file
is read (never written) to catch a broken file before a long run; input/ and
output/ are never touched.
"""

from pathlib import Path

import pytest

from fuzzy_system_research.config import (
    DEFAULT_CONFIG_PATH,
    MIN_BACKTEST_TRAINING_RETURNS,
    MIN_TRAINING_WINDOW_LENGTH,
    ConfigError,
    ModelingConfig,
    PortfolioBacktestConfig,
    PortfolioOptimizationConfig,
    TimeSeriesConfig,
    load_config,
    parse_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


PORTFOLIO = {
    "weight_cap_multiple": 4,
    "efficient_frontier_points": 100,
    "simulated_portfolios": 10_000,
    "random_seed": 7,
    "near_frontier_tolerance": 1.0,
}
PORTFOLIO_YAML = (
    "portfolio_optimization:\n"
    "  weight_cap_multiple: 4\n"
    "  efficient_frontier_points: 100\n"
    "  simulated_portfolios: 10000\n"
    "  random_seed: 7\n"
    "  near_frontier_tolerance: 1.0\n"
    "portfolio_backtest:\n"
    "  training_window_returns: 252\n"
    "  rebalance_interval_sessions: 5\n"
)
BACKTEST = {"training_window_returns": 252, "rebalance_interval_sessions": 5}


def raw(training=252, evaluation=20, portfolio=None, backtest=None, **extra) -> dict:
    """
    A parsed YAML document; extra keys go under time_series_models, and
    portfolio and backtest (dicts) replace or add keys in their sections.
    """
    return {
        "time_series_models": {
            "training_window_length": training,
            "evaluation_window_length": evaluation,
            **extra,
        },
        "portfolio_optimization": {**PORTFOLIO, **(portfolio or {})},
        "portfolio_backtest": {**BACKTEST, **(backtest or {})},
    }


# parse_config and TimeSeriesConfig


def test_parse_config_reads_every_value():
    config = parse_config(raw())
    assert config == ModelingConfig(
        time_series_models=TimeSeriesConfig(
            training_window_length=252, evaluation_window_length=20
        ),
        portfolio_optimization=PortfolioOptimizationConfig(
            weight_cap_multiple=4,
            efficient_frontier_points=100,
            simulated_portfolios=10_000,
            random_seed=7,
            near_frontier_tolerance=1.0,
        ),
        portfolio_backtest=PortfolioBacktestConfig(
            training_window_returns=252, rebalance_interval_sessions=5
        ),
    )


def test_derived_window_sizes():
    ts = TimeSeriesConfig(training_window_length=252, evaluation_window_length=20)
    assert ts.training_window_returns == 251  # One fewer return than closes.
    assert ts.arima_window_sessions == 272  # 252 closes + 20 scored days.


def test_config_is_immutable():
    ts = TimeSeriesConfig(training_window_length=252, evaluation_window_length=20)
    with pytest.raises(AttributeError):
        ts.training_window_length = 100


@pytest.mark.parametrize("document", [None, [], "text", 5])
def test_document_must_be_a_mapping(document):
    with pytest.raises(ConfigError, match="mapping"):
        parse_config(document)


def test_missing_section_is_rejected():
    with pytest.raises(ConfigError, match="missing required key.*time_series_models"):
        parse_config({})


def test_unknown_top_level_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown key.*portfolio"):
        parse_config({**raw(), "portfolio": {}})


def test_section_must_be_a_mapping():
    with pytest.raises(ConfigError, match="time_series_models must be a mapping"):
        parse_config({**raw(), "time_series_models": None})


def test_portfolio_section_must_be_a_mapping():
    with pytest.raises(ConfigError, match="portfolio_optimization must be a mapping"):
        parse_config({**raw(), "portfolio_optimization": None})


@pytest.mark.parametrize("key", ["training_window_length", "evaluation_window_length"])
def test_missing_key_is_rejected(key):
    document = raw()
    del document["time_series_models"][key]
    with pytest.raises(ConfigError, match=f"missing required key.*{key}"):
        parse_config(document)


def test_misspelled_key_is_rejected_not_ignored():
    document = raw()
    document["time_series_models"]["training_window_lenght"] = 100
    with pytest.raises(ConfigError, match="unknown key.*training_window_lenght"):
        parse_config(document)


def test_non_string_unknown_key_is_reported():
    document = raw()
    document["time_series_models"][7] = "x"  # YAML keys need not be strings.
    with pytest.raises(ConfigError, match="unknown key.*7"):
        parse_config(document)


@pytest.mark.parametrize("value", [252.0, "252", True, None, [252]])
def test_training_window_length_must_be_an_integer(value):
    with pytest.raises(ConfigError, match="training_window_length must be an integer"):
        parse_config(raw(training=value))


@pytest.mark.parametrize("value", [20.0, "20", True, None])
def test_evaluation_window_length_must_be_an_integer(value):
    with pytest.raises(ConfigError, match="evaluation_window_length must be an integer"):
        parse_config(raw(evaluation=value))


def test_training_window_length_has_a_minimum():
    parse_config(raw(training=MIN_TRAINING_WINDOW_LENGTH))  # Accepted.
    with pytest.raises(ConfigError, match="at least"):
        parse_config(raw(training=MIN_TRAINING_WINDOW_LENGTH - 1))


@pytest.mark.parametrize("value", [0, -1])
def test_evaluation_window_length_must_be_positive(value):
    with pytest.raises(ConfigError, match="at least 1"):
        parse_config(raw(evaluation=value))


def test_direct_construction_is_validated_too():
    with pytest.raises(ConfigError):
        TimeSeriesConfig(training_window_length=10, evaluation_window_length=20)


# portfolio_optimization


def test_missing_portfolio_section_is_rejected():
    document = raw()
    del document["portfolio_optimization"]
    with pytest.raises(ConfigError, match="missing required key.*portfolio_optimization"):
        parse_config(document)


@pytest.mark.parametrize("key", sorted(PORTFOLIO))
def test_missing_portfolio_key_is_rejected(key):
    document = raw()
    del document["portfolio_optimization"][key]
    with pytest.raises(ConfigError, match=f"missing required key.*{key}"):
        parse_config(document)


def test_unknown_portfolio_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key.*weight_cap"):
        parse_config(raw(portfolio={"weight_cap": 0.4}))


@pytest.mark.parametrize("value", [1.5, 2, 4, 10])
def test_weight_cap_multiple_accepts_numbers_above_one(value):
    config = parse_config(raw(portfolio={"weight_cap_multiple": value}))
    assert config.portfolio_optimization.weight_cap_multiple == value


@pytest.mark.parametrize("value", [1, 0.5, 0, -2])
def test_weight_cap_multiple_must_exceed_one(value):
    with pytest.raises(ConfigError, match="weight_cap_multiple must be greater than 1"):
        parse_config(raw(portfolio={"weight_cap_multiple": value}))


@pytest.mark.parametrize("value", ["4", True, None, float("nan"), float("inf")])
def test_weight_cap_multiple_must_be_a_finite_number(value):
    with pytest.raises(ConfigError, match="weight_cap_multiple must be a number"):
        parse_config(raw(portfolio={"weight_cap_multiple": value}))


@pytest.mark.parametrize(
    "key, too_small",
    [("efficient_frontier_points", 1), ("simulated_portfolios", 0), ("random_seed", -1)],
)
def test_portfolio_counts_have_minimums(key, too_small):
    with pytest.raises(ConfigError, match=f"portfolio_optimization.{key} must be at least"):
        parse_config(raw(portfolio={key: too_small}))


@pytest.mark.parametrize(
    "key", ["efficient_frontier_points", "simulated_portfolios", "random_seed"]
)
@pytest.mark.parametrize("value", [10.0, "10", True, None])
def test_portfolio_counts_must_be_integers(key, value):
    with pytest.raises(ConfigError, match=f"portfolio_optimization.{key} must be an integer"):
        parse_config(raw(portfolio={key: value}))


@pytest.mark.parametrize("value", [0.5, 1, 2.0])
def test_near_frontier_tolerance_accepts_positive_numbers(value):
    config = parse_config(raw(portfolio={"near_frontier_tolerance": value}))
    assert config.portfolio_optimization.near_frontier_tolerance == value


@pytest.mark.parametrize("value", [0, -1.0])
def test_near_frontier_tolerance_must_be_positive(value):
    with pytest.raises(ConfigError, match="near_frontier_tolerance must be greater than 0"):
        parse_config(raw(portfolio={"near_frontier_tolerance": value}))


@pytest.mark.parametrize("value", ["1", True, None, float("nan")])
def test_near_frontier_tolerance_must_be_a_number(value):
    with pytest.raises(ConfigError, match="near_frontier_tolerance must be a number"):
        parse_config(raw(portfolio={"near_frontier_tolerance": value}))


# portfolio_backtest


def test_missing_backtest_section_is_rejected():
    document = raw()
    del document["portfolio_backtest"]
    with pytest.raises(ConfigError, match="missing required key.*portfolio_backtest"):
        parse_config(document)


@pytest.mark.parametrize("key", sorted(BACKTEST))
def test_missing_backtest_key_is_rejected(key):
    document = raw()
    del document["portfolio_backtest"][key]
    with pytest.raises(ConfigError, match=f"missing required key.*{key}"):
        parse_config(document)


def test_unknown_backtest_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key.*rebalance_days"):
        parse_config(raw(backtest={"rebalance_days": 5}))


@pytest.mark.parametrize("key", sorted(BACKTEST))
@pytest.mark.parametrize("value", [5.0, "5", True, None])
def test_backtest_values_must_be_integers(key, value):
    with pytest.raises(ConfigError, match=f"portfolio_backtest.{key} must be an integer"):
        parse_config(raw(backtest={key: value}))


def test_backtest_training_window_has_a_minimum():
    parse_config(raw(backtest={"training_window_returns": MIN_BACKTEST_TRAINING_RETURNS}))
    with pytest.raises(ConfigError, match="at least"):
        parse_config(
            raw(backtest={"training_window_returns": MIN_BACKTEST_TRAINING_RETURNS - 1})
        )


def test_rebalance_interval_must_be_positive():
    with pytest.raises(ConfigError, match="at least 1"):
        parse_config(raw(backtest={"rebalance_interval_sessions": 0}))


def test_rebalance_interval_must_divide_the_scored_days():
    parse_config(raw(evaluation=60, backtest={"rebalance_interval_sessions": 20}))
    with pytest.raises(ConfigError, match="multiple of.*rebalance_interval_sessions"):
        parse_config(raw(evaluation=60, backtest={"rebalance_interval_sessions": 7}))


def test_backtest_sessions_needed():
    config = PortfolioBacktestConfig(
        training_window_returns=252, rebalance_interval_sessions=5
    )
    assert config.training_window_closes == 253
    assert config.sessions_needed(60) == 313  # 253 closes + 60 scored days.


# load_config


def test_load_config_reads_a_yaml_file(tmp_path):
    path = tmp_path / "settings.yml"
    path.write_text(
        "time_series_models:\n"
        "  training_window_length: 100\n"
        "  evaluation_window_length: 10\n" + PORTFOLIO_YAML,
        encoding="utf-8",
    )
    config = load_config(str(path))
    assert config.time_series_models == TimeSeriesConfig(
        training_window_length=100, evaluation_window_length=10
    )
    assert config.portfolio_optimization == PortfolioOptimizationConfig(**PORTFOLIO)
    assert config.portfolio_backtest == PortfolioBacktestConfig(**BACKTEST)


def test_load_config_missing_file_names_the_path(tmp_path):
    path = tmp_path / "nowhere.yml"
    with pytest.raises(ConfigError, match="not found.*nowhere.yml"):
        load_config(str(path))


def test_load_config_invalid_yaml_names_the_path(tmp_path):
    path = tmp_path / "broken.yml"
    path.write_text("time_series_models: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="broken.yml.*not valid YAML"):
        load_config(str(path))


def test_load_config_validation_error_names_the_path(tmp_path):
    path = tmp_path / "bad.yml"
    path.write_text(
        "time_series_models:\n  training_window_length: 252\n" + PORTFOLIO_YAML,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="bad.yml.*evaluation_window_length"):
        load_config(str(path))


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.yml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(str(path))


# The repository's own configuration file


def test_repository_config_file_is_valid():
    # Checks that the file parses and validates, not its values, so they
    # can be changed freely.
    config = load_config(str(REPO_ROOT / DEFAULT_CONFIG_PATH))
    assert isinstance(config.time_series_models, TimeSeriesConfig)
    assert isinstance(config.portfolio_optimization, PortfolioOptimizationConfig)
    assert isinstance(config.portfolio_backtest, PortfolioBacktestConfig)
