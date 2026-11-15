"""
Orchestration tests for aliciadata_finance_backend.pipeline.run_pipeline.

Downloads, ARIMA training, the risk-free rate history, the NYSE sessions, and
S3 are replaced by in-memory doubles; files are written only under pytest's tmp_path. Nothing
here touches the network, a bucket, or the real input/ and output/ folders.
"""

import datetime
import os

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import SYNTHETIC_TICKERS, make_long_prices
from aliciadata_finance_backend import pipeline
from aliciadata_finance_backend.optimization.workflow import InvariantError
from aliciadata_finance_backend.config import (
    PortfolioBacktestConfig,
    PortfolioOptimizationConfig,
    TimeSeriesConfig,
)
from aliciadata_finance_backend.data.prices import pivot_close_wide
from aliciadata_finance_backend.pipeline import (
    PORTFOLIO_FOLDER_ENV,
    TIME_SERIES_FOLDER_ENV,
    first_rebalance_date,
    price_download_sessions,
    required_sessions,
    run_pipeline,
    utc_timestamp,
    with_provenance,
)
from aliciadata_finance_backend.storage.layout import CONTRACT_VERSION
from aliciadata_finance_backend.storage.s3_uploader import BUCKET_NAME_ENV

SPACE = "test-space"
COMPUTED_AT = "2026-11-14T12:00:00Z"
TS_FOLDER = "test-time-series"
PO_FOLDER = "test-portfolio"
STUB_UPLIFT = 1.01  # Fake ARIMA forecast = 1 % above the previous close.
# A representative configuration (252 closes, 20 scored days). The tests do
# not depend on the values in config/modeling_and_optimization.yml.
TS_CONFIG = TimeSeriesConfig(training_window_length=252, evaluation_window_length=20)
ARIMA_WINDOW_SESSIONS = TS_CONFIG.arima_window_sessions
EVALUATION_DAYS = TS_CONFIG.evaluation_window_length
TRAINING_WINDOW_RETURNS = TS_CONFIG.training_window_returns
# Small optimization settings keep the many pipeline runs fast; the
# optimization itself is tested in tests/test_workflow.py.
PO_CONFIG = PortfolioOptimizationConfig(
    weight_cap_multiple=4,
    efficient_frontier_points=20,
    simulated_portfolios=500,
    random_seed=0,
    near_frontier_tolerance=1.0,
)
# The backtest scores its own number of days (30), different from the
# forecasts' (20), so the tests can tell the two settings apart.
BT_CONFIG = PortfolioBacktestConfig(
    training_window_returns=252,
    evaluation_window_length=30,
    rebalance_interval_sessions=5,
)
BACKTEST_DAYS = BT_CONFIG.evaluation_window_length
BACKTEST_SESSIONS = BT_CONFIG.sessions_needed
REQUIRED_SESSIONS = required_sessions(TS_CONFIG, BT_CONFIG)
PRICE_DOWNLOAD_SESSIONS = price_download_sessions(TS_CONFIG, BT_CONFIG)
# A full download: as many sessions as the pipeline requests.
N_SESSIONS = PRICE_DOWNLOAD_SESSIONS


def fake_risk_free_history(dm, first_date: pd.Timestamp) -> pd.DataFrame:
    # Covers every session of the synthetic prices, starting before them, and
    # changes daily, so a rate from the wrong date is visible. first_date is
    # ignored; test_risk_free_history_is_requested_from_the_first_rebalance
    # checks it.
    dates = pd.bdate_range("2026-06-01", periods=N_SESSIONS + 60)
    return pd.DataFrame({"DTB3": np.linspace(3.0, 5.0, len(dates))}, index=dates)


def fake_sessions(last_session: pd.Timestamp, n_sessions: int) -> pd.DatetimeIndex:
    # The synthetic prices are on weekdays, without NYSE holidays.
    return pd.bdate_range(end=last_session.normalize(), periods=n_sessions)


class FakeTickerDownloadManager:
    """
    Returns long_df, or cached_df (if given) when the cache is allowed, so a
    test can make today's cached download differ from a fresh one.
    """

    def __init__(self, long_df: pd.DataFrame, cached_df: pd.DataFrame | None = None):
        self.long_df = long_df
        self.cached_df = cached_df
        self.tickers = list(SYNTHETIC_TICKERS)
        self.calls: list[dict] = []

    def get_latest_tickers(self, days_in_past=40, use_cache=True):
        self.calls.append({"days_in_past": days_in_past, "use_cache": use_cache})
        if use_cache and self.cached_df is not None:
            return self.cached_df, "start", "end"
        return self.long_df, "start", "end"


class FakeTrainer:
    """
    Produces a forecasts table shaped like ArimaWalkForward's: one row per
    window end's next session plus the live row, with T_arima, T_naive (the
    previous close) and T (the actual, empty on the live row).
    """

    def __init__(
        self, fail: bool = False, training_window_returns: int = TRAINING_WINDOW_RETURNS
    ):
        self.fail = fail
        self.training_window_returns = training_window_returns
        self.calls: list[pd.DataFrame] = []

    def train_arima_models(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append(df)
        if self.fail:
            raise RuntimeError("ARIMA failed")
        n = self.training_window_returns
        window_ends = df.index[n:]
        pred_dates = list(df.index[n + 1 :]) + [
            df.index[-1] + pd.offsets.BDay(1)
        ]
        out = pd.DataFrame(index=pd.DatetimeIndex(pred_dates, name="pred_date"))
        for t in df.columns:
            naive = df.loc[window_ends, t].to_numpy()
            out[f"{t}_arima"] = STUB_UPLIFT * naive
            out[f"{t}_naive"] = naive
            out[t] = df[t].reindex(out.index).to_numpy()
        return out


class FakeUploader:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def upload_file(self, local_filename, object_key):
        # Every file must already be written when it is uploaded.
        assert os.path.isfile(local_filename), local_filename
        self.calls.append((local_filename, object_key))


@pytest.fixture
def upload_env(monkeypatch):
    monkeypatch.setenv(BUCKET_NAME_ENV, SPACE)
    monkeypatch.setenv(TIME_SERIES_FOLDER_ENV, TS_FOLDER)
    monkeypatch.setenv(PORTFOLIO_FOLDER_ENV, PO_FOLDER)


@pytest.fixture
def deps(tmp_path):
    """Fresh doubles plus the keyword arguments run_pipeline needs."""
    tdm = FakeTickerDownloadManager(make_long_prices(n_sessions=N_SESSIONS))
    trainer = FakeTrainer()
    uploader = FakeUploader()
    kwargs = dict(
        ts_config=TS_CONFIG,
        tdm=tdm,
        trainer=trainer,
        uploader=uploader,
        risk_free_history_getter=fake_risk_free_history,
        sessions_getter=fake_sessions,
        dm=object(),
        output_root=str(tmp_path / "output"),
        po_config=PO_CONFIG,
        bt_config=BT_CONFIG,
        # Fixed, so files from two runs can be compared byte for byte.
        computed_at=COMPUTED_AT,
    )
    return tdm, trainer, uploader, kwargs


def wide_prices() -> pd.DataFrame:
    return pivot_close_wide(make_long_prices(n_sessions=N_SESSIONS))


def expected_as_of() -> str:
    return wide_prices().index[-1].strftime("%Y-%m-%d")


def test_full_run_writes_six_files_and_uploads_in_order(upload_env, deps):
    _, _, uploader, kwargs = deps
    result = run_pipeline(**kwargs)
    as_of = expected_as_of()
    assert result.as_of == as_of
    for path in (
        result.forecasts_path,
        result.forecasts_metadata_path,
        result.plot_data_path,
        result.metadata_path,
        result.backtest_daily_path,
        result.backtest_metadata_path,
    ):
        assert os.path.isfile(path)
        assert f"{os.sep}{as_of}{os.sep}" in path
    assert [key for _, key in uploader.calls] == [
        f"{TS_FOLDER}/{as_of}/all_forecasts.csv",
        f"{TS_FOLDER}/{as_of}/all_forecasts_metadata.yml",
        f"{PO_FOLDER}/{as_of}/portfolio_optimization_plot_data.h5",
        f"{PO_FOLDER}/{as_of}/portfolio_backtest_daily.csv",
        f"{PO_FOLDER}/{as_of}/portfolio_backtest_metadata.yml",
        f"{PO_FOLDER}/{as_of}/optimization_metadata.yml",
    ]
    assert result.uploaded == tuple(key for _, key in uploader.calls)
    assert result.trained_arima is True


def test_keys_never_contain_the_space_name(upload_env, deps):
    # The Space is the S3 bucket; it must not also appear inside a key.
    _, _, uploader, kwargs = deps
    run_pipeline(**kwargs)
    assert all(SPACE not in key for _, key in uploader.calls)


def test_local_layout_mirrors_object_keys(upload_env, deps, tmp_path):
    # output/<kind>/<date>/<file> and <folder>/<date>/<file> share <date>/<file>.
    _, _, uploader, kwargs = deps
    run_pipeline(**kwargs)
    kinds_and_folders = [
        ("time_series", TS_FOLDER),
        ("time_series", TS_FOLDER),
        ("portfolio_optimization", PO_FOLDER),
        ("portfolio_optimization", PO_FOLDER),
        ("portfolio_optimization", PO_FOLDER),
        ("portfolio_optimization", PO_FOLDER),
    ]
    for (local, key), (kind, folder) in zip(uploader.calls, kinds_and_folders):
        relative = os.path.relpath(local, tmp_path / "output" / kind)
        assert f"{folder}/" + relative.replace(os.sep, "/") == key


def test_metadata_date_to_equals_folder_date(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    with open(result.metadata_path) as f:
        metadata = yaml.safe_load(f)
    assert metadata["date_updated"]["date_to"] == result.as_of
    assert metadata["tickers"] == SYNTHETIC_TICKERS


def test_single_download_of_enough_sessions_with_cache(upload_env, deps):
    tdm, _, _, kwargs = deps
    run_pipeline(**kwargs)
    assert ARIMA_WINDOW_SESSIONS == 272
    assert REQUIRED_SESSIONS == BACKTEST_SESSIONS == 283  # 253 closes + 30 days.
    assert PRICE_DOWNLOAD_SESSIONS > REQUIRED_SESSIONS  # A small margin.
    assert tdm.calls == [{"days_in_past": PRICE_DOWNLOAD_SESSIONS, "use_cache": True}]


def test_settings_flow_from_the_config_into_every_step(upload_env, deps):
    # A different configuration changes the download, the ARIMA input, and
    # the metadata, with no other code change.
    _, _, _, kwargs = deps
    ts_config = TimeSeriesConfig(training_window_length=100, evaluation_window_length=10)
    trainer = FakeTrainer(training_window_returns=ts_config.training_window_returns)
    tdm = FakeTickerDownloadManager(make_long_prices(n_sessions=N_SESSIONS))
    kwargs.update(ts_config=ts_config, trainer=trainer, tdm=tdm)
    result = run_pipeline(**kwargs)
    # The backtest's 283 sessions are still the largest need, plus the margin.
    assert required_sessions(ts_config, BT_CONFIG) == 283
    assert tdm.calls[0]["days_in_past"] == price_download_sessions(ts_config, BT_CONFIG)
    (window,) = trainer.calls
    assert len(window) == 110  # 100 closes + 10 scored days.
    meta = load_forecasts_metadata(result)
    assert meta["forecast_window"]["training_window_returns"] == 99
    assert meta["models"]["arima"]["order_selection_subwindow_returns"] == 99
    assert meta["evaluation_window"]["target_trading_days"] == 10
    for metrics in meta["ticker_metrics"].values():
        assert metrics["n_evaluated"] == 10
    # The live forecast's training window is its last 100 closes.
    assert meta["forecast_window"]["training_window_start_date"] == window.index[
        -100
    ].strftime("%Y-%m-%d")


@pytest.mark.parametrize("config_kwarg", ["ts_config", "po_config", "bt_config"])
def test_configs_are_required(upload_env, deps, config_kwarg):
    _, _, _, kwargs = deps
    del kwargs[config_kwarg]
    with pytest.raises(TypeError, match=config_kwarg):
        run_pipeline(**kwargs)


def test_optimization_settings_flow_into_the_plot_data(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    with h5py.File(result.plot_data_path, "r") as hf:
        assert hf["efficient_frontier/xs"].shape == (PO_CONFIG.efficient_frontier_points,)
        assert hf["simulated_portfolios/xs"].shape == (PO_CONFIG.simulated_portfolios,)
        # The capital allocation line ends at the max Sharpe ratio point.
        assert hf["tangency_line/xs"][-1] == hf["max_sharpe_ratio/xs"][0]


def test_mismatched_optimization_files_upload_nothing(upload_env, deps, monkeypatch):
    # verify_optimization_files() runs after both files are written and
    # before any upload; if it fails, nothing (not even the forecasts) is
    # uploaded.
    _, _, uploader, kwargs = deps

    def mismatch(plot_data_path, metadata_path):
        assert os.path.isfile(plot_data_path) and os.path.isfile(metadata_path)
        raise InvariantError("files disagree")

    monkeypatch.setattr(pipeline, "verify_optimization_files", mismatch)
    with pytest.raises(InvariantError, match="files disagree"):
        run_pipeline(**kwargs)
    assert uploader.calls == []


def test_short_cached_download_is_downloaded_again_once(upload_env, deps):
    _, trainer, uploader, kwargs = deps
    full = make_long_prices(n_sessions=N_SESSIONS)
    short = make_long_prices(n_sessions=REQUIRED_SESSIONS - 1)
    tdm = FakeTickerDownloadManager(full, cached_df=short)
    kwargs["tdm"] = tdm
    result = run_pipeline(**kwargs)
    assert [call["use_cache"] for call in tdm.calls] == [True, False]
    (window,) = trainer.calls
    assert len(window) == ARIMA_WINDOW_SESSIONS
    assert len(uploader.calls) == 6
    assert result.as_of == expected_as_of()


def test_too_short_fresh_download_fails_and_uploads_nothing(upload_env, deps):
    _, trainer, uploader, kwargs = deps
    short = make_long_prices(n_sessions=REQUIRED_SESSIONS - 1)
    tdm = FakeTickerDownloadManager(short)
    kwargs["tdm"] = tdm
    with pytest.raises(ValueError, match="fewer than"):
        run_pipeline(**kwargs)
    assert [call["use_cache"] for call in tdm.calls] == [True, False]
    assert trainer.calls == []
    assert uploader.calls == []


def test_refresh_data_with_short_download_fails_without_retrying(upload_env, deps):
    _, _, uploader, kwargs = deps
    tdm = FakeTickerDownloadManager(make_long_prices(n_sessions=REQUIRED_SESSIONS - 1))
    kwargs["tdm"] = tdm
    with pytest.raises(ValueError, match="fewer than"):
        run_pipeline(refresh_data=True, **kwargs)
    assert [call["use_cache"] for call in tdm.calls] == [False]
    assert uploader.calls == []


def test_optimization_uses_the_backtest_training_window(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    with open(result.metadata_path) as f:
        metadata = yaml.safe_load(f)
    wide = wide_prices()
    assert BT_CONFIG.training_window_closes == 253
    assert len(wide) > 253  # More was downloaded.
    assert metadata["date_updated"]["date_from"] == wide.index[-253].strftime("%Y-%m-%d")
    assert metadata["date_updated"]["date_to"] == result.as_of


def test_optimization_uses_the_last_rate_dated_before_the_as_of_date(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    with open(result.metadata_path) as f:
        rate = yaml.safe_load(f)["risk_free_rate"]
    history = fake_risk_free_history(None, pd.Timestamp("2026-06-01"))
    previous_session = wide_prices().index[-2].normalize()
    assert rate["risk_free_rate_date"] == str(previous_session)
    assert rate["risk_free_rate"] == pytest.approx(history.loc[previous_session, "DTB3"])
    assert rate["daily_risk_free_rate"] == pytest.approx(rate["risk_free_rate"] / 252)


def test_refresh_data_bypasses_download_cache(upload_env, deps):
    tdm, _, _, kwargs = deps
    run_pipeline(refresh_data=True, **kwargs)
    assert tdm.calls[-1]["use_cache"] is False


def test_arima_trains_on_last_272_sessions(upload_env, deps):
    assert ARIMA_WINDOW_SESSIONS == 272
    _, trainer, _, kwargs = deps
    run_pipeline(**kwargs)
    (window,) = trainer.calls
    pd.testing.assert_frame_equal(window, wide_prices().iloc[-ARIMA_WINDOW_SESSIONS:])


def test_cached_forecasts_skip_training(upload_env, deps):
    _, trainer, uploader, kwargs = deps
    first = run_pipeline(**kwargs)
    with open(first.forecasts_path) as f:
        cached = f.read()
    trainer.calls.clear()
    second = run_pipeline(**kwargs)
    assert trainer.calls == []
    assert second.trained_arima is False
    with open(second.forecasts_path) as f:
        assert f.read() == cached
    assert len(uploader.calls) == 12  # The cached file is still uploaded.


def test_retrain_arima_ignores_cache(upload_env, deps):
    _, trainer, _, kwargs = deps
    first = run_pipeline(**kwargs)
    with open(first.forecasts_path, "w") as f:
        f.write("stale")
    trainer.calls.clear()
    second = run_pipeline(retrain_arima=True, **kwargs)
    assert len(trainer.calls) == 1
    assert second.trained_arima is True
    with open(second.forecasts_path) as f:
        assert f.read() != "stale"


def test_no_upload_writes_files_without_uploading_or_env(monkeypatch, deps):
    for env in (BUCKET_NAME_ENV, TIME_SERIES_FOLDER_ENV, PORTFOLIO_FOLDER_ENV):
        monkeypatch.delenv(env, raising=False)
    _, _, uploader, kwargs = deps
    result = run_pipeline(upload=False, **kwargs)
    assert uploader.calls == []
    assert result.uploaded == ()
    assert os.path.isfile(result.metadata_path)
    assert os.path.isfile(result.forecasts_metadata_path)


@pytest.mark.parametrize(
    "missing", [BUCKET_NAME_ENV, TIME_SERIES_FOLDER_ENV, PORTFOLIO_FOLDER_ENV]
)
def test_missing_upload_env_fails_before_any_work(monkeypatch, upload_env, deps, missing):
    monkeypatch.delenv(missing)
    tdm, trainer, uploader, kwargs = deps
    with pytest.raises(RuntimeError, match=missing):
        run_pipeline(**kwargs)
    assert tdm.calls == []
    assert trainer.calls == []
    assert uploader.calls == []


def test_arima_failure_uploads_nothing(upload_env, deps):
    _, _, uploader, kwargs = deps
    kwargs["trainer"] = FakeTrainer(fail=True)
    with pytest.raises(RuntimeError, match="ARIMA failed"):
        run_pipeline(**kwargs)
    assert uploader.calls == []


def test_upload_failure_stops_before_completion_marker(upload_env, deps):
    # If an upload fails part way, the error propagates and
    # optimization_metadata.yml (the completion marker) is never uploaded,
    # so the frontend does not list the incomplete date.
    _, _, _, kwargs = deps

    class FailOnH5Uploader(FakeUploader):
        def upload_file(self, local_filename, object_key):
            if object_key.endswith(".h5"):
                raise ConnectionError("upload failed")
            super().upload_file(local_filename, object_key)

    uploader = FailOnH5Uploader()
    kwargs["uploader"] = uploader
    with pytest.raises(ConnectionError):
        run_pipeline(**kwargs)
    keys = [key for _, key in uploader.calls]
    assert [k.rsplit("/", 1)[-1] for k in keys] == [
        "all_forecasts.csv",
        "all_forecasts_metadata.yml",
    ]


def test_optimization_failure_uploads_nothing_but_keeps_forecast_cache(
    upload_env, deps
):
    _, _, uploader, kwargs = deps

    def failing_rf(dm, first_date):
        raise RuntimeError("risk-free rate unavailable")

    kwargs["risk_free_history_getter"] = failing_rf
    with pytest.raises(RuntimeError, match="risk-free rate"):
        run_pipeline(**kwargs)
    assert uploader.calls == []
    forecasts = os.path.join(
        kwargs["output_root"], "time_series", expected_as_of(), "all_forecasts.csv"
    )
    assert os.path.isfile(forecasts)  # A retry will not retrain ARIMA.


# The portfolio backtest


def load_backtest_metadata(result) -> dict:
    with open(result.backtest_metadata_path) as f:
        return yaml.safe_load(f)


def long_prices_without(date: str, ticker: str | None = None) -> pd.DataFrame:
    """A full download with one session removed, or one ticker's close on it."""
    long_df = make_long_prices(n_sessions=N_SESSIONS)
    removed = long_df["datetime"].str.startswith(date)
    if ticker is not None:
        removed &= long_df["ticker"] == ticker
    return long_df[~removed].reset_index(drop=True)


def first_backtest_session() -> str:
    # Older than the ARIMA and optimization windows, so only the backtest
    # sees a gap on this date.
    return wide_prices().index[-BACKTEST_SESSIONS].strftime("%Y-%m-%d")


def test_backtest_uses_its_windows_and_the_settings_of_the_config(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    meta = load_backtest_metadata(result)
    wide = wide_prices()
    assert meta["schedule"]["training_window_returns"] == 252
    assert meta["schedule"]["rebalance_interval_sessions"] == 5
    assert meta["schedule"]["rebalance_count"] == BACKTEST_DAYS // 5
    assert meta["constraints"]["weight_cap_multiple"] == PO_CONFIG.weight_cap_multiple
    assert meta["evaluation_window"]["trading_days"] == BACKTEST_DAYS
    assert meta["evaluation_window"]["end_date"] == result.as_of
    assert meta["evaluation_window"]["initial_wealth_date"] == wide.index[
        -(BACKTEST_DAYS + 1)
    ].strftime("%Y-%m-%d")
    first = meta["rebalances"][0]
    assert first["training_window"]["first_close_date"] == first_backtest_session()
    assert first["training_window"]["last_close_date"] == first["date"]
    assert meta["tickers"] == SYNTHETIC_TICKERS


def test_backtest_scores_its_own_days_ending_with_the_forecasts(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    backtest = load_backtest_metadata(result)["evaluation_window"]
    forecasts = load_forecasts_metadata(result)["evaluation_window"]
    wide = wide_prices()
    assert backtest["trading_days"] == BACKTEST_DAYS
    assert forecasts["target_trading_days"] == EVALUATION_DAYS
    assert backtest["trading_days"] != forecasts["target_trading_days"]
    # Both end on the as-of date; each starts its own number of days back.
    assert backtest["end_date"] == forecasts["end_date"] == result.as_of
    assert backtest["start_date"] == wide.index[-BACKTEST_DAYS].strftime("%Y-%m-%d")
    assert forecasts["start_date"] == wide.index[-EVALUATION_DAYS].strftime("%Y-%m-%d")
    daily = pd.read_csv(result.backtest_daily_path, index_col="date")
    assert len(daily) == BACKTEST_DAYS + 1


def test_risk_free_history_is_requested_from_the_first_rebalance(upload_env, deps):
    # The earliest optimization is the backtest's first rebalance, at the
    # close where wealth is $1.00; the history must reach back before it.
    _, _, _, kwargs = deps
    requested = []

    def recording_rf(dm, first_date):
        requested.append(first_date)
        return fake_risk_free_history(dm, first_date)

    kwargs["risk_free_history_getter"] = recording_rf
    result = run_pipeline(**kwargs)
    meta = load_backtest_metadata(result)
    assert len(requested) == 1
    first_rebalance = meta["rebalances"][0]["date"]
    assert requested[0].strftime("%Y-%m-%d") == first_rebalance
    assert first_rebalance == meta["evaluation_window"]["initial_wealth_date"]


def test_backtest_days_flow_from_the_config(upload_env, deps):
    # A shorter backtest changes only the backtest, and the ARIMA window
    # becomes the largest need.
    _, _, _, kwargs = deps
    bt_config = PortfolioBacktestConfig(
        training_window_returns=252,
        evaluation_window_length=10,
        rebalance_interval_sessions=5,
    )
    kwargs.update(bt_config=bt_config)
    result = run_pipeline(**kwargs)
    assert required_sessions(TS_CONFIG, bt_config) == ARIMA_WINDOW_SESSIONS
    meta = load_backtest_metadata(result)
    assert meta["evaluation_window"]["trading_days"] == 10
    assert meta["schedule"]["rebalance_count"] == 2
    forecasts = load_forecasts_metadata(result)["evaluation_window"]
    assert forecasts["target_trading_days"] == EVALUATION_DAYS


def test_backtest_failure_uploads_nothing(upload_env, deps, monkeypatch):
    _, _, uploader, kwargs = deps

    def failing_backtest(*args, **kwargs):
        raise RuntimeError("backtest failed")

    monkeypatch.setattr(pipeline, "run_portfolio_backtest", failing_backtest)
    with pytest.raises(RuntimeError, match="backtest failed"):
        run_pipeline(**kwargs)
    assert uploader.calls == []


def test_final_optimization_matches_both_optimization_files(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    final = load_backtest_metadata(result)["final_optimization"]
    with open(result.metadata_path) as f:
        metadata = yaml.safe_load(f)
    live = metadata["optimum_portfolio"]
    assert final["date"] == result.as_of
    assert final["annualized_return"] == live["annualized_return"]
    assert final["risk"] == live["risk"]
    assert final["sharpe_ratio"] == live["sharpe_ratio"]
    assert final["weights"] == live["weights"]
    assert final["training_window"]["first_close_date"] == metadata["date_updated"]["date_from"]
    assert final["training_window"]["returns"] == BT_CONFIG.training_window_returns
    assert final["risk_free_rate"]["risk_free_rate"] == (
        metadata["risk_free_rate"]["risk_free_rate"]
    )
    with h5py.File(result.plot_data_path, "r") as hf:
        assert final["risk"] == pytest.approx(hf["max_sharpe_ratio/xs"][0])
        assert final["annualized_return"] == pytest.approx(hf["max_sharpe_ratio/ys"][0])


def test_final_optimization_mismatch_uploads_nothing(upload_env, deps, monkeypatch):
    _, _, uploader, kwargs = deps

    def mismatch(backtest_metadata_path, optimization_metadata_path, plot_data_path):
        assert os.path.isfile(backtest_metadata_path)
        assert os.path.isfile(optimization_metadata_path)
        assert os.path.isfile(plot_data_path)
        raise InvariantError("files disagree")

    monkeypatch.setattr(pipeline, "verify_final_optimization", mismatch)
    with pytest.raises(InvariantError, match="files disagree"):
        run_pipeline(**kwargs)
    assert uploader.calls == []


def test_mismatched_backtest_files_upload_nothing(upload_env, deps, monkeypatch):
    _, _, uploader, kwargs = deps

    def mismatch(daily_path, metadata_path):
        assert os.path.isfile(daily_path) and os.path.isfile(metadata_path)
        raise InvariantError("files disagree")

    monkeypatch.setattr(pipeline, "verify_backtest_files", mismatch)
    with pytest.raises(InvariantError, match="files disagree"):
        run_pipeline(**kwargs)
    assert uploader.calls == []


def test_a_missing_close_in_the_backtest_window_uploads_nothing(upload_env, deps):
    _, _, uploader, kwargs = deps
    kwargs["tdm"] = FakeTickerDownloadManager(
        long_prices_without(first_backtest_session(), ticker="QQQ")
    )
    with pytest.raises(ValueError, match="QQQ"):
        run_pipeline(**kwargs)
    assert uploader.calls == []


def test_a_missing_session_in_the_backtest_window_uploads_nothing(upload_env, deps):
    _, _, uploader, kwargs = deps
    session = first_backtest_session()
    kwargs["tdm"] = FakeTickerDownloadManager(long_prices_without(session))
    with pytest.raises(ValueError, match=session):
        run_pipeline(**kwargs)
    assert uploader.calls == []


# all_forecasts_metadata.yml


def load_forecasts_metadata(result) -> dict:
    with open(result.forecasts_metadata_path) as f:
        return yaml.safe_load(f)


def test_forecasts_metadata_windows_match_the_data(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    meta = load_forecasts_metadata(result)
    window = wide_prices().iloc[-ARIMA_WINDOW_SESSIONS:]
    fw = meta["forecast_window"]
    assert fw["training_window_end_date"] == result.as_of
    assert fw["training_window_start_date"] == window.index[
        -(TRAINING_WINDOW_RETURNS + 1)
    ].strftime("%Y-%m-%d")
    forecasts = pd.read_csv(result.forecasts_path, index_col="pred_date", parse_dates=True)
    assert fw["forecast_date"] == forecasts.index[-1].strftime("%Y-%m-%d")
    assert fw["forecast_date"] > fw["training_window_end_date"]
    assert fw["training_window_returns"] == 251
    assert fw["horizon_trading_days"] == 1
    ev = meta["evaluation_window"]
    assert ev["target_trading_days"] == EVALUATION_DAYS
    assert ev["end_date"] == result.as_of  # The last day with an actual close.


def test_forecasts_metadata_has_lowercase_tickers_scored_on_20_days(upload_env, deps):
    _, _, _, kwargs = deps
    meta = load_forecasts_metadata(run_pipeline(**kwargs))
    assert set(meta["ticker_metrics"]) == {t.lower() for t in SYNTHETIC_TICKERS}
    for metrics in meta["ticker_metrics"].values():
        assert metrics["n_evaluated"] == EVALUATION_DAYS
        # The fake ARIMA is 1 % above the previous close; naive is the close.
        assert metrics["relative_mae"] == pytest.approx(
            metrics["arima"]["mae"] / metrics["naive"]["mae"]
        )
        assert metrics["skill"] == pytest.approx(1 - metrics["relative_mae"])


def test_forecasts_metadata_keys_are_lowercase_snake_case(upload_env, deps):
    _, _, _, kwargs = deps
    meta = load_forecasts_metadata(run_pipeline(**kwargs))

    def keys(d):
        for k, v in d.items():
            yield k
            if isinstance(v, dict):
                yield from keys(v)

    assert all(k == k.lower() and " " not in k for k in keys(meta))


def test_forecasts_metadata_is_written_for_cached_forecasts(upload_env, deps):
    _, trainer, _, kwargs = deps
    run_pipeline(**kwargs)
    trainer.calls.clear()
    second = run_pipeline(**kwargs)  # Reuses the cached CSV.
    assert trainer.calls == []
    assert load_forecasts_metadata(second)["ticker_metrics"]


# Runs for a past as-of date (as_of_date)

# Sessions of synthetic data after the as-of date in these tests.
LATER_SESSIONS = 7


def prices_with_later_sessions() -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    A full download plus LATER_SESSIONS more sessions, and the as-of date
    that leaves exactly a full download's sessions up to it.
    """
    long_df = make_long_prices(n_sessions=N_SESSIONS + LATER_SESSIONS)
    as_of = pivot_close_wide(long_df).index[-(LATER_SESSIONS + 1)].normalize()
    return long_df, as_of


def prices_up_to(long_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """The rows of a long download dated on or before as_of."""
    dates = pd.to_datetime(long_df["datetime"]).dt.normalize()
    return long_df[dates <= as_of].reset_index(drop=True)


def text_files(result) -> dict[str, str]:
    """Every text output file of a run, by file name."""
    paths = (
        result.forecasts_path,
        result.forecasts_metadata_path,
        result.metadata_path,
        result.backtest_daily_path,
        result.backtest_metadata_path,
    )
    contents = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            contents[os.path.basename(path)] = f.read()
    return contents


def h5_contents(path: str) -> tuple[dict, dict]:
    """The root attributes and every dataset of an .h5 file."""
    with h5py.File(path, "r") as hf:
        attrs = {key: hf.attrs[key] for key in hf.attrs}
        datasets = {}
        hf.visititems(
            lambda name, obj: datasets.__setitem__(name, obj[()])
            if isinstance(obj, h5py.Dataset)
            else None
        )
    return attrs, datasets


def assert_same_outputs(first, second) -> None:
    assert first.as_of == second.as_of
    assert text_files(first) == text_files(second)
    first_attrs, first_data = h5_contents(first.plot_data_path)
    second_attrs, second_data = h5_contents(second.plot_data_path)
    assert first_attrs == second_attrs
    assert first_data.keys() == second_data.keys()
    for name in first_data:
        assert np.array_equal(first_data[name], second_data[name], equal_nan=True), name


def test_past_as_of_date_gives_the_files_of_data_ending_that_day(deps, tmp_path):
    # No look-ahead: a run for a past as-of date, on data that continues after
    # it, writes exactly the files of a run on data that genuinely ends then.
    _, _, _, kwargs = deps
    long_df, as_of = prices_with_later_sessions()

    trainer_past = FakeTrainer()
    past = run_pipeline(
        upload=False,
        **{
            **kwargs,
            "tdm": FakeTickerDownloadManager(long_df),
            "trainer": trainer_past,
            "output_root": str(tmp_path / "past"),
        },
        as_of_date=as_of,
    )
    trainer_ended = FakeTrainer()
    ended = run_pipeline(
        upload=False,
        **{
            **kwargs,
            "tdm": FakeTickerDownloadManager(prices_up_to(long_df, as_of)),
            "trainer": trainer_ended,
            "output_root": str(tmp_path / "ended"),
        },
    )
    assert past.as_of == as_of.strftime("%Y-%m-%d")
    # ARIMA saw the same windows, none reaching past the as-of date.
    (past_window,) = trainer_past.calls
    (ended_window,) = trainer_ended.calls
    assert past_window.equals(ended_window)
    assert past_window.index.max().normalize() == as_of
    assert_same_outputs(past, ended)


def test_past_as_of_date_is_uploaded_under_its_own_date(upload_env, deps):
    _, _, uploader, kwargs = deps
    long_df, as_of = prices_with_later_sessions()
    kwargs["tdm"] = FakeTickerDownloadManager(long_df)
    result = run_pipeline(**kwargs, as_of_date=as_of)
    date = as_of.strftime("%Y-%m-%d")
    assert all(f"/{date}/" in key for _, key in uploader.calls)
    assert [key for _, key in uploader.calls][-1] == (
        f"{PO_FOLDER}/{date}/optimization_metadata.yml"
    )
    forecasts = pd.read_csv(result.forecasts_path, index_col="pred_date")
    assert pd.isna(forecasts.iloc[-1][SYNTHETIC_TICKERS[0]])  # The live row.


def test_as_of_the_latest_session_is_a_normal_run(deps, tmp_path):
    _, _, _, kwargs = deps
    latest = wide_prices().index[-1].normalize()
    normal = run_pipeline(
        upload=False, **{**kwargs, "output_root": str(tmp_path / "normal")}
    )
    as_of = run_pipeline(
        upload=False,
        **{**kwargs, "trainer": FakeTrainer(), "output_root": str(tmp_path / "as_of")},
        as_of_date=latest,
    )
    assert_same_outputs(normal, as_of)


def test_cache_too_short_for_a_past_as_of_date_downloads_once_more(deps):
    # The cached download ends on the latest session but has only a normal
    # run's length, so it is short by the sessions after the as-of date; the
    # new download asks for those too (counted as weekdays).
    _, _, _, kwargs = deps
    long_df, as_of = prices_with_later_sessions()
    cached = long_df[
        pd.to_datetime(long_df["datetime"])
        >= pd.to_datetime(long_df["datetime"]).drop_duplicates().iloc[LATER_SESSIONS]
    ].reset_index(drop=True)
    tdm = FakeTickerDownloadManager(long_df, cached_df=cached)
    kwargs["tdm"] = tdm
    result = run_pipeline(upload=False, **kwargs, as_of_date=as_of)
    assert result.as_of == as_of.strftime("%Y-%m-%d")
    assert tdm.calls == [
        {"days_in_past": PRICE_DOWNLOAD_SESSIONS, "use_cache": True},
        {"days_in_past": PRICE_DOWNLOAD_SESSIONS + LATER_SESSIONS, "use_cache": False},
    ]


def test_as_of_date_after_the_latest_session_fails_before_any_output(
    upload_env, deps, tmp_path
):
    tdm, trainer, uploader, kwargs = deps
    later = wide_prices().index[-1].normalize() + pd.Timedelta(days=7)
    with pytest.raises(ValueError, match="after the latest session"):
        run_pipeline(**kwargs, as_of_date=later)
    assert trainer.calls == []
    assert uploader.calls == []
    assert not os.path.exists(tmp_path / "output")


def test_as_of_date_that_is_not_a_session_fails_before_any_output(
    upload_env, deps, tmp_path
):
    # A weekend inside the data: the run must not silently use the Friday.
    _, trainer, uploader, kwargs = deps
    long_df, as_of = prices_with_later_sessions()
    kwargs["tdm"] = FakeTickerDownloadManager(long_df)
    saturday = next(
        day
        for day in pd.date_range(as_of, periods=7)
        if day.dayofweek == 5
    )
    with pytest.raises(ValueError, match="not a session in the price data"):
        run_pipeline(**kwargs, as_of_date=saturday)
    assert trainer.calls == []
    assert uploader.calls == []
    assert not os.path.exists(tmp_path / "output")


def test_as_of_date_with_too_little_history_fails(deps):
    _, _, _, kwargs = deps
    early = wide_prices().index[REQUIRED_SESSIONS - 2].normalize()
    with pytest.raises(ValueError, match="fewer than"):
        run_pipeline(upload=False, **kwargs, as_of_date=early)


# Provenance in the metadata files (contract v48)


def test_every_metadata_file_records_when_and_by_which_contract(deps):
    _, _, _, kwargs = deps
    result = run_pipeline(upload=False, **kwargs)
    for path in (
        result.forecasts_metadata_path,
        result.metadata_path,
        result.backtest_metadata_path,
    ):
        with open(path) as f:
            metadata = yaml.safe_load(f)
        assert metadata["computed_at"] == COMPUTED_AT, path
        assert metadata["contract_version"] == CONTRACT_VERSION, path


def test_provenance_defaults_to_the_time_of_the_run(deps):
    _, _, _, kwargs = deps
    del kwargs["computed_at"]
    before = utc_timestamp()
    result = run_pipeline(upload=False, **kwargs)
    after = utc_timestamp()
    with open(result.metadata_path) as f:
        computed_at = yaml.safe_load(f)["computed_at"]
    assert before <= computed_at <= after  # ISO strings sort by time.


def test_optimization_metadata_keeps_its_comment_with_provenance(deps):
    _, _, _, kwargs = deps
    result = run_pipeline(upload=False, **kwargs)
    with open(result.metadata_path) as f:
        text = f.read()
    comment = "# optimum_portfolio is the maximum Sharpe ratio (tangency) portfolio."
    assert comment in text


def test_with_provenance_adds_both_keys_without_changing_the_input():
    metadata = {"tickers": ["XBI"]}
    assert with_provenance(metadata, COMPUTED_AT) == {
        "tickers": ["XBI"],
        "computed_at": COMPUTED_AT,
        "contract_version": CONTRACT_VERSION,
    }
    assert metadata == {"tickers": ["XBI"]}


def test_utc_timestamp_format_and_time_zone():
    utc = datetime.datetime(2026, 11, 14, 9, 5, 3, tzinfo=datetime.timezone.utc)
    assert utc_timestamp(utc) == "2026-11-14T09:05:03Z"
    eastern = datetime.timezone(datetime.timedelta(hours=-5))
    assert utc_timestamp(utc.astimezone(eastern)) == "2026-11-14T09:05:03Z"
    naive = datetime.datetime(2026, 11, 14, 9, 5, 3)  # Taken to be UTC.
    assert utc_timestamp(naive) == "2026-11-14T09:05:03Z"


def test_first_rebalance_date_is_the_close_ending_the_first_training_window():
    wide = wide_prices()
    expected = wide.index[-BACKTEST_SESSIONS + BT_CONFIG.training_window_returns]
    assert first_rebalance_date(wide, BT_CONFIG) == expected
