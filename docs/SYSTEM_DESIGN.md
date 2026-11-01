# System Design

This document describes the system design of the backend.

Also of interest to an engineering audience is the [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md) file, which describes the contract of data transmission from the backend to the frontend.

## How the backend works

One run of `scripts/run_optimization_and_time_series.py` downloads prices, then follows two paths, the time series forecasts and the portfolio optimization with its backtest, and finally uploads their results. The first diagram shows which module owns each part of a run; the second shows which files each step reads and writes, and the functions that write them.

## From data sources to upload

```text
 legend: [external]   module.py   ─► data flow
         module paths are relative to src/fuzzy_system_research/

 scripts/run_optimization_and_time_series.py      entry point
   ├─ config.py ◄── config/modeling_and_optimization.yml
   └─ pipeline.py                                 runs the steps below
        │
 [Massive.com API] ──► 1. PRICES
                         data/ticker_download_manager.py   download, cache in input/annual/
                         data/date_manager.py              NYSE trading calendar
                         data/prices.py                    wide price table, recent sessions
        │
        ├────────────────────────────────┐   (run in order: 2, then 3)
        ▼                                ▼
 2. TIME SERIES                     3. OPTIMIZATION
 forecasting/ticker_predict.py      optimization/problem.py
   walk-forward windows,              PortfolioProblem: estimates, constraints,
   worker pool                        portfolio_stats(), feasible sampler
 forecasting/arima_models.py        optimization/portfolio.py  ◄── [FRED: DTB3]
   ARIMA fits, order by BIC           optimizers, risk-free rate (cache in input/)
 forecasting/metrics.py             optimization/workflow.py
   accuracy vs. naive, .yml writer    full optimization run, invariant checks,
                                      .h5 and .yml writers, file cross-check
                                    optimization/backtest.py
                                      walk-forward max Sharpe vs. equal weight,
                                      invariant checks, .csv and .yml writers,
                                      cross-checks of its files with the
                                      optimization's files
        │                                │
        └────────────────┬───────────────┘
                         ▼
 4. UPLOAD   only if 2 and 3 both succeeded; skipped with --no-upload
    storage/layout.py        dated local paths and bucket keys
    storage/s3_uploader.py   uploads to the DigitalOcean Space
                         │
                         ▼
             [DigitalOcean Space]   (files in the next diagram)
```

- The two paths run one after the other in a single process. Within the time series path, the ARIMA fits for every ticker and day run at the same time in a pool of worker processes (`--workers`).
- The ARIMA window, the number of scored days, and the optimization's weight cap, plot array lengths, and random seed come from the configuration file (see "Model settings" in [INSTALLATION_AND_EXECUTION.md](INSTALLATION_AND_EXECUTION.md#model-settings)). The optimization uses the backtest's training window length.
- The optimization has one definition, `PortfolioProblem` in `optimization/problem.py`. Its constraint set is used by every optimizer and by the random portfolios, and its `portfolio_stats()` computes every return, volatility, and Sharpe ratio that is uploaded, so the plot cannot drift from the optimization. `workflow.py` checks the invariants (see its `check_optimization_invariants()`) before the files are written, then reads both written files back and checks that they agree.
- The backtest (`optimization/backtest.py`) replays past rebalance dates. It passes each `PortfolioProblem` only the closes up to that date, and uses the same optimizer and constraints as the live run, so it evaluates the method the dashboard shows. Its training window, rebalance interval, and scored days come from the configuration file. It needs complete prices: a missing close or NYSE session stops the run. Its metadata also records the live optimization as `final_optimization`, without solving it again, and `verify_final_optimization()` checks that record against both optimization files. The last rebalance is one holding period older than the live portfolio, so it differs from it.
- Nothing is uploaded unless every step succeeds, including those checks.

## Files read and written

```text
 legend: ◄── written by    [n] upload order    <as-of> = last trading day in the data

 READS
   config/modeling_and_optimization.yml     load_config()
   input/annual/Tickers YYYY-MM-DD.csv      price cache, reused for the run date
   input/Risk Free Rate History YYYY-MM-DD.csv   T-bill cache, reused for the run date
        │
        ▼
 WRITES (local)                                                         UPLOAD
   output/time_series/<as-of>/
     ├── all_forecasts.csv                    ◄── load_or_train_forecasts()      [1]
     │     (reused if present, unless --retrain-arima)
     └── all_forecasts_metadata.yml           ◄── write_forecast_metadata()      [2]
   output/portfolio_optimization/<as-of>/
     ├── portfolio_optimization_plot_data.h5  ◄── write_plot_data_h5()           [3]
     ├── portfolio_backtest_daily.csv         ◄── write_backtest_daily()         [4]
     ├── portfolio_backtest_metadata.yml      ◄── write_backtest_metadata()      [5]
     └── optimization_metadata.yml            ◄── write_optimization_metadata()  [6]
                                                  (uploaded last: marks the date complete)
        │
        ▼
 UPLOAD   S3Uploader.upload_file(), to the Space named by FSF_FRONT_END_BUCKET_NAME
   output/time_series/<as-of>/*
       ─►  <FSF_FRONT_END_TIME_SERIES_FOLDER>/<as-of>/*
   output/portfolio_optimization/<as-of>/*
       ─►  <FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER>/<as-of>/*
```

The local folders mirror the bucket keys (`storage/layout.py` builds both), so each local date folder maps to the same date folder in the Space. `INTERFACE_CONTRACT.md` describes what each uploaded file contains.

## Project structure

The repository separates code that you **run** from code that you **import**:

```
fuzzy-system-research/
├── pyproject.toml                # Dependencies, Python version, build config
├── uv.lock                       # Exact resolved dependency versions
├── config/
│   └── modeling_and_optimization.yml  # Model settings (checked in)
├── docs/                         # Documentation, including the interface contract
├── scripts/                      # RUN these: entry points that wire the library together
│   ├── run_optimization_and_time_series.py
│   └── ticker_download_to_csv.py
├── src/fuzzy_system_research/    # IMPORT these: reusable library code, no entry points
│   ├── config.py                 # Loads and validates the model settings file
│   ├── pipeline.py               # One full run: prices → ARIMA → metrics → optimization → backtest → uploads
│   ├── data/                     # Market data download, caching, price reshaping, and date logic
│   ├── optimization/             # Portfolio optimization math and its backtest
│   ├── forecasting/              # ARIMA models and the walk-forward training workflow
│   └── storage/                  # Output/bucket key layout and S3/Spaces uploads
├── tests/                        # Tests (use mocks and temporary directories only)
├── input/                        # Cached market data (not committed)
└── output/                       # Results of runs (not committed)
```

Scripts contain argument handling and orchestration: they decide *what* to run. The package contains the computation and I/O helpers: it knows *how*. Scripts import from the package (for example, `from fuzzy_system_research.pipeline import run_pipeline`), and the package never imports from `scripts/`. This keeps the quantitative code importable and testable without triggering a download or an upload.

### Data folders

The backend uses the following folders. You don't need to create them: the scripts create each one (and any missing parent folders) the first time they write to it. Their content is not committed to GitHub.

1. `input/`: Holds the risk free rate history (one file per run date), from which every optimization takes its rate. Older `Risk Free Rate YYYY-MM-DD.json` files are no longer used and can be deleted.

2. `input/annual/`: Holds the cached price downloads (one file per run date), long enough for the optimization, the ARIMA forecasts, and the backtest. ARIMA uses the last training window plus scored backtest days, the portfolio backtest its own training window plus the same scored days, and the optimization the last of those training windows (see "Model settings" in [INSTALLATION_AND_EXECUTION.md](INSTALLATION_AND_EXECUTION.md#model-settings) for where those lengths are set). A cached file that is too short, for example one written before the ARIMA window grew, is downloaded again automatically.

3. `output/`: Holds the results of runs, in one folder per data as-of date (the last NYSE trading day in the run's data). Subfolders are created automatically. The layout mirrors the frontend buckets, where each file's key is its path below `output/<kind>/`:

   ```
   output/portfolio_optimization/YYYY-MM-DD/optimization_metadata.yml
   output/portfolio_optimization/YYYY-MM-DD/portfolio_optimization_plot_data.h5
   output/portfolio_optimization/YYYY-MM-DD/portfolio_backtest_daily.csv
   output/portfolio_optimization/YYYY-MM-DD/portfolio_backtest_metadata.yml
   output/time_series/YYYY-MM-DD/all_forecasts.csv
   output/time_series/YYYY-MM-DD/all_forecasts_metadata.yml
   ```

   Re-running for the same date overwrites that date's files, except that ARIMA training is skipped when that date's `all_forecasts.csv` already exists (unless `--retrain-arima` is given). See `INTERFACE_CONTRACT.md` for the bucket layout the frontend reads.

An older `input/monthly/` folder, if present, is no longer used and can be deleted.

### Scripts

Run these directly with `uv run python scripts/<filename>` from the repository root.

| Filename                                | Purpose                                                                                                                                                                                                                   | Uploads to front end? |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `scripts/run_optimization_and_time_series.py` | The whole backend in one run: obtain adjusted close prices, fit walk-forward ARIMA forecasts, compute max Sharpe ratio and min variance portfolios, the efficient frontier, and a Monte Carlo simulation, backtest the max Sharpe ratio portfolio against equal weights, then upload. See "Running the backend" in [INSTALLATION_AND_EXECUTION.md](INSTALLATION_AND_EXECUTION.md#running-the-backend) for flags. | Yes |
| `scripts/ticker_download_to_csv.py`     | Download a supported symbol from Massive.com and put the results into a `.csv` file. Usage: `uv run python scripts/ticker_download_to_csv.py <ticker> <date_from> <date_to> <filename>`.                                    | No                    |

### Library modules

These modules are imported by the scripts and are not meant to be executed directly.

| Module (under `src/fuzzy_system_research/`) | Purpose                                                                                                                                                 | Uploads to front end?                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `config.py`                                 | Reads and strictly validates `config/modeling_and_optimization.yml` into `TimeSeriesConfig`, `PortfolioOptimizationConfig`, and `PortfolioBacktestConfig`. | No |
| `data/ticker_download_manager.py`           | Downloads new data from Massive.com or retrieves old data from the cache.                                                                               | No                                   |
| `data/date_manager.py`                      | Date and business day calculations on the NYSE trading calendar (via `exchange_calendars`).                                                             | No                                   |
| `data/prices.py`                            | Pivots downloaded prices to a wide table and selects the most recent trading sessions.                                                                 | No                                   |
| `optimization/problem.py`                   | `PortfolioProblem`, the one definition of the optimization: mean and covariance estimates, risk-free rate, the constraint set (`bounds()`, `constraints()`: long-only, sum to 1, configurable cap), `portfolio_stats()` (every uploaded return, volatility, and Sharpe ratio), a seeded sampler of feasible random portfolios, and the annualization helpers. | No |
| `optimization/portfolio.py`                 | Max Sharpe ratio and minimum variance optimizations under the shared constraints (raising `OptimizationError` if one does not converge), the maximum return portfolio (exact, in closed form), the efficient frontier (unreachable targets become `NaN`), and retrieval of the risk-free rate history, with the rate available before a given date. | No |
| `optimization/workflow.py`                  | One full optimization run from prices to annualized results, all from one `PortfolioProblem`: the capital allocation line, the invariant checks (`InvariantError`), the random-portfolio summary, the `.h5` and `.yml` writers, and a read-back check that the two files agree. The `.yml` holds the weights of both the max Sharpe ratio and the minimum variance portfolios, with a comment marking `optimum_portfolio` as the max Sharpe ratio portfolio. | No |
| `optimization/backtest.py`                  | Walk-forward backtest of the max Sharpe ratio portfolio against equal weights: the rebalance schedule, drifting holdings, wealth and metrics (cumulative return, CAGR, maximum drawdown, annualized volatility), a documented fallback for failed optimizations, the invariant checks (`InvariantError`), the `.csv` and `.yml` writers, and read-back checks that the two files agree and that `final_optimization` matches the optimization's files. | No |
| `forecasting/arima_models.py`               | Trains ARIMA models on price data.                                                                                                                      | No                                   |
| `forecasting/ticker_predict.py`             | `ArimaWalkForward`: walk-forward ARIMA training over rolling windows; `load_or_train_forecasts` caches the result.                                      | No                                   |
| `forecasting/metrics.py`                    | Accuracy of the ARIMA and naive forecasts over the configured number of most recent trading days (RMSE, MAE, MAPE, relative MAE, skill, directional accuracy) for `all_forecasts_metadata.yml`. | No |
| `pipeline.py`                               | Orchestrates one backend run and its uploads (`all_forecasts.csv`, `all_forecasts_metadata.yml`, `.h5`, the two backtest files, then `optimization_metadata.yml` last). | Yes                                  |
| `storage/layout.py`                         | Builds the dated `output/` paths and matching bucket keys (`<folder>/YYYY-MM-DD/<file>`).                                                               | No                                   |
| `storage/s3_uploader.py`                    | Uploads files to the one DigitalOcean Space (`FSF_FRONT_END_BUCKET_NAME`) by full object key; rejects non-region-only endpoints.                       | Yes (other modules use it to upload) |

## More Engineering Reading

[INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md) describes the contract between the backend and frontend.
