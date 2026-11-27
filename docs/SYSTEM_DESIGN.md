# System Design

This document describes the system design of the backend.

Also of interest to an engineering audience is the [INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md) file, which describes the contract of data transmission from the backend to the frontend.

## Deployment

```text
 [backend VPS, Hetzner]                       [frontend VPS, Hetzner]
   cron, after each trading day's close         Caddy (reverse proxy)
     └─ backend container    ── S3 uploads ──►  Garage (S3-compatible storage)
        (this repository's                        ▲ reads
         Dockerfile)                            frontend Gradio app

 Every container on both VPSs is built and installed with Ansible.
```

- The results are stored in a self-hosted, S3-compatible object store, [Garage](https://garagehq.deuxfleurs.fr/), on a virtual private server (VPS) hosted by Hetzner. Three services run on that VPS, each in its own Docker container: Garage, the frontend Gradio app, and a Caddy reverse proxy. The frontend's deployment is documented in the frontend repository.
- **This backend runs on a separate Hetzner VPS**, in a Docker container built from the `Dockerfile` at the root of this repository (see "Running in Docker" in [INSTALLATION_AND_EXECUTION.md](INSTALLATION_AND_EXECUTION.md#running-in-docker)). Like the frontend's containers, it is built and installed onto its VPS with Ansible.
- **cron runs it unattended**, once per trading day after the market closes, for the most recent trading day: the container's default command, the normal run, which uploads that day's results. Past dates are filled in only by running `--past-days` by hand (see "Backfilling past dates" in [INSTALLATION_AND_EXECUTION.md](INSTALLATION_AND_EXECUTION.md#backfilling-past-dates)).
- The backend reaches Garage over the network, through Garage's S3 API, as for any S3 service, and needs network access to Massive.com and FRED for its data. For development, it can also be run directly on a local machine with `uv`.
- Keeping the backend on its own VPS separates the daily computation, which can use several processes and a few hundred MiB per process (see "Peak memory" in [INSTALLATION_AND_EXECUTION.md](INSTALLATION_AND_EXECUTION.md#peak-memory)), from the services the dashboard's users reach. For a proof-of-concept project, two small VPSs keep deployment simple and hosting costs low and predictable.
- **Because the run is unattended, a change reaches the production bucket on the first scheduled run after the image with that change is installed**, without anyone watching it. Check changes against the dev folders first.

**The image.** The `Dockerfile` builds on the official Python image, pinned to the patch version in `.python-version`, and adds `uv` at a pinned version. It installs the production dependencies exactly as locked in `uv.lock` (`uv sync --locked --no-dev`), first on their own so that layer is reused until they change, then the project itself. The container runs as an unprivileged user, and its default command is the normal daily run, `uv run --locked --no-dev python ./scripts/run_optimization_and_time_series.py`. Before running, `uv run` checks the environment against `uv.lock` with the same options, so the container always uses the configured production dependencies, installing anything missing from the network rather than running without it. `.dockerignore` keeps `.env`, `input/`, `output/`, and development files out of the image, so settings and secrets are passed to the container at run time.

## How the backend works

One run of `scripts/run_optimization_and_time_series.py` downloads prices, then follows two paths, the time series forecasts and the portfolio optimization with its backtest, and finally uploads their results. The first diagram shows which module owns each part of a run; the second shows which files each step reads and writes, and the functions that write them.

## From data sources to upload

```text
 legend: [external]   module.py   ─► data flow
         module paths are relative to src/aliciadata_finance_backend/

 scripts/run_optimization_and_time_series.py      entry point
   ├─ config.py ◄── config/modeling_and_optimization.yml
   ├─ backfill.py                                 --past-days: one run per date, oldest first
   ├─ memory_usage.py                             peak memory, printed after the run
   └─ pipeline.py                                 runs the steps below
        │
 [Massive.com API] ──► 1. PRICES AND EXOGENOUS VARIABLES
                         data/ticker_download_manager.py   download; tickers cached in input/annual/,
                                                           exogenous variables in input/exogenous/
                         data/date_manager.py              NYSE trading calendar
                         data/prices.py                    wide price table, cut off at the as-of date, recent sessions,
                                                           exogenous gaps
                         pipeline.py, load_exogenous()     exogenous values checked against the forecasts' sessions
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
    storage/s3_uploader.py   uploads to the S3 bucket
                         │
                         ▼
             [Garage S3 bucket on the VPS]   (files in the next diagram)
```

- The two paths run one after the other in a single process. Within the time series path, the ARIMA fits for every ticker and day run at the same time in a pool of worker processes (`--workers`).
- The ARIMA window and the number of scored forecast days, and the optimization's weight cap, plot array lengths, and random seed come from the configuration file (see "Model settings" in [INSTALLATION_AND_EXECUTION.md](INSTALLATION_AND_EXECUTION.md#model-settings)). The optimization uses the backtest's training window length.
- The optimization has one definition, `PortfolioProblem` in `optimization/problem.py`. Its constraint set is used by every optimizer and by the random portfolios, and its `portfolio_stats()` computes every return, volatility, and Sharpe ratio that is uploaded, so the plot cannot drift from the optimization. `workflow.py` checks the invariants (see its `check_optimization_invariants()`) before the files are written, then reads both written files back and checks that they agree.
- The backtest (`optimization/backtest.py`) replays past rebalance dates. It passes each `PortfolioProblem` only the closes up to that date, and uses the same optimizer and constraints as the live run, so it evaluates the method the dashboard shows. Its training window, rebalance interval, and scored days come from the configuration file; its scored days are set independently of the forecasts', so the two evaluations can cover different periods ending on the same as-of date. It needs complete prices: a missing close or NYSE session stops the run. Its metadata also records the live optimization as `final_optimization`, without solving it again, and `verify_final_optimization()` checks that record against both optimization files. The last rebalance is one holding period older than the live portfolio, so it differs from it.
- Nothing is uploaded unless every step succeeds, including those checks.

## Files read and written

```text
 legend: ◄── written by    [n] upload order    <as-of> = last trading day in the data

 READS
   config/modeling_and_optimization.yml     load_config()
   input/annual/Tickers YYYY-MM-DD.csv      price cache, reused for the run date
   input/exogenous/Exogenous YYYY-MM-DD.csv   exogenous variables' cache (e.g. I:SPX), reused for the run date
   input/Risk Free Rate History YYYY-MM-DD.csv   T-bill cache, reused for the run date
        │
        ▼
 WRITES (local)                                                         UPLOAD
   output/time_series/<as-of>/
     ├── all_forecasts.csv                    ◄── load_or_train_forecasts()      [1]
     │     (reused if it matches the data and settings, unless --retrain-arima)
     └── all_forecasts_metadata.yml           ◄── write_forecast_metadata()      [2]
   output/portfolio_optimization/<as-of>/
     ├── portfolio_optimization_plot_data.h5  ◄── write_plot_data_h5()           [3]
     ├── portfolio_backtest_daily.csv         ◄── write_backtest_daily()         [4]
     ├── portfolio_backtest_metadata.yml      ◄── write_backtest_metadata()      [5]
     └── optimization_metadata.yml            ◄── write_optimization_metadata()  [6]
                                                  (uploaded last: marks the date complete)
        │
        ▼
 UPLOAD   S3Uploader.upload_file(), to the bucket named by S3_BUCKET_NAME
   output/time_series/<as-of>/*
       ─►  <S3_TIME_SERIES_FOLDER>/<as-of>/*
   output/portfolio_optimization/<as-of>/*
       ─►  <S3_PORTFOLIO_OPTIMIZATION_FOLDER>/<as-of>/*
```

The local folders mirror the bucket keys (`storage/layout.py` builds both), so each local date folder maps to the same date folder in the bucket. `INTERFACE_CONTRACT.md` describes what each uploaded file contains.

## Project structure

The repository separates code that you **run** from code that you **import**:

```
aliciadata-finance-backend/
├── Dockerfile                    # Image that runs the backend on the VPS
├── .dockerignore                 # Keeps .env, input/, output/, and dev files out of the image
├── pyproject.toml                # Dependencies, Python version, build config
├── uv.lock                       # Exact resolved dependency versions
├── config/
│   └── modeling_and_optimization.yml  # Model settings (checked in)
├── docs/                         # Documentation, including the interface contract
├── scripts/                      # RUN these: entry points that wire the library together
│   ├── run_optimization_and_time_series.py
│   └── ticker_download_to_csv.py
├── src/aliciadata_finance_backend/ # IMPORT these: reusable library code, no entry points
│   ├── backfill.py               # --past-days: one download, the checks, the per-date loop and log
│   ├── config.py                 # Loads and validates the model settings file
│   ├── memory_usage.py           # Peak memory of a run, printed by the entry script
│   ├── pipeline.py               # One full run: prices → ARIMA → metrics → optimization → backtest → uploads
│   ├── data/                     # Market data download, caching, price reshaping, and date logic
│   ├── optimization/             # Portfolio optimization math and its backtest
│   ├── forecasting/              # ARIMA models and the walk-forward training workflow
│   └── storage/                  # Output/bucket key layout and S3 uploads
├── tests/                        # Tests (use mocks and temporary directories only)
├── input/                        # Cached market data (not committed)
└── output/                       # Results of runs (not committed)
```

Scripts contain argument handling and orchestration: they decide *what* to run. The package contains the computation and I/O helpers: it knows *how*. Scripts import from the package (for example, `from aliciadata_finance_backend.pipeline import run_pipeline`), and the package never imports from `scripts/`. This keeps the quantitative code importable and testable without triggering a download or an upload.

### Data folders

The backend uses the following folders. You don't need to create them: the scripts create each one (and any missing parent folders) the first time they write to it. Their content is not committed to GitHub.

1. `input/`: Holds the risk free rate history (one file per run date), from which every optimization takes its rate. It reaches back to shortly before the backtest's first rebalance, the earliest optimization date; a cached file that does not reach that far (for example, one written before the backtest grew) is downloaded again automatically. Older `Risk Free Rate YYYY-MM-DD.json` files are no longer used and can be deleted.

2. `input/annual/`: Holds the cached price downloads (one file per run date), long enough for the optimization, the ARIMA forecasts, and the backtest. ARIMA uses the last training window plus its scored forecast days, the portfolio backtest its own training window plus its own scored days, and the optimization the last of the backtest's training windows (see "Model settings" in [INSTALLATION_AND_EXECUTION.md](INSTALLATION_AND_EXECUTION.md#model-settings) for where those lengths are set). A cached file that is too short, for example one written before the ARIMA window grew, is downloaded again automatically.

3. `input/exogenous/`: Holds the cached downloads of the exogenous variables (one file per run date, `Exogenous YYYY-MM-DD.csv`): the regressors of the ARIMAX forecasts, such as the S&P 500 index (`I:SPX`), listed under `exogenous_variables` in `config/modeling_and_optimization.yml`. They are downloaded with the tickers, over the same range, but kept apart from the tickers being modeled, in the same long format (the symbol in the `ticker` column, its value in `close`). Like the prices, they are cut off at a run's as-of date. A cached file that lacks a configured variable, or starts after the sessions the forecasts use, is downloaded again automatically; a session missing inside the range is reported, and a variable with no value at all fails the run.

4. `output/`: Holds the results of runs, in one folder per data as-of date (the last NYSE trading day in the run's data). Subfolders are created automatically. The layout mirrors the frontend buckets, where each file's key is its path below `output/<kind>/`:

   ```
   output/portfolio_optimization/YYYY-MM-DD/optimization_metadata.yml
   output/portfolio_optimization/YYYY-MM-DD/portfolio_optimization_plot_data.h5
   output/portfolio_optimization/YYYY-MM-DD/portfolio_backtest_daily.csv
   output/portfolio_optimization/YYYY-MM-DD/portfolio_backtest_metadata.yml
   output/time_series/YYYY-MM-DD/all_forecasts.csv
   output/time_series/YYYY-MM-DD/all_forecasts_metadata.yml
   ```

   Re-running for the same date overwrites that date's files, except that ARIMA training is skipped when that date's `all_forecasts.csv` already exists (unless `--retrain-arima` is given). A run for a past date (`--as-of`, or each date of a `--past-days` backfill) writes that date's folder the same way. See `INTERFACE_CONTRACT.md` for the bucket layout the frontend reads.

   A backfill also writes one table of its dates, which is not uploaded:

   ```
   output/backfill/backfill_YYYYMMDDTHHMMSSZ.csv   (UTC start time; one row per date)
   ```

An older `input/monthly/` folder, if present, is no longer used and can be deleted.

### Scripts

Run these directly with `uv run python scripts/<filename>` from the repository root.

| Filename                                | Purpose                                                                                                                                                                                                                   | Uploads to front end? |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `scripts/run_optimization_and_time_series.py` | The whole backend in one run: obtain adjusted close prices, fit walk-forward ARIMA forecasts, compute max Sharpe ratio and min variance portfolios, the efficient frontier, and a Monte Carlo simulation, backtest the max Sharpe ratio portfolio against equal weights, then upload. See "Running the backend" in [INSTALLATION_AND_EXECUTION.md](INSTALLATION_AND_EXECUTION.md#running-the-backend) for flags. | Yes |
| `scripts/ticker_download_to_csv.py`     | Download a supported symbol from Massive.com and put the results into a `.csv` file. Usage: `uv run python scripts/ticker_download_to_csv.py <ticker> <date_from> <date_to> <filename>`.                                    | No                    |

### Library modules

These modules are imported by the scripts and are not meant to be executed directly.

| Module (under `src/aliciadata_finance_backend/`) | Purpose                                                                                                                                                 | Uploads to front end?                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `backfill.py`                               | One-time backfill (`--past-days N`): one price download long enough for the oldest date, an early check that every session and close the runs need is present, the exogenous variables downloaded once over the same range, the risk-free cache filled once, then one run per date, oldest first, each in its own process (the entry script with `--as-of` and the pinned cache date). Dates already complete (in the bucket's folders when uploading) are skipped unless `--force`; a failed date does not stop the others; each date's status, time, and peak memory go to `output/backfill/`. | No (each run uploads its own date) |
| `config.py`                                 | Reads and strictly validates `config/modeling_and_optimization.yml` into `TimeSeriesConfig`, `PortfolioOptimizationConfig`, `PortfolioBacktestConfig`, and the list of `ExogenousVariable`s (symbol, source, transform, lag). | No |
| `data/ticker_download_manager.py`           | Downloads new data from Massive.com or retrieves old data from the cache: the tickers (`input/annual/`) and, apart from them, the exogenous variables (`input/exogenous/`).                                                                              | No                                   |
| `data/date_manager.py`                      | Date and business day calculations on the NYSE trading calendar (via `exchange_calendars`).                                                             | No                                   |
| `data/prices.py`                            | Pivots downloaded prices to a wide table, cuts it off at an as-of date (so a run for a past date sees no later price), and selects the most recent trading sessions; lists the sessions an exogenous variable has no value on (`exogenous_gaps()`).                                                              | No                                   |
| `optimization/problem.py`                   | `PortfolioProblem`, the one definition of the optimization: mean and covariance estimates, risk-free rate, the constraint set (`bounds()`, `constraints()`: long-only, sum to 1, configurable cap), `portfolio_stats()` (every uploaded return, volatility, and Sharpe ratio), a seeded sampler of feasible random portfolios, and the annualization helpers. | No |
| `optimization/portfolio.py`                 | Max Sharpe ratio and minimum variance optimizations under the shared constraints (raising `OptimizationError` if one does not converge), the maximum return portfolio (exact, in closed form), the efficient frontier (unreachable targets become `NaN`), and retrieval of the risk-free rate history, with the rate available before a given date. | No |
| `optimization/workflow.py`                  | One full optimization run from prices to annualized results, all from one `PortfolioProblem`: the capital allocation line, the invariant checks (`InvariantError`), the random-portfolio summary, the `.h5` and `.yml` writers, and a read-back check that the two files agree. The `.yml` holds the weights of both the max Sharpe ratio and the minimum variance portfolios, with a comment marking `optimum_portfolio` as the max Sharpe ratio portfolio. | No |
| `optimization/backtest.py`                  | Walk-forward backtest of the max Sharpe ratio portfolio against equal weights: the rebalance schedule, drifting holdings, wealth and metrics (cumulative return, CAGR, maximum drawdown, annualized volatility), a documented fallback for failed optimizations, the invariant checks (`InvariantError`), the `.csv` and `.yml` writers, and read-back checks that the two files agree and that `final_optimization` matches the optimization's files. | No |
| `forecasting/arima_models.py`               | Trains ARIMA models on price data.                                                                                                                      | No                                   |
| `forecasting/ticker_predict.py`             | `ArimaWalkForward`: walk-forward ARIMA training over rolling windows; `load_or_train_forecasts` caches the result, and retrains a cached file whose rows do not match the data and settings (`cached_forecasts_match`).                                      | No                                   |
| `forecasting/metrics.py`                    | Accuracy of the ARIMA and naive forecasts over the configured number of most recent trading days (RMSE, MAE, MAPE, relative MAE, skill, directional accuracy) for `all_forecasts_metadata.yml`. | No |
| `memory_usage.py`                           | Peak resident set size of the main process and of the largest ARIMA worker, from the operating system's counters, and an upper bound for all processes together; printed by the entry script after a run. macOS and Linux only: on Windows nothing is measured. | No |
| `pipeline.py`                               | Orchestrates one backend run, including downloading the exogenous variables with the tickers and checking them (`load_exogenous()`), and its uploads (`all_forecasts.csv`, `all_forecasts_metadata.yml`, `.h5`, the two backtest files, then `optimization_metadata.yml` last). | Yes                                  |
| `storage/layout.py`                         | Builds the dated `output/` paths and matching bucket keys (`<folder>/YYYY-MM-DD/<file>`).                                                               | No                                   |
| `storage/s3_uploader.py`                    | Uploads files to the one S3 bucket (`S3_BUCKET_NAME`) by full object key; requires every bucket variable, and rejects an endpoint that names the bucket or has a path. Also checks whether an object exists (`object_exists()`), which a backfill uses to skip dates already complete.                       | Yes (other modules use it to upload) |

## More Engineering Reading

[INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md) describes the contract between the backend and frontend.
