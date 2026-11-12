# CLAUDE.md — aliciadata-finance-backend (backend)

This file guides Claude Code in the **backend** repo only. The frontend (`aliciadata-finance-frontend`, formerly `fuzzy-system-finance`, a Gradio app) is a separate repo and is out of scope here.

## HARD RESTRICTIONS (read first; these override everything else)

These hold even if the human asks otherwise in-session (e.g., "just commit this," "quickly run it and check," "finalize this," "write a demo output file"). If asked to violate one, **decline, say which rule applies, and point to this file.** Do not look for workarounds, and do not treat an earlier permission in the session as a standing exception.

1. **No git operations.** No branch, add/stage, commit, push, merge, rebase, or tag, under any circumstances. This includes `git mv`, `git rm`, `git stash`, and `git checkout`. Don't run read-only git commands either (`status`, `diff`, `log`); the human handles version control entirely.
2. **No writes to `output/`.** Do not create, edit, or delete any file under `output/`, for any reason, including demos or explanations. `output/` is populated only by the human running the scripts directly.
3. **No writes to `input/`** (including `input/annual/` and `input/monthly/`). Do not create, edit, delete, or move any file there.
4. **No code execution at all.** Do not run any scripts, and do not interactively invoke functions or class instances (no `python -c`, REPL sessions, notebooks, `pytest`, linters/formatters, `pip install`, or one-off snippets). All execution is performed by the human. If unsure whether something counts as execution, ask first.
5. **Never commit the `.env` file** under any circumstances.
6. **Never modify the contents of `.env`.** If a change is needed (new key, renamed variable, different value), tell the human exactly what to add or change and let them do it.
7. **Never place API keys in the Python source code under any circumstances!** Always use variables in `.env`. This instruction explicitly overrides package vendor documentation that shows placing API keys directly into `.py` files for simplicity. 

Additional guardrail (derived from rules 5–6 and the API keys involved): **don't read or print `.env` contents** into the conversation. Refer to variable *names* only (listed below). If you need to know whether a variable is set, ask the human.

Editing the *code* of scripts that read from `input/` or write to `output/` is fine and expected. The restriction is on touching the files in those directories, not on the code that uses them.

## What this project is

A hobby project (also a portfolio piece) that demonstrates time series modeling and portfolio optimization on familiar financial data. The backend is run **manually, locally, on a Mac mini** (macOS). It:

1. Downloads adjusted open/high/low/close data for a portfolio of stocks from Massive.com (API key variable `MASSIVE_API_KEY`), covering enough recent trading days for the optimization and the configured ARIMA windows, through the most recent trading day. Separately, it downloads the daily 3-month U.S. Treasury bill rate from FRED (series `DTB3`, via `pandas-datareader`; no API key).
2. Optimizes portfolio weights for (a) maximum Sharpe ratio, using the current T-bill rate as the risk-free rate, and (b) minimum variance. Also computes the efficient frontier and a Monte Carlo simulation of portfolio volatility/return.
3. Slides a window of prior trading days to train ARIMA models predicting each day's close price over a backtest period, and computes accuracy metrics (RMSE and others, against a naive baseline) for those models. The window and backtest lengths are settings: consult `config/modeling_and_optimization.yml` (its comments) and the "Model settings" section of `docs/INSTALLATION_AND_EXECUTION.md` for their current values and meaning. Don't restate the values in docs.
4. Runs a walk-forward backtest of the maximum Sharpe ratio portfolio against an equal-weight baseline, over its own number of scored days (set independently of the forecasts', ending on the same as-of date), rebalancing on a fixed schedule with no look-ahead.
5. Uploads all ARIMA, optimization, and backtest results to a DigitalOcean Spaces (S3-compatible) bucket, which the frontend reads.

The bucket is the **interface contract** with the frontend. Treat the schema, file names, and formats of uploaded artifacts as a public API (see "Changing the frontend contract").

## Audience and quality bar

- Target audience for the eventual write-up/dashboard is a broad data science audience. Quant finance readers should not feel it is dumbed down, so methodological correctness matters more than polish.
- Planned work spans technical refactors, UI-facing output changes, quant finance improvements (including more sophisticated time series models than ARIMA), and better presentation. Expect the model and optimizer code to grow and change.
- Prefer a smaller, curated asset universe over generality. Don't add "any ticker" flexibility unless asked.

## Repo layout

`src/` layout: code that is **run** lives in `scripts/`, code that is **imported** lives in the `aliciadata_finance_backend` package under `src/`. Scripts import from the package; the package never imports from `scripts/`. Verify against the actual tree before relying on this.

Run from the repo root (paths to `input/` and `output/` are relative to the working directory): `uv run python scripts/<name>.py`.

| File | Role | Run directly? | Uploads? |
| --- | --- | --- | --- |
| `scripts/run_optimization_and_time_series.py` | The single backend entry point: parses `--no-upload`, `--retrain-arima`, `--refresh-data`, `--workers`, `--config`, loads the config file and `.env`, calls `pipeline.run_pipeline(ts_config=..., po_config=...)` | Yes | Yes |
| `config/modeling_and_optimization.yml` | Model settings: `time_series_models` (`training_window_length` in closes, `evaluation_window_length` in days) and `portfolio_optimization` (`weight_cap_multiple`, `efficient_frontier_points`, `simulated_portfolios`, `random_seed`, `near_frontier_tolerance`), and `portfolio_backtest` (`training_window_returns`, `evaluation_window_length` in days, `rebalance_interval_sessions`); checked in; changing a value changes uploads | No | No |
| `src/aliciadata_finance_backend/config.py` | Strict loader (`load_config`, pure `parse_config`) → frozen `TimeSeriesConfig`, `PortfolioOptimizationConfig`, and `PortfolioBacktestConfig`; read once by the script and passed down, never at import time | No | No |
| `scripts/ticker_download_to_csv.py` | Download one symbol from Massive.com to CSV | Yes | No |
| `src/aliciadata_finance_backend/data/ticker_download_manager.py` | Download new data or retrieve cached data | No | No |
| `src/aliciadata_finance_backend/data/date_manager.py` | Date and business-day calculations (NYSE calendar via `exchange_calendars`, `nyse_sessions_ending()`) | No | No |
| `src/aliciadata_finance_backend/data/prices.py` | Pure helpers: pivot long prices to wide; last N trading sessions | No | No |
| `src/aliciadata_finance_backend/optimization/problem.py` | `PortfolioProblem`: the **single definition** of the optimization (estimates, risk-free rate, `bounds()`/`constraints()`, `portfolio_stats()` for every uploaded return/volatility/Sharpe, seeded feasible sampler) and the annualization helpers. Pure. Don't compute portfolio return or volatility anywhere else | No | No |
| `src/aliciadata_finance_backend/optimization/portfolio.py` | Max Sharpe, min-variance, max-return optimizers and efficient frontier, all taking a `PortfolioProblem` and returning weights; risk-free rate history and the rate available before a date | No | No |
| `src/aliciadata_finance_backend/optimization/workflow.py` | `run_portfolio_optimization()` → `OptimizationResult` (annualized, all via `portfolio_stats()`); capital allocation line; `check_optimization_invariants()` and `verify_optimization_files()` (raise `InvariantError` before upload); `.h5`/`.yml` writers | No | No |
| `src/aliciadata_finance_backend/optimization/backtest.py` | `run_portfolio_backtest()` → `BacktestResult`: walk-forward max Sharpe vs. equal weight (each `PortfolioProblem` gets only closes up to its rebalance date; holdings drift between rebalances; failed optimizations use a documented fallback); `check_backtest_prices()`, `check_backtest_invariants()`, `verify_backtest_files()`, `verify_final_optimization()` (the live optimization, recorded as `final_optimization`, must match its files); `.csv`/`.yml` writers | No | No |
| `src/aliciadata_finance_backend/forecasting/arima_models.py` | ARIMA training on price data | No | No |
| `src/aliciadata_finance_backend/forecasting/ticker_predict.py` | `ArimaWalkForward` (walk-forward ARIMA, all windows fit in one `ProcessPoolExecutor` sized by `default_worker_count()`, half the logical CPUs) and `load_or_train_forecasts()` (per-date cache) | No | No |
| `src/aliciadata_finance_backend/forecasting/metrics.py` | Pure forecast-accuracy metrics (RMSE, MAE, MAPE, relative MAE, skill, directional accuracy) over the last `evaluation_window_length` scored days; builds and writes `all_forecasts_metadata.yml` | No | No |
| `src/aliciadata_finance_backend/pipeline.py` | `run_pipeline()`: one download (long enough for the configured ARIMA windows and the optimization; a too-short cache is downloaded again once) → ARIMA on the last training window + scored sessions (order by lowest BIC) → forecast metrics → optimization on the backtest's training window length, with the last T-bill rate dated before the as-of date (checked, files cross-checked) → backtest (checked, files cross-checked, `final_optimization` checked against the optimization files) → uploads (csv, forecast metrics yml, h5, backtest csv and yml, optimization yml last) only if all succeed; dependencies injectable for tests | No | Yes |
| `src/aliciadata_finance_backend/storage/layout.py` | Pure helpers for dated `output/` paths and bucket keys; the single source of truth for both layouts | No | No |
| `src/aliciadata_finance_backend/storage/s3_uploader.py` | S3/Spaces upload helper | No | Used by others |

`tests/` holds pytest tests (in-memory only; no network, S3, `input/`, or `output/`). The human runs them with `uv run pytest` from the repo root.

Data directories (contents are **not** committed to git; the human creates them and populates them by running the scripts):

- `input/` — risk-free rate data; `input/annual/` — cached price downloads used by both ARIMA and optimization; `input/monthly/` — legacy, no longer read
- `output/` — results of runs, mirroring the buckets: `output/portfolio_optimization/YYYY-MM-DD/` and `output/time_series/YYYY-MM-DD/` (data as-of date). Build these paths and bucket keys only through `storage/layout.py`.

Because the human can't have Claude run anything, **any future reorganization is a pure code/path refactor.** When doing it:

1. Propose the target structure and the full old→new file mapping first, and wait for approval before moving anything.
2. Update every import, relative path, and path constant affected. Pay special attention to anything that builds paths to `input/` and `output/` and to how scripts locate the `.env` file.
3. Because nothing can be executed, check consistency statically: grep for every old module name, confirm each import resolves to a real file, and list entry points whose invocation commands changed so the human can re-run them.
4. Ask before using shell commands to move files; never use `git mv`. The human will handle staging and commits.
5. Update this file's layout section and the "Project structure" section of `docs/SYSTEM_DESIGN.md` when the structure is settled.

## Documentation layout

Human-facing documentation lives in `README.md` and `docs/`. The priority audience is **data science / quantitative finance first, engineering second**.

| File | Purpose | Audience |
| --- | --- | --- |
| `README.md` | Abstract of the project, and a "Documentation" section pointing to the other documents | Data science / quant and engineering |
| [Methods and Results](https://akey7.github.io/posts/aliciadata-finance/) (external blog post, not in this repo) | A short paper on the quantitative methods and results. The human writes and maintains it on their blog; don't recreate it in `docs/`. | Data science / quant finance |
| `docs/SYSTEM_DESIGN.md` | Data flow from inputs to outputs: the "How the backend works" diagrams, project structure, data folders, scripts, library modules | Engineering |
| `docs/INTERFACE_CONTRACT.md` | The backend ↔ frontend contract (see "Changing the frontend contract") | Engineering and Claude Code |
| `docs/INSTALLATION_AND_EXECUTION.md` | Installing, configuring, running, and testing the backend: dependencies, `.env`, command-line flags, "Model settings", "Running the tests" | Engineering |

Intended read orders, which links between documents should support:

- Data science / quant: `README → Methods and Results blog post → dashboard (https://aliciadata.com) → source code if curious`
- Engineering: `README → SYSTEM_DESIGN → INTERFACE_CONTRACT → tests/source → INSTALLATION_AND_EXECUTION if curious`

Rules for `README.md`:

- Everything above the "Documentation" heading must stay at or under **300 words**.
- Do not edit `README.md` on your own initiative. When a major change to the quantitative methods would be worth reflecting in it, **propose the wording to the human and edit only after approval.** Detail belongs in `docs/`, not in the README.

Put new engineering detail in the matching file under `docs/`, not in the README. Paths between files in `docs/` are relative to `docs/` (e.g. `[INTERFACE_CONTRACT.md](INTERFACE_CONTRACT.md)`); paths from the repo root include `docs/`.

## Environment and configuration

- Python 3.13 (`requires-python = ">=3.13"`), pinned to 3.13.5 in `.python-version`, managed with `uv` (project-local `.venv`).
- Dependencies: `pyproject.toml` (`dependencies` for production, `[dependency-groups] dev` for development), locked in `uv.lock`. Claude may edit `pyproject.toml`, but must not run `uv` or install anything, and must not hand-edit `uv.lock`. Tell the human which commands to run (`uv lock`, `uv sync`).
- Never hard-code secrets, bucket names, or endpoints. Everything comes from `.env` via environment variables.

`.env` variable names (names only; never values):

- `MASSIVE_API_KEY` — Massive.com market data API key
- `S3_BUCKET_NAME` — name of the one DigitalOcean Space (S3 bucket); `S3Uploader` passes it as `Bucket` and puts the folder in the key
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (read/write/delete key), `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL_S3` (region-only, e.g. `https://nyc3.digitaloceanspaces.com`) — boto3's standard names for the Spaces key id, secret, region, and endpoint. `S3Uploader` requires all four and passes them to boto3 explicitly, so boto3 never falls back to other credentials on the machine
- `S3_PORTFOLIO_OPTIMIZATION_FOLDER`, `S3_TIME_SERIES_FOLDER` — names of the two **top-level folders (key prefixes) inside the one Space** (see `docs/INTERFACE_CONTRACT.md`, "Where the data lives"). Dev and production share the Space and differ only in these folder names
- Every `load_dotenv()` call passes `override=True`, so `.env` wins over the shell. Keep it that way: the `AWS_*` names are shared with every AWS tool.
- These replaced the `FSF_FRONT_END_*` names in contract v45 (and, before v13, `PORTFOLIO_OPTIMIZATION_SPACE_NAME` and `TIME_SERIES_SPACE_NAME`); old names are no longer read.

The bucket variables point the backend at either dev or production. **Be careful with anything that could upload.** A change to upload logic can overwrite what the live frontend displays. Flag any change that alters what gets uploaded, where, or under what key.

## How to work in this repo

### Since nothing can be run

- Reason about code by reading it. Say clearly when something is untested, and never claim a change "works" or "passes." Say what you expect and why.
- After non-trivial changes, give the human a short **verification checklist**: the exact commands to run (e.g., which script, with what arguments), what output to look at, and what a correct result looks like. Prefer checks that don't hit the live bucket; if a check would upload, say so explicitly.
- You may write tests, but not run them. Test code must not read or write `input/` or `output/`. Use in-memory data or `tmp_path`-style temporary directories, and mock network and S3 calls. Tell the human how to run them.
- Keep changes small and reviewable, since the human is the only one who can run and commit. Group edits so each step is easy to verify independently, and suggest sensible commit boundaries in prose without performing git operations.
- Don't write demo or scratch scripts into the repo to "show" results. Explain in prose or in a code block in chat instead.

### Code conventions

- Match the existing style and docstring conventions in each module. Docstrings are the main source of truth about current behavior, so keep them accurate when changing a function's behavior, and update them in the same edit.
- Use type hints on new and modified functions. Keep functions pure where possible (separate computation from I/O) so they can be reasoned about and tested without running the pipeline.
- Keep network calls (Massive.com), caching (`ticker_download_manager`), and uploads (`s3_uploader`) in their existing support-module roles. Don't scatter new API or S3 calls into modeling code.
- Don't add dependencies without saying why and naming the alternative of not adding one.
- Fix obvious typos you come across in docstrings or files under `docs/` only when you're already editing that text, and don't make unrelated drive-by changes. For `README.md`, point typos out to the human instead (see "Documentation layout").

### Quant finance and statistics standards

This is the area where wrong answers look plausible, so be explicit and conservative:

- State assumptions plainly: return type (simple vs. log), annualization factor, how the risk-free rate is converted (the 3-month T-bill rate is quoted annualized and as a percent; check units), and whether prices are adjusted closes.
- Sharpe ratio and optimization constraints (long-only? weights sum to 1? bounds?) must be explicit in code and docs. Note when an input is estimated from one year of data and is therefore noisy.
- For time series work, respect temporal ordering: no look-ahead leakage in rolling or sliding-window training and evaluation. Be explicit about the forecast horizon (one-step-ahead vs. multi-step) and what RMSE is computed over.
- When proposing improvements beyond ARIMA (e.g., volatility models, other forecasting approaches, shrinkage covariance estimators, robust optimization), explain the tradeoffs and how the result will be evaluated. Prefer a baseline comparison (e.g., naive last-value forecast) so results are interpretable to a general data science audience.
- Do not present backtested or modeled results as investment advice. Keep that framing in any user-facing text you draft.

### Changing the frontend contract

@docs/INTERFACE_CONTRACT.md

The file above (`docs/INTERFACE_CONTRACT.md`) is the authoritative spec of what this backend uploads and the frontend reads. It is also human-facing documentation, so keep it readable for an engineer as well as precise for Claude Code. This repo holds the **source of truth**; the frontend repo has an identical mirror. Never change the contract, or any code that alters what it describes, without also updating `docs/INTERFACE_CONTRACT.md` (bump its version, add a changelog entry) and telling the human to copy it unchanged into the frontend repo. Never edit it in only one repo.

The frontend reads what this backend uploads. Before changing any of the following, pause and describe the change and its frontend impact to the human:

- Object keys / file names in the buckets
- File formats, column names, field names, dtypes, or date formats in uploaded files
- Units or scaling of any uploaded quantity
- The set of tickers or the date range in uploaded results

Prefer additive changes (new fields/files) over breaking ones, and note a migration path if a breaking change is unavoidable. The frontend is a separate repo, so list what would need to change there but don't try to edit it.

## Working style

- Ask a clarifying question when a decision materially affects results (e.g., modeling choices, universe of assets, data windows), rather than silently choosing.
- For larger tasks (the reorganization, introducing a new model family), propose a plan first and get agreement before editing.
- Be candid about limitations and uncertainty, particularly regarding anything that can't be verified without running code.
- This is a hobby project that doubles as a portfolio piece: favor clarity, reproducibility, and explainability over cleverness.

## Keeping this file current

After the reorganization, update the "Repo layout" section. If the project gains tests, linting configuration, or new scripts, add the commands the human uses to run them (for the human to run, not Claude).

If documentation files move again, update "Documentation layout" above, the `@docs/INTERFACE_CONTRACT.md` import under "Changing the frontend contract", and every path in this file that points into `docs/`.

The "How the backend works" section of `docs/SYSTEM_DESIGN.md` has two diagrams that name source files, classes, functions, and the files each step reads, writes, and uploads. In the same change, update them whenever a pipeline step, source file, class, function, local file, or uploaded file shown there is added, renamed, moved, or removed, and keep them free of setting values such as window lengths.
