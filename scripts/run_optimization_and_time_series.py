"""
Run the whole backend in one command: download prices (or use today's
cached download), fit walk-forward ARIMA forecasts (or reuse the cached
forecasts for the data's as-of date), optimize the portfolio, backtest it
against equal weights, and upload all results for the frontend.

Run from the repository root:

    uv run python scripts/run_optimization_and_time_series.py [--no-upload]
        [--retrain-arima] [--refresh-data] [--workers N] [--config PATH]

Model settings are read from config/modeling_and_optimization.yml (or
--config PATH).
"""

import argparse
from dotenv import load_dotenv
from fuzzy_system_research.config import DEFAULT_CONFIG_PATH, load_config
from fuzzy_system_research.pipeline import run_pipeline


def positive_int(value: str) -> int:
    """
    argparse type for an integer of at least 1.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def parse_args() -> argparse.Namespace:
    """
    Parse the command line options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download prices, fit ARIMA forecasts, optimize and backtest the "
            "portfolio, and upload the results under a YYYY-MM-DD/ prefix (the "
            "data as-of date)."
        )
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Write the output/ files but do not upload anything.",
    )
    parser.add_argument(
        "--retrain-arima",
        action="store_true",
        help="Retrain ARIMA models even if forecasts for the as-of date are cached.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download prices again even if today's download is cached.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=None,
        metavar="N",
        help=(
            "Worker processes for ARIMA training. Defaults to half the logical "
            "CPUs; 1 trains everything in this process."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help=(
            "Model settings file, relative to the working directory. Defaults to "
            f"{DEFAULT_CONFIG_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Load the configuration file and .env, then run the pipeline with the
    command line options. The configuration is checked first, so a bad file
    fails before any download.
    """
    args = parse_args()
    config = load_config(args.config)
    print(f"Using settings from {args.config}: {config.time_series_models}")
    print(f"Using settings from {args.config}: {config.portfolio_optimization}")
    print(f"Using settings from {args.config}: {config.portfolio_backtest}")
    load_dotenv()
    result = run_pipeline(
        upload=not args.no_upload,
        retrain_arima=args.retrain_arima,
        refresh_data=args.refresh_data,
        arima_workers=args.workers,
        ts_config=config.time_series_models,
        po_config=config.portfolio_optimization,
        bt_config=config.portfolio_backtest,
    )
    print(f"Done: {result.as_of} (ARIMA {'trained' if result.trained_arima else 'cached'})")
    for key in result.uploaded:
        print(f"  uploaded {key}")


if __name__ == "__main__":
    main()
