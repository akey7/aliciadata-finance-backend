import sys
from aliciadata_finance_backend.data.ticker_download_manager import (
    TickerDownloadManager,
)


def main():
    """
    Using the TickerDownloadManager class, download a ticker to a csv
    file. Useful for testing API connectivity.
    """
    ticker = sys.argv[1]
    date_from = sys.argv[2]
    date_to = sys.argv[3]
    filename = sys.argv[4]
    tdm = TickerDownloadManager()
    df = tdm.download_tickers([ticker], date_from, date_to)
    print(df.head())
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    main()
