import sys
import os

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import init_db, get_db_connection, get_tracked_stocks
from src.ingestion.stock_fetcher import fetch_and_store_stock_metadata, fetch_and_store_stock_prices, fetch_and_store_reference_data
from src.ingestion.news_fetcher import fetch_and_store_news

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BTC-USD"]


def main():
    print("==========================================")
    print("Starting News Sentiment Analyzer Ingestion")
    print("==========================================\n")

    # 1. Initialize DB
    init_db()

    # Clear old daily price records to prevent mixed interval datasets
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM stock_prices;")
        conn.commit()
        print("Cleared old stock_prices records for clean hourly data.")
    except Exception as e:
        print(f"Warning: Failed to clear stock_prices: {e}")
    finally:
        conn.close()

    # 2. Process each ticker
    for ticker in DEFAULT_TICKERS:
        print(f"\n--- Processing {ticker} ---")
        try:
            # Step A: metadata & get stock_id
            stock_id = fetch_and_store_stock_metadata(ticker)

            # Step B: prices (period="2mo" for hourly data)
            fetch_and_store_stock_prices(ticker, stock_id, period="2mo")

            # Step C: news articles
            fetch_and_store_news(ticker, stock_id)

        except Exception as e:
            print(f"Failed to process ticker {ticker}: {e}")

    # Step D: Sync S&P 500 and sector ETFs
    try:
        fetch_and_store_reference_data(period="2mo")
    except Exception as e:
        print(f"Failed to sync market reference data: {e}")

    # 3. Print database summary
    print("\n==========================================")
    print("Pipeline Execution Complete. Summary:")
    print("==========================================")

    conn = get_db_connection()
    try:
        stocks_count = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
        prices_count = conn.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
        news_count = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]

        print(f"Tracked Stocks: {stocks_count}")
        print(f"Price Records:  {prices_count}")
        print(f"News Articles:  {news_count}")

        # Show a few sample news articles
        print("\nLatest 5 news articles inserted:")
        rows = conn.execute(
            """
            SELECT s.ticker, n.headline, n.published_at
            FROM news_articles n
            JOIN stocks s ON n.stock_id = s.id
            ORDER BY n.published_at DESC
            LIMIT 5
            """
        ).fetchall()

        for idx, row in enumerate(rows, 1):
            print(f"{idx}. [{row['ticker']}] {row['headline']} ({row['published_at']})")

    except Exception as e:
        print(f"Error reading summary statistics: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
