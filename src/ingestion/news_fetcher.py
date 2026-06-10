import sys
import os
import yfinance as yf
from datetime import datetime, timezone

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import insert_news_articles


def fetch_and_store_news(ticker, stock_id):
    """
    Fetch recent news headlines from yfinance for a given ticker
    and insert them into the database. Gracefully avoids duplicate urls.
    """
    print(f"Fetching news articles for {ticker}...")
    try:
        ticker_obj = yf.Ticker(ticker)
        news_items = ticker_obj.news

        if not news_items:
            print(f"No news found for {ticker}.")
            return 0

        articles_to_insert = []
        for item in news_items:
            # Handle nested 'content' structure or fallback to flat structure
            content = item.get("content")
            if not isinstance(content, dict):
                content = item

            headline = content.get("title")
            
            # Extract publisher source
            provider = content.get("provider")
            if isinstance(provider, dict):
                source = provider.get("displayName")
            else:
                source = provider or content.get("publisher")

            # Extract URL
            url = None
            canonical_url_obj = content.get("canonicalUrl")
            if isinstance(canonical_url_obj, dict):
                url = canonical_url_obj.get("url")
            
            if not url:
                click_through_obj = content.get("clickThroughUrl")
                if isinstance(click_through_obj, dict):
                    url = click_through_obj.get("url")
            
            if not url:
                url = content.get("link")

            # Extract and parse publication date/time
            publish_time = content.get("pubDate") or content.get("providerPublishTime")
            published_at_str = None

            if publish_time:
                if isinstance(publish_time, (int, float)):
                    published_dt = datetime.fromtimestamp(publish_time, timezone.utc)
                    published_at_str = published_dt.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(publish_time, str):
                    try:
                        # Convert ISO format to standard SQLite DATETIME format
                        dt = datetime.fromisoformat(publish_time.replace("Z", "+00:00"))
                        published_at_str = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        published_at_str = publish_time

            # Validate required fields
            if not headline or not url or not published_at_str:
                continue

            articles_to_insert.append(
                {
                    "stock_id": stock_id,
                    "headline": headline,
                    "source": source,
                    "url": url,
                    "published_at": published_at_str,
                }
            )

        if not articles_to_insert:
            print(f"No valid new articles parsed for {ticker}.")
            return 0

        inserted_count = insert_news_articles(articles_to_insert)
        print(
            f"Successfully inserted {inserted_count} new articles for {ticker} "
            f"(out of {len(articles_to_insert)} total fetched)."
        )
        return inserted_count
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return 0


if __name__ == "__main__":
    # Test script execution
    ticker = "AAPL"
    # Assuming AAPL stock_id = 1 for a manual test run
    fetch_and_store_news(ticker, 1)
