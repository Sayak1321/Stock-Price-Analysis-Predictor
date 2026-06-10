import os
import sqlite3
from datetime import datetime

# Absolute path to the database file relative to db.py
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/stock_sentiment.db"))


def get_db_connection():
    """Establish a connection to the SQLite database and enable foreign keys."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initialize the database by running schema.sql if tables do not exist."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    with open(schema_path, "r") as f:
        schema_sql = f.read()

    conn = get_db_connection()
    try:
        conn.executescript(schema_sql)
        conn.commit()
        print("Database initialized successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error initializing database: {e}")
        raise e
    finally:
        conn.close()


def insert_stock(ticker, company_name=None, sector=None):
    """Insert a stock ticker. Return the stock's database ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO stocks (ticker, company_name, sector)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name = COALESCE(excluded.company_name, company_name),
                sector = COALESCE(excluded.sector, sector)
            """,
            (ticker.upper(), company_name, sector),
        )
        conn.commit()
        # Retrieve the ID
        cursor.execute("SELECT id FROM stocks WHERE ticker = ?", (ticker.upper(),))
        stock_id = cursor.fetchone()[0]
        return stock_id
    except Exception as e:
        conn.rollback()
        print(f"Error inserting stock {ticker}: {e}")
        raise e
    finally:
        conn.close()


def get_tracked_stocks():
    """Return all tracked stocks as a dict of {ticker: id}."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, ticker FROM stocks")
        rows = cursor.fetchall()
        return {row["ticker"]: row["id"] for row in rows}
    finally:
        conn.close()


def insert_news_articles(articles):
    """
    Insert a list of news articles.
    Each article should be a dict: {
        'stock_id': int,
        'headline': str,
        'source': str,
        'url': str,
        'published_at': str (YYYY-MM-DD HH:MM:SS or datetime)
    }
    Gracefully ignores duplicate URLs.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    inserted_count = 0
    try:
        for article in articles:
            try:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO news_articles (stock_id, headline, source, url, published_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        article["stock_id"],
                        article["headline"],
                        article.get("source"),
                        article["url"],
                        article["published_at"],
                    ),
                )
                if cursor.rowcount > 0:
                    inserted_count += 1
            except Exception as item_error:
                print(f"Skipping article insertion due to error: {item_error}")
        conn.commit()
        return inserted_count
    except Exception as e:
        conn.rollback()
        print(f"Error batch inserting news articles: {e}")
        raise e
    finally:
        conn.close()


def insert_stock_prices(prices):
    """
    Insert a list of stock price rows.
    Each price row should be a tuple or list:
    (stock_id, timestamp, open, high, low, close, volume)
    Replaces older entries on duplicate (stock_id, timestamp).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    inserted_count = 0
    try:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO stock_prices (stock_id, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            prices,
        )
        inserted_count = cursor.rowcount
        conn.commit()
        return inserted_count
    except Exception as e:
        conn.rollback()
        print(f"Error batch inserting stock prices: {e}")
        raise e
    finally:
        conn.close()


def insert_sentiment_scores(scores):
    """
    Insert a list of sentiment scores.
    Each score should be a dict: {
        'article_id': int,
        'model_name': str,
        'sentiment_label': str,
        'sentiment_score': float,
        'confidence': float
    }
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    inserted_count = 0
    try:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO sentiment_scores (article_id, model_name, sentiment_label, sentiment_score, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    s["article_id"],
                    s["model_name"],
                    s["sentiment_label"],
                    s["sentiment_score"],
                    s.get("confidence"),
                )
                for s in scores
            ],
        )
        inserted_count = cursor.rowcount
        conn.commit()
        return inserted_count
    except Exception as e:
        conn.rollback()
        print(f"Error batch inserting sentiment scores: {e}")
        raise e
    finally:
        conn.close()


def get_unscored_headlines():
    """Fetch all news articles that do not have sentiment scores."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, headline
            FROM news_articles
            WHERE id NOT IN (SELECT article_id FROM sentiment_scores)
            """
        )
        return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching unscored headlines: {e}")
        return []
    finally:
        conn.close()


def get_stock_price_history(ticker):
    """Return a Pandas DataFrame of stock price history for a given ticker."""
    import pandas as pd
    conn = get_db_connection()
    try:
        query = """
            SELECT p.timestamp, p.open, p.high, p.low, p.close, p.volume
            FROM stock_prices p
            JOIN stocks s ON p.stock_id = s.id
            WHERE s.ticker = ?
            ORDER BY p.timestamp ASC
        """
        return pd.read_sql_query(query, conn, params=(ticker.upper(),))
    except Exception as e:
        print(f"Error getting stock price history for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_stock_news_sentiment(ticker):
    """Return a Pandas DataFrame of news articles and their sentiment scores for a given ticker."""
    import pandas as pd
    conn = get_db_connection()
    try:
        query = """
            SELECT n.id as article_id, n.headline, n.source, n.url, n.published_at,
                   s.model_name, s.sentiment_label, s.sentiment_score, s.confidence
            FROM news_articles n
            JOIN stocks st ON n.stock_id = st.id
            LEFT JOIN sentiment_scores s ON n.id = s.article_id
            WHERE st.ticker = ?
            ORDER BY n.published_at DESC
        """
        return pd.read_sql_query(query, conn, params=(ticker.upper(),))
    except Exception as e:
        print(f"Error getting news sentiment for {ticker}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()
