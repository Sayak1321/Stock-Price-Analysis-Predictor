import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/stock_sentiment.db"))


def migrate():
    print(f"Running database schema migration 004 on {DB_PATH}...")
    if not os.path.exists(DB_PATH):
        print("Database file does not exist yet. Migration skipped.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION;")

        # Drop and recreate feature_store table with new quantitative feature columns
        cursor.execute("DROP TABLE IF EXISTS feature_store;")
        cursor.execute(
            """
            CREATE TABLE feature_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                headline_length INTEGER NOT NULL,
                hour_of_day INTEGER NOT NULL,
                news_count_last_1h INTEGER NOT NULL DEFAULT 0,
                news_count_last_24h INTEGER NOT NULL,
                avg_sentiment_last_1h REAL NOT NULL DEFAULT 0.0,
                avg_sentiment_last_24h REAL NOT NULL,
                volume_ratio REAL,
                market_return REAL,
                sector_return REAL,
                volatility REAL,
                prev_day_return REAL,
                return_5d REAL,
                return_10d REAL,
                rsi REAL,
                FOREIGN KEY(article_id) REFERENCES news_articles(id),
                UNIQUE(article_id, model_name)
            );
            """
        )

        conn.commit()
        print("Database migration 004 completed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error during database migration 004: {e}")
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
