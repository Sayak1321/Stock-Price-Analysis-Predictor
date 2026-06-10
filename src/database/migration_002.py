import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/stock_sentiment.db"))


def migrate():
    print(f"Running database schema migration 002 on {DB_PATH}...")
    if not os.path.exists(DB_PATH):
        print("Database file does not exist yet. Migration skipped.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Start transaction
        conn.execute("BEGIN TRANSACTION;")

        # Drop old tables if they exist
        cursor.execute("DROP TABLE IF EXISTS feature_store;")
        cursor.execute("DROP TABLE IF EXISTS daily_sentiment_index;")

        # Recreate daily_sentiment_index with model_name support
        cursor.execute(
            """
            CREATE TABLE daily_sentiment_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_id INTEGER NOT NULL,
                date DATE NOT NULL,
                model_name TEXT NOT NULL,
                avg_sentiment REAL NOT NULL,
                news_count INTEGER NOT NULL,
                sentiment_index REAL NOT NULL,
                FOREIGN KEY(stock_id) REFERENCES stocks(id),
                UNIQUE(stock_id, date, model_name)
            );
            """
        )

        # Recreate feature_store with model_name support
        cursor.execute(
            """
            CREATE TABLE feature_store (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                headline_length INTEGER NOT NULL,
                hour_of_day INTEGER NOT NULL,
                news_count_last_24h INTEGER NOT NULL,
                avg_sentiment_last_24h REAL NOT NULL,
                volume_ratio REAL,
                FOREIGN KEY(article_id) REFERENCES news_articles(id),
                UNIQUE(article_id, model_name)
            );
            """
        )

        conn.commit()
        print("Database migration 002 completed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error during database migration 002: {e}")
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
