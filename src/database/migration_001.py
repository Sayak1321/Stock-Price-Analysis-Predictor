import sqlite3
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/stock_sentiment.db"))


def migrate():
    print(f"Running database schema migration on {DB_PATH}...")
    if not os.path.exists(DB_PATH):
        print("Database file does not exist yet. Migration skipped.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_scores'")
        if not cursor.fetchone():
            print("sentiment_scores table does not exist. No migration needed.")
            return

        # Check if the table has already been migrated (check sqlite_master SQL definition)
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sentiment_scores'")
        table_sql = cursor.fetchone()[0]
        
        # If UNIQUE(article_id, model_name) is already in the SQL string, skip migration
        if "UNIQUE(article_id, model_name)" in table_sql or "UNIQUE (article_id, model_name)" in table_sql:
            print("sentiment_scores table is already migrated.")
            return

        print("Migrating sentiment_scores table constraint...")
        
        # Start transaction
        conn.execute("BEGIN TRANSACTION;")

        # 1. Rename old table
        cursor.execute("ALTER TABLE sentiment_scores RENAME TO sentiment_scores_old;")

        # 2. Create new table with UNIQUE(article_id, model_name)
        cursor.execute(
            """
            CREATE TABLE sentiment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                sentiment_label TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL,
                analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(article_id) REFERENCES news_articles(id),
                UNIQUE(article_id, model_name)
            );
            """
        )

        # 3. Copy data from old table to new table
        cursor.execute(
            """
            INSERT INTO sentiment_scores (id, article_id, model_name, sentiment_label, sentiment_score, confidence, analyzed_at)
            SELECT id, article_id, model_name, sentiment_label, sentiment_score, confidence, analyzed_at
            FROM sentiment_scores_old;
            """
        )

        # 4. Drop old table
        cursor.execute("DROP TABLE sentiment_scores_old;")

        conn.commit()
        print("Database migration completed successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error during database migration: {e}")
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
