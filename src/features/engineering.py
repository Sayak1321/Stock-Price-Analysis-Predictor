import sys
import os
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import get_db_connection

SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE"
}


def calculate_future_returns():
    """
    Calculate 1-hour, 4-hour, and 1-day future stock returns after a news headline
    is published, and store the targets in the database.
    """
    print("Calculating future stock returns for news articles...")
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Fetch all news articles
        cursor.execute("SELECT id, stock_id, published_at, headline FROM news_articles")
        articles = cursor.fetchall()
        print(f"Found {len(articles)} total articles in database.")

        targets_to_insert = []
        skipped_count = 0

        for article in articles:
            article_id = article["id"]
            stock_id = article["stock_id"]
            published_at = article["published_at"]

            # 1. Find P_current (closest price timestamp <= published_at)
            cursor.execute(
                """
                SELECT close FROM stock_prices
                WHERE stock_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (stock_id, published_at),
            )
            row_current = cursor.fetchone()
            if not row_current:
                # No baseline price found (article might be outside historical price range)
                skipped_count += 1
                continue

            p_current = row_current["close"]

            # 2. Find P_1h (closest price timestamp >= published_at + 1 hour)
            cursor.execute(
                """
                SELECT close FROM stock_prices
                WHERE stock_id = ? AND timestamp >= datetime(?, '+1 hour')
                ORDER BY timestamp ASC LIMIT 1
                """,
                (stock_id, published_at),
            )
            row_1h = cursor.fetchone()
            p_1h = row_1h["close"] if row_1h else None

            # 3. Find P_4h (closest price timestamp >= published_at + 4 hours)
            cursor.execute(
                """
                SELECT close FROM stock_prices
                WHERE stock_id = ? AND timestamp >= datetime(?, '+4 hour')
                ORDER BY timestamp ASC LIMIT 1
                """,
                (stock_id, published_at),
            )
            row_4h = cursor.fetchone()
            p_4h = row_4h["close"] if row_4h else None

            # 4. Find P_1d (closest price timestamp >= published_at + 1 day)
            cursor.execute(
                """
                SELECT close FROM stock_prices
                WHERE stock_id = ? AND timestamp >= datetime(?, '+1 day')
                ORDER BY timestamp ASC LIMIT 1
                """,
                (stock_id, published_at),
            )
            row_1d = cursor.fetchone()
            p_1d = row_1d["close"] if row_1d else None

            # Calculate return percentages
            return_1h = (p_1h - p_current) / p_current if p_1h is not None else None
            return_4h = (p_4h - p_current) / p_current if p_4h is not None else None
            return_1d = (p_1d - p_current) / p_current if p_1d is not None else None

            # Determine direction binary label (1: positive return, 0: negative/flat return)
            direction_1d = 1 if (return_1d is not None and return_1d > 0) else (0 if return_1d is not None else None)

            targets_to_insert.append(
                (article_id, return_1h, return_4h, return_1d, direction_1d)
            )

        # Bulk insert/replace into sentiment_targets
        if targets_to_insert:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO sentiment_targets (article_id, return_1h, return_4h, return_1d, direction_1d)
                VALUES (?, ?, ?, ?, ?)
                """,
                targets_to_insert,
            )
            conn.commit()
            print(f"Stored returns for {len(targets_to_insert)} articles (skipped {skipped_count} due to missing base price).")
        else:
            print("No targets generated.")

    except Exception as e:
        conn.rollback()
        print(f"Error calculating returns: {e}")
        raise e
    finally:
        conn.close()


def evaluate_sentiment_signal():
    """
    Compute correlation statistics between VADER sentiment scores and future returns.
    """
    print("\n==========================================")
    print("Evaluating Sentiment Predictive Signal")
    print("==========================================")

    conn = get_db_connection()
    try:
        # Load merged sentiment and returns data
        query = """
            SELECT n.headline, s.sentiment_label, s.sentiment_score,
                   t.return_1h, t.return_4h, t.return_1d, t.direction_1d
            FROM news_articles n
            JOIN sentiment_scores s ON n.id = s.article_id
            JOIN sentiment_targets t ON n.id = t.article_id
        """
        df = pd.read_sql_query(query, conn)

        if df.empty:
            print("No matching sentiment and target returns records found.")
            return

        print(f"Total evaluated articles: {len(df)}")
        print("\nSentiment Label Distribution:")
        print(df["sentiment_label"].value_counts())

        # 1. Average returns by sentiment label
        print("\nAverage Returns by Sentiment Label:")
        summary_returns = df.groupby("sentiment_label")[["return_1h", "return_4h", "return_1d"]].mean()
        # Convert to percentage for readable output
        print(summary_returns * 100)

        # 2. Correlation coefficient
        print("\nCorrelation (Pearson) between Sentiment Score and Returns:")
        for col in ["return_1h", "return_4h", "return_1d"]:
            valid_df = df.dropna(subset=[col, "sentiment_score"])
            if len(valid_df) > 1:
                corr = np.corrcoef(valid_df["sentiment_score"], valid_df[col])[0, 1]
                print(f"  Sentiment Score vs {col:9}: {corr:+.4f} (N={len(valid_df)})")
            else:
                print(f"  Sentiment Score vs {col:9}: Insufficient data")

        # 3. Accuracy of sentiment predicting direction
        # Positive sentiment predicting positive return (>0)
        # Negative sentiment predicting negative return (<=0)
        valid_dir = df.dropna(subset=["direction_1d", "sentiment_label"])
        valid_dir = valid_dir[valid_dir["sentiment_label"].isin(["positive", "negative"])]

        if not valid_dir.empty:
            correct = 0
            for _, row in valid_dir.iterrows():
                label = row["sentiment_label"]
                direction = row["direction_1d"]
                if (label == "positive" and direction == 1) or (label == "negative" and direction == 0):
                    correct += 1
            accuracy = correct / len(valid_dir)
            print(f"\nPrediction Accuracy (Direction 1D): {accuracy * 100:.2f}% (N={len(valid_dir)})")
            print("  (Positive sentiment -> Price up | Negative sentiment -> Price down)")
        else:
            print("\nPrediction Accuracy: Insufficient directional labels.")

    except Exception as e:
        print(f"Error evaluating sentiment signal: {e}")
    finally:
        conn.close()


def calculate_features_and_store(model_name="VADER"):
    """
    Calculate and store headline features in the feature_store table.
    """
    print(f"Calculating and storing features for model {model_name}...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Fetch all articles and their sentiment scores for this model
        cursor.execute(
            """
            SELECT n.id as article_id, n.stock_id, n.headline, n.published_at, s.sentiment_score
            FROM news_articles n
            JOIN sentiment_scores s ON n.id = s.article_id
            WHERE s.model_name = ?
            """,
            (model_name,),
        )
        articles = cursor.fetchall()
        print(f"Found {len(articles)} articles with {model_name} sentiment scores.")
        
        # Calculate stock-level average hourly volume for volume ratio normalization
        cursor.execute("SELECT stock_id, AVG(volume) as avg_vol FROM stock_prices GROUP BY stock_id")
        stock_avg_vol = {row["stock_id"]: row["avg_vol"] for row in cursor.fetchall()}
        
        # Get market index ID
        cursor.execute("SELECT id FROM stocks WHERE ticker = '^GSPC'")
        market_id_row = cursor.fetchone()
        market_id = market_id_row[0] if market_id_row else None

        def get_ref_return(ref_stock_id, ref_pub_at, offset_str="-24 hour"):
            if ref_stock_id is None:
                return 0.0
            # Closest price <= ref_pub_at
            cursor.execute(
                """
                SELECT close FROM stock_prices
                WHERE stock_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (ref_stock_id, ref_pub_at),
            )
            row_now = cursor.fetchone()
            if not row_now:
                return 0.0
            p_now = row_now["close"]
            
            # Closest price <= ref_pub_at - offset
            cursor.execute(
                f"""
                SELECT close FROM stock_prices
                WHERE stock_id = ? AND timestamp <= datetime(?, '{offset_str}')
                ORDER BY timestamp DESC LIMIT 1
                """,
                (ref_stock_id, ref_pub_at),
            )
            row_prior = cursor.fetchone()
            if not row_prior:
                return 0.0
            p_prior = row_prior["close"]
            
            return (p_now - p_prior) / p_prior if p_prior > 0 else 0.0

        def calculate_rsi(prices, period=14):
            if len(prices) <= period:
                return 50.0  # fallback neutral RSI
            
            deltas = np.diff(prices)
            seed = deltas[:period]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            
            if down == 0:
                rs = 1e9
            else:
                rs = up / down
            
            last_rsi = 100. - 100. / (1. + rs)
            
            for i in range(period, len(prices)):
                delta = deltas[i - 1]
                if delta > 0:
                    up_val = delta
                    down_val = 0.
                else:
                    up_val = 0.
                    down_val = -delta
                    
                up = (up * (period - 1) + up_val) / period
                down = (down * (period - 1) + down_val) / period
                
                if down == 0:
                    rs = 1e9
                else:
                    rs = up / down
                last_rsi = 100. - 100. / (1. + rs)
                
            return last_rsi

        features_to_insert = []
        for article in articles:
            article_id = article["article_id"]
            stock_id = article["stock_id"]
            headline = article["headline"]
            published_at = article["published_at"]
            sentiment_score = article["sentiment_score"]
            
            # 1. Headline length
            headline_length = len(headline)
            
            # 2. Hour of day
            try:
                dt = datetime.strptime(published_at, "%Y-%m-%d %H:%M:%S")
                hour_of_day = dt.hour
            except Exception:
                hour_of_day = 12 # fallback
                
            # 3. News count & average sentiment in the last 1h and 24 hours prior to published_at
            cursor.execute(
                """
                SELECT COUNT(*), AVG(s.sentiment_score)
                FROM news_articles n
                JOIN sentiment_scores s ON n.id = s.article_id
                WHERE n.stock_id = ?
                  AND s.model_name = ?
                  AND n.published_at >= datetime(?, '-1 hour')
                  AND n.published_at < ?
                """,
                (stock_id, model_name, published_at, published_at),
            )
            count_1h_row = cursor.fetchone()
            news_count_last_1h = count_1h_row[0] if count_1h_row else 0
            avg_sentiment_last_1h = count_1h_row[1] if (count_1h_row and count_1h_row[1] is not None) else 0.0

            cursor.execute(
                """
                SELECT COUNT(*), AVG(s.sentiment_score)
                FROM news_articles n
                JOIN sentiment_scores s ON n.id = s.article_id
                WHERE n.stock_id = ?
                  AND s.model_name = ?
                  AND n.published_at >= datetime(?, '-24 hour')
                  AND n.published_at < ?
                """,
                (stock_id, model_name, published_at, published_at),
            )
            count_24h_row = cursor.fetchone()
            news_count_last_24h = count_24h_row[0] if count_24h_row else 0
            avg_sentiment_last_24h = count_24h_row[1] if (count_24h_row and count_24h_row[1] is not None) else 0.0
            
            # 4. Volume ratio
            cursor.execute(
                """
                SELECT volume FROM stock_prices
                WHERE stock_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1
                """,
                (stock_id, published_at),
            )
            vol_row = cursor.fetchone()
            if not vol_row:
                cursor.execute(
                    """
                    SELECT volume FROM stock_prices
                    WHERE stock_id = ? AND timestamp >= ?
                    ORDER BY timestamp ASC LIMIT 1
                    """,
                    (stock_id, published_at),
                )
                vol_row = cursor.fetchone()
            vol = vol_row["volume"] if vol_row else None
            
            base_vol = stock_avg_vol.get(stock_id, 0.0)
            if vol is not None and base_vol > 0:
                volume_ratio = vol / base_vol
            else:
                volume_ratio = 1.0

            # 5. Market Return (preceding 24h market index return)
            market_return = get_ref_return(market_id, published_at, '-24 hour')

            # 6. Sector Return (preceding 24h sector ETF return)
            cursor.execute("SELECT sector FROM stocks WHERE id = ?", (stock_id,))
            sector_row = cursor.fetchone()
            sector = sector_row[0] if sector_row else "Technology"
            
            etf_ticker = SECTOR_ETF_MAP.get(sector, "XLK")
            cursor.execute("SELECT id FROM stocks WHERE ticker = ?", (etf_ticker,))
            etf_id_row = cursor.fetchone()
            etf_id = etf_id_row[0] if etf_id_row else None
            
            sector_return = get_ref_return(etf_id, published_at, '-24 hour')

            # 7. Volatility (rolling std dev of stock returns over past 14 days)
            cursor.execute(
                """
                SELECT close FROM stock_prices
                WHERE stock_id = ? AND timestamp >= datetime(?, '-14 day') AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (stock_id, published_at, published_at),
            )
            price_rows = cursor.fetchall()
            prices = [r["close"] for r in price_rows]
            if len(prices) > 2:
                pct_changes = [(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
                volatility = np.std(pct_changes)
            else:
                volatility = 0.0

            # 8. Previous Day Return, Return 5D, Return 10D
            prev_day_return = get_ref_return(stock_id, published_at, '-24 hour')
            return_5d = get_ref_return(stock_id, published_at, '-5 day')
            return_10d = get_ref_return(stock_id, published_at, '-10 day')

            # 9. Calculate RSI (14 hourly periods prior to published_at)
            cursor.execute(
                """
                SELECT close FROM stock_prices
                WHERE stock_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 150
                """,
                (stock_id, published_at),
            )
            rsi_price_rows = cursor.fetchall()
            rsi_prices = [r["close"] for r in reversed(rsi_price_rows)]
            rsi = calculate_rsi(rsi_prices, period=14) if len(rsi_prices) > 2 else 50.0
                
            features_to_insert.append(
                (
                    article_id,
                    model_name,
                    sentiment_score,
                    headline_length,
                    hour_of_day,
                    news_count_last_1h,
                    news_count_last_24h,
                    avg_sentiment_last_1h,
                    avg_sentiment_last_24h,
                    volume_ratio,
                    market_return,
                    sector_return,
                    volatility,
                    prev_day_return,
                    return_5d,
                    return_10d,
                    rsi,
                )
            )
            
        if features_to_insert:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO feature_store (
                    article_id, model_name, sentiment_score, headline_length,
                    hour_of_day, news_count_last_1h, news_count_last_24h,
                    avg_sentiment_last_1h, avg_sentiment_last_24h, volume_ratio,
                    market_return, sector_return, volatility, prev_day_return,
                    return_5d, return_10d, rsi
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                features_to_insert,
            )
            conn.commit()
            print(f"Successfully stored features for {len(features_to_insert)} articles ({model_name}).")
        else:
            print(f"No features to store for {model_name}.")
            
    except Exception as e:
        conn.rollback()
        print(f"Error calculating features for {model_name}: {e}")
        raise e
    finally:
        conn.close()


def calculate_daily_sentiment_index(model_name="VADER"):
    """
    Calculate the daily average sentiment and volume, and store it
    in the daily_sentiment_index table.
    """
    print(f"Calculating daily sentiment index for model {model_name}...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Group news articles by stock and date
        cursor.execute(
            """
            SELECT n.stock_id, date(n.published_at) as article_date,
                   AVG(s.sentiment_score) as avg_sent, COUNT(*) as cnt
            FROM news_articles n
            JOIN sentiment_scores s ON n.id = s.article_id
            WHERE s.model_name = ?
            GROUP BY n.stock_id, article_date
            """,
            (model_name,),
        )
        rows = cursor.fetchall()
        
        index_rows = []
        for row in rows:
            stock_id = row["stock_id"]
            article_date = row["article_date"]
            avg_sent = row["avg_sent"]
            cnt = row["cnt"]
            
            sentiment_index = avg_sent
            
            index_rows.append(
                (stock_id, article_date, model_name, avg_sent, cnt, sentiment_index)
            )
            
        if index_rows:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO daily_sentiment_index (
                    stock_id, date, model_name, avg_sentiment, news_count, sentiment_index
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                index_rows,
            )
            conn.commit()
            print(f"Successfully updated daily sentiment index for {len(index_rows)} rows ({model_name}).")
        else:
            print(f"No daily sentiment indices to update for {model_name}.")
            
    except Exception as e:
        conn.rollback()
        print(f"Error calculating daily sentiment index for {model_name}: {e}")
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    calculate_future_returns()
    for model in ["VADER", "FinBERT"]:
        calculate_features_and_store(model)
        calculate_daily_sentiment_index(model)
    evaluate_sentiment_signal()
