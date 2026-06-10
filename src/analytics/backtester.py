import os
import sys
import sqlite3
import pandas as pd
import numpy as np

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import get_db_connection


def run_sentiment_backtest(ticker, model_name="VADER", sentiment_threshold=0.15, holding_period="1d", short_selling=False):
    """
    Simulate a trading strategy based on headline sentiment.
    - Long when sentiment_score >= sentiment_threshold
    - Short when sentiment_score <= -sentiment_threshold (if short_selling is True)
    - Hold for the selected period (1h, 4h, 1d) and compute cumulative returns.
    """
    print(f"Running sentiment backtest for {ticker} using {model_name}...")
    
    conn = get_db_connection()
    try:
        # Determine the target column based on the holding period
        return_col = f"return_{holding_period}"
        
        query = f"""
            SELECT n.published_at, n.headline, s.sentiment_score,
                   t.{return_col} as target_return
            FROM news_articles n
            JOIN sentiment_scores s ON n.id = s.article_id
            JOIN sentiment_targets t ON n.id = t.article_id
            JOIN stocks st ON n.stock_id = st.id
            WHERE st.ticker = ?
              AND s.model_name = ?
            ORDER BY n.published_at ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(ticker.upper(), model_name))
        
        if df.empty:
            return pd.DataFrame(), {}
            
        # Drop rows where target return or sentiment score is missing
        df = df.dropna(subset=["target_return", "sentiment_score"]).copy()
        
        if df.empty:
            return pd.DataFrame(), {}
            
        trade_returns = []
        positions = []
        
        for _, row in df.iterrows():
            score = row["sentiment_score"]
            ret = row["target_return"]
            
            if score >= sentiment_threshold:
                trade_returns.append(ret)
                positions.append("LONG")
            elif score <= -sentiment_threshold:
                if short_selling:
                    trade_returns.append(-ret)
                    positions.append("SHORT")
                else:
                    trade_returns.append(0.0)
                    positions.append("FLAT (CASH)")
            else:
                trade_returns.append(0.0)
                positions.append("NONE")
                
        df["position"] = positions
        df["trade_return"] = trade_returns
        
        # Calculate cumulative returns
        # Reinvested strategy equity curve
        df["strategy_cum_return"] = (1 + df["trade_return"]).cumprod() - 1
        
        # Buy & Hold baseline: invest at every news timestamp and hold for the same period
        # For an honest trade-by-trade comparison, we compare the sum or product of all underlying asset returns
        df["bh_cum_return"] = (1 + df["target_return"]).cumprod() - 1
        
        # Calculate summary statistics
        total_trades = len(df[df["position"].isin(["LONG", "SHORT"])])
        active_trades = df[df["position"].isin(["LONG", "SHORT"])]
        
        if total_trades > 0:
            win_rate = len(active_trades[active_trades["trade_return"] > 0]) / total_trades
            avg_trade_return = active_trades["trade_return"].mean()
            std_trade_return = active_trades["trade_return"].std()
            sharpe_ratio_proxy = (avg_trade_return / std_trade_return * np.sqrt(252)) if std_trade_return > 0 else 0.0
        else:
            win_rate = 0.0
            avg_trade_return = 0.0
            sharpe_ratio_proxy = 0.0
            
        strategy_final_return = df["strategy_cum_return"].iloc[-1] if not df.empty else 0.0
        bh_final_return = df["bh_cum_return"].iloc[-1] if not df.empty else 0.0
        
        summary = {
            "total_articles": len(df),
            "total_trades": total_trades,
            "win_rate": win_rate,
            "strategy_return": strategy_final_return,
            "bh_return": bh_final_return,
            "sharpe_ratio": sharpe_ratio_proxy,
            "ticker": ticker,
            "model_name": model_name,
            "threshold": sentiment_threshold,
            "holding_period": holding_period
        }
        
        return df, summary
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        return pd.DataFrame(), {}
    finally:
        conn.close()


if __name__ == "__main__":
    # Test execution
    df, summary = run_sentiment_backtest("AAPL", "VADER", 0.1, "4h", True)
    if not df.empty:
        print("\nBacktest Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("\nFirst 5 trades:")
        print(df[["published_at", "headline", "sentiment_score", "position", "trade_return", "strategy_cum_return"]].head())
