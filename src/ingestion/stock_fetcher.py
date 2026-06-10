import sys
import os
import yfinance as yf
from datetime import datetime

# Add the workspace root to Python path so we can import src.database.db
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import insert_stock, insert_stock_prices


def fetch_and_store_stock_metadata(ticker):
    """
    Fetch company name and sector using yfinance.
    Gracefully falls back to placeholder values if info request fails.
    """
    print(f"Fetching metadata for {ticker}...")
    company_name = ticker
    sector = "Unknown"

    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        if info:
            company_name = info.get("longName") or info.get("shortName") or company_name
            sector = info.get("sector") or sector
    except Exception as e:
        print(f"Warning: Failed to fetch metadata for {ticker} from Yahoo Finance: {e}")
        print("Using fallback metadata.")

    stock_id = insert_stock(ticker, company_name, sector)
    print(f"Stock {ticker} verified in DB with ID: {stock_id}")
    return stock_id


def fetch_and_store_stock_prices(ticker, stock_id, period="2mo"):
    """
    Fetch hourly historical stock price data for the specified period (e.g. '2mo', '1mo')
    and store it in the database.
    """
    print(f"Fetching hourly price history for {ticker} (period: {period})...")
    try:
        ticker_obj = yf.Ticker(ticker)
        # Fetch hourly historical prices (Yahoo Finance supports 1h data for up to 730 days)
        df = ticker_obj.history(period=period, interval="1h")

        if df.empty:
            print(f"No price data found for {ticker} for period {period}.")
            return 0

        # Reset index to access Date/Datetime column
        df = df.reset_index()

        import pandas as pd
        from datetime import timezone
        prices_to_insert = []
        
        # yfinance resets index to "Datetime" for hourly, or "Date" for daily
        time_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])

        for _, row in df.iterrows():
            # Skip rows with NaN values in Open, High, Low, Close
            if pd.isna(row["Open"]) or pd.isna(row["High"]) or pd.isna(row["Low"]) or pd.isna(row["Close"]):
                continue

            date_val = row[time_col]
            
            # Standardize timezone to UTC
            if hasattr(date_val, "tzinfo") and date_val.tzinfo is not None:
                dt_utc = date_val.astimezone(timezone.utc)
            else:
                if hasattr(date_val, "to_pydatetime"):
                    dt_utc = date_val.to_pydatetime().replace(tzinfo=timezone.utc)
                else:
                    dt_utc = date_val.replace(tzinfo=timezone.utc)

            # Format datetime as YYYY-MM-DD HH:MM:SS
            timestamp_str = dt_utc.strftime("%Y-%m-%d %H:%M:%S")

            prices_to_insert.append(
                (
                    stock_id,
                    timestamp_str,
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    int(row["Volume"]),
                )
            )

        inserted_count = insert_stock_prices(prices_to_insert)
        print(f"Successfully inserted/updated {inserted_count} price rows for {ticker}.")
        return inserted_count
    except Exception as e:
        print(f"Error fetching/storing stock prices for {ticker}: {e}")
        return 0


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


def fetch_and_store_reference_data(period="2mo"):
    """
    Ensure S&P 500 (^GSPC) and sector ETFs are registered in stocks table,
    and fetch their hourly price histories.
    """
    print("\n==========================================")
    print(f"Syncing market reference index & sector ETFs (period: {period})...")
    print("==========================================")
    
    reference_tickers = ["^GSPC"] + list(SECTOR_ETF_MAP.values())
    for ticker in reference_tickers:
        try:
            stock_id = insert_stock(ticker, company_name=ticker, sector="Reference Index")
            fetch_and_store_stock_prices(ticker, stock_id, period=period)
        except Exception as e:
            print(f"Warning: Failed to fetch reference data for {ticker}: {e}")


if __name__ == "__main__":
    # Test script execution
    ticker = "AAPL"
    stock_id = fetch_and_store_stock_metadata(ticker)
    fetch_and_store_stock_prices(ticker, stock_id, period="1mo")
    fetch_and_store_reference_data(period="1mo")
