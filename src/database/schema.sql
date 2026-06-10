-- Database Schema for Stock Sentiment Analyzer

CREATE TABLE IF NOT EXISTS stocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    company_name TEXT,
    sector TEXT
);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER NOT NULL,
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT UNIQUE,
    published_at DATETIME NOT NULL,
    collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(stock_id) REFERENCES stocks(id)
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
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

CREATE TABLE IF NOT EXISTS stock_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    FOREIGN KEY(stock_id) REFERENCES stocks(id),
    UNIQUE(stock_id, timestamp)
);

CREATE TABLE IF NOT EXISTS sentiment_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER UNIQUE NOT NULL,
    return_1h REAL,
    return_4h REAL,
    return_1d REAL,
    direction_1d INTEGER,
    FOREIGN KEY(article_id) REFERENCES news_articles(id)
);

CREATE TABLE IF NOT EXISTS daily_sentiment_index (
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

CREATE TABLE IF NOT EXISTS feature_store (
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

CREATE TABLE IF NOT EXISTS headline_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY(article_id) REFERENCES news_articles(id),
    UNIQUE(article_id, model_name)
);


