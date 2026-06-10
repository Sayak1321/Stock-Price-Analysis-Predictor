# 📈 Stock Price Analysis & Sentiment Predictor

<div align="center">

![DASHBOARD](/dashboard1.png)
![RECENT HEADLINES](/Recent%20Headline.png)
![MACHINE LEARNING PREDICTOR](/Machine%20Learning%20Predictor.png)
![PHASE COMPARISON SUMMARY](/Phase%20Comparision%20Summary.png)
![BACKTESTING SANDBOX](/Backtesting%20Sandbox.png)

**An end-to-end machine learning pipeline that combines financial news sentiment analysis (powered by FinBERT) with stock market data to predict next-day price direction.**

[Features](#-features) · [Architecture](#-architecture) · [Getting Started](#-getting-started) · [Usage](#-usage) · [Docker](#-docker-deployment) · [Model Details](#-model-details) · [Roadmap](#-roadmap)

</div>

---

## 🔍 Overview

This project builds a full data pipeline that:

1. **Collects** real-time financial news headlines via `yfinance` and news APIs
2. **Analyses** sentiment using the `FinBERT` transformer model (purpose-built for financial text)
3. **Engineers** features combining sentiment signals, technical indicators, and market context
4. **Predicts** whether a stock will go **UP** or **DOWN** the next trading session
5. **Visualises** everything in an interactive **Streamlit dashboard**

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is not financial advice. Never make investment decisions based solely on model outputs.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📰 **News Ingestion** | Automatic collection of financial headlines per ticker |
| 🤖 **FinBERT Sentiment** | State-of-the-art transformer sentiment scoring (Positive / Negative / Neutral) |
| 📊 **Technical Indicators** | RSI, Moving Averages, Volatility, Volume Ratio |
| 🌐 **Market Context** | Sector return, market index return, VIX integration |
| 🏗️ **Feature Store** | Precomputed ML-ready feature rows with look-ahead labels |
| 🔮 **Direction Prediction** | XGBoost / Random Forest / Logistic Regression ensemble |
| 📉 **Interactive Dashboard** | Streamlit UI with Plotly charts — sentiment vs price overlays |
| 🐳 **Docker Support** | One-command containerised deployment |
| 🗄️ **Structured Database** | SQLite schema with stocks, news, sentiment, prices, and ML targets |

---

## 🏗️ Architecture

```
Financial News API
        │
        ▼
┌──────────────────┐
│  News Ingestion  │  ← src/ingestion/news_fetcher.py
│  (news_fetcher)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌───────────────────┐
│   FinBERT NLP    │     │  Stock Price Data │  ← yfinance
│ Sentiment Engine │     │  (stock_fetcher)  │
└────────┬─────────┘     └────────┬──────────┘
         │                        │
         ▼                        ▼
┌──────────────────────────────────────────┐
│           SQLite / PostgreSQL DB          │
│  stocks | news_articles | sentiment_scores│
│  stock_prices | sentiment_targets        │
│  daily_sentiment_index | feature_store   │
└────────────────────┬─────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Feature Engineering  │  ← src/features/
         │  + Technical Indicators│
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   ML Prediction Model  │  ← src/models/
         │   XGBoost / RF / LR   │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Streamlit Dashboard  │  ← src/dashboard/streamlit_app.py
         │  (Port 8501)          │
         └───────────────────────┘
```

---

## 📁 Project Structure

```
Stock-Price-Analysis-Predictor/
│
├── src/
│   ├── ingestion/
│   │   ├── news_fetcher.py        # Fetches headlines from news APIs
│   │   ├── run_pipeline.py        
│   │   └── stock_fetcher.py       # Downloads OHLCV data via yfinance
│   │
│   ├── sentiment/
│   │   └── finbert_analyzer.py     # FinBERT inference + embedding extraction
│   │   └── vader_analyzer.py       # VADER inference + embedding extraction
│   │
│   ├── features/
│   │   └── feature_engineering.py # RSI, MA, sentiment aggregation, feature store
│   │
│   ├── models/
│   │   └── market_predictor.py    # ML model training, evaluation, persistence
│   │
│   ├── database/
│   │   ├── schema.sql             # Full SQL schema definition
│   │   └── db.py                  # DB connection helpers and CRUD operations
│   │
│   └── dashboard/
│       └── streamlit_app.py       # Interactive Streamlit web application
│
├── data/
│   └── stock_sentiment.db         # SQLite database (auto-created on first run)
│
├── Dockerfile                     # Container build instructions
├── docker-compose.yml             # Docker Compose orchestration
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- pip
- Git
- *(Optional)* Docker & Docker Compose for containerised setup

---

### Option 1 — Local Installation

#### 1. Clone the repository

```bash
git clone https://github.com/Sayak1321/Stock-Price-Analysis-Predictor.git
cd Stock-Price-Analysis-Predictor
```

#### 2. Create and activate a virtual environment

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Installing `torch` and `transformers` may take a few minutes. PyTorch (~2 GB) will be downloaded on first install. If you only have CPU, this is fine — FinBERT will still run, just slower.

#### 4. Download NLTK data (for VADER baseline)

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

#### 5. Initialise the database

```bash
python -c "
import sqlite3, os
os.makedirs('data', exist_ok=True)
conn = sqlite3.connect('data/stock_sentiment.db')
with open('src/database/schema.sql') as f:
    conn.executescript(f.read())
conn.close()
print('Database initialised successfully.')
"
```

#### 6. Run the Streamlit dashboard

```bash
streamlit run src/dashboard/streamlit_app.py
```

Open your browser at **http://localhost:8501**

---

### Option 2 — Docker Deployment

#### 1. Clone the repository

```bash
git clone https://github.com/Sayak1321/Stock-Price-Analysis-Predictor.git
cd Stock-Price-Analysis-Predictor
```

#### 2. Build and launch with Docker Compose

```bash
docker-compose up --build
```

The app will be available at **http://localhost:8501**

To run in detached (background) mode:

```bash
docker-compose up --build -d
```

To stop the container:

```bash
docker-compose down
```

#### Manual Docker build (without Compose)

```bash
# Build the image
docker build -t stock-predictor .

# Run the container
docker run -p 8501:8501 stock-predictor
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `yfinance` | ≥ 0.2.40 | Download stock OHLCV data |
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `nltk` | ≥ 3.8.1 | VADER baseline sentiment |
| `streamlit` | ≥ 1.35.0 | Interactive web dashboard |
| `plotly` | ≥ 5.22.0 | Charts and visualisations |
| `scikit-learn` | ≥ 1.4.0 | ML models (LR, RF, metrics) |
| `torch` | ≥ 2.2.0 | Deep learning backend for FinBERT |
| `transformers` | ≥ 4.38.0 | HuggingFace FinBERT model |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Details

### Sentiment Analysis — FinBERT

The project uses **ProsusAI/finbert**, a BERT model fine-tuned on financial text from analyst reports, financial news, and earnings call transcripts.

- **Output:** `Positive`, `Negative`, or `Neutral` with a confidence score in `[-1, 1]`
- **Advantage over VADER:** VADER is generic; FinBERT understands domain-specific financial language (e.g., "the company missed guidance" → correctly negative)

### ML Prediction Models

Direction prediction (UP / DOWN next day) is framed as a **binary classification** problem.

**Features used:**

| Category | Features |
|---|---|
| Sentiment | FinBERT score, 1h/24h average sentiment, news volume |
| Technical | RSI, MA20 distance, 5d/10d return, volatility |
| Market Context | Sector return, market index return, VIX |
| Time | Hour of day, day of week |

**Models trained and compared:**

- Logistic Regression (baseline)
- Random Forest
- XGBoost

**Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC — using **time-based train/test splits** (no data leakage).

> Realistic expected accuracy: **52–60%** out-of-sample. In quantitative finance, 58%+ with proper splits is strong.

---

## 📊 Dashboard Features

The Streamlit dashboard provides:

- **Sentiment Timeline** — Daily average sentiment index overlaid on stock price
- **Positive vs Negative News** — Bar breakdown per day
- **Prediction Output** — Model probability for UP/DOWN next session
- **Top Headlines** — Most recent positive and negative headlines
- **Feature Importance** — Which features matter most for the model

---

## 🗄️ Database Schema

The project uses a relational SQLite database with 6 tables:

```
stocks              → Tracked tickers and company info
news_articles       → Raw headlines with timestamps
sentiment_scores    → FinBERT scores per article
stock_prices        → OHLCV data per ticker
sentiment_targets   → Future return labels (1h, 4h, 1d direction)
daily_sentiment_index → Aggregated daily sentiment per ticker
feature_store       → Precomputed ML-ready feature rows
```

---

## 🔧 Configuration

To add new tickers to track, edit the relevant section in `src/ingestion/stock_fetcher.py` or pass them via the dashboard UI.

To switch from SQLite to PostgreSQL, update the connection string in `src/database/db.py`:

```python
# SQLite (default)
DATABASE_URL = "sqlite:///data/stock_sentiment.db"

# PostgreSQL (production)
DATABASE_URL = "postgresql://user:password@localhost:5432/stock_sentiment"
```

---

## 🗺️ Roadmap

- [x] Database schema design
- [x] Stock price ingestion (yfinance)
- [x] FinBERT sentiment pipeline
- [x] Streamlit dashboard
- [x] Docker deployment
- [ ] News API integration (NewsAPI / Alpha Vantage)
- [ ] FinBERT embedding features (768-dim vectors)
- [ ] XGBoost model training and evaluation
- [ ] Backtesting module
- [ ] REST API for predictions
- [ ] CI/CD with GitHub Actions
- [ ] Scheduled data ingestion (cron / Airflow)

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sayak** — [@Sayak1321](https://github.com/Sayak1321)

---

<div align="center">

⭐ If you found this project useful, please give it a star on GitHub!

</div>
