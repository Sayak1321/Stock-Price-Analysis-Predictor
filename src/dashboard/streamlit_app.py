import sys
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import init_db, get_tracked_stocks, get_stock_price_history, get_stock_news_sentiment
from src.ingestion.stock_fetcher import fetch_and_store_stock_metadata, fetch_and_store_stock_prices
from src.ingestion.news_fetcher import fetch_and_store_news
from src.sentiment.analyzer import run_sentiment_pipeline
from src.sentiment.finbert_analyzer import run_finbert_pipeline
from src.features.engineering import calculate_future_returns, calculate_features_and_store, calculate_daily_sentiment_index
from src.models.market_predictor import train_and_evaluate_model, predict_latest_news_direction
from src.analytics.backtester import run_sentiment_backtest

# Ensure database and tables exist at startup
init_db()

# -----------------------------------------------------------------------------
# PAGE CONFIG & PREMIUM CUSTOM STYLING (Dark Mode + Glassmorphism)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look & feel (Inter/Outfit typography, Glassmorphism, tailored colors)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        /* Base styles and fonts */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #080b10 !important;
            color: #c9d1d9;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            color: #ffffff;
            font-weight: 700;
        }
        
        /* Glassmorphism Metric Cards */
        .metric-card {
            background: rgba(18, 22, 32, 0.65) !important;
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border-radius: 16px;
            padding: 24px 16px;
            border: 1px solid rgba(56, 139, 253, 0.15);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            text-align: center;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        .metric-card:hover {
            border-color: rgba(88, 166, 255, 0.6);
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 12px 30px rgba(56, 139, 253, 0.25);
        }
        .metric-val {
            font-size: 30px;
            font-weight: 700;
            color: #ffffff;
            margin-top: 8px;
            letter-spacing: -0.5px;
        }
        .metric-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #8b949e;
            font-weight: 600;
        }
        
        /* Styled News Cards with Sentiment Left-Borders */
        .news-card {
            background: rgba(18, 22, 32, 0.45);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 14px;
            border: 1px solid rgba(48, 54, 61, 0.3);
            border-left: 5px solid #8b949e; /* Default left border */
            transition: all 0.25s ease;
        }
        .news-card.positive {
            border-left-color: #2ea043 !important;
        }
        .news-card.negative {
            border-left-color: #f85149 !important;
        }
        .news-card.neutral {
            border-left-color: #8b949e !important;
        }
        .news-card:hover {
            border-color: rgba(88, 166, 255, 0.45);
            transform: translateX(4px);
            background: rgba(22, 27, 39, 0.65);
        }
        .news-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 8px;
            text-decoration: none;
            color: #58a6ff !important;
        }
        .news-title:hover {
            color: #79c0ff !important;
            text-decoration: underline;
        }
        .news-meta {
            font-size: 12px;
            color: #8b949e;
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
            align-items: center;
        }
        
        /* Pill badges */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .badge-positive {
            background-color: rgba(46, 160, 67, 0.12);
            color: #3fb950;
            border: 1px solid rgba(46, 160, 67, 0.25);
        }
        .badge-neutral {
            background-color: rgba(139, 148, 158, 0.12);
            color: #8b949e;
            border: 1px solid rgba(139, 148, 158, 0.25);
        }
        .badge-negative {
            background-color: rgba(248, 81, 73, 0.12);
            color: #f85149;
            border: 1px solid rgba(248, 81, 73, 0.25);
        }
        
        /* Sidebar styling */
        .css-1d391tw {
            background-color: #0b0e14 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
selected_model = st.sidebar.selectbox(
    "Select Sentiment Model",
    ["FinBERT", "VADER"],
    help="Choose which AI model's sentiment scores to display and analyze."
)
st.sidebar.markdown("---")

# Get list of stocks dynamically from DB
stocks_dict = get_tracked_stocks()
tickers = sorted(list(stocks_dict.keys()))

POPULAR_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "BTC-USD", "RELIANCE.NS", "SAP.F", "BP.L"]
dropdown_tickers = sorted(list(set(tickers + POPULAR_TICKERS)))

selected_ticker = st.sidebar.selectbox(
    "Select Stock Ticker",
    dropdown_tickers,
    help="Choose a stock ticker to view sentiment and market metrics. If it's not initialized, click Initialize below.",
)

is_initialized = selected_ticker in stocks_dict

if selected_ticker and not is_initialized:
    st.sidebar.warning(f"⚠️ {selected_ticker} is not loaded locally.")
    if st.sidebar.button(f"🚀 Initialize {selected_ticker}", use_container_width=True):
        with st.sidebar.status(f"Initializing {selected_ticker}...", expanded=True) as status:
            try:
                stock_id = fetch_and_store_stock_metadata(selected_ticker)
                st.write("Fetching stock price history...")
                fetch_and_store_stock_prices(selected_ticker, stock_id, period="3mo")
                st.write("Fetching latest headlines...")
                fetch_and_store_news(selected_ticker, stock_id)
                st.write("Running VADER analysis...")
                run_sentiment_pipeline()
                st.write("Running FinBERT analysis...")
                run_finbert_pipeline()
                st.write("Rebuilding feature stores & indices...")
                calculate_future_returns()
                for model in ["VADER", "FinBERT"]:
                    calculate_features_and_store(model)
                    calculate_daily_sentiment_index(model)
                status.update(label=f"Successfully initialized {selected_ticker}!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status.update(label=f"Initialization failed: {e}", state="error")

st.sidebar.markdown("---")
st.sidebar.markdown("### Track Custom Stock")
custom_ticker = st.sidebar.text_input("Enter Ticker Symbol", placeholder="e.g. NFLX, RELIANCE.NS, BTC-USD")
if st.sidebar.button("➕ Add Custom Ticker", use_container_width=True):
    if custom_ticker:
        ticker_clean = custom_ticker.strip().upper()
        if ticker_clean in stocks_dict:
            st.sidebar.info(f"{ticker_clean} is already tracked!")
        else:
            with st.sidebar.status(f"Initializing {ticker_clean}...", expanded=True) as status:
                try:
                    stock_id = fetch_and_store_stock_metadata(ticker_clean)
                    st.write("Fetching prices...")
                    fetch_and_store_stock_prices(ticker_clean, stock_id, period="3mo")
                    st.write("Fetching news...")
                    fetch_and_store_news(ticker_clean, stock_id)
                    st.write("Running VADER...")
                    run_sentiment_pipeline()
                    st.write("Running FinBERT...")
                    run_finbert_pipeline()
                    st.write("Rebuilding feature stores...")
                    calculate_future_returns()
                    for model in ["VADER", "FinBERT"]:
                        calculate_features_and_store(model)
                        calculate_daily_sentiment_index(model)
                    status.update(label=f"Added {ticker_clean} successfully!", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    status.update(label=f"Failed to add {ticker_clean}: {e}", state="error")


st.sidebar.markdown("---")
st.sidebar.markdown("### Update Data Pipeline")
st.sidebar.info("Pull latest headlines and stock price records from Yahoo Finance.")

if st.sidebar.button("🔄 Fetch & Analyze Live Data", use_container_width=True):
    with st.sidebar.status("Processing Data Ingestion & Sentiment Analysis...", expanded=True) as status:
        st.write("Initializing databases...")
        # Fallback list if no tickers in database yet
        active_tickers = tickers if tickers else ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        for ticker in active_tickers:
            st.write(f"Syncing stock info & price history for {ticker}...")
            stock_id = fetch_and_store_stock_metadata(ticker)
            fetch_and_store_stock_prices(ticker, stock_id, period="3mo")

            st.write(f"Fetching latest headlines for {ticker}...")
            fetch_and_store_news(ticker, stock_id)

        st.write("Running VADER Sentiment Intensity Analyzer...")
        run_sentiment_pipeline()
        st.write("Running FinBERT Sequence Classifier (CPU)...")
        run_finbert_pipeline()
        st.write("Calculating future stock returns...")
        calculate_future_returns()
        st.write("Rebuilding feature stores & daily indexes...")
        for model in ["VADER", "FinBERT"]:
            calculate_features_and_store(model)
            calculate_daily_sentiment_index(model)
        status.update(label="Sync Completed successfully!", state="complete", expanded=False)

    # Force app re-run with new data
    st.rerun()

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div style='text-align: center; font-size: 11px; color: #8b949e;'>"
    "Stock Sentiment Analyzer MVP<br>Powered by NLTK VADER & yfinance"
    "</div>",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# MAIN APP BODY
# -----------------------------------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom: 0px;'>📈 News Sentiment Analyzer</h1>"
    "<p style='color: #8b949e; font-size: 15px; margin-top: 5px; margin-bottom: 25px;'>"
    "Cross-referencing financial news sentiment index with daily market closing prices"
    "</p>",
    unsafe_allow_html=True,
)

if selected_ticker and not is_initialized:
    st.markdown(f"## 🚀 Initialize {selected_ticker} to begin analysis")
    st.info("This stock has not been loaded in your local database yet. "
            "Use the **🚀 Initialize** button in the sidebar to download historical prices, headlines, and compute sentiment metrics.")
elif selected_ticker:
    # 1. Fetch data from DB
    df_prices = get_stock_price_history(selected_ticker)
    df_news_all = get_stock_news_sentiment(selected_ticker)

    # Filter news articles by the selected model, preserving any completely unscored ones
    if not df_news_all.empty:
        df_news = df_news_all[(df_news_all["model_name"] == selected_model) | (df_news_all["model_name"].isna())].copy()
    else:
        df_news = pd.DataFrame()

    # Convert timestamps to python date format
    if not df_prices.empty:
        df_prices["date"] = pd.to_datetime(df_prices["timestamp"]).dt.date
    if not df_news.empty:
        df_news["date"] = pd.to_datetime(df_news["published_at"]).dt.date
        # Ensure we only work with articles that have sentiment scores computed
        df_scored_news = df_news.dropna(subset=["sentiment_score"])
    else:
        df_scored_news = pd.DataFrame()

    # Create interactive tabs for the advanced features
    tab1, tab2, tab3 = st.tabs([
        "📊 Live Sentiment Dashboard",
        "🧠 Machine Learning Predictor",
        "📈 Backtesting Sandbox"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: LIVE SENTIMENT DASHBOARD (Existing features)
    # -------------------------------------------------------------------------
    with tab1:
        # 2. Render KPI metrics
        if not df_scored_news.empty:
            avg_sentiment = df_scored_news["sentiment_score"].mean()
            total_articles = len(df_scored_news)
            pos_count = len(df_scored_news[df_scored_news["sentiment_label"] == "positive"])
            neu_count = len(df_scored_news[df_scored_news["sentiment_label"] == "neutral"])
            neg_count = len(df_scored_news[df_scored_news["sentiment_label"] == "negative"])
        else:
            avg_sentiment = 0.0
            total_articles = 0
            pos_count = neu_count = neg_count = 0

        # Determine visual sentiment badge color mapping
        if avg_sentiment >= 0.05:
            sentiment_status = "Positive"
            sentiment_css = "color: #3fb950;"
        elif avg_sentiment <= -0.05:
            sentiment_status = "Negative"
            sentiment_css = "color: #f85149;"
        else:
            sentiment_status = "Neutral"
            sentiment_css = "color: #8b949e;"

        # Injecting custom metric layout
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Overall Sentiment</div>
                    <div class='metric-val' style='{sentiment_css}'>{sentiment_status}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Avg Score</div>
                    <div class='metric-val'>{avg_sentiment:+.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>News Volume</div>
                    <div class='metric-val'>{total_articles}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Positive News</div>
                    <div class='metric-val' style='color: #3fb950;'>{pos_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <div class='metric-label'>Negative News</div>
                    <div class='metric-val' style='color: #f85149;'>{neg_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Charts Area
        st.markdown("### 📈 Price & Sentiment Correlation Chart")

        if not df_prices.empty and not df_scored_news.empty:
            # Group news sentiment by date to find daily averages
            df_daily_sentiment = (
                df_scored_news.groupby("date")["sentiment_score"].mean().reset_index()
            )
            df_daily_sentiment.rename(columns={"sentiment_score": "avg_sentiment"}, inplace=True)

            # Merge price history and daily sentiment indices
            df_merged = pd.merge(df_prices, df_daily_sentiment, on="date", how="left")
            df_merged["avg_sentiment"] = df_merged["avg_sentiment"].ffill().fillna(0.0)

            # Creating Subplots: Line chart for close price and Bar chart for average sentiment
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Plot stock closing price with premium area fill
            fig.add_trace(
                go.Scatter(
                    x=df_merged["date"],
                    y=df_merged["close"],
                    name=f"{selected_ticker} Close Price",
                    line=dict(color="#58a6ff", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(88, 166, 255, 0.08)",
                    hoverinfo="x+y",
                ),
                secondary_y=False,
            )

            # Color-coded bar values based on daily sentiment
            bar_colors = [
                "#2ea043" if s > 0.05 else ("#f85149" if s < -0.05 else "#8b949e")
                for s in df_merged["avg_sentiment"]
            ]

            # Plot daily average sentiment index
            fig.add_trace(
                go.Bar(
                    x=df_merged["date"],
                    y=df_merged["avg_sentiment"],
                    name="Daily Sentiment Index",
                    marker_color=bar_colors,
                    opacity=0.45,
                    hoverinfo="x+y",
                ),
                secondary_y=True,
            )

            # Layout Styling
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(
                    showgrid=True,
                    gridcolor="rgba(48, 54, 61, 0.4)",
                    tickformat="%Y-%m-%d",
                ),
                yaxis=dict(
                    title=dict(text="Stock Price ($)", font=dict(color="#58a6ff")),
                    showgrid=True,
                    gridcolor="rgba(48, 54, 61, 0.4)",
                    tickfont=dict(color="#58a6ff"),
                ),
                yaxis2=dict(
                    title=dict(text="Sentiment Score (-1.0 to 1.0)", font=dict(color="#8b949e")),
                    showgrid=False,
                    tickfont=dict(color="#8b949e"),
                    range=[-1.05, 1.05],
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Insufficient stock price or sentiment data to display the correlation chart.")

        # 4. News Articles Details
        st.markdown("### 📰 Recent Headlines Detail")

        if not df_news.empty:
            # Loop through headlines and print styled cards
            for _, row in df_news.iterrows():
                sentiment_label = row.get("sentiment_label")
                score = row.get("sentiment_score")

                # Format published date
                pub_date = row["published_at"]
                if isinstance(pub_date, str):
                    try:
                        dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                        pub_date_formatted = dt.strftime("%b %d, %Y at %H:%M")
                    except ValueError:
                        pub_date_formatted = pub_date
                else:
                    pub_date_formatted = str(pub_date)

                # Map badges and card border classes
                card_class = "news-card"
                if pd.isna(score):
                    badge_html = "<span class='badge badge-neutral'>Unanalyzed</span>"
                    card_class += " neutral"
                elif sentiment_label == "positive":
                    badge_html = f"<span class='badge badge-positive'>Positive ({score:+.2f})</span>"
                    card_class += " positive"
                elif sentiment_label == "negative":
                    badge_html = f"<span class='badge badge-negative'>Negative ({score:.2f})</span>"
                    card_class += " negative"
                else:
                    badge_html = f"<span class='badge badge-neutral'>Neutral ({score:+.2f})</span>"
                    card_class += " neutral"

                st.markdown(
                    f"""
                    <div class='{card_class}'>
                        <div class='news-meta'>
                            <span>{pub_date_formatted}</span>
                            <span>•</span>
                            <span>Source: <b>{row['source'] or 'Unknown'}</b></span>
                            <span>•</span>
                            {badge_html}
                        </div>
                        <div>
                            <a class='news-title' href='{row['url']}' target='_blank'>{row['headline']}</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No headlines found for this stock ticker.")

    # -------------------------------------------------------------------------
    # TAB 2: MACHINE LEARNING PREDICTORS
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 🧠 ML Stock Direction Predictor")
        st.markdown("Predict future stock return direction (UP vs DOWN/FLAT) based on historical article features, context, and embeddings.")

        # Training panel
        train_col, info_col = st.columns([1, 2])
        with train_col:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Retrain ML Models", use_container_width=True):
                with st.spinner("Training models on current feature store..."):
                    # Train on 4h horizon to match the prediction layout
                    train_and_evaluate_model(selected_model, "direction_4h")
                    st.success("Models retrained successfully!")
                    st.rerun()

        with info_col:
            st.info("Features are organized in three stages. Retraining fits Phase 1 (Baseline features), "
                    "Phase 2 (Market context), and Phase 3 (Transformers CLS embeddings + market context).")

        # Load trained weights and check for model
        import pickle
        MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/models"))
        model_path = os.path.join(MODEL_DIR, f"market_predictor_{selected_model.lower()}.pkl")

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                payload = pickle.load(f)

            st.markdown("---")
            # Dynamic Real-Time Forecast Section
            st.markdown("### 🔮 Real-Time Directional Forecast")
            st.markdown("Generate stock direction forecasts dynamically by selecting the pipeline phase and underlying classifier.")
            
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                phase_options = ["Phase 1 (Baseline)", "Phase 2 (Market Context)"]
                if selected_model == "FinBERT" and payload.get("phase3") is not None:
                    phase_options.append("Phase 3 (FinBERT Embeddings)")
                selected_phase_label = st.selectbox("Select Model Phase", phase_options, key="ml_forecast_phase")
                
                phase_map = {
                    "Phase 1 (Baseline)": "phase1",
                    "Phase 2 (Market Context)": "phase2",
                    "Phase 3 (FinBERT Embeddings)": "phase3"
                }
                phase_key = phase_map[selected_phase_label]
                
            with pcol2:
                if phase_key in ["phase1", "phase2"]:
                    model_options = {"Random Forest": "rf", "Logistic Regression": "lr"}
                else:
                    model_options = {"XGBoost": "xgb", "LightGBM": "lgb"}
                selected_sub_model_label = st.selectbox("Select Classifier", list(model_options.keys()), key="ml_forecast_classifier")
                sub_model_key = model_options[selected_sub_model_label]
                
            pred_result = predict_latest_news_direction(selected_ticker, selected_model, phase=phase_key, sub_model=sub_model_key)
            
            if "error" not in pred_result:
                pred = pred_result["prediction"]
                prob = pred_result["probability_up"]
                cnt = pred_result["articles_count"]
                
                color = "#3fb950" if pred == "UP" else "#f85149"
                bg_alpha = "rgba(46, 160, 67, 0.08)" if pred == "UP" else "rgba(248, 81, 73, 0.08)"
                
                st.markdown(
                    f"""
                    <div class='metric-card' style='text-align: left; max-width: 650px; border-left: 5px solid {color}; background: {bg_alpha} !important;'>
                        <div class='metric-label'>Next 4h Forecast for <b>{selected_ticker}</b> ({selected_phase_label} - {selected_sub_model_label})</div>
                        <div class='metric-val' style='color: {color}; text-align: left; margin-top: 5px;'>{pred}</div>
                        <div style='font-size: 13px; color: #8b949e; margin-top: 10px;'>
                            Probability of Upward Return: <b>{prob*100:.1f}%</b><br>
                            Evaluated over the latest <b>{cnt}</b> articles' averaged features.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning(pred_result["error"])

            st.markdown("---")
            st.markdown("### 📊 Phase Comparison Summary")
            st.markdown(f"**Trained Sample Size:** {payload.get('sample_count', 0)} headlines (Train: {payload.get('train_samples', 0)} / Test: {payload.get('test_samples', 0)}) | **Validation:** Chronological Out-of-Sample Split (80/20)")

            # Create columns for Phase 1, Phase 2, Phase 3
            cols = st.columns(3) if selected_model == "FinBERT" and payload.get("phase3") is not None else st.columns(2)
            
            # Phase 1 Baseline
            with cols[0]:
                st.markdown("#### 🟥 Phase 1: Baseline")
                st.markdown("<p style='font-size:12px; color:#8b949e; margin-bottom:15px;'>Features: Sentiment Score, Hour of Day, News Volume, Avg Sentiment, Volume Ratio</p>", unsafe_allow_html=True)
                p1_metrics = payload["phase1"]["metrics"]
                
                st.markdown("##### Random Forest Classifier")
                m_rf = p1_metrics["rf"]
                st.markdown(f"**Accuracy:** {m_rf['accuracy']*100:.1f}% | **Precision:** {m_rf['precision']*100:.1f}% | **Recall:** {m_rf['recall']*100:.1f}% | **F1:** {m_rf['f1']*100:.1f}% | **ROC-AUC:** {m_rf.get('auc', 0.5)*100:.1f}%")
                
                st.markdown("##### Logistic Regression")
                m_lr = p1_metrics["lr"]
                st.markdown(f"**Accuracy:** {m_lr['accuracy']*100:.1f}% | **Precision:** {m_lr['precision']*100:.1f}% | **Recall:** {m_lr['recall']*100:.1f}% | **F1:** {m_lr['f1']*100:.1f}% | **ROC-AUC:** {m_lr.get('auc', 0.5)*100:.1f}%")
                
            # Phase 2 Context
            with cols[1]:
                st.markdown("#### 🟩 Phase 2: Context")
                st.markdown("<p style='font-size:12px; color:#8b949e; margin-bottom:15px;'>Features: Baseline + Market Return, Sector Return, Volatility, Prev Day Return</p>", unsafe_allow_html=True)
                p2_metrics = payload["phase2"]["metrics"]
                
                st.markdown("##### Random Forest Classifier")
                m_rf_p2 = p2_metrics["rf"]
                st.markdown(f"**Accuracy:** {m_rf_p2['accuracy']*100:.1f}% | **Precision:** {m_rf_p2['precision']*100:.1f}% | **Recall:** {m_rf_p2['recall']*100:.1f}% | **F1:** {m_rf_p2['f1']*100:.1f}% | **ROC-AUC:** {m_rf_p2.get('auc', 0.5)*100:.1f}%")
                
                st.markdown("##### Logistic Regression")
                m_lr_p2 = p2_metrics["lr"]
                st.markdown(f"**Accuracy:** {m_lr_p2['accuracy']*100:.1f}% | **Precision:** {m_lr_p2['precision']*100:.1f}% | **Recall:** {m_lr_p2['recall']*100:.1f}% | **F1:** {m_lr_p2['f1']*100:.1f}% | **ROC-AUC:** {m_lr_p2.get('auc', 0.5)*100:.1f}%")

            # Phase 3 Advanced (FinBERT only)
            if len(cols) > 2:
                with cols[2]:
                    st.markdown("#### 🟦 Phase 3: Advanced")
                    st.markdown("<p style='font-size:12px; color:#8b949e; margin-bottom:15px;'>Features: 768-dim FinBERT CLS Embeddings + 14 Scalar Context Features</p>", unsafe_allow_html=True)
                    p3_metrics = payload["phase3"]["metrics"]
                    
                    st.markdown("##### XGBoost Classifier")
                    m_xgb = p3_metrics["xgb"]
                    st.markdown(f"**Accuracy:** {m_xgb['accuracy']*100:.1f}% | **Precision:** {m_xgb['precision']*100:.1f}% | **Recall:** {m_xgb['recall']*100:.1f}% | **F1:** {m_xgb['f1']*100:.1f}% | **ROC-AUC:** {m_xgb.get('auc', 0.5)*100:.1f}%")
                    
                    st.markdown("##### LightGBM Classifier")
                    m_lgb = p3_metrics["lgb"]
                    st.markdown(f"**Accuracy:** {m_lgb['accuracy']*100:.1f}% | **Precision:** {m_lgb['precision']*100:.1f}% | **Recall:** {m_lgb['recall']*100:.1f}% | **F1:** {m_lgb['f1']*100:.1f}% | **ROC-AUC:** {m_lgb.get('auc', 0.5)*100:.1f}%")

            st.markdown("---")
            # Feature Importance Section
            st.markdown("### 📊 Feature Importance Insights")
            st.markdown("Analyze which factors are most critical in predicting stock direction across the three pipeline phases.")
            
            # Choose which phase's feature importance to plot
            imp_phase_options = ["Phase 1 (Baseline)", "Phase 2 (Market Context)"]
            if selected_model == "FinBERT" and payload.get("phase3") is not None:
                imp_phase_options.append("Phase 3 (FinBERT Embeddings)")
            selected_imp_phase_label = st.selectbox("Select Phase for Feature Importance", imp_phase_options, key="ml_imp_phase")
            imp_phase_key = phase_map[selected_imp_phase_label]
            
            if imp_phase_key == "phase1":
                importances = payload["phase1"]["metrics"]["rf"]["feature_importances"]
                feature_names = [n.replace("_", " ").title() for n in payload["phase1"]["features"]]
                title_suffix = "Random Forest (Baseline)"
            elif imp_phase_key == "phase2":
                importances = payload["phase2"]["metrics"]["rf"]["feature_importances"]
                feature_names = [n.replace("_", " ").title() for n in payload["phase2"]["features"]]
                title_suffix = "Random Forest (Context)"
            else:
                # For Phase 3, display the summarized importances (embed sum + 9 scalars)
                selected_p3_clf = st.radio("Select Classifier Importance Source", ["XGBoost", "LightGBM"], horizontal=True)
                p3_metrics = payload["phase3"]["metrics"]
                if selected_p3_clf == "XGBoost":
                    importances = p3_metrics["xgb"]["feature_importances"]
                    title_suffix = "XGBoost (Embeddings + Context)"
                else:
                    importances = p3_metrics["lgb"]["feature_importances"]
                    title_suffix = "LightGBM (Embeddings + Context)"
                    
                feature_names = ["FinBERT Embeddings (768 dims, summed)"] + [n.replace("_", " ").title() for n in payload["phase3"]["scalar_features"]]

            # Create feature importance chart
            fig_imp = go.Figure(go.Bar(
                x=importances,
                y=feature_names,
                orientation='h',
                marker_color='#58a6ff'
            ))
            fig_imp.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text=f"Feature Importances - {title_suffix}", font=dict(color="#58a6ff")),
                xaxis=dict(title="Relative Importance", showgrid=True, gridcolor="rgba(48, 54, 61, 0.4)"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("No models have been trained yet. Click 'Retrain ML Models' above to start.")

    # -------------------------------------------------------------------------
    # TAB 3: BACKTESTING SANDBOX
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### 📈 Sentiment Trading Backtester")
        st.markdown("Simulate returns from trading stock based on headline sentiment scores compared to a baseline Buy & Hold strategy.")

        # Input Parameters
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            threshold = st.slider(
                "Sentiment Threshold Score",
                min_value=0.05,
                max_value=0.90,
                value=0.15,
                step=0.05,
                help="Absolute sentiment score above which to trigger trades."
            )
        with bcol2:
            period = st.selectbox(
                "Strategy Holding Period",
                ["1h", "4h", "1d"],
                index=1,
                help="Duration to hold the position after a news headline trigger."
            )
        with bcol3:
            shorting = st.checkbox(
                "Enable Short Selling on Negative News",
                value=False,
                help="Short the stock when sentiment <= -threshold. If unchecked, go to cash (0% return)."
            )

        df_bt, summary = run_sentiment_backtest(selected_ticker, selected_model, threshold, period, shorting)

        if not df_bt.empty and summary:
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Trades Triggered</div>
                        <div class='metric-val'>{summary['total_trades']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m2:
                strat_ret = summary['strategy_return']
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Strategy Cumulative Return</div>
                        <div class='metric-val' style='color: {"#3fb950" if strat_ret >= 0 else "#f85149"};'>{strat_ret*100:+.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m3:
                bh_ret = summary['bh_return']
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Buy & Hold Baseline Return</div>
                        <div class='metric-val' style='color: {"#3fb950" if bh_ret >= 0 else "#f85149"};'>{bh_ret*100:+.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m4:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Trade Win Rate</div>
                        <div class='metric-val'>{summary['win_rate']*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Cumulative Return Chart
            st.markdown("#### Cumulative Performance Curve")
            df_bt["timestamp_dt"] = pd.to_datetime(df_bt["published_at"])
            
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=df_bt["timestamp_dt"],
                y=df_bt["strategy_cum_return"] * 100,
                name="Sentiment Strategy",
                line=dict(color="#3fb950", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(46, 160, 67, 0.05)",
                hoverinfo="x+y"
            ))
            fig_bt.add_trace(go.Scatter(
                x=df_bt["timestamp_dt"],
                y=df_bt["bh_cum_return"] * 100,
                name="Buy & Hold Baseline",
                line=dict(color="#8b949e", width=2, dash='dash'),
                fill="tozeroy",
                fillcolor="rgba(139, 148, 158, 0.03)",
                hoverinfo="x+y"
            ))
            
            fig_bt.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor="rgba(48, 54, 61, 0.4)"),
                yaxis=dict(title="Cumulative Return (%)", showgrid=True, gridcolor="rgba(48, 54, 61, 0.4)")
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            # Table showing details
            st.markdown("#### 📋 Position & Trade Details")
            df_trades = df_bt[df_bt["position"].isin(["LONG", "SHORT"])].copy()
            if not df_trades.empty:
                df_trades_display = df_trades[[
                    "published_at", "headline", "sentiment_score", "position", "target_return"
                ]].rename(columns={
                    "published_at": "Trigger Time",
                    "headline": "Triggering Headline",
                    "sentiment_score": "Sentiment Score",
                    "position": "Position Triggered",
                    "target_return": "Trade Return"
                })
                df_trades_display["Trade Return"] = df_trades_display["Trade Return"].map(lambda x: f"{x*100:+.2f}%")
                st.dataframe(df_trades_display, hide_index=True, use_container_width=True)
            else:
                st.info("No trades triggered under these parameters.")
        else:
            st.warning("Insufficient trades or historical data to display backtester analytics.")

else:
    st.warning("Please select a stock ticker in the sidebar control panel.")

