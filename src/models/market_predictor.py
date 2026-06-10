import os
import sys
import sqlite3
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import get_db_connection

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/models"))


def get_training_data(model_name="VADER"):
    """
    Fetch features from feature_store and targets from sentiment_targets for training,
    along with embeddings from headline_embeddings if model_name is FinBERT.
    """
    conn = get_db_connection()
    try:
        # Load joined feature store and sentiment targets
        query = """
            SELECT f.article_id, f.sentiment_score, f.headline_length, f.hour_of_day,
                   f.news_count_last_1h, f.news_count_last_24h,
                   f.avg_sentiment_last_1h, f.avg_sentiment_last_24h, f.volume_ratio,
                   f.market_return, f.sector_return, f.volatility, f.prev_day_return,
                   f.return_5d, f.return_10d, f.rsi,
                   t.return_1h, t.return_4h, t.return_1d, t.direction_1d,
                   n.published_at
            FROM feature_store f
            JOIN sentiment_targets t ON f.article_id = t.article_id
            JOIN news_articles n ON f.article_id = n.id
            WHERE f.model_name = ?
        """
        df = pd.read_sql_query(query, conn, params=(model_name,))
        
        # If model is FinBERT, fetch embeddings
        if model_name == "FinBERT" and not df.empty:
            emb_query = """
                SELECT article_id, embedding
                FROM headline_embeddings
                WHERE model_name = ?
            """
            emb_df = pd.read_sql_query(emb_query, conn, params=(model_name,))
            if not emb_df.empty:
                df = pd.merge(df, emb_df, on="article_id", how="left")
        
        return df
    except Exception as e:
        print(f"Error reading training data: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def instantiate_model(model_type, random_state=42):
    """Instantiate a model with custom hyperparameters to avoid overfitting on small datasets."""
    if model_type == "lr":
        return LogisticRegression(C=0.5, random_state=random_state, max_iter=1000)
    elif model_type == "rf":
        return RandomForestClassifier(n_estimators=30, max_depth=3, random_state=random_state)
    elif model_type == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=random_state, eval_metric="logloss")
    elif model_type == "lgb":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, random_state=random_state, verbosity=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model_time_split(model_type, X_train, y_train, X_test, y_test):
    """Train model on train split, evaluate on test split, and return metrics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = instantiate_model(model_type)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    # Calculate metrics
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    
    # Calculate ROC-AUC if test target has both classes
    classes_test = np.unique(y_test)
    if len(classes_test) >= 2:
        try:
            pred_probs = model.predict_proba(X_test_scaled)[:, 1]
            auc = float(roc_auc_score(y_test, pred_probs))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5
        
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    }


def train_and_evaluate_model(model_name="VADER", target_col="direction_1d"):
    """
    Train and save models for Phase 1 (Baseline), Phase 2 (Context), and Phase 3 (Advanced).
    Uses chronological time-based split (80% train, 20% test) for robust validation.
    """
    print(f"\nTraining multi-phase predictive model for: {model_name}, target: {target_col}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    df = get_training_data(model_name)
    if df.empty:
        print("No training data found.")
        return None

    # Handle target definition and cleaning
    if target_col == "direction_1d":
        target_check = ["direction_1d"]
    elif target_col == "direction_4h":
        target_check = ["return_4h"]
    else:
        target_check = ["return_1h"]

    # Define features for each phase
    features_p1 = [
        "sentiment_score",
        "hour_of_day",
        "news_count_last_24h",
        "avg_sentiment_last_24h",
        "volume_ratio"
    ]
    features_p2 = features_p1 + [
        "market_return",
        "sector_return",
        "volatility",
        "prev_day_return"
    ]
    features_p3_scalars = [
        "headline_length",
        "hour_of_day",
        "news_count_last_1h",
        "news_count_last_24h",
        "avg_sentiment_last_1h",
        "avg_sentiment_last_24h",
        "volume_ratio",
        "market_return",
        "sector_return",
        "volatility",
        "prev_day_return",
        "return_5d",
        "return_10d",
        "rsi"
    ]

    # Clean the dataset to ensure no missing values on columns needed
    cols_to_check = features_p2 + ["headline_length", "news_count_last_1h", "avg_sentiment_last_1h", "return_5d", "return_10d", "rsi"] + target_check
    if model_name == "FinBERT":
        cols_to_check += ["embedding"]
        
    df_clean = df.dropna(subset=cols_to_check)
    if len(df_clean) < 10:
        print(f"Insufficient training samples (only {len(df_clean)} rows). Training skipped.")
        return None

    # Sort chronologically by publication time to prevent look-ahead bias
    df_clean = df_clean.sort_values(by="published_at").reset_index(drop=True)

    # Set up class label
    if target_col == "direction_1d":
        y = df_clean["direction_1d"].astype(int).values
    elif target_col == "direction_4h":
        y = (df_clean["return_4h"] > 0).astype(int).values
    else:
        y = (df_clean["return_1h"] > 0).astype(int).values

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        print(f"Single-class or extremely unbalanced target distribution: {dict(zip(classes, counts))}. Skip training.")
        return None

    # Split dataset chronologically: 80% train, 20% test
    split_idx = int(len(df_clean) * 0.8)
    print(f"Dataset clean rows: {len(df_clean)} | Train rows: {split_idx} | Test rows: {len(df_clean) - split_idx}")

    # Targets for splits
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    # Initialize the payload dictionary
    payload = {
        "model_name": model_name,
        "target_col": target_col,
        "sample_count": len(y),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "class_distribution": {int(k): int(v) for k, v in zip(classes, counts)},
        "phase1": None,
        "phase2": None,
        "phase3": None
    }

    # -------------------------------------------------------------
    # PHASE 1: Baseline (LR + RF)
    # -------------------------------------------------------------
    X_p1 = df_clean[features_p1].values
    X_p1_train = X_p1[:split_idx]
    X_p1_test = X_p1[split_idx:]

    # Train final models on entire dataset for real-time inference
    scaler_p1 = StandardScaler()
    X_p1_scaled = scaler_p1.fit_transform(X_p1)
    lr_p1 = instantiate_model("lr")
    rf_p1 = instantiate_model("rf")
    lr_p1.fit(X_p1_scaled, y)
    rf_p1.fit(X_p1_scaled, y)
    
    metrics_lr_p1 = evaluate_model_time_split("lr", X_p1_train, y_train, X_p1_test, y_test)
    metrics_rf_p1 = evaluate_model_time_split("rf", X_p1_train, y_train, X_p1_test, y_test)
    
    payload["phase1"] = {
        "features": features_p1,
        "scaler": scaler_p1,
        "lr_model": lr_p1,
        "rf_model": rf_p1,
        "metrics": {
            "lr": metrics_lr_p1,
            "rf": {
                **metrics_rf_p1,
                "feature_importances": rf_p1.feature_importances_.tolist()
            }
        }
    }

    # -------------------------------------------------------------
    # PHASE 2: Context (LR + RF)
    # -------------------------------------------------------------
    X_p2 = df_clean[features_p2].values
    X_p2_train = X_p2[:split_idx]
    X_p2_test = X_p2[split_idx:]

    scaler_p2 = StandardScaler()
    X_p2_scaled = scaler_p2.fit_transform(X_p2)
    lr_p2 = instantiate_model("lr")
    rf_p2 = instantiate_model("rf")
    lr_p2.fit(X_p2_scaled, y)
    rf_p2.fit(X_p2_scaled, y)
    
    metrics_lr_p2 = evaluate_model_time_split("lr", X_p2_train, y_train, X_p2_test, y_test)
    metrics_rf_p2 = evaluate_model_time_split("rf", X_p2_train, y_train, X_p2_test, y_test)
    
    payload["phase2"] = {
        "features": features_p2,
        "scaler": scaler_p2,
        "lr_model": lr_p2,
        "rf_model": rf_p2,
        "metrics": {
            "lr": metrics_lr_p2,
            "rf": {
                **metrics_rf_p2,
                "feature_importances": rf_p2.feature_importances_.tolist()
            }
        }
    }

    # -------------------------------------------------------------
    # PHASE 3: Advanced Embeddings + Context (XGBoost + LightGBM)
    # -------------------------------------------------------------
    if model_name == "FinBERT":
        # Extract embeddings
        embeddings_list = []
        for val in df_clean["embedding"]:
            arr = np.frombuffer(val, dtype=np.float32)
            embeddings_list.append(arr)
        X_emb = np.stack(embeddings_list)
        
        # Get metadata scalars
        X_scal = df_clean[features_p3_scalars].values
        
        # Concatenate: 768 dims + 14 scalar dimensions = 782 features
        X_p3 = np.hstack([X_emb, X_scal])
        X_p3_train = X_p3[:split_idx]
        X_p3_test = X_p3[split_idx:]
        
        scaler_p3 = StandardScaler()
        X_p3_scaled = scaler_p3.fit_transform(X_p3)
        xgb_p3 = instantiate_model("xgb")
        lgb_p3 = instantiate_model("lgb")
        xgb_p3.fit(X_p3_scaled, y)
        lgb_p3.fit(X_p3_scaled, y)
        
        metrics_xgb_p3 = evaluate_model_time_split("xgb", X_p3_train, y_train, X_p3_test, y_test)
        metrics_lgb_p3 = evaluate_model_time_split("lgb", X_p3_train, y_train, X_p3_test, y_test)
        
        # Summarize feature importances for UI:
        # Sum importances of the first 768 features to represent the "FinBERT Embeddings" importance
        xgb_importances = xgb_p3.feature_importances_
        xgb_emb_sum = float(np.sum(xgb_importances[:768]))
        xgb_scal_imps = xgb_importances[768:].tolist()
        xgb_summarized_importances = [xgb_emb_sum] + xgb_scal_imps
        
        lgb_importances = lgb_p3.feature_importances_
        lgb_emb_sum = float(np.sum(lgb_importances[:768]))
        lgb_scal_imps = lgb_importances[768:].tolist()
        lgb_summarized_importances = [lgb_emb_sum] + lgb_scal_imps
        
        payload["phase3"] = {
            "scalar_features": features_p3_scalars,
            "scaler": scaler_p3,
            "xgb_model": xgb_p3,
            "lgb_model": lgb_p3,
            "metrics": {
                "xgb": {
                    **metrics_xgb_p3,
                    "feature_importances": xgb_summarized_importances
                },
                "lgb": {
                    **metrics_lgb_p3,
                    "feature_importances": lgb_summarized_importances
                }
            }
        }

    # Save to disk
    model_path = os.path.join(MODEL_DIR, f"market_predictor_{model_name.lower()}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Successfully trained and saved {model_name} models to {model_path}.")
    print(f"Phase 1 RF OS Accuracy: {payload['phase1']['metrics']['rf']['accuracy']:.2%}")
    print(f"Phase 2 RF OS Accuracy: {payload['phase2']['metrics']['rf']['accuracy']:.2%}")
    if payload["phase3"]:
        print(f"Phase 3 XGB OS Accuracy: {payload['phase3']['metrics']['xgb']['accuracy']:.2%}")
    return payload["phase1"]["metrics"]


def predict_latest_news_direction(ticker, model_name="VADER", phase="phase2", sub_model="rf"):
    """
    Make a directional prediction for the selected ticker based on its latest headlines.
    """
    model_path = os.path.join(MODEL_DIR, f"market_predictor_{model_name.lower()}.pkl")
    if not os.path.exists(model_path):
        return {"error": f"Model payload for {model_name} has not been trained yet."}

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    if phase not in payload or payload[phase] is None:
        return {"error": f"{phase.upper()} is not available for model {model_name}."}

    phase_payload = payload[phase]
    
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Fetch latest articles features for the stock
        cursor.execute(
            """
            SELECT f.article_id, f.sentiment_score, f.headline_length, f.hour_of_day,
                   f.news_count_last_1h, f.news_count_last_24h,
                   f.avg_sentiment_last_1h, f.avg_sentiment_last_24h, f.volume_ratio,
                   f.market_return, f.sector_return, f.volatility, f.prev_day_return,
                   f.return_5d, f.return_10d, f.rsi
            FROM feature_store f
            JOIN news_articles n ON f.article_id = n.id
            JOIN stocks s ON n.stock_id = s.id
            WHERE s.ticker = ? AND f.model_name = ?
            ORDER BY n.published_at DESC
            LIMIT 5
            """,
            (ticker.upper(), model_name),
        )
        rows = cursor.fetchall()
        if not rows:
            return {"error": f"No recent headlines found for ticker {ticker}."}

        feats_list = []
        
        if phase == "phase1" or phase == "phase2":
            feature_cols = phase_payload["features"]
            for r in rows:
                feats_list.append([r[col] for col in feature_cols])
        elif phase == "phase3":
            article_ids = [r["article_id"] for r in rows]
            placeholders = ",".join("?" for _ in article_ids)
            cursor.execute(
                f"""
                SELECT article_id, embedding
                FROM headline_embeddings
                WHERE model_name = ? AND article_id IN ({placeholders})
                """,
                [model_name] + article_ids
            )
            emb_rows = {r["article_id"]: r["embedding"] for r in cursor.fetchall()}
            
            scalar_cols = phase_payload["scalar_features"]
            for r in rows:
                art_id = r["article_id"]
                emb_bytes = emb_rows.get(art_id)
                if emb_bytes is None:
                    continue
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                if len(emb) != 768:
                    continue
                scalars = [r[col] for col in scalar_cols]
                feat = np.concatenate([emb, scalars])
                feats_list.append(feat)

        if not feats_list:
            return {"error": f"No complete feature records found for ticker {ticker} in {phase.upper()}."}

        # Average feature vectors
        mean_feats = np.mean(feats_list, axis=0).reshape(1, -1)
        
        # Scale and predict
        scaler = phase_payload["scaler"]
        
        if sub_model == "lr":
            model = phase_payload["lr_model"]
        elif sub_model == "rf":
            model = phase_payload["rf_model"]
        elif sub_model == "xgb":
            model = phase_payload["xgb_model"]
        elif sub_model == "lgb":
            model = phase_payload["lgb_model"]
        else:
            return {"error": f"Model {sub_model} is not supported in {phase.upper()}."}

        scaled_feats = scaler.transform(mean_feats)
        pred_class = int(model.predict(scaled_feats)[0])
        pred_probs = model.predict_proba(scaled_feats)[0].tolist()

        return {
            "ticker": ticker.upper(),
            "model_name": model_name,
            "phase": phase,
            "sub_model": sub_model,
            "prediction": "UP" if pred_class == 1 else "DOWN/FLAT",
            "probability_up": pred_probs[1] if len(pred_probs) > 1 else 0.5,
            "articles_count": len(feats_list)
        }

    except Exception as e:
        return {"error": f"Prediction error: {e}"}
    finally:
        conn.close()


if __name__ == "__main__":
    # Test script by training VADER and FinBERT models
    for m in ["VADER", "FinBERT"]:
        train_and_evaluate_model(m, "direction_4h")
