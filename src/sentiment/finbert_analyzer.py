import sys
import os
import ssl
import sqlite3
import numpy as np
import torch

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import get_db_connection, insert_sentiment_scores

# Bypass SSL verification for model downloads from Hugging Face if needed
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


def get_todo_articles():
    """Fetch all news articles that lack either a FinBERT sentiment score or a FinBERT embedding."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, headline
            FROM news_articles
            WHERE id NOT IN (
                SELECT article_id FROM sentiment_scores WHERE model_name = 'FinBERT'
            )
            OR id NOT IN (
                SELECT article_id FROM headline_embeddings WHERE model_name = 'FinBERT'
            )
            """
        )
        return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching todo articles for FinBERT: {e}")
        return []
    finally:
        conn.close()


def insert_headline_embeddings(embeddings):
    """Insert or replace a list of headline embeddings in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO headline_embeddings (article_id, model_name, embedding)
            VALUES (?, ?, ?)
            """,
            embeddings,
        )
        conn.commit()
        return cursor.rowcount
    except Exception as e:
        conn.rollback()
        print(f"Error batch inserting embeddings: {e}")
        raise e
    finally:
        conn.close()


def run_finbert_pipeline():
    """
    Load the FinBERT model, perform sequence classification and embedding extraction,
    and save both the sentiment scores and the float embeddings to the database.
    """
    print("Starting FinBERT sentiment & embedding analysis pipeline...")

    # 1. Fetch todo articles
    todo = get_todo_articles()
    if not todo:
        print("No articles missing FinBERT scores or embeddings. Pipeline is up-to-date!")
        return 0

    print(f"Found {len(todo)} headlines to process with FinBERT.")
    print("Loading ProsusAI/finbert model & tokenizer from Hugging Face...")

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()  # Put model in evaluation mode
    except Exception as e:
        print(f"Failed to load FinBERT model/tokenizer: {e}")
        return 0

    print("Model loaded. Running forward inference passes...")

    scores_to_insert = []
    embeddings_to_insert = []
    
    # Process each headline
    for article in todo:
        article_id = article["id"]
        headline = article["headline"]

        try:
            # Tokenize headline
            inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True)
            
            # Forward pass with hidden states output enabled
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # 1. Decode Sentiment Scores
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [positive_prob, negative_prob, neutral_prob]
            
            # Label mapping from model config (0: positive, 1: negative, 2: neutral)
            labels = ["positive", "negative", "neutral"]
            best_idx = np.argmax(probs)
            label = labels[best_idx]
            confidence = float(probs[best_idx])

            if label == "positive":
                score = confidence
            elif label == "negative":
                score = -confidence
            else:
                score = 0.0

            scores_to_insert.append(
                {
                    "article_id": article_id,
                    "model_name": "FinBERT",
                    "sentiment_label": label,
                    "sentiment_score": score,
                    "confidence": confidence,
                }
            )

            # 2. Extract [CLS] Token Embedding
            # Hidden states is a tuple of layers; outputs.hidden_states[-1] is the last layer's hidden states
            last_hidden_state = outputs.hidden_states[-1]
            # Extract first token (CLS) of first batch item, size: [768]
            cls_embedding = last_hidden_state[0, 0, :].cpu().numpy()
            
            # Serialize numpy array to raw float32 binary blob
            embedding_bytes = cls_embedding.astype(np.float32).tobytes()
            embeddings_to_insert.append((article_id, "FinBERT", embedding_bytes))

        except Exception as item_err:
            print(f"Error processing headline '{headline}': {item_err}")

    # Save scores and embeddings to database
    inserted_scores = 0
    inserted_embeddings = 0

    if scores_to_insert:
        try:
            inserted_scores = insert_sentiment_scores(scores_to_insert)
            print(f"Successfully stored {inserted_scores} FinBERT sentiment scores in the database.")
        except Exception as e:
            print(f"Failed to store sentiment scores: {e}")

    if embeddings_to_insert:
        try:
            inserted_embeddings = insert_headline_embeddings(embeddings_to_insert)
            print(f"Successfully stored {inserted_embeddings} FinBERT headline embeddings in the database.")
        except Exception as e:
            print(f"Failed to store headline embeddings: {e}")

    return max(inserted_scores, inserted_embeddings)


if __name__ == "__main__":
    run_finbert_pipeline()
