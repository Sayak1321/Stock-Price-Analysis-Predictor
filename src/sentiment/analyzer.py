import sys
import os
import ssl
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.database.db import get_unscored_headlines, insert_sentiment_scores

# Bypass SSL verification for NLTK downloads if needed
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download("vader_lexicon", quiet=True)

# Initialize the VADER sentiment intensity analyzer
_analyzer = SentimentIntensityAnalyzer()


def analyze_headline(headline):
    """
    Analyze the sentiment of a headline.
    Returns:
        score: float (compound score from -1.0 to 1.0)
        label: str ('positive', 'neutral', 'negative')
        confidence: float (absolute compound score as proxy for sentiment strength)
    """
    scores = _analyzer.polarity_scores(headline)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # Compound is between -1.0 and 1.0. The absolute value is the sentiment strength (confidence proxy)
    confidence = abs(compound)

    return compound, label, confidence


def run_sentiment_pipeline():
    """
    Fetch headlines that lack sentiment scores, analyze them, and store
    the resulting scores in the database.
    """
    print("Starting VADER sentiment analysis pipeline...")

    # 1. Get articles without sentiment scores
    unscored = get_unscored_headlines()
    if not unscored:
        print("No unscored news articles found in the database. Sentiment is up-to-date!")
        return 0

    print(f"Found {len(unscored)} unscored news articles.")

    # 2. Score articles
    scores_to_insert = []
    for article in unscored:
        article_id = article["id"]
        headline = article["headline"]

        score, label, confidence = analyze_headline(headline)

        scores_to_insert.append(
            {
                "article_id": article_id,
                "model_name": "VADER",
                "sentiment_label": label,
                "sentiment_score": score,
                "confidence": confidence,
            }
        )

    # 3. Save scores to DB
    inserted_count = insert_sentiment_scores(scores_to_insert)
    print(f"Successfully processed and stored {inserted_count} sentiment scores in the database.")
    return inserted_count


if __name__ == "__main__":
    run_sentiment_pipeline()
