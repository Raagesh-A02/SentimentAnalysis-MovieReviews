# utils/sentiment_predictor.py
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon if not available
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """
    Use VADER to classify text sentiment
    Returns: 1 for Positive, 0 for Negative
    """
    score = sia.polarity_scores(text)['compound']
    return 1 if score >= 0 else 0