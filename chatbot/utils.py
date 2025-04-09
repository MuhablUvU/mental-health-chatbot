from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the input text.
    Returns a sentiment score using VADER.
    """
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score
