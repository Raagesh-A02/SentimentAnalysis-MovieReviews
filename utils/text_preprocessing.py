import nltk
import re
import string

# Ensure the 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure stopwords are also available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and tokenizes input text.
    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Tokenize
    4. Remove stopwords
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Return cleaned text
    return ' '.join(tokens)