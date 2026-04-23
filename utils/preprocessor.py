"""
preprocessor.py
===============
NLP preprocessing pipeline for the College Chatbot.

Pipeline steps:
  1. Lowercase conversion
  2. Punctuation removal
  3. Tokenization (NLTK word_tokenize)
  4. Stopword removal (English stopwords from NLTK)
  5. Stemming (Porter Stemmer)

Usage:
    from utils.preprocessor import preprocess_text
    clean_text = preprocess_text("How do I apply for Admission?")
    # Returns: "appl admiss"
"""

import re
import string
import nltk

# ─── Download required NLTK resources on first import ──────────────────────────
# These are small and downloaded once; subsequent imports use the cached version.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─── Initialize NLP objects once (module-level, not per-call) ──────────────────
stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))

# Keep question words — they can carry intent meaning
KEEP_WORDS = {"what", "when", "where", "who", "how", "which", "why"}
EFFECTIVE_STOP_WORDS = STOP_WORDS - KEEP_WORDS


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Args:
        text (str): Raw user input string.

    Returns:
        str: Cleaned, stemmed text ready for TF-IDF vectorization.
    """
    # Step 1 – Lowercase
    text = text.lower()

    # Step 2 – Remove URLs and emails (not useful for intent classification)
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)

    # Step 3 – Remove punctuation (keep spaces so tokenizer works correctly)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 4 – Tokenize into individual words
    tokens = word_tokenize(text)

    # Step 5 – Remove stopwords (keeping question words) and non-alphabetic tokens
    tokens = [
        tok for tok in tokens
        if tok.isalpha() and tok not in EFFECTIVE_STOP_WORDS
    ]

    # Step 6 – Stem each token to its root form
    # e.g. "admissions" → "admiss", "applying" → "appli"
    tokens = [stemmer.stem(tok) for tok in tokens]

    # Return as a single space-joined string (expected by TF-IDF vectorizer)
    return " ".join(tokens)


def get_tokens(text: str) -> list:
    """
    Return a list of individual processed tokens for debugging or analysis.

    Args:
        text (str): Raw input.

    Returns:
        list[str]: List of stemmed, cleaned tokens.
    """
    processed = preprocess_text(text)
    return processed.split()
