"""
train.py
========
Training script for the College Chatbot ML model.

What this script does:
  1. Loads the intents dataset from data/intents.json
  2. Prepares training data using the NLP preprocessor
  3. Converts text to numerical features using TF-IDF
  4. Trains a Logistic Regression classifier (primary)
  5. Evaluates model accuracy and prints a classification report
  6. Saves the trained model, vectorizer, and label encoder as .pkl files

Run this script before starting the Flask app:
    python model/train.py

Output files (in model/):
    chatbot_model.pkl   – Trained Logistic Regression classifier
    vectorizer.pkl      – Fitted TF-IDF vectorizer
    label_encoder.pkl   – Label encoder for intent tags
"""

import json
import os
import pickle
import sys

# ─── Add project root to path so we can import utils ───────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from utils.preprocessor import preprocess_text

# ─── Paths ─────────────────────────────────────────────────────────────────────
INTENTS_PATH = os.path.join(PROJECT_ROOT, "data", "intents.json")
MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(MODEL_DIR, "chatbot_model.pkl")
VECTOR_PATH  = os.path.join(MODEL_DIR, "vectorizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load dataset
# ──────────────────────────────────────────────────────────────────────────────
def load_intents(path: str) -> dict:
    """Load and return the intents JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Prepare training corpus
# ──────────────────────────────────────────────────────────────────────────────
def prepare_training_data(intents: dict) -> tuple[list, list]:
    """
    Convert intents JSON into parallel lists of (texts, labels).

    Each pattern (example query) is:
      - preprocessed through the NLP pipeline
      - paired with its intent tag as the label

    Returns:
        texts  (list[str]): Preprocessed training sentences
        labels (list[str]): Corresponding intent tags
    """
    texts, labels = [], []

    for intent in intents["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            processed = preprocess_text(pattern)
            if processed.strip():  # skip empty strings after preprocessing
                texts.append(processed)
                labels.append(tag)

    print(f"  Total training samples: {len(texts)}")
    print(f"  Unique intents: {len(set(labels))} -> {sorted(set(labels))}")
    return texts, labels


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Build TF-IDF feature vectors
# ──────────────────────────────────────────────────────────────────────────────
def build_vectorizer(texts: list) -> tuple:
    """
    Fit a TF-IDF vectorizer on the training texts.

    TF-IDF (Term Frequency – Inverse Document Frequency) scores each word
    by how important it is to a document relative to the whole corpus.
    This works better than simple Bag of Words for short texts.

    ngram_range=(1,2) captures single words AND two-word phrases, improving
    accuracy for queries like "admission deadline" vs just "admission".

    Returns:
        vectorizer (TfidfVectorizer): Fitted vectorizer
        X          (sparse matrix):  Feature matrix for all training texts
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        max_features=5000,       # vocabulary cap to prevent overfitting
        sublinear_tf=True,       # apply log normalization to TF
        min_df=1,                # include terms appearing in at least 1 doc
    )
    X = vectorizer.fit_transform(texts)
    print(f"  Feature matrix shape: {X.shape}")
    return vectorizer, X


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Train and evaluate classifiers
# ──────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(X, y_encoded: np.ndarray, label_names: list) -> object:
    """
    Train Logistic Regression (primary) and compare with RandomForest.

    We use cross-validation to get a reliable accuracy estimate,
    then train on the full dataset for the saved production model.

    Args:
        X            : TF-IDF feature matrix
        y_encoded    : Encoded integer labels
        label_names  : Original string intent tags

    Returns:
        best_model: The trained production classifier
    """
    # ── 4a. Cross-validation comparison ─────────────────────────────────────
    print("\n  ── Cross-Validation (5-fold) ──")

    lr_model = LogisticRegression(
        max_iter=1000,
        C=10,             # regularization strength (higher = less regularization)
        solver="lbfgs",
        random_state=42,
    )
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    for name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
        cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring="accuracy")
        print(f"  {name}: mean={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # ── 4b. Train-test split evaluation ─────────────────────────────────────
    # Only split if we have enough data; otherwise train on full set
    if X.shape[0] > 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        print(f"\n  Train/Test Split Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_names))
    else:
        print("  (Too few samples for train/test split; using cross-val only)")

    # ── 4c. Final model: train on full dataset ───────────────────────────────
    print("  Training final model on full dataset...")
    final_model = LogisticRegression(
        max_iter=1000,
        C=10,
        solver="lbfgs",
        random_state=42,
    )
    final_model.fit(X, y_encoded)
    full_acc = accuracy_score(y_encoded, final_model.predict(X))
    print(f"  Full dataset accuracy: {full_acc:.4f}")

    return final_model


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Save artifacts
# ──────────────────────────────────────────────────────────────────────────────
def save_artifacts(model, vectorizer, label_encoder) -> None:
    """Serialize and save the three model artifacts using pickle."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTOR_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"\n  [OK] Saved model      -> {MODEL_PATH}")
    print(f"  [OK] Saved vectorizer -> {VECTOR_PATH}")
    print(f"  [OK] Saved encoder    -> {ENCODER_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  College Chatbot – Model Training")
    print("=" * 60)

    # 1. Load intents
    print("\n[1/5] Loading intents dataset...")
    intents = load_intents(INTENTS_PATH)

    # 2. Prepare data
    print("\n[2/5] Preparing training data...")
    texts, labels = prepare_training_data(intents)

    # 3. Encode labels
    print("\n[3/5] Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    label_names = list(label_encoder.classes_)
    print(f"  Encoded {len(label_names)} intent classes")

    # 4. Build TF-IDF features
    print("\n[4/5] Building TF-IDF feature vectors...")
    vectorizer, X = build_vectorizer(texts)

    # 5. Train and evaluate
    print("\n[5/5] Training model...")
    model = train_and_evaluate(X, y_encoded, label_names)

    # 6. Save
    print("\n[Saving] Saving model artifacts...")
    save_artifacts(model, vectorizer, label_encoder)

    print("\n" + "=" * 60)
    print("  Training complete! You can now run: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
