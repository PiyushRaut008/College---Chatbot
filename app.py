"""
app.py
======
Flask backend for the College Chatbot.

API Endpoints:
  GET  /          → Serves the chat UI (index.html)
  POST /chat      → Accepts user message, returns chatbot response
  GET  /health    → Health check endpoint
  GET  /admin/logs → View recent query logs (admin only, no auth for demo)

Architecture:
  - ChatbotEngine class encapsulates all ML inference logic
  - Confidence threshold: if < CONFIDENCE_THRESHOLD → fallback response
  - Optional spell correction via pyspellchecker
  - All queries logged to logs/queries.log
"""

import json
import os
import pickle
import random
import sys
import uuid
import traceback

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ─── Project root on sys.path so utils/ is importable ─────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessor import preprocess_text
from utils.logger import log_query, get_recent_logs

# ─── Optional spell correction ─────────────────────────────────────────────────
try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    SPELL_CHECK_ENABLED = True
except ImportError:
    SPELL_CHECK_ENABLED = False

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35   # Below this → show fallback response
MODEL_DIR            = os.path.join(PROJECT_ROOT, "model")
DATA_PATH            = os.path.join(PROJECT_ROOT, "data", "intents.json")

FALLBACK_RESPONSES = [
    "I'm not sure I understood that. Could you rephrase your question?",
    "Hmm, I couldn't find an answer to that. Try asking about admissions, fees, courses, exams, faculty, or events!",
    "That's outside my knowledge area. Please contact us at info@college.edu for detailed assistance.",
    "I didn't quite get that! You can ask me about: 📋 Admissions | 💰 Fees | 📚 Courses | 📝 Exams | 👩‍🏫 Faculty | 🎉 Events",
]

# ──────────────────────────────────────────────────────────────────────────────
# ChatbotEngine – loads models and performs inference
# ──────────────────────────────────────────────────────────────────────────────
class ChatbotEngine:
    """
    Encapsulates model loading and intent prediction.

    Separation of concerns: Flask routes only call engine.get_response();
    all ML logic lives here. This makes swapping models easy (e.g., LLM later).
    """

    def __init__(self):
        self.model         = None
        self.vectorizer    = None
        self.label_encoder = None
        self.intents       = None
        self.is_ready      = False
        self._load_models()
        self._load_intents()

    def _load_models(self):
        """Load the three pickle artifacts saved by train.py."""
        model_path   = os.path.join(MODEL_DIR, "chatbot_model.pkl")
        vector_path  = os.path.join(MODEL_DIR, "vectorizer.pkl")
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

        missing = [p for p in [model_path, vector_path, encoder_path]
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Model files missing: {missing}\n"
                f"Run: python model/train.py   to generate them."
            )

        with open(model_path,   "rb") as f: self.model         = pickle.load(f)
        with open(vector_path,  "rb") as f: self.vectorizer    = pickle.load(f)
        with open(encoder_path, "rb") as f: self.label_encoder = pickle.load(f)

        self.is_ready = True
        print("  ✅ ML models loaded successfully.")

    def _load_intents(self):
        """Load intents.json to retrieve response lists per intent."""
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.intents = json.load(f)

        # Build a quick-lookup dict: tag → list of responses
        self.response_map = {
            intent["tag"]: intent["responses"]
            for intent in self.intents["intents"]
        }
        print(f"  ✅ Intents loaded: {len(self.response_map)} tags")

    def correct_spelling(self, text: str) -> str:
        """
        Apply spell correction word-by-word.
        Only corrects if spell checker is available.
        """
        if not SPELL_CHECK_ENABLED:
            return text
        words = text.split()
        corrected = []
        for word in words:
            correction = _spell.correction(word)
            corrected.append(correction if correction else word)
        corrected_text = " ".join(corrected)
        return corrected_text

    def predict(self, user_message: str) -> dict:
        """
        Full prediction pipeline for a single user message.

        Steps:
          1. Optional spell correction
          2. NLP preprocessing (tokenize, stem, etc.)
          3. TF-IDF vectorization
          4. Model prediction + probability scores
          5. Confidence threshold check
          6. Random response selection from matched intent

        Args:
            user_message (str): Raw text from the user.

        Returns:
            dict with keys: response, intent, confidence, corrected_message
        """
        if not user_message or not user_message.strip():
            return {
                "response": "Please type a message!",
                "intent": "empty",
                "confidence": 0.0,
                "corrected_message": user_message,
            }

        # Step 1 – Spell correction (optional)
        corrected = self.correct_spelling(user_message)

        # Step 2 – Preprocess
        processed = preprocess_text(corrected)

        # Handle case where preprocessing removes all content (e.g. pure numbers)
        if not processed.strip():
            return {
                "response": random.choice(FALLBACK_RESPONSES),
                "intent": "unknown",
                "confidence": 0.0,
                "corrected_message": corrected,
            }

        # Step 3 – Vectorize
        X = self.vectorizer.transform([processed])

        # Step 4 – Predict class and probabilities
        y_pred    = self.model.predict(X)[0]
        y_proba   = self.model.predict_proba(X)[0]
        confidence = float(y_proba.max())
        intent_tag = self.label_encoder.inverse_transform([y_pred])[0]

        # Step 5 – Confidence threshold check
        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "response": random.choice(FALLBACK_RESPONSES),
                "intent": "unknown",
                "confidence": confidence,
                "corrected_message": corrected,
            }

        # Step 6 – Pick a random response from the matched intent
        responses = self.response_map.get(intent_tag, FALLBACK_RESPONSES)
        response  = random.choice(responses)

        return {
            "response": response,
            "intent": intent_tag,
            "confidence": confidence,
            "corrected_message": corrected,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Flask Application
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow cross-origin requests (useful during development)

# Load the chatbot engine at startup (not per-request, for performance)
print("\n  Initializing College Chatbot Engine...")
try:
    chatbot = ChatbotEngine()
except FileNotFoundError as e:
    print(f"\n  ❌ ERROR: {e}")
    chatbot = None


# ─── Route: Home ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


# ─── Route: Chat ───────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Request body (JSON):
        {
            "message": "How do I apply for admission?",
            "session_id": "abc123"   (optional)
        }

    Response (JSON):
        {
            "response": "Admissions are open from June...",
            "intent": "admissions",
            "confidence": 0.92,
            "corrected_message": "How do I apply for admission?"
        }
    """
    if chatbot is None:
        return jsonify({
            "error": "Model not loaded. Run 'python model/train.py' first.",
            "response": "Service is temporarily unavailable. Please try again later.",
        }), 503

    try:
        data        = request.get_json(force=True, silent=True) or {}
        user_message = data.get("message", "").strip()
        session_id   = data.get("session_id", str(uuid.uuid4()))

        if not user_message:
            return jsonify({"response": "Please type a message!", "intent": "empty"}), 400

        # Get prediction from the engine
        result = chatbot.predict(user_message)

        # Log this query for analytics
        log_query(
            user_message=user_message,
            intent=result["intent"],
            confidence=result["confidence"],
            response=result["response"],
            session_id=session_id,
        )

        return jsonify(result), 200

    except Exception:
        traceback.print_exc()
        return jsonify({
            "response": "Something went wrong on my end. Please try again!",
            "intent": "error",
            "confidence": 0.0,
        }), 500


# ─── Route: Health Check ───────────────────────────────────────────────────────
@app.route("/health")
def health():
    """Simple health check for monitoring / deployment."""
    return jsonify({
        "status": "ok" if chatbot and chatbot.is_ready else "degraded",
        "model_loaded": chatbot is not None and chatbot.is_ready,
    })


# ─── Route: Admin Logs ─────────────────────────────────────────────────────────
@app.route("/admin/logs")
def admin_logs():
    """
    View recent query logs.
    In production, protect this with authentication!
    """
    n = request.args.get("n", 20, type=int)
    logs = get_recent_logs(n)
    return jsonify({"count": len(logs), "logs": logs})


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🎓 College Chatbot – Flask Server")
    print("=" * 60)
    print("  Running at: http://127.0.0.1:5000")
    print("  Press Ctrl+C to stop\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,      # Set to False in production
        use_reloader=False,  # Prevents double model loading in debug mode
    )
