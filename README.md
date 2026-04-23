# 🎓 College Chatbot — Complete Setup & Run Guide

A production-quality **College Information Chatbot** built with:
- **ML**: scikit-learn (Logistic Regression + TF-IDF)
- **NLP**: NLTK (tokenization, stemming, stopword removal)
- **Backend**: Python Flask
- **Frontend**: HTML + CSS + Vanilla JS

---

## 📁 Folder Structure

```
PROJECT/
├── data/
│   └── intents.json            ← Intent dataset (patterns + responses)
├── model/
│   ├── train.py                ← ML training script
│   ├── chatbot_model.pkl       ← Saved classifier (generated after training)
│   ├── vectorizer.pkl          ← Saved TF-IDF vectorizer
│   └── label_encoder.pkl       ← Saved label encoder
├── static/
│   ├── css/style.css           ← Dark-mode chat UI styles
│   └── js/chat.js              ← Frontend AJAX / fetch logic
├── templates/
│   └── index.html              ← Chat page
├── utils/
│   ├── preprocessor.py         ← NLP pipeline
│   └── logger.py               ← Query logging utility
├── logs/
│   └── queries.log             ← Auto-created at runtime
├── app.py                      ← Flask application (entry point)
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

---

## ⚙️ Prerequisites

- Python 3.9 or higher
- pip (comes with Python)
- Git (optional)

---

## 🚀 Step-by-Step Setup

### Step 1 — Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Download NLTK data

The training script does this automatically, but you can also run manually:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### Step 4 — Train the ML model

```bash
python model/train.py
```

**Expected output:**
```
============================================================
  College Chatbot – Model Training
============================================================
[1/5] Loading intents dataset...
[2/5] Preparing training data...
  Total training samples: 130+
  Unique intents: 13
[3/5] Encoding labels...
[4/5] Building TF-IDF feature vectors...
[5/5] Training model...
  Logistic Regression: mean=0.98xx ± 0.0xxx
  ...
  ✅ Saved model      → model/chatbot_model.pkl
  ✅ Saved vectorizer → model/vectorizer.pkl
  ✅ Saved encoder    → model/label_encoder.pkl

Training complete! You can now run: python app.py
```

### Step 5 — Start the Flask server

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

---

## 🤖 API Reference

### `POST /chat`

**Request:**
```json
{
  "message": "How do I apply for admission?",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "response": "Admissions are open from June 1st to July 31st...",
  "intent": "admissions",
  "confidence": 0.9287,
  "corrected_message": "How do I apply for admission?"
}
```

### `GET /health`
Returns `{ "status": "ok", "model_loaded": true }` when the model is running.

### `GET /admin/logs?n=20`
Returns the last 20 query log entries (JSON Lines format).

---

## 🧠 How the ML Pipeline Works

```
User Input
    ↓
Spell Correction (pyspellchecker)
    ↓
NLP Preprocessing (NLTK)
  • Lowercase
  • Remove punctuation
  • Tokenize
  • Remove stopwords
  • Porter Stemmer
    ↓
TF-IDF Vectorization (ngram 1-2, 5000 features)
    ↓
Logistic Regression Classifier
    ↓
Confidence Check (threshold = 0.35)
  • High → pick random response for matched intent
  • Low  → fallback response
    ↓
Response + Logging
```

---

## 🗂️ Supported Intents

| Intent      | Example Query                          |
|-------------|----------------------------------------|
| greeting    | "Hello", "Hi there"                    |
| admissions  | "How do I apply for admission?"        |
| fees        | "What is the fee structure?"           |
| courses     | "What programs are available?"         |
| exams       | "When is the semester exam?"           |
| faculty     | "Who is the HOD of CSE?"              |
| events      | "Any upcoming hackathons?"             |
| placement   | "What is the average salary package?"  |
| hostel      | "Is hostel available for girls?"       |
| library     | "What are library timings?"            |
| contact     | "What is the college phone number?"    |
| thanks      | "Thank you so much"                    |
| goodbye     | "Bye, see you later"                   |

---

## ➕ Adding New Intents

1. Open `data/intents.json`
2. Add a new object to the `"intents"` array:

```json
{
  "tag": "scholarships",
  "patterns": [
    "Tell me about scholarships",
    "Is there any financial aid?",
    "Merit scholarship details"
  ],
  "responses": [
    "We offer merit scholarships for students scoring above 90%...",
    "Scholarships available: Merit (up to 50%), SC/ST, Sports..."
  ]
}
```

3. Re-train: `python model/train.py`
4. Restart Flask: `python app.py`

---

## 📋 Query Logs

Every conversation is logged to `logs/queries.log` in JSON Lines format:

```json
{"timestamp": "2024-10-15T09:30:00Z", "session_id": "sess_abc123", "user_message": "What are fees?", "intent": "fees", "confidence": 0.9231, "response_snippet": "Our fee structure..."}
```

View recent logs via API: `GET /admin/logs?n=20`

---

## 🔮 Future Improvements

| Feature | Description |
|---------|-------------|
| 🗄️ Database | Move intents + logs to SQLite/PostgreSQL |
| 🔐 Auth | Add JWT-based auth for admin routes |
| 🤖 LLM Fallback | Connect GPT/Gemini for unknown queries |
| 📊 Analytics Dashboard | Visualize intent distribution, popular queries |
| 🌐 Multi-language | Add Hindi/regional language support |
| 🔈 Voice Input | Web Speech API integration |
| 📱 PWA | Service worker for offline access |
| 🐳 Docker | Containerize for easy deployment |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: chatbot_model.pkl` | Run `python model/train.py` first |
| `ModuleNotFoundError: nltk` | Run `pip install -r requirements.txt` |
| Port 5000 in use | Run `python app.py` and change port in `app.py` |
| NLTK data missing | Run `python -c "import nltk; nltk.download('all')"` |
| Low accuracy | Add more pattern examples to `data/intents.json` and retrain |

---

## 📜 License

MIT License — Free for academic and personal use.
