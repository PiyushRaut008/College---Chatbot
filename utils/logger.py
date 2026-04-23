"""
logger.py
=========
Logging utility for the College Chatbot.

Logs every user query with:
  - Timestamp (ISO format)
  - User message
  - Detected intent
  - Confidence score
  - Bot response

Log file: logs/queries.log
Format: JSON Lines (one JSON object per line) for easy parsing and analysis.

This design allows future analytics (e.g., most-asked topics, low-confidence
queries that need new training data, etc.).
"""

import json
import logging
import os
from datetime import datetime, timezone

# ─── Ensure the logs directory exists ──────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "queries.log")

# ─── Set up a dedicated file logger ────────────────────────────────────────────
_file_logger = logging.getLogger("chatbot.queries")
_file_logger.setLevel(logging.INFO)

# Only add the handler once (avoids duplicate log lines on module reload)
if not _file_logger.handlers:
    _handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(message)s"))  # raw message only
    _file_logger.addHandler(_handler)
    _file_logger.propagate = False  # don't bubble up to root logger


def log_query(
    user_message: str,
    intent: str,
    confidence: float,
    response: str,
    session_id: str = "anonymous",
) -> None:
    """
    Write a single query event to the log file as a JSON line.

    Args:
        user_message (str): The raw message typed by the user.
        intent      (str): The detected intent tag.
        confidence  (float): Model confidence score (0.0 – 1.0).
        response    (str): The response sent back to the user.
        session_id  (str): Optional session identifier for tracking conversations.
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "user_message": user_message,
        "intent": intent,
        "confidence": round(confidence, 4),
        "response_snippet": response[:120],  # truncate long responses in log
    }
    _file_logger.info(json.dumps(log_entry, ensure_ascii=False))


def get_recent_logs(n: int = 20) -> list:
    """
    Read the last n log entries from the log file.

    Args:
        n (int): Number of recent entries to return.

    Returns:
        list[dict]: List of log entry dictionaries.
    """
    if not os.path.exists(LOG_FILE):
        return []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    for line in reversed(lines[-n:]):
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries
