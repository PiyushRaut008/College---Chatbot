/**
 * chat.js
 * =======
 * Frontend JavaScript for the College Chatbot.
 *
 * Responsibilities:
 *  - Manage conversation state (session ID, message history)
 *  - Send user messages to Flask /chat endpoint via fetch (AJAX)
 *  - Render user and bot message bubbles with animation
 *  - Show typing indicator while waiting for response
 *  - Handle sidebar toggle, quick-topic buttons, clear chat
 *  - Auto-resize textarea, enforce character limit
 *  - Display intent badge + confidence tooltip on bot messages
 *  - Keyboard shortcut: Enter to send, Shift+Enter for new line
 */

'use strict';

/* ════════════════════════════════════════════════════════
   Constants & Configuration
════════════════════════════════════════════════════════ */
const API_URL            = '/chat';
const HEALTH_URL         = '/health';
const CONFIDENCE_LOW     = 0.50;   // below this → show "low confidence" badge style
const MAX_CHARS          = 500;
const TYPING_DELAY_MS    = 600;    // artificial delay so typing indicator shows

// Map intent tags to emoji icons for the intent badge
const INTENT_ICONS = {
  admissions: '📋',
  fees:       '💰',
  courses:    '📚',
  exams:      '📝',
  faculty:    '👩‍🏫',
  events:     '🎉',
  placement:  '💼',
  hostel:     '🏠',
  library:    '📖',
  contact:    '📞',
  greeting:   '👋',
  goodbye:    '👋',
  thanks:     '🙏',
  unknown:    '❓',
};

/* ════════════════════════════════════════════════════════
   DOM Element References
════════════════════════════════════════════════════════ */
const $form          = document.getElementById('inputForm');
const $input         = document.getElementById('userInput');
const $sendBtn       = document.getElementById('sendBtn');
const $messagesArea  = document.getElementById('messagesArea');
const $typingWrapper = document.getElementById('typingWrapper');
const $charCount     = document.getElementById('charCount');
const $clearBtn      = document.getElementById('clearBtn');
const $menuToggle    = document.getElementById('menuToggle');
const $sidebar       = document.getElementById('sidebar');
const $statusDot     = document.getElementById('statusDot');
const $statusText    = document.getElementById('statusText');
const $headerSubtitle= document.getElementById('headerSubtitle');
const $tooltip       = document.getElementById('confidenceTooltip');

/* ════════════════════════════════════════════════════════
   State
════════════════════════════════════════════════════════ */
// Generate a unique session ID per page load for log tracking
const sessionId = 'sess_' + Math.random().toString(36).slice(2, 11);
let   isWaiting = false;

/* ════════════════════════════════════════════════════════
   Initialization
════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  renderWelcome();
  checkHealth();
  $input.focus();
});

/* ════════════════════════════════════════════════════════
   Health Check – marks bot as online/offline
════════════════════════════════════════════════════════ */
async function checkHealth() {
  try {
    const res  = await fetch(HEALTH_URL, { method: 'GET' });
    const data = await res.json();

    if (data.status === 'ok') {
      $statusDot.className    = 'status-dot online';
      $statusText.textContent = 'Online';
      $headerSubtitle.textContent = 'Powered by ML & NLP · Online';
    } else {
      setOffline('Model not loaded – run train.py');
    }
  } catch {
    setOffline('Server unreachable');
  }
}

function setOffline(msg) {
  $statusDot.className    = 'status-dot offline';
  $statusText.textContent = 'Offline';
  $headerSubtitle.textContent = msg;
}

/* ════════════════════════════════════════════════════════
   Welcome Screen
════════════════════════════════════════════════════════ */
function renderWelcome() {
  const chips = [
    { label: '📋 Admissions', query: 'How do I apply for admission?' },
    { label: '💰 Fees',       query: 'What is the fee structure?' },
    { label: '📚 Courses',    query: 'What courses do you offer?' },
    { label: '🎉 Events',     query: 'What events are happening?' },
  ];

  const chipsHTML = chips.map(c =>
    `<button class="welcome-chip" data-query="${escapeHtml(c.query)}" aria-label="Ask: ${escapeHtml(c.query)}">${c.label}</button>`
  ).join('');

  const card = document.createElement('div');
  card.className = 'welcome-card';
  card.id = 'welcomeCard';
  card.innerHTML = `
    <div class="welcome-emoji">🎓</div>
    <h2 class="welcome-title">Hi, I'm CollegeBot!</h2>
    <p class="welcome-subtitle">
      Your AI-powered campus assistant. Ask me anything about admissions,
      fees, courses, exams, faculty, events, and more!
    </p>
    <div class="welcome-chips">${chipsHTML}</div>
  `;

  $messagesArea.appendChild(card);

  // Quick-select chip clicks
  card.querySelectorAll('.welcome-chip').forEach(btn => {
    btn.addEventListener('click', () => sendMessage(btn.dataset.query));
  });
}

function removeWelcomeCard() {
  const card = document.getElementById('welcomeCard');
  if (card) {
    card.style.opacity = '0';
    card.style.transform = 'scale(0.95)';
    card.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
    setTimeout(() => card.remove(), 260);
  }
}

/* ════════════════════════════════════════════════════════
   Sidebar & Mobile Toggle
════════════════════════════════════════════════════════ */
$menuToggle.addEventListener('click', () => {
  const isOpen = $sidebar.classList.toggle('open');
  $menuToggle.setAttribute('aria-expanded', String(isOpen));
});

// Close sidebar when clicking outside on mobile
document.addEventListener('click', (e) => {
  if (window.innerWidth <= 680 &&
      $sidebar.classList.contains('open') &&
      !$sidebar.contains(e.target) &&
      !$menuToggle.contains(e.target)) {
    $sidebar.classList.remove('open');
    $menuToggle.setAttribute('aria-expanded', 'false');
  }
});

// Sidebar topic buttons
document.querySelectorAll('.topic-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    sendMessage(btn.dataset.query);
    // Close sidebar on mobile after selection
    if (window.innerWidth <= 680) {
      $sidebar.classList.remove('open');
      $menuToggle.setAttribute('aria-expanded', 'false');
    }
  });
});

/* ════════════════════════════════════════════════════════
   Clear Chat
════════════════════════════════════════════════════════ */
$clearBtn.addEventListener('click', () => {
  $messagesArea.innerHTML = '';
  renderWelcome();
});

/* ════════════════════════════════════════════════════════
   Input Handling
════════════════════════════════════════════════════════ */
// Auto-resize textarea as user types
$input.addEventListener('input', () => {
  // Reset height first so shrinking works
  $input.style.height = 'auto';
  $input.style.height = Math.min($input.scrollHeight, 140) + 'px';

  // Update character counter
  const len = $input.value.length;
  $charCount.textContent = `${len} / ${MAX_CHARS}`;
  $charCount.className = 'char-count' +
    (len > MAX_CHARS - 50 ? ' warn' : '') +
    (len >= MAX_CHARS     ? ' limit' : '');
});

// Enter to send, Shift+Enter for new line
$input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!isWaiting) $form.dispatchEvent(new Event('submit'));
  }
});

// Form submit handler
$form.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = $input.value.trim();
  if (!text || isWaiting) return;
  sendMessage(text);
});

/* ════════════════════════════════════════════════════════
   Core: Send Message → Fetch → Render
════════════════════════════════════════════════════════ */
async function sendMessage(text) {
  if (!text || isWaiting) return;

  // Remove welcome card on first real message
  removeWelcomeCard();

  // Render user bubble
  renderUserBubble(text);

  // Clear and reset textarea
  $input.value = '';
  $input.style.height = 'auto';
  $charCount.textContent = `0 / ${MAX_CHARS}`;
  $charCount.className = 'char-count';

  // Show typing indicator and lock input
  setWaiting(true);

  // Small artificial delay so typing indicator is visible
  await sleep(TYPING_DELAY_MS);

  try {
    const res  = await fetch(API_URL, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json();

    renderBotBubble(data.response, data.intent, data.confidence);

  } catch (err) {
    console.error('Chat error:', err);
    renderSystemMessage('⚠️ Could not reach the server. Is Flask running?', true);
  } finally {
    setWaiting(false);
    scrollToBottom();
    $input.focus();
  }
}

/* ════════════════════════════════════════════════════════
   Render Helpers
════════════════════════════════════════════════════════ */

/** Render a user message bubble (right-aligned) */
function renderUserBubble(text) {
  const row = document.createElement('div');
  row.className = 'message-row user-row';
  row.innerHTML = `
    <div class="bubble-col">
      <div class="bubble">${escapeHtml(text)}</div>
      <span class="msg-time">${formatTime()}</span>
    </div>
  `;
  $messagesArea.appendChild(row);
  scrollToBottom();
}

/** Render a bot response bubble (left-aligned) with intent badge */
function renderBotBubble(text, intent, confidence) {
  const icon       = INTENT_ICONS[intent] || '🤖';
  const pct        = Math.round((confidence || 0) * 100);
  const badgeClass = confidence < CONFIDENCE_LOW ? 'intent-badge low-conf' : 'intent-badge';
  const badgeId    = 'badge_' + Math.random().toString(36).slice(2, 9);

  const row = document.createElement('div');
  row.className = 'message-row bot-row';
  row.innerHTML = `
    <div class="bot-avatar small-avatar" aria-hidden="true">🤖</div>
    <div class="bubble-col">
      <div class="bubble">${formatBotText(text)}</div>
      <div style="display:flex;align-items:center;gap:0.5rem;">
        <span class="msg-time">${formatTime()}</span>
        <span class="${badgeClass}" id="${badgeId}"
              data-confidence="${pct}"
              data-intent="${escapeHtml(intent || 'unknown')}"
              aria-label="Intent: ${escapeHtml(intent || 'unknown')}, Confidence: ${pct}%">
          ${icon} ${capitalize(intent || 'unknown')}
        </span>
      </div>
    </div>
  `;

  $messagesArea.appendChild(row);

  // Attach tooltip to badge
  const badge = document.getElementById(badgeId);
  badge.addEventListener('mouseenter', (e) => showTooltip(e, `Intent: ${intent} · Confidence: ${pct}%`));
  badge.addEventListener('mousemove',  (e) => moveTooltip(e));
  badge.addEventListener('mouseleave', hideTooltip);
}

/** Render a system notification (e.g., error) */
function renderSystemMessage(text, isError = false) {
  const el = document.createElement('p');
  el.className = `system-msg${isError ? ' error' : ''}`;
  el.textContent = text;
  $messagesArea.appendChild(el);
}

/** Format bot text: handle newline chars and bullet points */
function formatBotText(text) {
  return escapeHtml(text)
    .replace(/\n/g, '<br>')
    .replace(/•/g, '<span aria-hidden="true">•</span>');
}

/* ════════════════════════════════════════════════════════
   Typing Indicator
════════════════════════════════════════════════════════ */
function setWaiting(state) {
  isWaiting = state;
  $sendBtn.disabled = state;
  $input.disabled   = state;

  if (state) {
    $typingWrapper.classList.add('visible');
    $typingWrapper.setAttribute('aria-hidden', 'false');
    scrollToBottom();
  } else {
    $typingWrapper.classList.remove('visible');
    $typingWrapper.setAttribute('aria-hidden', 'true');
  }
}

/* ════════════════════════════════════════════════════════
   Confidence Tooltip
════════════════════════════════════════════════════════ */
function showTooltip(e, text) {
  $tooltip.textContent = text;
  $tooltip.classList.add('visible');
  moveTooltip(e);
}
function moveTooltip(e) {
  $tooltip.style.left = (e.clientX + 12) + 'px';
  $tooltip.style.top  = (e.clientY - 32) + 'px';
}
function hideTooltip() {
  $tooltip.classList.remove('visible');
}

/* ════════════════════════════════════════════════════════
   Utilities
════════════════════════════════════════════════════════ */

/** Scroll the messages container to the bottom */
function scrollToBottom() {
  $messagesArea.scrollTop = $messagesArea.scrollHeight;
}

/** Format current time as HH:MM */
function formatTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Escape HTML to prevent XSS */
function escapeHtml(str) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return String(str || '').replace(/[&<>"']/g, m => map[m]);
}

/** Capitalize first letter */
function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/** Promise-based sleep */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
