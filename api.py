"""
Web wrapper around policy_rag.RagEngine.
=========================================
Lets multiple people ask questions over HTTP instead of one person
typing into a terminal. Only ever returns the final answer text to the
client — no retrieval debug info, no chunk contents, no section labels.

Run:
    export POLICY_FOLDER=./policies      # folder of your policy PDFs
    uvicorn api:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in a browser, or POST to /ask:
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d '{"question": "how many sick leaves do I get", "session_id": "abc123"}'

SESSION HANDLING
-----------------
Each conversation needs its own SessionState (conversation_history +
last_section_asked) so two different people's questions/follow-ups
don't bleed into each other. The client is responsible for sending a
stable `session_id` with every request (the bundled HTML page generates
one random ID per browser tab and reuses it for the whole chat). If no
session_id is sent, one request = one throwaway session with no memory
of anything before it.

Sessions are kept in memory only (a plain dict). Restarting the server
wipes all conversation history — fine for a small internal tool, not
fine if you need durability; swap SESSIONS for a real store (Redis,
a database) if that ever matters.

STARTUP
-------
The FastAPI app object is created — and uvicorn binds $PORT — BEFORE
the RagEngine (PDF parsing, model downloads, embedding all chunks) is
built. That heavy work happens in a background thread kicked off by
the "startup" event instead. This matters on platforms like Render
that scan for an open port shortly after the container starts: if the
engine load blocked the app/uvicorn from existing yet, the port scan
would time out and fail the deploy even though the app would have come
up fine a few seconds later. Requests to /ask return a 503 with a
clear message until the engine finishes loading.
"""

import os
import uuid
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from extract import RagEngine, SessionState

POLICY_FOLDER = os.getenv("POLICY_FOLDER", "./policies")

# ─────────────────────────────────────────────────────────────
# APP — created immediately so uvicorn can bind $PORT right away.
# The heavy RagEngine load happens in a background thread after
# startup, not before.
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Company Policy Assistant")

_engine_lock = threading.Lock()
engine: RagEngine | None = None
engine_error: str | None = None

# session_id -> SessionState. Plain dict + a lock since FastAPI can
# handle requests concurrently (e.g. via threadpool for sync code) and
# dict mutation isn't guaranteed atomic across all Python versions/impls.
SESSIONS: dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


def _load_engine() -> None:
    global engine, engine_error
    try:
        print(f"Loading policy documents from '{POLICY_FOLDER}'...")
        e = RagEngine(POLICY_FOLDER, verbose=False)
        with _engine_lock:
            engine = e
        print("Engine ready.")
    except Exception as exc:
        with _engine_lock:
            engine_error = str(exc)
        print(f"Engine failed to load: {exc}")


@app.on_event("startup")
def start_background_load() -> None:
    threading.Thread(target=_load_engine, daemon=True).start()


def get_session(session_id: str) -> SessionState:
    with _sessions_lock:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = SessionState()
        return SESSIONS[session_id]


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    session_id: str


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if engine is None:
        detail = (
            "Still starting up, please retry in a moment."
            if not engine_error
            else f"Engine failed to initialize: {engine_error}"
        )
        raise HTTPException(status_code=503, detail=detail)

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    session_id = req.session_id or str(uuid.uuid4())
    session = get_session(session_id)

    try:
        answer = engine.answer(question, session)
    except Exception as e:
        # Anything unexpected (e.g. OpenRouter down, a bad response
        # shape) becomes a clean error for the client instead of a
        # raw traceback. ask_llm() itself already returns "" on
        # failure rather than raising, so this is a last-resort net.
        raise HTTPException(status_code=502, detail=f"Failed to generate an answer: {e}")

    return AskResponse(answer=answer, session_id=session_id)


@app.get("/", response_class=HTMLResponse)
@app.head("/")
def home():
    return _CHAT_PAGE


# Minimal single-file chat UI. No frameworks, no build step — just
# enough to type a question and see the answer. session_id is generated
# once per page load (crypto.randomUUID) and reused for every message
# in that tab, so follow-up questions ("what about them?") work.
_CHAT_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Company Policy Assistant</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

  :root {
    --ink: #1c2430;
    --paper: #f6f3ec;
    --paper-raised: #ffffff;
    --line: #ddd6c7;
    --stamp: #8a2e2e;
    --stamp-dim: #8a2e2e22;
    --muted: #6b6558;
  }

  * { box-sizing: border-box; }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--paper);
    color: var(--ink);
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }

  .card {
    width: 100%;
    max-width: 640px;
    background: var(--paper-raised);
    border: 1px solid var(--line);
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(28,36,48,0.06);
    overflow: hidden;
  }

  header {
    padding: 28px 28px 20px;
    border-bottom: 1px solid var(--line);
    position: relative;
  }

  header::after {
    content: "";
    position: absolute;
    top: 24px;
    right: 28px;
    width: 34px;
    height: 34px;
    border: 2px solid var(--stamp);
    border-radius: 50%;
    opacity: 0.35;
  }

  h1 {
    font-family: 'Fraunces', serif;
    font-size: 22px;
    font-weight: 600;
    margin: 0 0 4px;
    letter-spacing: -0.01em;
  }

  .subtitle {
    font-size: 13px;
    color: var(--muted);
    margin: 0;
  }

  #log {
    height: 440px;
    overflow-y: auto;
    padding: 20px 28px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  #log::-webkit-scrollbar { width: 6px; }
  #log::-webkit-scrollbar-thumb { background: var(--line); border-radius: 3px; }

  .msg {
    max-width: 82%;
    padding: 11px 15px;
    border-radius: 10px;
    font-size: 14.5px;
    line-height: 1.5;
    animation: rise 0.28s ease;
    white-space: pre-wrap;
  }

  @keyframes rise {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .user {
    align-self: flex-end;
    background: var(--ink);
    color: var(--paper);
    border-bottom-right-radius: 3px;
  }

  .bot {
    align-self: flex-start;
    background: #efeade;
    color: var(--ink);
    border-bottom-left-radius: 3px;
    border-left: 2px solid var(--stamp-dim);
  }

  .bot.error {
    border-left-color: var(--stamp);
    color: #6e2323;
  }

  .typing {
    align-self: flex-start;
    display: flex;
    gap: 4px;
    padding: 13px 16px;
  }

  .typing span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--muted);
    animation: bounce 1.1s infinite;
  }

  .typing span:nth-child(2) { animation-delay: 0.15s; }
  .typing span:nth-child(3) { animation-delay: 0.3s; }

  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-4px); opacity: 1; }
  }

  form {
    display: flex;
    gap: 10px;
    padding: 18px 28px 24px;
    border-top: 1px solid var(--line);
  }

  input {
    flex: 1;
    padding: 12px 14px;
    font-size: 14.5px;
    font-family: inherit;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--paper);
    color: var(--ink);
    outline: none;
    transition: border-color 0.15s;
  }

  input:focus {
    border-color: var(--stamp);
  }

  button {
    padding: 0 20px;
    font-size: 14px;
    font-weight: 500;
    font-family: inherit;
    border: none;
    border-radius: 6px;
    background: var(--stamp);
    color: #fff;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
  }

  button:hover:not(:disabled) { opacity: 0.9; }
  button:active:not(:disabled) { transform: scale(0.97); }
  button:disabled { opacity: 0.4; cursor: default; }

  .empty {
    align-self: center;
    margin: auto;
    text-align: center;
    color: var(--muted);
    font-size: 13.5px;
    max-width: 260px;
    line-height: 1.6;
  }
</style>
</head>
<body>
  <div class="card">
    <header>
      <h1>Company Policy Assistant</h1>
      <p class="subtitle">Answers sourced from your company's policy documents.</p>
    </header>
    <div id="log">
      <div class="empty" id="empty-state">Ask about leave, benefits, conduct, or any company policy — I'll answer strictly from the documents on file.</div>
    </div>
    <form id="form">
      <input id="input" autocomplete="off" placeholder="Ask a question..." />
      <button id="send">Send</button>
    </form>
  </div>

<script>
const sessionId = crypto.randomUUID();
const log = document.getElementById("log");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const emptyState = document.getElementById("empty-state");

function addMessage(text, cls) {
  if (emptyState) emptyState.remove();
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function addTyping() {
  const div = document.createElement("div");
  div.className = "typing";
  div.innerHTML = "<span></span><span></span><span></span>";
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;

  const typingEl = addTyping();

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId })
    });
    const data = await res.json();
    typingEl.remove();
    if (res.ok) {
      addMessage(data.answer, "bot");
    } else {
      addMessage(data.detail || "Something went wrong.", "bot error");
    }
  } catch (err) {
    typingEl.remove();
    addMessage("Could not reach the server. Please try again.", "bot error");
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>
"""