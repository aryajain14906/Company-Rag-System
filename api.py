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
"""

import os
import uuid
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from extract import RagEngine, SessionState

# ─────────────────────────────────────────────────────────────
# ONE-TIME SETUP (happens once, when the server process starts —
# NOT per request. Parsing PDFs / embedding / loading models per
# request would make every single question take minutes.)
# ─────────────────────────────────────────────────────────────

POLICY_FOLDER = os.getenv("POLICY_FOLDER", "./policies")

print(f"Starting up — loading policy documents from '{POLICY_FOLDER}'...")
engine = RagEngine(POLICY_FOLDER, verbose=False)
print("Engine ready.")

# session_id -> SessionState. Plain dict + a lock since FastAPI can
# handle requests concurrently (e.g. via threadpool for sync code) and
# dict mutation isn't guaranteed atomic across all Python versions/impls.
SESSIONS: dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


def get_session(session_id: str) -> SessionState:
    with _sessions_lock:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = SessionState()
        return SESSIONS[session_id]


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Company Policy Assistant")


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    session_id: str


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
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
<title>Company Policy Assistant</title>
<style>
  body { font-family: -apple-system, Arial, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  #log { border: 1px solid #ddd; border-radius: 8px; padding: 16px; height: 420px; overflow-y: auto; margin-bottom: 12px; }
  .msg { margin-bottom: 14px; line-height: 1.4; }
  .user { font-weight: 600; }
  .bot { color: #222; white-space: pre-wrap; }
  form { display: flex; gap: 8px; }
  input { flex: 1; padding: 10px; font-size: 15px; border: 1px solid #ccc; border-radius: 6px; }
  button { padding: 10px 18px; font-size: 15px; border: none; border-radius: 6px; background: #111; color: #fff; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
</style>
</head>
<body>
  <h2>Company Policy Assistant</h2>
  <div id="log"></div>
  <form id="form">
    <input id="input" autocomplete="off" placeholder="Ask about leave, benefits, conduct..." />
    <button id="send">Send</button>
  </form>

<script>
const sessionId = crypto.randomUUID();
const log = document.getElementById("log");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

function addMessage(text, cls) {
  const div = document.createElement("div");
  div.className = "msg";
  div.innerHTML = `<div class="${cls}">${text}</div>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId })
    });
    const data = await res.json();
    addMessage(res.ok ? data.answer : ("Error: " + (data.detail || "something went wrong")), "bot");
  } catch (err) {
    addMessage("Error: could not reach the server.", "bot");
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