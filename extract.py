"""
Company Policy RAG Engine (multi-file) — server-ready version
================================================================
Same retrieval/answering logic as the original CLI script, restructured
so it can be used by a web server with multiple concurrent users:

  - RagEngine: everything that's expensive and shared (parsed chunks,
    embeddings, BM25 indexes, models, persona name). Built ONCE at
    server startup, reused for every request.

  - SessionState: everything that's per-conversation (history, last
    section asked). One instance per user/session_id, created fresh
    by the caller (see api.py) instead of living in module globals.
    This is what makes concurrent users safe — two people's follow-up
    questions can no longer bleed into each other the way they would
    with module-level globals.

  - engine.answer(question, session) replaces the old CLI while-loop
    body. Same routing / rewrite / retrieval / verification pipeline,
    just reading and writing `session` instead of module globals.

LLM BACKEND: OpenRouter instead of local Ollama, so this doesn't
require every laptop to run its own model server. Only the server
process needs OPENROUTER_API_KEY set — clients just hit the HTTP API.

PDF PARSING: pdfplumber instead of unstructured.partition_pdf. This
drops the entire onnxruntime / opencv / spacy / timm / transformers
layout-detection dependency chain that unstructured pulls in, which was
the main cause of out-of-memory crashes on Render's 512MB free tier.
Since policy PDFs here are plain single-column text (no scans, no
complex layouts), pdfplumber gives equivalent results at a fraction of
the memory. See extract_elements() below for details on the title
detection heuristic that replaces unstructured's ML layout model.
"""

import os
import re
import json
import glob
import time
import pickle
import hashlib
import statistics
from dataclasses import dataclass, field
from collections import Counter

import numpy as np
import requests
import pdfplumber

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

EMBED_MODEL_NAME  = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# OpenRouter model fallback list. Instead of pinning to ONE free model
# (which fails entirely if that model's current provider is down or
# rate-limited — as we saw happen with Venice on both Qwen and Llama),
# this gives OpenRouter a priority-ordered list of free models to try.
# If the first one's provider is down/rate-limited/refuses, OpenRouter
# automatically tries the next one in the list — no code-level retry
# or manual model-switching needed.
#
# Configure via OPENROUTER_MODELS as a comma-separated list, e.g.:
#   OPENROUTER_MODELS=qwen/qwen3-next-80b-a3b-instruct:free,meta-llama/llama-3.3-70b-instruct:free
# Falls back to a sensible built-in list of free models if not set.
#
# SAFETY DEFAULT: every entry defaults to a ":free" model, so forgetting
# to set this can never silently start billing you. If you want a paid
# model anywhere in the list, add it explicitly yourself.
_DEFAULT_FREE_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    # OpenRouter's own auto-router for free models. Unlike a hardcoded
    # slug (e.g. the DeepSeek one this used to be — DeepSeek currently
    # has ZERO free models on OpenRouter, which is what caused a 404
    # "model not found"), this always resolves to whatever free model
    # is currently live, so it can't go stale as free-tier availability
    # rotates. Kept last since a specific model is still preferred when
    # available.
    "openrouter/free",
]

_env_models = os.getenv("OPENROUTER_MODELS", "")
LLM_MODELS = (
    [m.strip() for m in _env_models.split(",") if m.strip()]
    if _env_models
    else _DEFAULT_FREE_MODELS
)

# Kept for anything that still refers to a single "current model" name
# (e.g. log messages) — always the first/primary one in the list.
LLM_MODEL = LLM_MODELS[0]

if not _env_models:
    print(
        f"[policy_rag] OPENROUTER_MODELS not set — defaulting to free "
        f"model fallback list: {LLM_MODELS}. Set OPENROUTER_MODELS "
        f"(comma-separated) to customize."
    )
non_free = [m for m in LLM_MODELS if ":free" not in m and m != "openrouter/free"]
if non_free:
    print(
        f"[policy_rag] WARNING: OPENROUTER_MODELS contains non-free "
        f"entries (no ':free' suffix): {non_free}. These will incur "
        f"OpenRouter charges."
    )

MAX_CHARS         = 400
OVERLAP_ELEMENTS  = 2
TOP_K_RETRIEVAL   = 10
TOP_K_FINAL       = 10
THRESHOLD         = 0.1
DENSE_ONLY_FLOOR  = 0.55
MAX_HISTORY       = 5
MAX_SUBQUERIES    = 2
SOFT_SECTION_BOOST = 0.15
DEFAULT_ASSISTANT_NAME = "Company HR Assistant"
MAX_SECTION_CONTEXT_CHARS = 6000
CHUNKING_VERSION  = 5  # bumped: pdfplumber chunks are not guaranteed
                       # identical to old unstructured-based cache entries

# ─────────────────────────────────────────────────────────────
# OPENROUTER REST CLIENT
# ─────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def ask_llm(prompt: str, _retries: int = 3) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it on the server before "
            "starting the app, e.g.: export OPENROUTER_API_KEY=sk-or-..."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "Company Policy Assistant"),
    }
    payload = {
        "models": LLM_MODELS,
        "messages": [{"role": "user", "content": prompt}],
        # Venice has been consistently returning immediate 429s for every
        # free model we've tried routed through it (confirmed via
        # OpenRouter's own Upstream Requests log — different models, same
        # provider, same instant rejection). Excluding it here so
        # OpenRouter routes to any other provider serving each model.
        "provider": {
            "ignore": ["Venice"]
        },
    }

    last_error = None
    for attempt in range(_retries + 1):
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 429:
            # Rate limited — short backoff then retry, instead of
            # immediately failing the whole answer. Free-tier
            # OpenRouter models can hit per-minute caps easily since
            # one user question triggers several ask_llm() calls.
            last_error = f"429 rate limited (attempt {attempt + 1})"
            print(f"[ask_llm] {last_error} — backing off before retry")
            if attempt < _retries:
                time.sleep(min(5 * (2 ** attempt), 30))  # 5s, 10s, 20s
            continue

        response.raise_for_status()
        data = response.json()

        if "error" in data:
            last_error = data["error"]
            print(f"[ask_llm] OpenRouter error (attempt {attempt + 1}): {last_error}")
            continue

        choices = data.get("choices") or []
        if not choices:
            last_error = f"empty 'choices' in response: {data}"
            print(f"[ask_llm] {last_error} (attempt {attempt + 1})")
            continue

        content = choices[0].get("message", {}).get("content")
        if content is None or (isinstance(content, str) and not content.strip()):
            last_error = f"null/empty content in response: {data}"
            print(f"[ask_llm] {last_error} (attempt {attempt + 1})")
            continue

        return content

    raise RuntimeError(
        f"OpenRouter returned no usable content after {_retries + 1} attempts "
        f"across models={LLM_MODELS}. Last issue: {last_error}. "
        f"All models in the fallback list may be under load — try again "
        f"shortly, or add more entries to OPENROUTER_MODELS."
    )


# ─────────────────────────────────────────────────────────────
# 1. FOLDER INGESTION  (one section per file, via filename)
# ─────────────────────────────────────────────────────────────

def filename_to_section(pdf_path: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    stem = re.sub(r'[_\-]+', ' ', stem).strip()
    return stem.title() if stem else "General"


def find_policy_pdfs(folder_path: str) -> list[str]:
    pdfs = sorted(glob.glob(os.path.join(folder_path, "*.pdf")) +
                  glob.glob(os.path.join(folder_path, "*.PDF")))
    seen, unique = set(), []
    for p in pdfs:
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ─────────────────────────────────────────────────────────────
# 1b. PDF ELEMENT EXTRACTION (pdfplumber-based)
# ─────────────────────────────────────────────────────────────

_SENTENCE_END_RE = re.compile(r'[.,;:]\s*$')
_MAX_TITLE_WORDS = 12


class _Element:
    """Mimics the small surface area of unstructured's element objects
    that chunk_file() actually uses: str(el) and el.category."""
    __slots__ = ("text", "category")

    def __init__(self, text: str, category: str):
        self.text = text
        self.category = category

    def __str__(self) -> str:
        return self.text


def _looks_like_title(line: str, median_size: float | None, line_size: float | None) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    word_count = len(stripped.split())
    if word_count == 0 or word_count > _MAX_TITLE_WORDS:
        return False

    if _SENTENCE_END_RE.search(stripped):
        return False

    is_all_caps = stripped.upper() == stripped and any(c.isalpha() for c in stripped)
    is_title_case = stripped.istitle()
    is_larger_font = (
        median_size is not None and line_size is not None
        and line_size > median_size * 1.15
    )

    return is_all_caps or is_title_case or is_larger_font


def extract_elements(pdf_path: str) -> list[_Element]:
    """
    Extracts text lines from a PDF, tagging each as "Title" or "Text".

    Replaces unstructured.partition_pdf's ML layout model with a
    heuristic (short line, no sentence punctuation, ALL CAPS / Title
    Case / larger-than-median font size). This is a heuristic, not a
    guarantee — fine for the plain single-column text policy PDFs this
    app is built for. `subsection` (derived from Title lines) is stored
    on each chunk but isn't read anywhere else in retrieval/answering,
    so any occasional misfire here is cosmetic only.
    """
    elements: list[_Element] = []

    with pdfplumber.open(pdf_path) as pdf:
        all_sizes = []
        for page in pdf.pages:
            for ch in page.chars:
                size = ch.get("size")
                if size:
                    all_sizes.append(size)
        median_size = statistics.median(all_sizes) if all_sizes else None

        for page in pdf.pages:
            lines = page.extract_text_lines(layout=False) or []

            for line in lines:
                text = line.get("text", "").strip()
                if not text:
                    continue

                chars = line.get("chars", [])
                sizes = [c["size"] for c in chars if c.get("size")]
                line_size = (sum(sizes) / len(sizes)) if sizes else None

                category = "Title" if _looks_like_title(text, median_size, line_size) else "Text"
                elements.append(_Element(text, category))

    return [el for el in elements if str(el).strip()]


# ─────────────────────────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_long_element(text: str, max_chars: int) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    pieces, current, current_len = [], [], 0
    for s in sentences:
        if current and current_len + len(s) > max_chars:
            pieces.append(" ".join(current))
            current, current_len = [], 0
        current.append(s)
        current_len += len(s)
    if current:
        pieces.append(" ".join(current))
    return pieces or [text]


def chunk_file(elements: list, section: str, max_chars: int = MAX_CHARS,
              overlap: int = OVERLAP_ELEMENTS) -> list[dict]:
    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0
    current_subsection = None

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append({
                "text": " ".join(current),
                "section": section,
                "subsection": current_subsection
            })
            current = current[-overlap:] if overlap else []
            current_len = sum(len(c) for c in current)

    for el in elements:
        text = str(el).strip()
        if not text:
            continue

        category = getattr(el, "category", None)
        if category == "Title":
            flush()
            current_subsection = text
            current.append(text)
            current_len += len(text)
            continue

        pieces = (
            [text] if len(text) <= max_chars
            else _split_long_element(text, max_chars)
        )

        for piece in pieces:
            if current and current_len + len(piece) > max_chars:
                flush()
            current.append(piece)
            current_len += len(piece)

    flush()
    return chunks


def build_all_chunks(folder_path: str) -> list[dict]:
    pdf_paths = find_policy_pdfs(folder_path)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {folder_path}")

    all_chunks = []
    for pdf_path in pdf_paths:
        section = filename_to_section(pdf_path)
        elements = extract_elements(pdf_path)
        all_chunks.extend(chunk_file(elements, section))
    return all_chunks


# ─────────────────────────────────────────────────────────────
# 3. EMBEDDINGS
# ─────────────────────────────────────────────────────────────

def _folder_signature(folder_path: str) -> str:
    pdf_paths = find_policy_pdfs(folder_path)
    parts = []
    for p in pdf_paths:
        stat = os.stat(p)
        parts.append(f"{os.path.basename(p)}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha256("|".join(sorted(parts)).encode()).hexdigest()


def load_or_create_embeddings(chunks: list[dict],
                               embed_model: SentenceTransformer,
                               cache_path: str,
                               folder_signature: str) -> list[dict]:
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)

        valid = (
            isinstance(cached, dict)
            and cached.get("version") == CHUNKING_VERSION
            and cached.get("signature") == folder_signature
            and cached.get("chunks") is not None
        )
        if valid:
            print(f"[embeddings] Cache hit — reusing {len(cached['chunks'])} cached embeddings.")
            return cached["chunks"]

    # Batched encoding instead of one encode() call per chunk. Calling
    # embed_model.encode() once per chunk in a Python loop is *dramatically*
    # slower on CPU than one batched call across all chunks at once (each
    # call has fixed overhead, and batching lets the model vectorize across
    # samples). For a few hundred chunks this can be the difference between
    # seconds and many minutes.
    #
    # BUT: a larger batch_size also means more sequences processed (and
    # padded to the same length) simultaneously, which spikes peak memory.
    # Render's free tier caps a container at 512MB total RAM, shared with
    # the embedding model + reranker already loaded in memory — a batch
    # size of 64 was enough to push it over that limit and crash the whole
    # process with an OOM kill (no Python exception, no traceback — the
    # container is just killed by the platform).
    #
    # EMBED_BATCH_SIZE lets you tune this without another code change:
    # lower it further (e.g. 4) if you still see "Out of memory", raise it
    # (e.g. 32-64) if you move to a plan with more RAM and want more speed.
    batch_size = int(os.getenv("EMBED_BATCH_SIZE", "8"))
    print(
        f"[embeddings] No valid cache — encoding {len(chunks)} chunks "
        f"(batch_size={batch_size})..."
    )
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_model.encode(
        texts,
        convert_to_tensor=False,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    print(f"[embeddings] Finished encoding {len(chunks)} chunks.")

    with open(cache_path, "wb") as f:
        pickle.dump({
            "version": CHUNKING_VERSION,
            "signature": folder_signature,
            "chunks": chunks
        }, f)
    return chunks


# ─────────────────────────────────────────────────────────────
# 4. SECTION SHORT-CIRCUIT
# ─────────────────────────────────────────────────────────────

def _merge_overlapping_texts(texts: list[str]) -> str:
    merged = ""
    for text in texts:
        if not merged:
            merged = text
            continue
        merged_words = merged.split()
        text_words = text.split()
        max_overlap = min(len(merged_words), len(text_words))
        overlap_len = 0
        for k in range(max_overlap, 0, -1):
            if merged_words[-k:] == text_words[:k]:
                overlap_len = k
                break
        remainder = " ".join(text_words[overlap_len:])
        merged = f"{merged} {remainder}".strip() if remainder else merged
    return merged


def batch_section_contexts(section: str, chunks: list[dict],
                            max_chars: int = MAX_SECTION_CONTEXT_CHARS) -> list[str]:
    section_chunks = [c for c in chunks if c.get("section") == section]
    if not section_chunks:
        return [f"[Section: {section}]\n(no content found)"]

    batches: list[list[str]] = [[]]
    batch_len = 0
    for c in section_chunks:
        text = c["text"]
        if batches[-1] and batch_len + len(text) > max_chars:
            batches.append([])
            batch_len = 0
        batches[-1].append(text)
        batch_len += len(text)

    return [
        f"[Section: {section} — part {i} of {len(batches)}]\n{_merge_overlapping_texts(batch)}"
        for i, batch in enumerate(batches, start=1)
    ]


# ─────────────────────────────────────────────────────────────
# 4b. HYBRID RETRIEVAL
# ─────────────────────────────────────────────────────────────

def build_bm25(chunks: list[dict]) -> BM25Okapi:
    tokenised = [chunk["text"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenised)


def build_embedding_matrix(chunks: list[dict]) -> np.ndarray:
    return np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)


def build_section_indexes(chunks: list[dict]) -> dict[str, dict]:
    by_section: dict[str, list[dict]] = {}
    for c in chunks:
        by_section.setdefault(c.get("section"), []).append(c)

    indexes: dict[str, dict] = {}
    for section, section_chunks in by_section.items():
        indexes[section] = {
            "chunks": section_chunks,
            "bm25": build_bm25(section_chunks),
            "matrix": build_embedding_matrix(section_chunks),
        }
    return indexes


def hybrid_retrieve(query: str,
                    chunks: list[dict],
                    bm25: BM25Okapi,
                    embed_model: SentenceTransformer,
                    embedding_matrix: np.ndarray,
                    section_hint: str | None,
                    top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    q_emb      = embed_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(q_emb, embedding_matrix)[0].tolist()

    all_bm25  = bm25.get_scores(query.lower().split())
    bm25_max  = max(all_bm25) or 1.0
    bm25_norm = [s / bm25_max for s in all_bm25]

    results = []
    for i, chunk in enumerate(chunks):
        if bm25_norm[i] == 0.0 and cos_scores[i] < DENSE_ONLY_FLOOR:
            continue

        hybrid_score = 0.5 * cos_scores[i] + 0.5 * bm25_norm[i]
        if section_hint and chunk.get("section") == section_hint:
            hybrid_score += SOFT_SECTION_BOOST
        results.append({"score": hybrid_score, "chunk": chunk})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def multi_query_retrieve(queries: list[str],
                         chunks: list[dict],
                         bm25: BM25Okapi,
                         embed_model: SentenceTransformer,
                         embedding_matrix: np.ndarray,
                         section_hint: str | None,
                         top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
    best_by_text: dict[str, dict] = {}
    for q in queries:
        if not q.strip():
            continue
        candidates = hybrid_retrieve(
            q, chunks, bm25, embed_model, embedding_matrix, section_hint, top_k=top_k
        )
        for c in candidates:
            text = c["chunk"]["text"]
            if text not in best_by_text or c["score"] > best_by_text[text]["score"]:
                best_by_text[text] = c

    merged = list(best_by_text.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


# ─────────────────────────────────────────────────────────────
# 5. RERANKER
# ─────────────────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict], reranker: CrossEncoder,
          top_k: int = TOP_K_FINAL) -> list[dict]:
    pairs  = [(query, c["chunk"]["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


# ─────────────────────────────────────────────────────────────
# 6. LLM ROUTER
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 6+6b. COMBINED ROUTE + REWRITE + EXPAND + SECTION/SCOPE
#        CLASSIFICATION — merged into ONE LLM call (was two:
#        route_query() + rewrite_expand_and_classify()) to cut
#        API usage roughly in half, since free-tier OpenRouter
#        models have tight rate limits.
# ─────────────────────────────────────────────────────────────

ROUTE_REWRITE_EXPAND_CLASSIFY_PROMPT = """
You are a query understanding assistant for a company policy chatbot
with multiple policy documents, one section per document.

Available sections in this company's policy documents:
{sections}

Conversation History:
{history}

Latest Message:
{query}

Do FOUR things:

1. Classify the message as "greeting" or "policy":
   greeting
     - Greetings, farewells, thanks, casual small talk, or any
       non-factual conversational message.
     - Includes ANY message that is clearly meant as a greeting or
       casual opener, even if it contains a typo or misspelling.
     - If a short message (1-4 words) looks like it could be a
       greeting attempt, classify it as greeting.
     - Examples: hi, hello, hey, hii, wello, helo, heyyy, show are
       you, how are you, how r u, hw are you, good morning, gm,
       thanks, ty, thank you, nice to meet you, bye, cya, who are
       you, sup, wassup.
   policy
     - Any clear question or request about company policy, HR,
       benefits, leave, conduct, insurance, or any related topic.
     - GENERAL RULE: any message that seeks a judgment, evaluation,
       recommendation, comparison, opinion, pro/con, or "should I..."
       verdict about the company, its policies, the job, or working
       there — in EITHER direction — is policy, never greeting.
     - If genuinely torn, default to policy.

   If intent is "greeting", leave every other field below as null /
   empty and do not attempt steps 2-4.

2. If intent is "policy": rewrite the latest message into a fully
   standalone question. Resolve pronouns and follow-ups using
   history. Never narrow scope. Keep any format instructions
   unchanged (e.g. if the user asked for a short/detailed/bulleted
   answer, preserve that instruction in the rewrite).

3. Generate up to {max_paraphrases} alternative phrasing(s) of the
   standalone question for search purposes.

4. Decide which ONE section this question is most likely about, as a
   soft hint to help retrieval (exact name from the list, or null if
   general/unclear/multi-section). This is just a bias, not a hard
   routing decision, so a reasonable best guess is fine.

Respond with ONLY valid JSON, no markdown fences, no preamble:

{{
  "intent": "greeting" or "policy",
  "standalone_question": "...",
  "paraphrases": ["...", "..."],
  "section": "<exact section name from the list, or null>"
}}
"""


def route_rewrite_expand_and_classify(query: str, available_sections: list[str],
                                       history_text: str) -> dict:
    section_lookup = {s.lower(): s for s in available_sections}
    prompt = ROUTE_REWRITE_EXPAND_CLASSIFY_PROMPT.format(
        sections="\n".join(f"- {s}" for s in available_sections) or "(none)",
        history=history_text,
        query=query,
        max_paraphrases=max(MAX_SUBQUERIES - 1, 0)
    )
    raw = ask_llm(prompt)
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    fallback = {
        "intent": "policy",
        "standalone_question": query,
        "paraphrases": [],
        "section": None
    }

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        return fallback

    intent_raw = parsed.get("intent")
    intent = "greeting" if isinstance(intent_raw, str) and "greeting" in intent_raw.lower() else "policy"

    if intent == "greeting":
        return {
            "intent": "greeting",
            "standalone_question": query,
            "paraphrases": [],
            "section": None
        }

    standalone = parsed.get("standalone_question")
    standalone = standalone.strip() if isinstance(standalone, str) and standalone.strip() else query

    paraphrases_raw = parsed.get("paraphrases", [])
    paraphrases = (
        [p.strip() for p in paraphrases_raw if isinstance(p, str) and p.strip()]
        if isinstance(paraphrases_raw, list) else []
    )
    paraphrases = paraphrases[:max(MAX_SUBQUERIES - 1, 0)]

    section = None
    if available_sections:
        section_raw = parsed.get("section")
        section = (
            section_lookup.get(section_raw.strip().lower())
            if isinstance(section_raw, str) else None
        )

    return {
        "intent": "policy",
        "standalone_question": standalone,
        "paraphrases": paraphrases,
        "section": section
    }


# ─────────────────────────────────────────────────────────────
# 6c. STANDALONE SECTION CLASSIFICATION (soft hint)
# ─────────────────────────────────────────────────────────────

SECTION_SCOPE_PROMPT = """
You are a query classifier for a company policy chatbot with multiple
policy documents, one section per document.

Available sections in this company's policy documents:
{sections}

Question:
{query}

Decide which ONE section this question is primarily about (exact name
from the list, or null if it doesn't clearly belong to one section).

Respond with ONLY valid JSON, no markdown fences, no preamble:

{{
  "section": "<exact section name from the list, or null>"
}}
"""


def soft_section_hint(query: str, available_sections: list[str] | None = None) -> str | None:
    if not available_sections:
        return None

    section_lookup = {s.lower(): s for s in available_sections}
    prompt = SECTION_SCOPE_PROMPT.format(
        sections="\n".join(f"- {s}" for s in available_sections),
        query=query
    )
    raw = ask_llm(prompt)
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        return None

    section_raw = parsed.get("section")
    if isinstance(section_raw, str):
        return section_lookup.get(section_raw.strip().lower())
    return None


# ─────────────────────────────────────────────────────────────
# 7. PER-SESSION CONVERSATION STATE
# ─────────────────────────────────────────────────────────────

_MORE_FOLLOWUP_RE = re.compile(
    r'\b(any\s*more|anything\s*else|any\s*other|what\s*else|else\??$|'
    r'more\s+(leaves?|benefits?|rules?|policies)|besides\s+that|'
    r'other\s+than\s+that)\b',
    re.IGNORECASE
)


def is_more_followup(query: str) -> bool:
    return bool(_MORE_FOLLOWUP_RE.search(query.strip()))


@dataclass
class SessionState:
    """
    Per-conversation state. One instance per session_id, kept by the
    caller (e.g. api.py's SESSIONS dict) — never shared across users.
    This replaces the old module-level `conversation_history` /
    `last_section_asked` globals from the CLI version.
    """
    conversation_history: list[dict] = field(default_factory=list)
    last_section_asked: str | None = None

    def build_history_text(self) -> str:
        if not self.conversation_history:
            return "No previous conversation."
        history = ""
        for turn in self.conversation_history:
            history += f"User: {turn['user']}\n"
            history += f"Assistant: {turn['assistant']}\n\n"
        return history

    def save_turn(self, user_query: str, assistant_answer: str) -> None:
        self.conversation_history.append({"user": user_query, "assistant": assistant_answer})
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history.pop(0)


# ─────────────────────────────────────────────────────────────
# 9. ANSWER GENERATION PROMPTS
# ─────────────────────────────────────────────────────────────

GREETING_PROMPT = """
You are a company policy assistant. Respond briefly and naturally to this
greeting or casual message.

Rules:
- Never give yourself a personal name. Refer to yourself only as "an
  assistant" or "the policy assistant".
- If asked what you are or what you're doing, say you're here to help
  answer company policy questions — nothing more.
- State no facts, opinions, or claims about the company. Greetings only.
- 1-2 sentences max.

Message: {query}

Response:
"""

PERSONA_PROMPT = """
You are {assistant_name}, a first-person AI assistant that answers
questions strictly from the company's policy documents. Always speak in
first person ("I", "my", "me") as the assistant — never claim to BE an
employee or the company itself, just its policy assistant.

Policy Document Context (with section labels):
{context}

Question:
{rewritten_query}

Behaviour rules:

0. State facts only. Never state an opinion, rating, or recommendation
   about the company — in either direction — even if asked directly.

1. FACTUAL QUESTIONS (context is relevant to the question)
   Answer using ONLY the policy context above. Use first-person language
   as the assistant. Every factual statement MUST appear explicitly in
   the context.

Never infer eligibility criteria, durations, amounts, dates, coverage
limits, approval processes, or exceptions.

If a fact is not written, say:
"I don't have that information in the company policy documents."

CRITICAL RULE — NEVER COMBINE FACTS ACROSS DIFFERENT [Chunk N] BLOCKS:
Each chunk is an independent, unrelated fact unless it explicitly says
otherwise. Before combining any two facts, check that both are stated
INSIDE THE SAME CHUNK.

2. FACTUAL QUESTIONS (context does NOT answer the question)
   Say: "I don't have that information in the company policy documents."

Never break character. Never claim personal employment status or
personal leave balances. Never invent factual information not present
in the context. If asked to disparage the company or the policy, say:
"I don't have that information in the company policy documents."

SELF-CHECK BEFORE ANSWERING (do this silently, do not show your work):
- Re-read your answer against the context above, sentence by sentence.
- If any sentence combines facts from two different [Chunk N] blocks
  without the context explicitly stating a connection between them,
  delete or rewrite that sentence so it only uses one chunk's facts.
- If any sentence states an opinion, rating, or recommendation about
  the company, delete it and replace the overall answer with:
  "I can't say — the policy documents don't state an opinion on that."
- If the context doesn't actually address the question, discard your
  draft and answer exactly:
  "I don't have that information in the company policy documents."
- Only after this check passes, output the final answer. Output ONLY
  the final answer text — no preamble, no notes about your check.

Answer:
"""


def answer_greeting(query: str) -> str:
    return ask_llm(GREETING_PROMPT.format(query=query))


def answer_as_assistant(context: str, rewritten_query: str, assistant_name: str) -> str:
    prompt = PERSONA_PROMPT.format(
        context=context if context.strip() else "(no policy context retrieved)",
        rewritten_query=rewritten_query,
        assistant_name=assistant_name
    )
    return ask_llm(prompt)


SECTION_ANSWER_PROMPT = """
You are {assistant_name}, a first-person AI assistant for company policy
questions.

Below is the COMPLETE, raw "{section}" section of the company's policy
documents as ONE continuous piece of text.

{context}

Question:
{query}

Instructions:
- Cover every distinct item actually present in the text above.
- Identify separate entries ONLY from the content itself.
- NEVER invent a name, title, or label not literally written in the text.
- For lists, name every item individually.
- Use ONLY facts literally written in the text — no inference.
- State facts only — never an opinion or recommendation.
- Never break character.

Answer:
"""

SECTION_SYNTHESIS_PROMPT = """
You are {assistant_name}, a first-person AI assistant for company policy
questions.

The "{section}" section was too long to process in one pass, so it was
answered in {n_parts} separate parts.

{partial_answers}

Original source text for each part:
{source_parts}

Question:
{query}

Combine the partial answers into ONE final, coherent answer, including
every distinct item across all parts (mention duplicates once). Never
invent new figures. Prefer source text over a partial answer if they
disagree. Never break character.

Final Answer:
"""


def answer_section_as_assistant(section_label: str, contexts: list[str], query: str,
                                 assistant_name: str) -> str:
    if len(contexts) == 1:
        prompt = SECTION_ANSWER_PROMPT.format(
            section=section_label, context=contexts[0], query=query,
            assistant_name=assistant_name
        )
        return ask_llm(prompt)

    partial_answers = []
    for ctx in contexts:
        prompt = SECTION_ANSWER_PROMPT.format(
            section=section_label, context=ctx, query=query,
            assistant_name=assistant_name
        )
        partial_answers.append(ask_llm(prompt))

    joined_partials = "\n\n".join(
        f"[Part {i}]\n{ans}" for i, ans in enumerate(partial_answers, start=1)
    )
    joined_sources = "\n\n".join(
        f"[Part {i} source]\n{ctx}" for i, ctx in enumerate(contexts, start=1)
    )
    synthesis_prompt = SECTION_SYNTHESIS_PROMPT.format(
        section=section_label,
        n_parts=len(contexts),
        partial_answers=joined_partials,
        source_parts=joined_sources,
        query=query,
        assistant_name=assistant_name
    )
    return ask_llm(synthesis_prompt)


SECTION_AGGREGATE_PROMPT = """
You are {assistant_name}, a first-person AI assistant for company policy
questions.

Below is the COMPLETE, raw "{section}" section of the company's policy
documents as ONE continuous piece of text.

{context}

Question:
{query}

The user wants a CONCISE but COMPLETE answer: every distinct
entitlement/type in this section, with its headline amount, nothing else.

Instructions:
- Name every distinct type present, each with its amount, concisely.
- Do NOT include procedural detail unless explicitly asked.
- Honor short-format requests without dropping entitlement types.
- If an amount varies by an unspecified criterion, name every variant.
- If the question specifies a qualifying detail, use the matching
  variant for that entitlement but still name every other type in full.
- Use ONLY facts literally written in the text.
- State facts only — never an opinion or recommendation.
- Never break character.

Answer:
"""

SECTION_AGGREGATE_SYNTHESIS_PROMPT = """
You are {assistant_name}, a first-person AI assistant for company policy
questions.

The "{section}" section was too long to process in one pass, so it was
answered in {n_parts} separate condensed parts.

{partial_answers}

Original source text for each part:
{source_parts}

Question:
{query}

Combine into ONE final, CONCISE answer covering every distinct
entitlement type (mention duplicates once). No procedural detail unless
asked. Never invent new figures — prefer source text if partials
disagree. Never break character.

Final Answer:
"""


def answer_section_aggregate(section_label: str, contexts: list[str], query: str,
                              assistant_name: str) -> str:
    if len(contexts) == 1:
        prompt = SECTION_AGGREGATE_PROMPT.format(
            section=section_label, context=contexts[0], query=query,
            assistant_name=assistant_name
        )
        return ask_llm(prompt)

    partial_answers = []
    for ctx in contexts:
        prompt = SECTION_AGGREGATE_PROMPT.format(
            section=section_label, context=ctx, query=query,
            assistant_name=assistant_name
        )
        partial_answers.append(ask_llm(prompt))

    joined_partials = "\n\n".join(
        f"[Part {i}]\n{ans}" for i, ans in enumerate(partial_answers, start=1)
    )
    joined_sources = "\n\n".join(
        f"[Part {i} source]\n{ctx}" for i, ctx in enumerate(contexts, start=1)
    )
    synthesis_prompt = SECTION_AGGREGATE_SYNTHESIS_PROMPT.format(
        section=section_label,
        n_parts=len(contexts),
        partial_answers=joined_partials,
        source_parts=joined_sources,
        query=query,
        assistant_name=assistant_name
    )
    return ask_llm(synthesis_prompt)


# ─────────────────────────────────────────────────────────────
# 10. ANSWER VERIFICATION
# ─────────────────────────────────────────────────────────────

VERIFY_PROMPT = """
You are a strict fact-checker for a company policy chatbot assistant.

Policy Document Context:
{context}

Question that was asked:
{question}

Answer to verify:
{answer}

Rules:
- If the context does not contain information relevant to the question,
  output exactly: "I don't have that information in the company policy
  documents."
- If the answer uses context about a different topic than what was
  asked, output exactly the same message.
- Chunks from different sections/files are unrelated unless one
  explicitly states the connection — delete any claim that combines them.
- Remove any subjective claims / opinions / recommendations about the
  company. If that leaves the question unanswered, output:
  "I can't say — the policy documents don't state an opinion on that."
- Never merge facts from different chunks unless the context states the
  connection. No inference, no invented numbers/names/dates.
- Otherwise rewrite the answer so every claim is grounded in the context.
  Keep first-person assistant language.

Output ONLY the final corrected answer. No preamble.

Corrected Answer:
"""


def verify_answer(context: str, answer: str, question: str = "") -> str:
    prompt = VERIFY_PROMPT.format(context=context, answer=answer, question=question)
    return ask_llm(prompt).strip()


# ─────────────────────────────────────────────────────────────
# 11. COMPANY NAME EXTRACTION
# ─────────────────────────────────────────────────────────────

# (company-name extraction removed — always uses DEFAULT_ASSISTANT_NAME
# now, saving one LLM call per unique section at startup)


def build_assistant_name(company_name: str | None = None) -> str:
    return DEFAULT_ASSISTANT_NAME


# ─────────────────────────────────────────────────────────────
# 12. RAG ENGINE  (built once at server startup, shared/read-only
#     across all requests; per-session mutable state lives in
#     SessionState instead)
# ─────────────────────────────────────────────────────────────

class RagEngine:
    def __init__(self, folder_path: str, verbose: bool = True):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        self.folder_path = folder_path
        self.verbose = verbose
        cache_path = os.path.join(folder_path, "_embeddings_cache.pkl")

        self._log("Scanning folder and parsing PDFs...")
        chunks = build_all_chunks(folder_path)
        self._log(f"Created {len(chunks)} chunks across all files.")

        section_counts = Counter(c["section"] for c in chunks)
        self._log(f"Section distribution: {dict(section_counts)}")
        self.available_sections = sorted(section_counts.keys())

        self.assistant_name = build_assistant_name()
        self._log(f"Assistant persona: {self.assistant_name}")

        self._log("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        self._log("Loading reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL_NAME)

        folder_signature = _folder_signature(folder_path)
        self._log("Loading or creating chunk embeddings...")
        self.chunks = load_or_create_embeddings(chunks, self.embed_model, cache_path, folder_signature)

        self._log("Building BM25 + section indexes...")
        self.bm25 = build_bm25(self.chunks)
        self.embedding_matrix = build_embedding_matrix(self.chunks)
        self.section_indexes = build_section_indexes(self.chunks)

        self._log("Engine ready.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def answer(self, query: str, session: "SessionState") -> str:
        query = query.strip()

        understanding = route_rewrite_expand_and_classify(
            query, self.available_sections, session.build_history_text()
        )

        if understanding["intent"] == "greeting":
            answer = answer_greeting(query)
            session.save_turn(query, answer)
            return answer

        standalone_query = understanding["standalone_question"]
        paraphrases = understanding["paraphrases"]
        section_hint = understanding["section"]
        all_queries = [standalone_query] + paraphrases

        # Safety net: if the LLM's rewrite of a vague follow-up (e.g.
        # "answer in detail") drifts away from the actual topic, retrieval
        # using only the rewritten query can come back empty even though
        # the conversation clearly has a topic. Adding the previous turn's
        # raw question as one more retrieval query costs a single extra
        # embed+BM25 lookup (negligible time/memory) and gives retrieval a
        # second, independent shot at the right topic.
        if session.conversation_history:
            prev_user_query = session.conversation_history[-1].get("user", "").strip()
            if prev_user_query and prev_user_query not in all_queries:
                all_queries.append(prev_user_query)

        context = ""
        candidates = multi_query_retrieve(
            all_queries, self.chunks, self.bm25, self.embed_model,
            self.embedding_matrix, section_hint
        )
        if candidates and candidates[0]["score"] >= THRESHOLD:
            top = rerank(standalone_query, candidates, self.reranker)
            context = "\n\n".join(
                f"[Chunk {i} — Section: {r['chunk'].get('section')}]\n{r['chunk']['text']}"
                for i, r in enumerate(top, start=1)
            )

        answer = answer_as_assistant(context, standalone_query, self.assistant_name)

        session.save_turn(query, answer)
        return answer