"""OffliQ - offline study assistant.
Run:  streamlit run app.py
Needs Ollama running locally with an LLM and an embedding model pulled.
"""
import math
import re
import socket
import sqlite3

import fitz  # PyMuPDF
import numpy as np
import ollama
import streamlit as st

DB_PATH = "offliq.db"
LLM = "llama3.2:3b"          # change to whatever model you pulled
EMBED = "nomic-embed-text"   # if missing, app falls back to keyword search
CTX = {"num_ctx": 4096}
MAX_WORDS = 2500             # keeps prompts inside the small model's context

st.set_page_config(page_title="OffliQ", page_icon="📚", layout="wide")


# ---------- storage ----------
@st.cache_resource
def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, name TEXT, pages INTEGER);
        CREATE TABLE IF NOT EXISTS pages(doc_id INTEGER, page INTEGER, text TEXT);
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY, doc_id INTEGER, page INTEGER, text TEXT, emb BLOB);
        """
    )
    return c


def list_docs():
    return conn().execute("SELECT id, name, pages FROM docs ORDER BY id DESC").fetchall()


def delete_doc(doc_id):
    c = conn()
    for t in ("chunks", "pages"):
        c.execute(f"DELETE FROM {t} WHERE doc_id=?", (doc_id,))
    c.execute("DELETE FROM docs WHERE id=?", (doc_id,))
    c.commit()


# ---------- ingestion ----------
def chunk_words(text, size=180, overlap=40):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step) if words[i:i + size]]


def embed(texts):
    """Return float32 array (n, d) or None if the embedding model is unavailable."""
    try:
        out = []
        for i in range(0, len(texts), 16):
            out.extend(ollama.embed(model=EMBED, input=texts[i:i + 16])["embeddings"])
        return np.array(out, dtype=np.float32)
    except Exception:
        return None


def ingest(name, data, progress):
    pdf = fitz.open(stream=data, filetype="pdf")
    pages = [(i + 1, p.get_text().strip()) for i, p in enumerate(pdf)]
    if not any(t for _, t in pages):
        return False, "No text found. This looks like a scanned PDF (OCR not included)."
    c = conn()
    doc_id = c.execute("INSERT INTO docs(name, pages) VALUES(?,?)", (name, len(pages))).lastrowid
    pieces = []
    for pno, text in pages:
        c.execute("INSERT INTO pages VALUES(?,?,?)", (doc_id, pno, text))
        pieces += [(pno, ch) for ch in chunk_words(text)]
    progress.progress(0.3, "Creating search index...")
    vecs = embed([t for _, t in pieces])
    for i, (pno, ch) in enumerate(pieces):
        blob = vecs[i].tobytes() if vecs is not None else None
        c.execute("INSERT INTO chunks(doc_id,page,text,emb) VALUES(?,?,?,?)", (doc_id, pno, ch, blob))
    c.commit()
    progress.progress(1.0, "Done")
    note = "" if vecs is not None else " (embedding model not found: using keyword search)"
    return True, f"Saved {len(pages)} pages, {len(pieces)} chunks{note}."


# ---------- retrieval ----------
STOP = set("the a an of and or to in is are was for on with as by it this that what which how why do does".split())


def toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def kw_score(q, text):
    counts = {}
    for w in toks(text):
        counts[w] = counts.get(w, 0) + 1
    return sum(math.log(1 + counts.get(w, 0)) for w in set(toks(q)))


def retrieve(doc_ids, query, k=4):
    ph = ",".join("?" * len(doc_ids))
    rows = conn().execute(
        f"SELECT c.doc_id, d.name, c.page, c.text, c.emb FROM chunks c "
        f"JOIN docs d ON d.id=c.doc_id WHERE c.doc_id IN ({ph})", doc_ids).fetchall()
    if not rows:
        return []
    qv = embed([query]) if all(r[4] for r in rows) else None
    if qv is not None:
        m = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        q = qv[0]
        sims = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-9)
    else:
        sims = np.array([kw_score(query, r[3]) for r in rows])
    top = np.argsort(sims)[::-1][:k]
    return [{"doc": rows[i][1], "page": rows[i][2], "text": rows[i][3]} for i in top if sims[i] > 0]


def page_range_text(doc_id, p1, p2):
    rows = conn().execute(
        "SELECT text FROM pages WHERE doc_id=? AND page BETWEEN ? AND ? ORDER BY page",
        (doc_id, p1, p2)).fetchall()
    words = " ".join(r[0] for r in rows).split()
    return " ".join(words[:MAX_WORDS]), len(words) > MAX_WORDS


# ---------- LLM ----------
BASE = ("You are OffliQ, a study assistant for students. Use ONLY the provided document "
        "content. If the answer is not in it, say you couldn't find it in the document.")
STYLES = {
    "Standard": "Give a clear, accurate answer.",
    "Simplify (like I'm 15)": "Explain with short sentences and one everyday analogy. Define any jargon.",
    "Exam-oriented": "Write a model exam answer: a one-line definition, key points as a short list, "
                     "then a one-line 'Remember this' tip.",
}


def stream(system, user):
    for part in ollama.chat(model=LLM, options=CTX, stream=True,
                            messages=[{"role": "system", "content": system},
                                      {"role": "user", "content": user}]):
        yield part["message"]["content"]


def is_online():
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=1).close()
        return True
    except OSError:
        return False


def llm_ready():
    try:
        names = [m.model for m in ollama.list().models]
        return any(n.startswith(LLM.split(":")[0]) for n in names)
    except Exception:
        return False


# ---------- UI ----------
st.title("📚 OffliQ")
st.caption("Your study materials and AI tutor, working without internet.")

with st.sidebar:
    st.subheader("Status")
    st.write("🟢 Offline mode: running locally" if not is_online() else "🔵 Online (everything still runs locally)")
    st.write("🟢 Local AI ready" if llm_ready() else f"🔴 Start Ollama and run `ollama pull {LLM}`")

docs = list_docs()
labels = {f"{n} ({p} pages)": i for i, n, p in docs}
tab_lib, tab_ask, tab_sum, tab_quiz = st.tabs(["Library", "Ask", "Chapter summary", "Practice questions"])

with tab_lib:
    up = st.file_uploader("Upload a PDF", type="pdf")
    if up and st.button("Save to library", type="primary"):
        bar = st.progress(0.05, "Reading PDF...")
        ok, msg = ingest(up.name, up.read(), bar)
        (st.success if ok else st.error)(msg)
        if ok:
            st.rerun()
    st.divider()
    if not docs:
        st.info("No documents yet. Upload a PDF to get started.")
    for i, n, p in docs:
        a, b = st.columns([6, 1])
        a.write(f"📄 **{n}** · {p} pages")
        if b.button("Delete", key=f"del{i}"):
            delete_doc(i)
            st.rerun()

with tab_ask:
    if not docs:
        st.info("Add a document in the Library tab first.")
    else:
        sel = st.multiselect("Search in", list(labels), default=list(labels)[:1])
        style = st.radio("Answer style", list(STYLES), horizontal=True)
        q = st.text_input("Ask a question about your notes")
        if st.button("Ask", type="primary") and q and sel:
            hits = retrieve([labels[s] for s in sel], q)
            if not hits:
                st.warning("Couldn't find anything relevant in the selected documents.")
            else:
                ctx = "\n\n".join(f"[{h['doc']} p.{h['page']}]\n{h['text']}" for h in hits)
                st.write_stream(stream(f"{BASE} {STYLES[style]} Mention page numbers when helpful.",
                                       f"Document content:\n{ctx}\n\nQuestion: {q}"))
                with st.expander("Sources"):
                    for h in hits:
                        st.markdown(f"**{h['doc']} · page {h['page']}**")
                        st.caption(h["text"][:400] + "...")


def range_picker(prefix):
    sel = st.selectbox("Document", list(labels), key=prefix + "d")
    doc_id = labels[sel]
    total = next(p for i, _, p in docs if i == doc_id)
    lo, hi = st.slider("Pages (smaller ranges give more detail)", 1, total, (1, min(total, 5)), key=prefix + "r")
    return doc_id, lo, hi


with tab_sum:
    if docs:
        doc_id, lo, hi = range_picker("s")
        if st.button("Summarize", type="primary"):
            text, cut = page_range_text(doc_id, lo, hi)
            if cut:
                st.caption("Range was long, so only the first part was used.")
            st.write_stream(stream(
                BASE, "Summarize this chapter for revision: a 2-sentence overview, then 5-8 key "
                      f"points, then key terms with one-line definitions.\n\n{text}"))

with tab_quiz:
    if docs:
        doc_id, lo, hi = range_picker("q")
        c1, c2 = st.columns(2)
        n = c1.slider("Number of questions", 3, 10, 5)
        kind = c2.radio("Type", ["Multiple choice", "Short answer"], horizontal=True)
        if st.button("Generate questions", type="primary"):
            text, _ = page_range_text(doc_id, lo, hi)
            fmt = ("4 options (A-D), then 'Answer:' with a one-line reason"
                   if kind == "Multiple choice" else "then 'Model answer:' in 1-3 sentences")
            st.write_stream(stream(
                BASE, f"Write {n} exam-style {kind.lower()} questions from this content. "
                      f"For each question: {fmt}. Number them.\n\n{text}"))
