import os
import io
import re
import time
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import torch

st.set_page_config(page_title="StudyMate – PDF Q&A", page_icon="📚", layout="wide")
def extract_pages(file_bytes: bytes, filename: str) -> List[Dict]:
    docs = []
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text("text")
            if not text:
                continue
            cleaned = re.sub(r"[ \t]+", " ", text).strip()
            cleaned = re.sub(r"\n{2,}", "\n", cleaned)
            docs.append({"text": cleaned, "page": i + 1, "source": filename})
    return docs

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
    # Simple char-based chunking (safe for mixed content)
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # try to end on sentence boundary
        sub = text[start:end]
        m = re.search(r"[.!?]\s", text[end-100:end]) if end - start > 100 else None
        if m:
            end = end - (100 - m.end())
            sub = text[start:end]
        chunks.append(sub.strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-dim, fast

def build_faiss(chunks_meta: List[Dict], embedder) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    texts = [c["text"] for c in chunks_meta]
    embs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via normalized dot product
    index.add(embs)
    return index, embs

def retrieve(query: str, embedder, index, k: int = 5) -> List[int]:
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(q, k)
    return ids[0].tolist()
@st.cache_resource(show_spinner=False)
def load_llm():
    """
    Tries IBM Granite (Hugging Face) locally.
    If too heavy or fails, falls back to FLAN-T5-base (lightweight).
    """
    try:
        model_id = "ibm-granite/granite-7b-instruct"  # IBM model (no IBM login)
        tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        mod = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        gen = pipeline("text-generation", model=mod, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
        return {"type": "granite", "pipe": gen}
    except Exception:
        model_id = "google/flan-t5-base"
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        gen = pipeline("text2text-generation", model=mod, tokenizer=tok)
        return {"type": "flan", "pipe": gen}

def generate_answer(llm, question: str, contexts: List[Dict]) -> str:
    context_block = ""
    for i, c in enumerate(contexts, 1):
        context_block += f"[{i}] (p.{c['page']} – {c['source']})\n{c['text']}\n\n"

    system = (
        "You are StudyMate, an academic assistant. Answer using ONLY the context.\n"
        "Cite sources like [1], [2] matching the context blocks. If not in context, say you don't know."
    )
    prompt = f"{system}\n\nContext:\n{context_block}\nQuestion: {question}\nAnswer:"

    pipe = llm["pipe"]
    if llm["type"] == "granite":
        out = pipe(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]
        # Granite will echo the prompt—extract answer after 'Answer:'
        ans = out.split("Answer:", 1)[-1].strip() if "Answer:" in out else out.strip()
        return ans
    else:  # FLAN
        out = pipe({"text": prompt}, max_new_tokens=300)[0]["generated_text"]
        return out.strip()

st.sidebar.title("📚 StudyMate")
st.sidebar.caption("Upload PDFs → Ask questions → Get cited answers")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None

files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

colL, colR = st.columns([1, 2])
with colL:
    st.header("Documents")
    if files:
        with st.spinner("Reading and indexing…"):
            all_chunks = []
            for f in files:
                data = f.read()
                pages = extract_pages(data, f.name)
                for p in pages:
                    for ch in chunk_text(p["text"]):
                        all_chunks.append({"text": ch, "page": p["page"], "source": p["source"]})
            st.session_state.embedder = load_embedder()
            index, _ = build_faiss(all_chunks, st.session_state.embedder)
            st.session_state.index = index
            st.session_state.chunks = all_chunks
        st.success(f"Indexed {len(st.session_state.chunks)} chunks ✅")
    else:
        st.info("Upload PDFs on the left to begin.")

with colR:
    st.header("Ask a question")
    q = st.text_input("Type your academic question here…")
    if st.button("Get Answer") and q:
        if not st.session_state.chunks or st.session_state.index is None:
            st.warning("Please upload and index PDFs first.")
        else:
            with st.spinner("Retrieving context…"):
                ids = retrieve(q, st.session_state.embedder, st.session_state.index, k=5)
                top_ctx = [st.session_state.chunks[i] for i in ids]

            with st.spinner("Generating answer… (IBM Granite if available)"):
                llm = load_llm()
                ans = generate_answer(llm, q, top_ctx)

            st.subheader("Answer")
            st.write(ans)

            st.subheader("Sources")
            for i, c in enumerate(top_ctx, 1):
                st.markdown(f"**[{i}]** p.{c['page']} — *{c['source']}*")
