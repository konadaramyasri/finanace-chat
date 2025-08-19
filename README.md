import os
import re
import io
import time
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
import torch
import speech_recognition as sr
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from googletrans import Translator

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="💰 FinanceMate – Your Personal Finance Assistant",
    page_icon="💸",
    layout="wide"
)

# Custom CSS for attractive background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .chat-bubble {
        padding: 12px;
        border-radius: 12px;
        margin: 6px 0;
    }
    .user {background: #4CAF50; text-align: right;}
    .bot {background: #1E90FF; text-align: left;}
    </style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ---------------- #
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
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        sub = text[start:end]
        chunks.append(sub.strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_faiss(chunks_meta: List[Dict], embedder):
    texts = [c["text"] for c in chunks_meta]
    embs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs

def retrieve(query: str, embedder, index, k: int = 5):
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, ids = index.search(q, k)
    return ids[0].tolist()

@st.cache_resource(show_spinner=False)
def load_llm():
    try:
        model_id = "ibm-granite/granite-7b-instruct"
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
    context_block = "\n\n".join([f"[{i+1}] (p.{c['page']} – {c['source']})\n{c['text']}" for i, c in enumerate(contexts)])
    system = "You are FinanceMate, a personal finance assistant. Use context + your knowledge. Provide clear, practical advice."
    prompt = f"{system}\n\nContext:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    pipe = llm["pipe"]
    if llm["type"] == "granite":
        out = pipe(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]
        return out.split("Answer:", 1)[-1].strip()
    else:
        out = pipe({"text": prompt}, max_new_tokens=300)[0]["generated_text"]
        return out.strip()

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Speak now…")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
    try:
        return r.recognize_google(audio)
    except:
        return "Sorry, could not recognize speech."

translator = Translator()

def translate_text(text, target_lang):
    return translator.translate(text, dest=target_lang).text

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("💰 FinanceMate")
mode = st.sidebar.radio("Mode", ["New Chat", "Search Docs", "Finance Advice"])

files = st.sidebar.file_uploader("📂 Upload financial docs (PDFs)", type=["pdf"], accept_multiple_files=True)

if "chunks" not in st.session_state:
    st.session_state.chunks, st.session_state.index, st.session_state.embedder = [], None, None
if "history" not in st.session_state:
    st.session_state.history = []

if files:
    all_chunks = []
    for f in files:
        data = f.read()
        pages = extract_pages(data, f.name)
        for p in pages:
            for ch in chunk_text(p["text"]):
                all_chunks.append({"text": ch, "page": p["page"], "source": p["source"]})
    st.session_state.embedder = load_embedder()
    index, _ = build_faiss(all_chunks, st.session_state.embedder)
    st.session_state.index, st.session_state.chunks = index, all_chunks
    st.sidebar.success(f"Indexed {len(st.session_state.chunks)} chunks ✅")

# ---------------- MAIN UI ---------------- #
st.title("💸 FinanceMate – Smart Personal Finance Chatbot")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Chat History")
    for role, msg in st.session_state.history:
        st.markdown(f"<div class='chat-bubble {role}'>{msg}</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Ask Me Anything 💬")

    input_method = st.radio("Input Method:", ["Text", "Speech"])
    if input_method == "Text":
        q = st.text_input("Type your finance question here…")
    else:
        if st.button("🎙 Speak"):
            q = speech_to_text()
            st.success(f"You said: {q}")
        else:
            q = ""

    lang = st.selectbox("🌍 Translate Answer To:", ["English", "Telugu", "Hindi", "Spanish"])

    if st.button("Get Answer") and q:
        if mode == "Search Docs" and st.session_state.index is not None:
            ids = retrieve(q, st.session_state.embedder, st.session_state.index, k=5)
            top_ctx = [st.session_state.chunks[i] for i in ids]
        else:
            top_ctx = []

        llm = load_llm()
        ans = generate_answer(llm, q, top_ctx)

        if lang != "English":
            ans = translate_text(ans, lang.lower())

        st.session_state.history.append(("user", q))
        st.session_state.history.append(("bot", ans))

        st.markdown(f"<div class='chat-bubble bot'>{ans}</div>", unsafe_allow_html=True)
