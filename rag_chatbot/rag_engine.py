# -*- coding: utf-8 -*-
"""
RAG engine: builds/loads FAISS indexes over the knowledge base (the book)
and the latest prediction output, and retrieves relevant context for a query.
"""
import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(__file__)
EMB_MODEL = "all-MiniLM-L6-v2"
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
BOOK_PATH = os.path.join(BASE_DIR, "knowledge_base", "the_intelligent_investor.txt")
PREDICTION_PATH = os.path.join(BASE_DIR, "..", "data", "predictions", "prediction_latest.txt")
os.makedirs(EMB_DIR, exist_ok=True)


def chunk_text(text, size=400):
    """Split text into chunks of approximately `size` words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_or_load_index(file_path, emb_model):
    """Build a FAISS index for `file_path`, or load it from cache if it exists."""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    index_file = os.path.join(EMB_DIR, f"{base_name}_index.faiss")
    chunks_file = os.path.join(EMB_DIR, f"{base_name}_chunks.pkl")

    if os.path.exists(index_file) and os.path.exists(chunks_file):
        print(f"Loading cached index for {os.path.basename(file_path)}...")
        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks

    print(f"Building new index for {os.path.basename(file_path)}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"ERROR: {file_path} not found!")
        return None, []

    chunks = chunk_text(text, size=400)
    if not chunks:
        print(f"WARNING: no chunks created from {file_path}")
        return None, []

    embeddings = emb_model.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index created: {len(chunks)} chunks")
    return index, chunks


def setup_rag():
    """Initialize the RAG system: embedding model + book index + prediction index."""
    print("Loading embedding model...")
    model_emb = SentenceTransformer(EMB_MODEL)

    print("Setting up book index...")
    book_index, book_chunks = build_or_load_index(BOOK_PATH, model_emb)

    print("Setting up prediction index...")
    pred_index, pred_chunks = build_or_load_index(PREDICTION_PATH, model_emb)

    return model_emb, book_index, book_chunks, pred_index, pred_chunks


def retrieve_context(query, model_emb, book_index=None, book_chunks=None,
                      pred_index=None, pred_chunks=None, k=2):
    """Retrieve the top-k relevant chunks from the requested indexes for `query`."""
    context_parts = []
    query_emb = model_emb.encode([query], show_progress_bar=False).astype("float32")

    if book_index is not None and book_chunks:
        try:
            distances, indices = book_index.search(query_emb, k)
            retrieved = [book_chunks[i] for i in indices[0] if i < len(book_chunks)]
            if retrieved:
                context_parts.append("Book Context:\n" + "\n".join(retrieved))
        except Exception as e:
            print(f"Error retrieving from book index: {e}")

    if pred_index is not None and pred_chunks:
        try:
            distances, indices = pred_index.search(query_emb, k)
            retrieved = [pred_chunks[i] for i in indices[0] if i < len(pred_chunks)]
            if retrieved:
                context_parts.append("Prediction Data:\n" + "\n".join(retrieved))
        except Exception as e:
            print(f"Error retrieving from prediction index: {e}")

    return "\n\n".join(context_parts) if context_parts else "No context found."
