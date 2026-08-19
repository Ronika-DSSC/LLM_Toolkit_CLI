"""
retrieval.py

RAG retrieval module:
Supports:
- NumPy cosine similarity
- ChromaDB vector search

Input: query + store
Output: List[DocumentChunk]
"""

import logging
import numpy as np
import os
import re
import json
import time
import argparse
import hashlib
from typing import List, Literal, Optional, Dict, Any, Tuple

import requests
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import chromadb

from embedding_api import embed_texts
from embedding import DocumentChunk, faiss_store_paths
from embedding import get_collection


# ========================================================= Faiss INIT ================================================================
def load_faiss_store(index_path: Optional[str] = None, metadata_path: Optional[str] = None,):
    """
    Load a FAISS index and its associated metadata.
    """
    default_index, default_metadata, _ = faiss_store_paths()
    index_path = index_path or default_index
    metadata_path = metadata_path or default_metadata
    logging.info("Loading FAISS index: %s", index_path,)
    logging.info("Loading FAISS metadata: %s", metadata_path,)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"FAISS metadata not found: {metadata_path}")
    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True,).tolist()
    logging.info("Loaded FAISS index with %d vectors", index.ntotal,)
    return index, metadata


# ============================================================ Numpy retrieval =========================================================
def numpy_retrieve(query: str, store: List[DocumentChunk], embedding_model: str, k: int, threshold: float):
    # Convert query text into embedding vector and reshape to 2D for similarity computation
    q_emb = embed_texts([query], embedding_model)[0].reshape(1, -1)
    # Stack all document embeddings into a single matrix (shape: num_docs x embedding_dim)
    matrix = np.vstack([c.embedding for c in store])
    # Cosine similarity between query embedding and all document embeddings
    sims = cosine_similarity(q_emb, matrix)[0]
    print("Query embedding shape:", q_emb.shape)
    print("Document matrix shape:", matrix.shape)
    print("Similarity scores:", sims[:10])
    # Get indices that would sort similarities in descending order (highest first)
    idxs = np.argsort(-sims)
    # Filter indices by similarity threshold
    idxs = [i for i in idxs if sims[i] >= threshold]
    # Return top-k chunks based on filtered similarity ranking
    return [store[i] for i in idxs[:k]]


# =========================================================== ChromaDB retrieval =========================================================
def chroma_retrieve(query: str, embedding_model: str, k: int, threshold: float, collection_name: str):
    # Get or create the ChromaDB collection
    collection = get_collection(collection_name)
    # Convert query text into embedding vector
    q_emb = embed_texts([query], embedding_model)[0]
    # Query ChromaDB for more candidates than needed (k * 5 for filtering later)
    results = collection.query(query_embeddings=[q_emb.tolist()], n_results=k * 5, include=["documents", "metadatas", "distances"],)
    # List to store final filtered chunks
    chunks = []
    # Iterate through returned documents, metadata, and distances together
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0],):
        # Convert distance to similarity score (Chroma returns distance, not similarity)
        similarity = 1 - dist
        print("Similarity:", similarity, "Source:", meta)
        print("Chroma distance:", dist, "similarity:", similarity)
        # Keep only results above similarity threshold
        if similarity >= threshold:
            chunks.append(DocumentChunk(content=doc, embedding=[], meta=meta or {}))
        # Stop early once we have enough results
        if len(chunks) >= k:
            break
    return chunks


# ============================================================== Faiss retrieval =========================================================
def faiss_retrieve(query: str, embedding_model: str, k: int, threshold: float, index_path: Optional[str] = None, metadata_path: Optional[str] = None,):
    """
    Retrieve the top-k most similar chunks from FAISS.
    """
    index, metadata = load_faiss_store(index_path, metadata_path,)
    q_emb = embed_texts([query], embedding_model)[0].astype(np.float32)
    q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb) # Need to discuss about the normalization approach
    scores, indices = index.search(q_emb, k)
    chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score < threshold:
            continue
        item = metadata[idx]
        chunks.append(DocumentChunk(content=item["content"], embedding=[], meta=item["meta"],))
    return chunks


# ========================================================= Unified retrieval API =========================================================
# Define allowed backend types for retrieval system
Backend = Literal["numpy", "chroma", "faiss"]

def retrieve_top_k(
    query: str,
    store: List[DocumentChunk] = None,
    *,
    embedding_model: str,
    backend: Backend = "numpy",
    cosine_threshold: float = 0.7,
    k: int = 3,
    collection_name: str = "rag_collection",
) -> List[DocumentChunk]:

    # If using NumPy backend, ensure in-memory store is provided
    if backend == "numpy":
        if store is None:
            raise ValueError("store required for numpy backend")
        # Run retrieval using NumPy-based cosine similarity search
        return numpy_retrieve(query, store, embedding_model, k, cosine_threshold,)

    # If using Chroma backend, delegate to vector database query
    elif backend == "chroma":
        return chroma_retrieve(query, embedding_model, k, cosine_threshold, collection_name,)

    # If using FAISS
    elif backend == "faiss":
        return faiss_retrieve(query, embedding_model, k, cosine_threshold,)

    # Raise error for unsupported backend types
    else:
        raise ValueError("Unknown backend")


################################ NumPy backend ##################################
# from retrieval import retrieve_top_k
# top_chunks = retrieve_top_k(query=rag_prompt, store=store, embedding_model=embedding_model, backend="numpy", cosine_threshold=0.7, k=3,)

################################ ChromaDB backend ###############################
# from retrieval import retrieve_top_k
# top_chunks = retrieve_top_k(query=rag_prompt, embedding_model=embedding_model, backend="chroma", cosine_threshold=0.7, k=3, collection_name="rag_collection",)




