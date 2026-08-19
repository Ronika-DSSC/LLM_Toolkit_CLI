"""
embedding.py

Responsible for:
1. Chunking documents
2. Creating embeddings
3. Storing vectors in:
   - NumPy in-memory store OR
   - ChromaDB vector database OR
   - FAISS

NOT responsible for retrieval.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable
import os
import logging
import numpy as np
import re
import json
import time
import argparse
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import faiss
import chromadb
import nltk

from embedding_api import embed_texts  # your API-based embedding function
#from text_pipeline import process

# ====================================== Optional: Impoting libraries to read different documents - pdf, docx =====================================
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None


# ========================================================= Vector-store and embedding-cache directories ===========================================

VECTOR_STORE_DIR = "./vector_store"

CHROMA_DB_PATH = os.path.join(VECTOR_STORE_DIR, "chroma")
NUMPY_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "numpy")
FAISS_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "faiss")
EMBEDDING_CACHE_PATH = os.path.join(VECTOR_STORE_DIR, "embedding_cache")

def configure_vector_store_paths(vector_store_dir: str):
    global VECTOR_STORE_DIR
    global CHROMA_DB_PATH
    global NUMPY_STORE_PATH
    global FAISS_STORE_PATH
    global EMBEDDING_CACHE_PATH

    VECTOR_STORE_DIR = os.path.abspath(vector_store_dir)

    CHROMA_DB_PATH = os.path.join(VECTOR_STORE_DIR, "chroma")
    NUMPY_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "numpy")
    FAISS_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "faiss")
    EMBEDDING_CACHE_PATH = os.path.join(VECTOR_STORE_DIR, "embedding_cache",)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    logging.info("Configured VECTOR_STORE_DIR = %s", VECTOR_STORE_DIR)
    logging.info("Configured CHROMA_DB_PATH = %s", CHROMA_DB_PATH)
    logging.info("Configured NUMPY_STORE_PATH = %s", NUMPY_STORE_PATH)
    logging.info("Configured FAISS_STORE_PATH = %s", FAISS_STORE_PATH)
    logging.info("Configured EMBEDDING_CACHE_PATH = %s", EMBEDDING_CACHE_PATH,)


# ------------------------- Data classes -------------------------
class DocumentChunk:
    def __init__(self, content: str, embedding: List[float], meta: Optional[dict] = None):
        self.content = content
        self.embedding = np.array(embedding, dtype=np.float32)
        self.meta = meta or {}

# ========================================================= Read different document types ==========================================================
def read_text_from_file(path: str, *, use_text_pipeline: bool = False, run_id: str = "embedding",) -> str:
    """
    Read text from a .txt, .pdf, or .docx file.
    """
    # If you directly import & use test_pipeline.py for generating the cleaned text then you can use the next 4 lines that are commented out.
    #if use_text_pipeline:
    #    ext = Path(path).suffix.lower().lstrip(".")
    #    result = process(file_path=path, file_type=ext, run_id=run_id,)
    #    return result["cleaned_text"]
    ext = Path(path).suffix.lower()
    if ext in {".txt", ".md"}:
        return Path(path).read_text(encoding="utf-8", errors="ignore",)
    elif ext == ".pdf":
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 is not installed.")
        pages = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                pages.append(page.extract_text() or "")
        return "\n".join(pages)
    elif ext == ".docx":
        if docx is None:
            raise RuntimeError("python-docx is not installed.")
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    raise ValueError(f"Unsupported file type: {ext}")


def read_pdf_pages(path: str) -> List[str]:
    """Return a list of extracted page texts (1 element per PDF page)."""
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 not installed. Run: pip install PyPDF2")
    pages: List[str] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i in range(len(reader.pages)):
            try:
                pages.append(reader.pages[i].extract_text() or "")
            except Exception:
                pages.append("")
    return pages


# ============================================================= Tokenizer/Encoder helper =========================================================
def get_tiktoken_encoder(tokenizer_name="cl100k_base"):
    try:
        import tiktoken
        # Splits text into tokens
        return tiktoken.get_encoding(tokenizer_name)
    except Exception: # # If tokenizer library is missing or fails, return None as fallback
        return None


# =============================================================== Embedding cache ================================================================
def safe_stem(path: str) -> str:
    """
    Convert filename into safe cache filename.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")


def file_signature(path: str) -> Dict[str, Any]:
    """
    Detect file changes.
    """
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime": float(st.st_mtime),}


def make_cache_key(sig: Dict[str, Any], embedding_model: str, chunk_params: Dict[str, Any],):
    payload = {"sig": sig, "embedding_model": embedding_model, "chunk_params": chunk_params,}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False,).encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:16]


def cache_paths(cache_dir: str, input_path: str, cache_key: str,):
    stem = safe_stem(input_path)
    base = os.path.join(cache_dir, f"{stem}__{cache_key}" )
    return (base + ".chunks.jsonl", base + ".embeddings.npz", base + ".info.json",)


def load_cached_embeddings(cache_dir: str, input_path: str, cache_key: str,):
    chunks_path, emb_path, info_path = cache_paths(cache_dir, input_path, cache_key,)
    if not (os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(info_path)):
        return None

    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    with np.load(emb_path) as data:
        embeddings = data["emb"]
    logging.info("Loaded cached embeddings: %d chunks", len(chunks))
    return chunks, embeddings


def save_cached_embeddings(cache_dir: str, input_path: str, cache_key: str, chunks, embeddings, info):
    os.makedirs(cache_dir, exist_ok=True)
    chunks_path, emb_path, info_path = cache_paths(cache_dir, input_path, cache_key,)

    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c,  ensure_ascii=False) + "\n")

    np.savez_compressed(emb_path, emb=embeddings)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    logging.info("Saved embedding cache: %s", input_path)


# ================================================================ Numpy store cache ==================================================================
def numpy_store_paths(cache_dir=None):
    if cache_dir is None:
        cache_dir = NUMPY_STORE_PATH
    os.makedirs(cache_dir, exist_ok=True)
    return (os.path.join(cache_dir, "numpy_store.npy"), os.path.join(cache_dir, "numpy_metadata.json"),)


def save_numpy_store(store, cache_key, cache_dir=None):
    if cache_dir is None:
        cache_dir = NUMPY_STORE_PATH
    os.makedirs(cache_dir, exist_ok=True)
    vector_path, meta_path = numpy_store_paths(cache_dir)
    embeddings = np.vstack([c.embedding for c in store])
    np.save(vector_path, embeddings)
    metadata = [{"content": c.content, "meta": c.meta,} for c in store]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    with open(os.path.join(cache_dir, "numpy.info.json"), "w", encoding="utf-8",) as f:
        json.dump({"cache_key": cache_key}, f)
    logging.info("Saved NumPy vector store")


def load_numpy_store(cache_key, cache_dir=None):
    if cache_dir is None:
        cache_dir = NUMPY_STORE_PATH
    vector_path, meta_path = numpy_store_paths(cache_dir)
    info_path = os.path.join(cache_dir, "numpy.info.json")
    if not (os.path.exists(vector_path) and os.path.exists(meta_path) and os.path.exists(info_path)):
        return None

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    if info.get("cache_key") != cache_key:
        logging.info("NumPy cache key changed. Rebuilding.")
        return None
    embeddings = np.load(vector_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    store = [DocumentChunk(item["content"], emb, item["meta"]) for item, emb in zip(metadata, embeddings)]
    logging.info("Loaded NumPy vector store: %d chunks", len(store))
    return store


# ============================================================ Build document chunk objects ==============================================================
def build_vector_store_from_chunks(chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> List[DocumentChunk]:
    texts = [c["content"] for c in chunks]
    metas = [c.get("meta", {}) for c in chunks]
    if embeddings.shape[0] != len(texts):
        raise RuntimeError("Mismatch between number of chunks and embeddings.")
    return [DocumentChunk(t, emb, meta=m) for t, emb, m in zip(texts, embeddings, metas)]


# ========================================================= Prepare vector store for one documemt =========================================================
def prepare_vector_store_for_record(
    input_path: str,
    embedding_model: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    tokenizer: str,
    cache_dir: Optional[str],
    use_cache: bool,
    force_reembed: bool,
    use_text_pipeline: bool = False,
) -> Tuple[List[DocumentChunk], str, str]:
    record_basename = os.path.basename(input_path)
    ext = os.path.splitext(input_path)[1].lower()

    chunk_params = {
        "max_tokens": int(max_tokens),
        "overlap_tokens": int(overlap_tokens),
        "tokenizer": tokenizer,
        "pdf_page_chunking": bool(ext == ".pdf"),
    }

    sig = file_signature(input_path)
    key = make_cache_key(sig, embedding_model, chunk_params)

    if cache_dir and use_cache and not force_reembed:
        cached = load_cached_embeddings(cache_dir, input_path, key)
        if cached is not None:
            logging.info("Using cached embedding for %s", record_basename)
            chunks, embeddings = cached
            store = build_vector_store_from_chunks(chunks, embeddings)
            return store, record_basename, key

    # Compute from scratch
    if ext == ".pdf":
        pages = read_pdf_pages(input_path)
        logging.info("Number of pages: %d", len(pages))
        for i, page in enumerate(pages, start=1):
            logging.info("Page %d: %d characters, preview=%r", i, len(page), page[:100],)
        chunks = chunk_pdf_pages(
            pages,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer_name=tokenizer,
            source=record_basename,
        )
    else:
        record_text = read_text_from_file(input_path, use_text_pipeline=use_text_pipeline,).strip()
        if not record_text:
            raise RuntimeError("No text extracted from the input document.")
        chunks = [
            {"content": c, "meta": {"source": record_basename}}
            for c in chunk_text_hybrid(
                record_text,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                tokenizer_name=tokenizer,
            )
        ]

    if not chunks:
        raise RuntimeError("Chunking produced zero chunks.")

    # Every chunk knows exactly which cache generated it
    for c in chunks:
        c["meta"]["cache_key"] = key

    texts = [c["content"] for c in chunks]
    logging.info(f"Embedding {len(texts)} chunks (once per record)...")
    emb = embed_texts(texts, embedding_model)
    store = build_vector_store_from_chunks(chunks, emb)

    if cache_dir and use_cache:
        info = {
            "file_signature": sig,
            "embedding_model": embedding_model,
            "chunk_params": chunk_params,
            "cache_key": key,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_cached_embeddings(cache_dir, input_path, key, chunks, emb, info)
    return store, record_basename, key


# ============================================================== Token based chunking ==============================================================
def chunk_text_tokens(text: str, max_tokens: int = 400, overlap_tokens: int = 40, tokenizer_name: str = "cl100k_base",) -> List[str]:
    """
    Chunk text into overlapping token windows.
    Falls back to word chunking if tiktoken isn't installed.
    """
    # Ensure text is not None and remove extra whitespace
    text = (text or "").strip()
    # Return empty list if no valid text is provided
    if not text:
        return []
    # Initialize tokenizer/encoder helper
    encoder = get_tiktoken_encoder(tokenizer_name)
    # Define step size (how far we move forward each chunk)
    step = max_tokens - overlap_tokens
    if encoder:
         # Convert full text into token IDs
        ids = encoder.encode(text)
        chunks = [] # Store final text chunks
        # Slide over token IDs with overlap
        for start in range(0, len(ids), step):
            chunk = ids[start:start + max_tokens]
            # Convert token IDs back into readable text
            chunks.append(encoder.decode(chunk))
         # Return token-based chunks
        return chunks
    # Fallback
    words = text.split()
    # Create word-based chunks (approximate tokenization)
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), step)]


# ========================================================= Hybrid / Sentence-Aware chunking =========================================================
def chunk_text_hybrid(text: str, max_tokens: int = 400, overlap_tokens: int = 40, tokenizer_name: str = "cl100k_base",) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    encoder = get_tiktoken_encoder(tokenizer_name)
    if encoder:
        sentences = nltk.sent_tokenize(text)
        chunks = []; current_chunk = []; current_tokens = 0;
        for sentence in sentences:
            sentence_tokens = len(encoder.encode(sentence))
            # If adding this sentence exceeds limit
            if (current_tokens + sentence_tokens > max_tokens and current_chunk):
                chunks.append(" ".join(current_chunk))
                # overlap using previous sentences
                overlap_chunk = []; overlap_count = 0;
                for s in reversed(current_chunk):
                    t = len(encoder.encode(s))
                    if overlap_count + t <= overlap_tokens:
                        overlap_chunk.insert(0, s)
                        overlap_count += t
                    else:
                        break
                current_chunk = overlap_chunk
                current_tokens = overlap_count
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    # fallback if no tokenizer available
    return chunk_text_tokens(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, tokenizer_name=tokenizer_name,)


# ============================================================= PDF page-aware chunking ===========================================================
def chunk_pdf_pages(
    pages: List[str],
    *,
    max_tokens: int = 400,
    overlap_tokens: int = 40,
    tokenizer_name: str = "cl100k_base",
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Chunk a PDF page-by-page so each chunk keeps page metadata."""
    chunks: List[Dict[str, Any]] = []
    for page_no, page_text in enumerate(pages, start=1):
        page_text = (page_text or "").strip()
        if not page_text:
            continue
        subchunks = chunk_text_hybrid(
            page_text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer_name=tokenizer_name,
        )
        for sub_idx, c in enumerate(subchunks):
            chunks.append(
                {
                    "content": c,
                    "meta": {
                        "source": source,
                        "page": page_no,
                        "page_start": page_no,
                        "page_end": page_no,
                        "subchunk": sub_idx,
                    },
                }
            )
    return chunks

# ========================================================= Read a document for tokenization & chunking =========================================================
def chunk_document(file_path: str, max_tokens: int = 400, overlap_tokens: int = 40,) -> List[Dict[str, Any]]:
    """
    Read one document and convert it into structured chunks.
    """
    # Read raw text from file (PDF, TXT, DOCX, etc.)
    text = read_text_from_file(file_path)
    # Calling the tokenization & chunking function on the document to chunk the document text into smaller overlapping pieces
    raw_chunks = chunk_text_tokens(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens,)
    # Add metadata (e.g., filename) to each chunk for tracking in retrieval systems
    return prepare_chunks(raw_chunks, default_meta={ "source": os.path.basename(file_path)},)


# ========================================================= Document ingestion and vector store creation =========================================================
def ingest_folder(folder: str, embedding_model: str, backend: str = "numpy", collection_name: str = "rag_collection", max_tokens: int = 400, overlap_tokens: int = 40, use_text_pipeline: bool = False, vector_store_dir: str = "./vector_store",):
    """
    Read every supported document in a folder, chunk it, embed it, and build a vector store.
    """
    # Configure vector-store locations
    configure_vector_store_paths(vector_store_dir)
    logging.info("Ingesting using vector store directory: %s", VECTOR_STORE_DIR,)
    all_chunks = [] # Collect chunks from all documents
    backend = backend.lower()
    logging.info("Scanning folder: %s", folder)
    #logging.info("Folder exists: %s", Path(folder).exists())
    # Iterate through all files in folder
    paths = list(Path(folder).rglob("*"))
    for path in paths:
        #logging.info("Found: %s", path)
        # Skip unsupported file types
        if path.name.startswith("~$"):
            logging.info("Skipping temporary file %s", path.name)
            continue
        if path.suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
            continue
        logging.info("Reading %s", path.name)
        store, record_name, cache_key = prepare_vector_store_for_record(
            input_path=str(path),
            embedding_model=embedding_model,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer="cl100k_base",
            cache_dir=EMBEDDING_CACHE_PATH,
            use_cache=True,
            force_reembed=False,
            use_text_pipeline=use_text_pipeline,
        )
        logging.info("%s -> %d chunks", record_name, len(store))

        if backend == "numpy":
            all_chunks.extend(store)
        elif backend.lower() == "chroma":
            collection = get_collection(collection_name)
            existing = collection.get(where={"source": record_name}, include=["metadatas"])
            if existing["metadatas"]:
                old_key = existing["metadatas"][0].get("cache_key")
                if old_key == cache_key:
                    logging.info("%s unchanged. Skipping Chroma update.", record_name)
                    continue
                else:
                    logging.info("%s changed. Re-indexing.", record_name)
                    collection.delete(where={"source": record_name})
            all_chunks.extend({"content": c.content, "embedding": c.embedding, "meta": c.meta,} for c in store)
        elif backend == "faiss":
            all_chunks.extend({"content": c.content, "embedding": c.embedding, "meta": c.meta,} for c in store)
    # Log total number of chunks created
    logging.info("Prepared %d chunks.", len(all_chunks),)
    # Build vector store depending on backend choice
    if backend == "numpy":
        global_cache_key = hashlib.sha256(json.dumps(sorted(c.meta["cache_key"] for c in all_chunks), sort_keys=True).encode()).hexdigest()[:16]
        cached_numpy = load_numpy_store(global_cache_key)
        if cached_numpy is not None:
            return cached_numpy
        store = create_numpy_vector_store(all_chunks)
        save_numpy_store(store, global_cache_key)
        return store
    elif backend == "chroma":
        return create_chroma_vector_store(all_chunks, collection_name,)
    elif backend == "faiss":
        if not all_chunks:
            raise RuntimeError("FAISS requested but no chunks were generated.")
        cache_key = hashlib.sha256(json.dumps(sorted(c.get("meta", {}).get("cache_key") for c in all_chunks)).encode()).hexdigest()[:16]
        existing = load_faiss_store(cache_key)
        if existing is not None:
            logging.info("Using existing FAISS index")
            return existing
        return create_faiss_vector_store(all_chunks, embedding_model, cache_key,)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ================================================================== Chroma setup ==================================================================
def get_collection(name: str):
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"},)


# ================================================================ Numpy vector store ==============================================================
# Build an in-memory vector store using document chunks and their embeddings
def build_numpy_store(chunks: List[Dict[str, Any]], embeddings: np.ndarray,) -> List[DocumentChunk]:
    """
    Build in-memory vector store.
    """
    # Extract the text content from each chunk
    texts = [c["content"] for c in chunks]
    # Extract metadata from each chunk (use empty dictionary if missing)
    metas = [c.get("meta", {}) for c in chunks]
    # Ensure the number of embeddings matches the number of text chunks
    if embeddings.shape[0] != len(texts):
        raise ValueError("Mismatch between chunks and embeddings")
    # Create a DocumentChunk object for every text, embedding, and metadata pair
    store = [DocumentChunk(text, emb, meta=m) for text, emb, m in zip(texts, embeddings, metas)]
    # Log the number of chunks stored in memory
    logging.info("Built NumPy vector store with %d chunks", len(store))
    # Return the completed in-memory vector store
    return store


# ============================================================== ChromaDB storage ==================================================================
# Store document embeddings in a ChromaDB collection
def build_chroma_store(chunks: List[Dict[str, Any]], embeddings: np.ndarray, collection_name: str = "rag_collection", batch_size: int = 5000,):
    """
    Store embeddings in ChromaDB in batches.
    Chroma has a maximum batch size, so large collections must be uploaded incrementally.
    """
    collection = get_collection(collection_name)
    if not chunks:
        logging.info("No chunks to store in ChromaDB.")
        return collection
    if embeddings.shape[0] != len(chunks):
        raise ValueError(f"Mismatch: {len(chunks)} chunks but " f"{embeddings.shape[0]} embeddings.")
    total = len(chunks)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        batch_embeddings = embeddings[start:end]
        ids, documents, metadatas = [],[],[]
        for local_idx, c in enumerate(batch_chunks):
            m = c.get("meta", {})
            source = m.get("source", "unknown")
            page = m.get("page", 0)
            subchunk = m.get("subchunk", local_idx)
            # Global index makes IDs unique across batches
            global_idx = start + local_idx
            ids.append(f"{source}::{page}::{subchunk}::{global_idx}")
            documents.append(c["content"])
            metadatas.append(m)
        collection.upsert(ids=ids, embeddings=batch_embeddings.tolist(), documents=documents, metadatas=metadatas,)
        logging.info("Stored Chroma batch: %d-%d / %d chunks", start + 1, end, total,)
    logging.info("Stored %d chunks in ChromaDB (%s)", total, collection_name,)
    return collection

# ================================================================= Faiss storage =================================================================
def faiss_store_paths():
    os.makedirs(FAISS_STORE_PATH, exist_ok=True)
    return (os.path.join(FAISS_STORE_PATH, "faiss.index"), os.path.join(FAISS_STORE_PATH, "faiss_metadata.npy"), os.path.join(FAISS_STORE_PATH, "faiss.info.json"),)


def build_faiss_store(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    info_path: Optional[str] = None,
    cache_key: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    """
    Store embeddings in a FAISS index with metadata and cache information.
    """
    if index_path is None or metadata_path is None or info_path is None:
        default_index, default_metadata, default_info = faiss_store_paths()
        index_path = index_path or default_index
        metadata_path = metadata_path or default_metadata
        info_path = info_path or default_info
    embeddings = embeddings.astype(np.float32)
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    # Save FAISS index
    faiss.write_index(index, index_path)
    # Save chunk metadata
    metadata = [{"content": c["content"], "meta": c.get("meta", {}) } for c in chunks]
    np.save(metadata_path, metadata, allow_pickle=True)
    # Save cache information
    info = {"cache_key": cache_key, "embedding_model": embedding_model, "num_vectors": len(chunks), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    logging.info("Saved FAISS index with %d chunks", len(chunks))
    return index


# ================================================================== Embedding pipeline ==================================================================
def embed_chunks(chunks: List[Dict[str, Any]], embedding_model: str, batch_size: int = 16,) -> np.ndarray:
    """
    Convert text chunks into embeddings.
    """
    # Extract text content from each chunk
    texts = [c["content"] for c in chunks]
    # Log the number of chunks being embedded
    logging.info("Embedding %d chunks", len(texts))
    # Generate embeddings in batches using the selected embedding model
    embeddings = embed_texts(texts, embedding_model=embedding_model, batch_size=batch_size,)
    # Return the embedding matrix
    return embeddings


# ========================================================= Vector store creation: NUMPY ==================================================================
def create_numpy_vector_store(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """
    Build NumPy store from already embedded chunks.
    Embeddings come from embedding cache.
    """
    logging.info("Using cached embeddings for NumPy store: %d chunks", len(chunks))
    return chunks

# ========================================================= Vector store creation: ChromaDB ===============================================================
# Complete pipeline for storing embeddings in ChromaDB
def create_chroma_vector_store(chunks: List[Dict[str, Any]], collection_name: str = "rag_collection",):
    if not chunks:
        logging.info("No new Chroma chunks. Returning existing collection.")
        return get_collection(collection_name)
    embeddings = np.vstack([c["embedding"] for c in chunks])
    return build_chroma_store(chunks, embeddings, collection_name)

# ============================================================ Vector store creation: Faiss ===============================================================
# Complete pipeline for storing embeddings in FAISS
def create_faiss_vector_store(chunks: List[Dict[str, Any]], embedding_model: str, cache_key: str, index_path: str = None, metadata_path: str = None,):
    if index_path is None or metadata_path is None:
        default_index, default_metadata, default_info = faiss_store_paths()
        index_path = index_path or default_index
        metadata_path = metadata_path or default_metadata
    if not chunks:
        logging.info("No FAISS chunks to create.")
        return None
    embeddings = np.vstack([c["embedding"] for c in chunks])
    index = build_faiss_store(chunks, embeddings, index_path=index_path, metadata_path=metadata_path, cache_key=cache_key, embedding_model=embedding_model,)
    metadata = np.load(metadata_path, allow_pickle=True)
    return index, metadata

def load_faiss_store(cache_key: str, index_path: Optional[str] = None, metadata_path: Optional[str] = None, info_path: Optional[str] = None,):
    if index_path is None or metadata_path is None or info_path is None:
        default_index, default_metadata, default_info = faiss_store_paths()
        index_path = index_path or default_index
        metadata_path = metadata_path or default_metadata
        info_path = info_path or default_info
    if not (os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(info_path)):
        return None
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    if info.get("cache_key") != cache_key:
        logging.info("FAISS cache key changed. Rebuilding.")
        return None
    index = faiss.read_index(index_path)
    metadata = np.load(metadata_path, allow_pickle=True)
    logging.info("Loaded FAISS index with %d vectors", index.ntotal)
    return index, metadata


# ================================= OPTIONAL: PDF / TEXT chunk wrapper hook (You already have chunk_text_tokens in main code) ============================
# Convert raw text chunks into the standard dictionary format
def prepare_chunks(raw_chunks: List[str], default_meta: Optional[Dict[str, Any]] = None,) -> List[Dict[str, Any]]:
    """
    Convert raw text chunks into structured format.
    """
    # Use an empty metadata dictionary if none is provided
    default_meta = default_meta or {}
    # Create a structured dictionary for each non-empty text chunk. Ignores empty or whitespace-only chunks.
    return [{"content": c, "meta": dict(default_meta),} for c in raw_chunks if c.strip()]


################################ NumPy backend ##################################
# from embedding import create_numpy_vector_store
# store = create_numpy_vector_store(chunks, embedding_model)

################################ ChromaDB backend ###############################
# from embedding import create_chroma_vector_store
# collection = create_chroma_vector_store(chunks, embedding_model, collection_name="rag_collection")


