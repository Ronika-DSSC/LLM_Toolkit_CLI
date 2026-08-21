from RAG.embedding import ingest_folder, configure_vector_store_paths
from RAG.retrieval import retrieve_top_k
from RAG.generate_markdown import generate_folder
from RAG.llm import call_chat_llm
from RAG.prompt_loader import load_promptsets
import logging
import numpy as np
import os
import re
import json
import time
import argparse
import logging
import hashlib
import faiss
import chromadb
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class RAGConfig:
    # Original documents
    input_folder: str = "./test_input"
    # Document ingestion
    folder: str = "./data/markdown" # Generated markdown
    markdown_dir: str = "./data/markdown" # Generate markdown
    vector_store_dir: str = "./vector_store"
    output_dir: str = "./output"
    embedding_model: str = "embeddinggemma:latest"
    backend: str = "numpy"
    collection_name: str = "clinical_chromaDB_test_docs"
    # Chunking
    max_tokens: int = 400
    overlap_tokens: int = 40
    # Retrieval
    top_k: int = 5
    cosine_threshold: float = 0.3
    # Prompts
    prompt_dir: str = "prompts/promptset"
    # LLM
    llm_model: str = "phi4:14b"
    system_prompt: str = "You are a clinical information extraction assistant."
    temperature: float = 0.1
    # Pipeline
    use_text_pipeline: bool = False



def format_source(meta: Dict[str, Any]) -> str:
    source = meta.get("source", "unknown")
    if "page" in meta:
        return f"{source}, page {meta['page']}"
    return source


def format_context(chunks):
    context_parts = []
    for chunk in chunks:
        source = format_source(chunk.meta)
        context_parts.append(f"[{source}]\n{chunk.content}")
    return "\n\n".join(context_parts)


def run_rag_pipeline(config: RAGConfig):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    # Generate Markdown files first
    markdown_results = generate_folder(
        input_dir=config.input_folder, 
        output_dir=config.markdown_dir,
        run_id=run_id,
        llm_model=config.llm_model, 
        temperature=config.temperature, 
        max_tokens=config.max_tokens,)

    successful_markdown = [r for r in markdown_results if r.get("status") != "error"]
    logging.info("Markdown generation complete: %d successful file(s), %d failed", len(successful_markdown), len(markdown_results) - len(successful_markdown),)
    # Configure all vector-store paths before ingestion/retrieval
    configure_vector_store_paths(config.vector_store_dir)
    logging.info("Using vector store directory: %s", config.vector_store_dir,)
    # -------------------------
    # Load all Prompts
    # -------------------------
    promptsets = load_promptsets(config.prompt_dir)
    logging.info("Loaded %d prompt set(s) from %s", len(promptsets), config.prompt_dir,)
    # -------------------------------------
    # Build Vector Stores and Retrieve
    # -------------------------------------
    collection_size = None
    retrieved_chunks = None
    if config.backend == 'chroma':
        collection = ingest_folder(
        folder=config.folder, 
        vector_store_dir=config.vector_store_dir,
        embedding_model=config.embedding_model, 
        backend="chroma", 
        collection_name=config.collection_name, 
        max_tokens=config.max_tokens, 
        overlap_tokens=config.overlap_tokens, 
        use_text_pipeline=config.use_text_pipeline,)
        collection_size = collection.count()
    
    elif config.backend == 'numpy':
        store = ingest_folder(
        folder=config.folder, 
        vector_store_dir= config.vector_store_dir,
        embedding_model=config.embedding_model, 
        backend="numpy", max_tokens=config.max_tokens, 
        overlap_tokens=config.overlap_tokens, 
        use_text_pipeline=config.use_text_pipeline,)
        collection_size = len(store)
    
    elif config.backend == 'faiss':
        faiss_index, faiss_metadata = ingest_folder(
            folder=config.folder,
            vector_store_dir=config.vector_store_dir,
            embedding_model=config.embedding_model,
            backend="faiss",
            max_tokens=config.max_tokens,
            overlap_tokens=config.overlap_tokens,
            use_text_pipeline=config.use_text_pipeline,
        )
        collection_size = faiss_index.ntotal
    
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")
    # -----------------------------------------------------
    # Run EVERY prompt set
    # -----------------------------------------------------
    results = []
    for prompt_number, prompt in enumerate(promptsets, start=1,):
        prompt_id = prompt["id"]
        logging.info("Running prompt %d/%d: %s", prompt_number, len(promptsets), prompt_id,)
        rag_prompt = prompt["rag"]
        user_prompt = prompt["user"]
        # -----------------------------------------------------
        # Retrieve evidence for this prompt
        # -----------------------------------------------------
        if config.backend == "chroma":
            top_retrieved_chunks = retrieve_top_k(
                query=rag_prompt,
                embedding_model=config.embedding_model,
                backend="chroma",
                collection_name=config.collection_name,
                cosine_threshold=config.cosine_threshold,
                k=config.top_k,
            )

        elif config.backend == "numpy":
            top_retrieved_chunks = retrieve_top_k(
                query=rag_prompt,
                store=store,
                embedding_model=config.embedding_model,
                backend="numpy",
                cosine_threshold=config.cosine_threshold,
                k=config.top_k,
            )

        elif config.backend == "faiss":
            top_retrieved_chunks = retrieve_top_k(
                query=rag_prompt,
                embedding_model=config.embedding_model,
                backend="faiss",
                cosine_threshold=config.cosine_threshold,
                k=config.top_k,
            )
        # -----------------------------------------------------
        # Build context
        # -----------------------------------------------------
        context = format_context(top_retrieved_chunks)
        combined_user_prompt = f"""{user_prompt} Evidence from patient records: {context}"""
        # Call LLM
        llm_response = call_chat_llm(model=config.llm_model, system_prompt=config.system_prompt, user_prompt=combined_user_prompt, temperature=config.temperature,)
        # Store result
        results.append({"prompt_id": prompt_id, "source_file": prompt.get("source_file"), "response": llm_response, "context": context, "retrieved_chunks": top_retrieved_chunks,})
        logging.info("Completed prompt: %s", prompt_id,)
    # =========================================================
    # 6. Return all results
    # =========================================================
    return {"results": results, "collection_size": collection_size, "backend": config.backend,}


def save_rag_output(result: Dict[str, Any], output_dir: str,) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    for item in result["results"]:
        prompt_id = item["prompt_id"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{prompt_id}_{timestamp}.txt",)
        with open(output_path, "w", encoding="utf-8",) as f:
            f.write("========================================\n")
            f.write("              RAG RESPONSE\n")
            f.write("========================================\n\n")
            f.write(f"Prompt: {prompt_id}\n")
            if item.get("source_file"):
                f.write(f"Prompt file: {item['source_file']}\n")
            f.write("\n")
            f.write(item["response"])
            f.write("\n\n========================================\n")
            f.write("           RETRIEVED CONTEXT\n")
            f.write("========================================\n\n")
            f.write(item["context"])
            f.write("\n\n========================================\n")
            f.write("           PIPELINE INFO\n")
            f.write("========================================\n\n")
            f.write(f"Backend: {result['backend']}\n")
            f.write(f"Collection size: " f"{result['collection_size']}\n")
            f.write(f"Retrieved chunks: " f"{len(item['retrieved_chunks'])}\n")
        logging.info("RAG output saved to: %s", output_path,)
        saved_files.append(output_path)
    return saved_files


if __name__ == "__main__":
    config = RAGConfig()
    result = run_rag_pipeline(config)
    print("\n===== LLM RESPONSE =====")
    print(result["response"])




