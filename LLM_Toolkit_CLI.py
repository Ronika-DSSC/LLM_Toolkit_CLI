"""
LLM_Toolkit_CLI.py

Command-line interface for the RAG pipeline.

Examples:

    python LLM_Toolkit_CLI.py

    python LLM_Toolkit_CLI.py \
        --markdown-dir ./data/markdown \
        --folder ./data/markdown \
        --backend numpy \
        --embedding-model embeddinggemma:latest \
        --top-k 5 \
        --cosine-threshold 0.3 \
        --llm-model phi4:14b

    python LLM_Toolkit_CLI.py \
        --markdown-dir ./data/markdown \
        --folder ./data/markdown \
        --backend faiss \
        --collection-name clinical_chromaDB_test_docs \
        --max-tokens 400 \
        --overlap-tokens 40 \
        --top-k 5 \
        --cosine-threshold 0.3 \
        --prompt-file prompts/promptset/patient_diagnosis_prompts_v3_with_page_evidence.txt \
        --prompt-index 0 \
        --llm-model phi4:14b \
        --temperature 0.1 \
        --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys

from RAG_pipeline import RAGConfig, run_rag_pipeline, save_rag_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-cli",
        description="Command-line interface for the RAG pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ================================================= Document ingestion ====================================================
    ingestion = parser.add_argument_group("Document ingestion")
    ingestion.add_argument("--input-folder", type=str, default="./test_input", help="Folder containing the original patient documents to convert to Markdown.",)
    ingestion.add_argument("--folder", type=str, default="./data/markdown", help="Folder containing documents to ingest.",)
    ingestion.add_argument("--markdown-dir", type=str, default="./data/markdown", help="Directory where generated Markdown files/reports are written.",)
    ingestion.add_argument("--vector-store-dir", type=str, default="./vector_store", help="Root directory for vector stores and embedding cache.",)
    ingestion.add_argument("--embedding-model", type=str, default="embeddinggemma:latest", help="Embedding model used to generate document/query embeddings.",)
    ingestion.add_argument("--backend", type=str, choices=["numpy", "chroma", "faiss"], default="numpy", help="Vector store backend.",)
    ingestion.add_argument("--collection-name", type=str, default="clinical_chromaDB_test_docs", help="ChromaDB collection name. Used only with the chroma backend.",)
    ingestion.add_argument("--max-tokens", type=int, default=400, help="Maximum number of tokens per chunk.",)
    ingestion.add_argument("--overlap-tokens", type=int, default=40, help="Number of overlapping tokens between chunks.",)
    ingestion.add_argument("--use-text-pipeline", action="store_true", help="Run the optional text pipeline before chunking.",)

    # ===================================================== Retrieval =======================================================
    retrieval = parser.add_argument_group("Retrieval")
    retrieval.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.",)
    retrieval.add_argument("--cosine-threshold", type=float, default=0.3, help="Minimum cosine similarity required for a retrieved chunk.",)

    # ============================================== Prompt configuration ===================================================
    prompts = parser.add_argument_group("Prompt configuration")
    prompts.add_argument("--prompt-file", type=str, default=("prompts/promptset/patient_diagnosis_prompts_v3_with_page_evidence.txt"), help="Prompt-set file.",)
    prompts.add_argument("--prompt-index", type=int, default=0, help="Zero-based index of the prompt set to use.",)

    # ========================================================= LLM =========================================================
    llm = parser.add_argument_group("LLM")
    llm.add_argument("--llm-model", type=str, default="phi4:14b", help="LLM model used to generate the final response.",)
    llm.add_argument("--system-prompt", type=str, default="You are a clinical information extraction assistant.", help="System prompt sent to the LLM.",)
    llm.add_argument("--temperature", type=float, default=0.1, help="LLM sampling temperature.",)

    # ============================================== Output / debugging =====================================================
    output = parser.add_argument_group("Output")
    output.add_argument("--show-context", action="store_true", help="Print the retrieved evidence/context.",)
    output.add_argument("--show-metadata", action="store_true", help="Print metadata for retrieved chunks.",)
    output.add_argument("--verbose", action="store_true", help="Enable verbose logging.",)
    output.add_argument("--output-dir", type=str, default="./output", help="Directory where RAG output files are saved.",)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before starting the pipeline."""
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than 0.")
    if args.overlap_tokens < 0:
        raise ValueError("--overlap-tokens cannot be negative.")
    if args.overlap_tokens >= args.max_tokens:
        raise ValueError("--overlap-tokens must be smaller than --max-tokens.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if not 0.0 <= args.cosine_threshold <= 1.0:
        raise ValueError("--cosine-threshold must be between 0.0 and 1.0.")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative.")
    if args.prompt_index < 0:
        raise ValueError("--prompt-index cannot be negative.")


def configure_logging(verbose: bool) -> None:
    """Configure application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",)


def print_retrieved_chunks(chunks) -> None:
    """Print retrieved chunks and their metadata."""
    print("\n===== RETRIEVED CHUNKS =====")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print("Metadata:", chunk.meta)
        print("\nContent:", chunk.content)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        configure_logging(args.verbose)

        # -----------------------------------------------------
        # Build RAG configuration
        # -----------------------------------------------------

        config = RAGConfig(
            input_folder=args.input_folder,
            folder=args.folder,
            markdown_dir=args.markdown_dir,
            vector_store_dir=args.vector_store_dir,
            output_dir=args.output_dir,
            embedding_model=args.embedding_model,
            backend=args.backend,
            collection_name=args.collection_name,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            top_k=args.top_k,
            cosine_threshold=args.cosine_threshold,
            prompt_file=args.prompt_file,
            prompt_index=args.prompt_index,
            llm_model=args.llm_model,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            use_text_pipeline=args.use_text_pipeline,
        )

        logging.info("Starting RAG pipeline")
        logging.info("Input folder: %s", config.input_folder)
        logging.info("Folder: %s", config.folder)
        logging.info("Backend: %s", config.backend)
        logging.info("Embedding model: %s", config.embedding_model)
        logging.info("LLM model: %s", config.llm_model)

        # -----------------------------------------------------
        # Run RAG
        # -----------------------------------------------------

        result = run_rag_pipeline(config)
        output_path = save_rag_output(result, config.output_dir,)
        print(f"\nRAG output saved to: {output_path}")

        # -----------------------------------------------------
        # Final response
        # -----------------------------------------------------

        print("\n========================================")
        print("           RAG RESPONSE")
        print("========================================\n")

        print(result["response"])

        # -----------------------------------------------------
        # Optional context
        # -----------------------------------------------------

        if args.show_context:
            print("\n========================================")
            print("           RETRIEVED CONTEXT")
            print("========================================\n")

            print(result["context"])

        # -----------------------------------------------------
        # Optional metadata
        # -----------------------------------------------------

        if args.show_metadata:
            print_retrieved_chunks(result["retrieved_chunks"])

        # -----------------------------------------------------
        # Pipeline information
        # -----------------------------------------------------

        print("\n========================================")
        print("           PIPELINE INFO")
        print("========================================")

        print(f"Backend:        {result['backend']}")
        print(f"Collection size: {result['collection_size']}")
        print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
        print("RAG pipeline completed successfully")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        logging.exception("RAG pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
