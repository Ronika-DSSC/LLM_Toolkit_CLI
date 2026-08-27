"""
LLM_Toolkit_CLI.py

Command-line interface for the RAG pipeline.

Examples:

    python LLM_Toolkit_CLI.py

"""

from __future__ import annotations

import argparse
import logging
import sys

from RAG.RAG_pipeline import RAGConfig, run_rag_pipeline, save_rag_output

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="LLM_Toolkit_CLI.py",
        description=(
            "LLM Toolkit CLI - Retrieval-Augmented Generation (RAG) pipeline.\n\n"
            "This tool runs the complete RAG workflow:\n"
            "  1. Read source medical records (knowledgebase).\n"
            "  2. Generate Markdown representations.\n"
            "  3. Chunk the documents.\n"
            "  4. Generate and cache embeddings.\n"
            "  5. Build/update the selected vector store.\n"
            "  6. Retrieve relevant chunks for each prompt set.\n"
            "  7. Send retrieved evidence to the configured LLM.\n"
            "  8. Save LLM responses and retrieved context.\n"
        ),
        epilog=r"""
======================================================================
HOW TO RUN
======================================================================

Basic execution:

    python LLM_Toolkit_CLI.py


======================================================================
RECOMMENDED CHROMADB EXAMPLE
======================================================================

    python LLM_Toolkit_CLI.py \
        --input-folder ./test_input \
        --markdown-dir ./CHROMA/markdown \
        --folder ./CHROMA/markdown \
        --backend chroma \
        --vector-store-dir ./chroma_data_store \
        --output-dir ./output_chroma_test_run \
        --max-tokens 400 \
        --overlap-tokens 40 \
        --top-k 5 \
        --cosine-threshold 0.3 \
        --collection-name clinical_chromaDB_test_docs \
        --prompt-dir ./test_prompts/ \
        --llm-model phi4:14b \
        --temperature 0.1 \
        --verbose


======================================================================
OPTION DESCRIPTIONS
======================================================================

--input-folder
    Directory containing the original medical records.

    Example:
        --input-folder ./test_input


--markdown-dir
    Directory where generated Markdown files are stored.

    Example:
        --markdown-dir ./CHROMA/markdown


--folder
    Directory containing the documents that will be ingested into
    the vector store.

    Example:
        --folder ./CHROMA/markdown

    In the standard workflow this is the same directory as
    --markdown-dir.


--backend
    Vector-store backend.

    Supported values:
        numpy
        chroma
        faiss

    Example:
        --backend chroma


--vector-store-dir
    Root directory used for vector stores and embedding caches.

    Example:
        --vector-store-dir ./chroma_data_store


--output-dir
    Directory where RAG responses and retrieved context are saved.

    Example:
        --output-dir ./output_chroma_test_run


--max-tokens
    Maximum number of tokens per document chunk.

    Example:
        --max-tokens 400


--overlap-tokens
    Number of overlapping tokens between adjacent chunks.

    Example:
        --overlap-tokens 40


--top-k
    Number of document chunks returned by retrieval.

    Example:
        --top-k 5


--cosine-threshold
    Minimum similarity score required for a chunk to be included
    in the retrieved context.

    Example:
        --cosine-threshold 0.3


--collection-name
    Name of the ChromaDB collection.

    Used when:
        --backend chroma

    Example:
        --collection-name clinical_chromaDB_test_docs


--prompt-dir
    Directory containing the RAG prompt-set files.

    Example:
        --prompt-dir ./test_prompts/


--embedding-model
    Embedding model used for document and query embeddings.

    Default:
        embeddinggemma:latest

    Example:
        --embedding-model embeddinggemma:latest


--llm-model
    LLM used to generate the final response.

    Example:
        --llm-model phi4:14b


--temperature
    LLM sampling temperature.

    Example:
        --temperature 0.1


--verbose
    Enable detailed logging during execution.


--show-context
    Print the retrieved evidence/context to the console.


--show-metadata
    Print metadata for the retrieved chunks to the console.


======================================================================
VECTOR STORE EXAMPLES
======================================================================

ChromaDB:

    python LLM_Toolkit_CLI.py \
        --input-folder ./test_input \
        --markdown-dir ./CHROMA/markdown \
        --folder ./CHROMA/markdown \
        --backend chroma \
        --vector-store-dir ./chroma_data_store \
        --output-dir ./output_chroma_test_run \
        --max-tokens 400 \
        --overlap-tokens 40 \
        --top-k 5 \
        --cosine-threshold 0.3 \
        --collection-name clinical_chromaDB_test_docs \
        --prompt-dir ./test_prompts/ \
        --llm-model phi4:14b \
        --temperature 0.1 \
        --verbose


NumPy:

    python LLM_Toolkit_CLI.py \
        --input-folder ./test_input \
        --markdown-dir ./NUMPY/markdown \
        --folder ./NUMPY/markdown \
        --backend numpy \
        --vector-store-dir ./numpy_data_store \
        --output-dir ./output_numpy_test_run \
        --max-tokens 400 \
        --overlap-tokens 40 \
        --top-k 5 \
        --cosine-threshold 0.3 \
        --prompt-dir ./test_prompts/ \
        --llm-model phi4:14b \
        --temperature 0.1 \
        --verbose


FAISS:

    python LLM_Toolkit_CLI.py \
        --input-folder ./test_input \
        --markdown-dir ./FAISS/markdown \
        --folder ./FAISS/markdown \
        --backend faiss \
        --vector-store-dir ./faiss_data_store \
        --output-dir ./output_faiss_test_run \
        --max-tokens 400 \
        --overlap-tokens 40 \
        --top-k 5 \
        --cosine-threshold 0.3 \
        --prompt-dir ./test_prompts/ \
        --llm-model phi4:14b \
        --temperature 0.1 \
        --verbose


======================================================================
DEBUGGING
======================================================================

Show retrieved context:

    python LLM_Toolkit_CLI.py \
        --backend chroma \
        --show-context


Show retrieved chunks and metadata:

    python LLM_Toolkit_CLI.py \
        --backend chroma \
        --show-metadata


Show both:

    python LLM_Toolkit_CLI.py \
        --backend chroma \
        --show-context \
        --show-metadata


Enable verbose logging:

    python LLM_Toolkit_CLI.py \
        --backend chroma \
        --verbose


======================================================================
TYPICAL DIRECTORY STRUCTURE
======================================================================

    ./test_input/
        |
        |-- Original medical records
        |
        v
    ./CHROMA/markdown/
        |
        |-- Generated Markdown documents
        |
        v
    ./chroma_data_store/
        |
        |-- ChromaDB
        |-- Embedding cache
        |
        v
    ./output_chroma_test_run/
        |
        |-- RAG responses
        |-- Retrieved context


======================================================================
QUICK REFERENCE
======================================================================

    --input-folder       Original medical records or knowledgebase
    --markdown-dir       Generated Markdown directory
    --folder             Documents to ingest
    --backend             numpy | chroma | faiss
    --vector-store-dir   Vector store/cache location
    --output-dir         RAG output location
    --max-tokens         Chunk size
    --overlap-tokens     Chunk overlap
    --top-k              Number of retrieved chunks
    --cosine-threshold   Retrieval similarity threshold
    --collection-name    ChromaDB collection
    --prompt-dir         Prompt-set directory
    --embedding-model    Embedding model
    --llm-model          Generation model
    --temperature        LLM temperature
    --show-context       Print retrieved context
    --show-metadata      Print chunk metadata
    --verbose            Enable detailed logging

For the complete list of options:

    python LLM_Toolkit_CLI.py --help

""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    prompts.add_argument("--prompt-dir", type=str, default="./prompts/promptset", help="Directory containing prompt-set .txt files.",)

    # ========================================================= LLM =========================================================
    llm = parser.add_argument_group("LLM")
    llm.add_argument("--llm-model", type=str, default="phi4:14b", help="LLM model used to generate the final response.",)
    llm.add_argument("--system-prompt", type=str, default="You are a clinical information extraction assistant.", help="System prompt sent to the LLM.",)
    llm.add_argument("--temperature", type=float, default=0.1, help="LLM sampling temperature.",)

    # ============================================== Output / debugging =====================================================
    output = parser.add_argument_group("Output / debugging")
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
            prompt_dir=args.prompt_dir,
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
        # -----------------------------------------------------
        # Save all prompt-set results
        # -----------------------------------------------------
        saved_files = save_rag_output(result, config.output_dir,)
        print("\n========================================")
        print("           RAG RESULTS")
        print("========================================\n")
        print(f"Prompt sets processed: {len(result['results'])}")
        print(f"Backend:               {result['backend']}")
        print(f"Collection size:       {result['collection_size']}")
        print("\nSaved output files:")
        for path in saved_files:
            print(f"  {path}")
        # -----------------------------------------------------
        # Print responses
        # -----------------------------------------------------
        for item in result["results"]:
            print("\n========================================")
            print(f"           {item['prompt_id']}")
            print("========================================\n")
            print(item["response"])
            if args.show_context:
                print("\n---------------------------------------- RETRIEVED CONTEXT ----------------------------------------\n")
                print(item["context"])
            if args.show_metadata:
                print("\n---------------------------------------- RETRIEVED CHUNKS ----------------------------------------\n")
                print_retrieved_chunks(item["retrieved_chunks"])
        print("\nRAG pipeline completed successfully")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr,)
        return 130
    except Exception as exc:
        logging.exception("RAG pipeline failed: %s", exc,)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())




