from pipelines import text_pipeline
from pathlib import Path
import argparse
import logging


def generate_markdown(
    input_path: str,
    output_dir: str,
    run_id: str | None = None,
    llm_model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Run the existing text pipeline and generate a Markdown file
    containing the YAML metadata/front matter, including confidence_score.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_type = input_path.suffix.lower().lstrip(".")

    if file_type not in {"pdf", "docx", "txt", "csv"}:
        raise ValueError(f"Unsupported text file type: {file_type}")

    logging.info("Processing: %s", input_path)

    # Use your EXISTING text pipeline.
    result = text_pipeline.process(
        file_path=str(input_path),
        file_type=file_type,
        run_id=run_id,
    )

    logging.info("Pipeline result: %s", result)

    # Your text_pipeline should already generate markdown_path.
    markdown_path = result.get("markdown_path")

    if not markdown_path:
        raise RuntimeError(
            f"text_pipeline did not return markdown_path for {input_path}"
        )

    markdown_path = Path(markdown_path)

    # If the pipeline generated it somewhere else, optionally copy/move it
    # into the RAG markdown directory.
    target_path = output_dir / markdown_path.name

    if markdown_path.resolve() != target_path.resolve():
        target_path.write_text(markdown_path.read_text(encoding="utf-8"), encoding="utf-8",)

    return {
        "source": str(input_path),
        "markdown_path": str(target_path),
        "confidence_score": result.get("confidence_score"),
        "confidence_metrics": result.get("confidence_metrics", {}),
        "status": result.get("status", "done"),
    }


def generate_folder(
    input_dir: str,
    output_dir: str,
    run_id: str | None = None,
    llm_model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """
    Process all supported text documents in a directory.
    """
    input_dir = Path(input_dir)
    supported = {".pdf", ".docx", ".txt", ".csv",}
    results = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported:
            continue
        try:
            result = generate_markdown(
                input_path=str(path),
                output_dir=output_dir,
                run_id=run_id,
                llm_model=llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            results.append(result)
            print(f"OK: {path.name} -> " f"{result['markdown_path']} " f"(confidence={result['confidence_score']})")
        except Exception as exc:
            logging.exception("Failed to process %s: %s", path, exc,)
            results.append({"source": str(path), "status": "error", "error": str(exc),})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Markdown files using the existing text pipeline."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input file or folder.",
    )

    parser.add_argument(
        "--output-dir",
        default="./data/markdown",
        help="Directory where Markdown files should be placed.",
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional pipeline run ID.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_path = Path(args.input)

    if input_path.is_file():
        result = generate_markdown(
            input_path=str(input_path),
            output_dir=args.output_dir,
            run_id=args.run_id,
        )

        print("\nGenerated Markdown:")
        print(result["markdown_path"])
        print("Confidence:", result["confidence_score"])

    elif input_path.is_dir():
        results = generate_folder(
            input_dir=str(input_path),
            output_dir=args.output_dir,
            run_id=args.run_id,
        )

        print(f"\nProcessed {len(results)} file(s).")

    else:
        raise FileNotFoundError(
            f"Input does not exist: {input_path}"
        )



