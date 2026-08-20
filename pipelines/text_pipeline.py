"""
Text pipeline: extract text from PDF/DOCX/TXT, run a deterministic
mechanical cleaning pass (fix spacing, de-hyphenate, strip control
characters).

CSV is handled separately (`_process_csv`): it's structural data, not
prose, so instead of "cleaning" it's parsed and rendered as a markdown
table 

"""

import csv
import os
import re
from typing import Any

from docx import Document
from pypdf import PdfReader

#from utils.llm_client import LLMClient
from utils.markdown_writer import write_markdown
from utils.confidence_score import score_extraction

SUPPORTED_TYPES = {"pdf", "docx", "txt", "csv"}

def process(file_path: str, file_type: str ,run_id: str,) -> dict[str, Any]:
    if file_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"text_pipeline.process() received unsupported file_type "
            f"'{file_type}'. Expected one of {sorted(SUPPORTED_TYPES)}."
        )

    if file_type == "csv":
        return _process_csv(file_path, run_id)

    raw_text, metadata = _extract_raw_text(file_path, file_type)
    mechanically_cleaned = _mechanical_clean(raw_text)

    final_text = mechanically_cleaned
    model = "n/a (non-LLM pipeline)"

    confidence = score_extraction(
        final_text,
        raw_text=raw_text,
        num_pages=metadata.get("num_pages"),
    )
    confidence_score = confidence.score

    markdown_path = write_markdown(
        source_path=file_path,
        metadata={
            "pipeline": "text",
            "source_type": file_type,
            "model": model,
            "confidence_score": confidence_score,
            **metadata,
        },
        body=final_text if final_text.strip() else "*(no text extracted)*",
        run_id=run_id,
    )

    return {
            "raw_text": raw_text,
            "cleaned_text": final_text,
            "confidence_score": confidence_score,
            "confidence_metrics": confidence.as_dict()["metrics"],
            "markdown_path": markdown_path,
            "model": "n/a (non-LLM pipeline)",
            "metadata": metadata,
        }

# Raw extraction per format
def _extract_raw_text(file_path: str, file_type: str) -> tuple[str, dict]:
    if file_type == "pdf":
        return _extract_pdf(file_path)
    if file_type == "docx":
        return _extract_docx(file_path)
    if file_type == "txt":
        return _extract_txt(file_path)
    raise ValueError(f"No raw-text extractor for '{file_type}'")


def _extract_pdf(file_path: str) -> tuple[str, dict]:
    reader = PdfReader(file_path)
    pages_text = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages_text), {"num_pages": len(reader.pages)}


def _extract_docx(file_path: str) -> tuple[str, dict]:
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs), {"num_paragraphs": len(paragraphs)}


def _extract_txt(file_path: str) -> tuple[str, dict]:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read(), {"encoding": encoding}
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read(), {"encoding": "utf-8 (with replacement)"}


# Mechanical (non-LLM) cleaning

def _mechanical_clean(text: str) -> str:
   
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    # De-hyphenate words split across a line break, e.g. "exam-\nple" -> "example"
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)
    # Collapse runs of spaces/tabs
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # Collapse 3+ blank lines down to a single blank line
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Strip stray non-printable / control characters commonly left by OCR
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", cleaned)
    # Trim trailing whitespace on each line
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    return cleaned.strip()


#CSV

def _process_csv(file_path: str, run_id: str,) -> dict[str, Any]:
    with open(file_path, "r", newline="", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(f, dialect))

    if not rows:
        return _csv_result(
            file_path, run_id, raw_text="", table_md="*(empty CSV)*",
            num_rows=0, num_cols=0, model="n/a (empty file)",
            confidence=0.0, cleaning_note="empty file",
        )

    header, *data_rows = rows
    num_rows, num_cols = len(data_rows), len(header)
    raw_text = "\n".join(",".join(row) for row in rows)
    structural_md = _rows_to_markdown_table(header, data_rows)

    if num_cols == 0:
        return _csv_result(
            file_path, run_id, raw_text=raw_text, table_md=structural_md,
            num_rows=num_rows, num_cols=num_cols, model="n/a (no columns parsed)",
            confidence=0.0, cleaning_note="no columns parsed",
        )

    # Ragged rows (cells != header width) indicate a malformed source and
    # are the main structural quality signal for a CSV.
    ragged = sum(1 for r in data_rows if len(r) != num_cols)
    ragged_ratio = ragged / max(num_rows, 1)

    final_md = structural_md
    model = "n/a (skipped LLM cleaning)"
    note = ""

    final_md = structural_md
    model = "n/a (non-LLM pipeline)"
    note = "deterministic CSV parsing"

    confidence = _score_csv(ragged_ratio)

    return _csv_result(
        file_path, run_id, raw_text=raw_text, table_md=final_md,
        num_rows=num_rows, num_cols=num_cols, model=model,
        confidence=confidence, cleaning_note=note, ragged_rows=ragged,
    )

def _validate_cleaned_table(
    candidate: str, header: list[str], expected_rows: int
) -> tuple[bool, str]:
    """
    Confirm the LLM returned the same table shape it was given.
    Checks row count, column count, and that header names are unchanged.
    """
    if not candidate or not candidate.strip():
        return False, "empty response"

    lines = [ln for ln in candidate.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return False, "not a markdown table"

    # lines[0] = header, lines[1] = separator, rest = data
    data_lines = lines[2:]
    if len(data_lines) != expected_rows:
        return False, f"row count {len(data_lines)} != {expected_rows}"

    returned_header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    if len(returned_header) != len(header):
        return False, f"column count {len(returned_header)} != {len(header)}"

    expected_header = [(h or "").strip() for h in header]
    if returned_header != expected_header:
        return False, "header names altered"

    return True, ""


def _score_csv(ragged_ratio: float) -> float:
    """
    Confidence Score for a CSV.
    """
    score = 1.0 - min(ragged_ratio, 1.0) * 0.5
    return round(max(0.0, min(1.0, score)), 3)


def _csv_result(
        file_path: str,
        run_id: str,
        *,
        raw_text: str,
        table_md: str,
        num_rows: int,
        num_cols: int,
        model: str,
        confidence: float,
        cleaning_note: str,
        ragged_rows: int = 0,
    ) -> dict[str, Any]:

    markdown_path = write_markdown(
        source_path=file_path,
        metadata={
            "pipeline": "text",
            "source_type": "csv",
            "model": model,
            "confidence_score": confidence,
            "num_rows": num_rows,
            "num_columns": num_cols,
            "ragged_rows": ragged_rows,
            "cleaning": cleaning_note or "n/a",
        },
        body=table_md,
        run_id=run_id,
    )
    return {
        "raw_text": raw_text,
        "cleaned_text": table_md,
        "confidence_score": confidence,
        "confidence_metrics": {},
        "markdown_path": markdown_path,
        "model": model,
        "metadata": {
            "num_rows": num_rows,
            "num_columns": num_cols,
            "ragged_rows": ragged_rows,
            "cleaning": cleaning_note or "n/a",
        },
    }



def _rows_to_markdown_table(header: list[str], data_rows: list[list[str]]) -> str:
    def esc(cell: str) -> str:
        return (cell or "").replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(esc(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in data_rows:
        padded = row + [""] * (len(header) - len(row))  # pad ragged rows
        lines.append("| " + " | ".join(esc(c) for c in padded[: len(header)]) + " |")
    return "\n".join(lines)