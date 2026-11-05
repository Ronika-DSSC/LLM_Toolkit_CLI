"""
Confidence scoring for pipeline output, and generation of a single
batch-level quality report (as a PDF) once all files in a run have
finished.
"""

import datetime
import os
from typing import Any, Optional

from utils.markdown_writer import REPORTS_DIR, ensure_data_dirs
from utils.pdf_report_writer import render_pdf_report

LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("QUALITY_THRESHOLD", "0.6"))

def generate_quality_report(records: list[dict[str, Any]], run_id: Optional[str] = None) -> str:

    ensure_data_dirs()
    timestamp = run_id or datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(REPORTS_DIR, f"quality_report_{timestamp}.pdf")
    generated_at = f"{datetime.datetime.utcnow().isoformat()}Z"

    scored = [r for r in records if isinstance(r.get("confidence_score"), (int, float))]
    avg_score = round(sum(r["confidence_score"] for r in scored) / len(scored), 3) if scored else None
    low_confidence = [r for r in scored if r["confidence_score"] < LOW_CONFIDENCE_THRESHOLD]
    failed = [r for r in records if r.get("status") == "error"]

    render_pdf_report(
        pdf_path=report_path,
        records=records,
        threshold=LOW_CONFIDENCE_THRESHOLD,
        avg_score=avg_score,
        scored_count=len(scored),
        failed_count=len(failed),
        low_confidence=low_confidence,
        failed=failed,
        generated_at=generated_at,
    )

    return report_path
