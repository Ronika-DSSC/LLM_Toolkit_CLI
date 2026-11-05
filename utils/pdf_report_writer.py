"""
Renders the batch quality report as a formatted PDF
(data/reports/quality_report_<timestamp>.pdf).
"""

from typing import Any, Optional
from xml.sax.saxutils import escape as _esc

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_HEADER_BG = colors.HexColor("#2d3142")
_HEADER_FG = colors.white
_ROW_OK_BG = colors.HexColor("#eef7ee")
_ROW_FLAG_BG = colors.HexColor("#fdecec")
_BORDER = colors.HexColor("#d9d9df")
_MUTED = colors.HexColor("#666666")
_TEXT = colors.HexColor("#222222")

_MAX_DETAIL_CHARS = 300  # truncate long error messages so the PDF stays readable


def render_pdf_report(
    pdf_path: str,
    records: list[dict[str, Any]],
    threshold: float,
    avg_score: Optional[float],
    scored_count: int,
    failed_count: int,
    low_confidence: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    generated_at: str,
) -> None:
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, spaceAfter=2, textColor=_TEXT
    )
    meta_style = ParagraphStyle(
        "Meta", parent=styles["Normal"], textColor=_MUTED, fontSize=9, spaceAfter=2
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], spaceBefore=18, spaceAfter=8, textColor=_TEXT
    )
    cell_style = ParagraphStyle(
        "Cell", parent=styles["Normal"], fontSize=8, leading=10, alignment=TA_LEFT
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=styles["Normal"], fontSize=9, leading=13, spaceAfter=3
    )

    story: list[Any] = []
    story.append(Paragraph("Quality Score Report", title_style))
    story.append(Paragraph(f"Generated: {_esc(generated_at)}", meta_style))
    story.append(Spacer(1, 14))

    # --- Summary box ---
    summary_rows = [
        ["Total files", str(len(records))],
        ["Successfully converted", str(scored_count)],
        ["Failed", str(failed_count)],
        ["Average confidence", f"{avg_score:.2f}" if avg_score is not None else "n/a"],
        [f"Flagged for review (< {threshold:.2f})", str(len(low_confidence))],
    ]
    summary_table = Table(summary_rows, colWidths=[2.6 * inch, 2.0 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -2), 0.5, _BORDER),
                ("TEXTCOLOR", (0, 0), (-1, -1), _TEXT),
            ]
        )
    )
    story.append(summary_table)

    # --- Per-file table ---
    story.append(Paragraph("Per-file Results", section_style))

    header = ["File", "Type", "Pipeline", "Status", "Confidence Score", "Output"]
    table_data: list[list[Any]] = [header]
    row_bg: list[colors.Color] = []

    for r in records:
        conf = r.get("confidence_score")
        conf_display = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        is_flagged = isinstance(conf, (int, float)) and conf < threshold
        is_error = r.get("status") == "error"
        if is_flagged:
            conf_display += " *"  

        table_data.append(
            [
                Paragraph(_esc(r["name"]), cell_style),
                Paragraph(_esc(str(r["type"])), cell_style),
                Paragraph(_esc(str(r.get("pipeline") or "-")), cell_style),
                Paragraph(_esc(str(r["status"])), cell_style),
                Paragraph(_esc(conf_display), cell_style),
                Paragraph(_esc(str(r.get("markdown_path") or "-")), cell_style),
            ]
        )
        row_bg.append(_ROW_FLAG_BG if (is_flagged or is_error) else _ROW_OK_BG)

    col_widths = [1.5 * inch, 0.55 * inch, 0.65 * inch, 0.65 * inch, 0.75 * inch, 2.25 * inch]
    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), _HEADER_FG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, _BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
    ]
    for i, bg in enumerate(row_bg, start=1):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
    table.setStyle(TableStyle(style_cmds))
    story.append(table)

    if any(bg == _ROW_FLAG_BG for bg in row_bg):
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                f"* below the {threshold:.2f} confidence threshold, or failed",
                meta_style,
            )
        )

    # Flagged section 
    if low_confidence:
        story.append(Paragraph("Flagged for Manual Review", section_style))
        for r in low_confidence:
            conf = r["confidence_score"]
            story.append(
                Paragraph(
                    f"\u2022 <b>{_esc(r['name'])}</b> \u2014 confidence {conf:.2f}",
                    bullet_style,
                )
            )

    # Failed section 
    if failed:
        story.append(Paragraph("Failed Files", section_style))
        for r in failed:
            detail = r.get("detail") or "unknown error"
            if len(detail) > _MAX_DETAIL_CHARS:
                detail = detail[:_MAX_DETAIL_CHARS] + "..."
            story.append(
                Paragraph(f"\u2022 <b>{_esc(r['name'])}</b> \u2014 {_esc(detail)}", bullet_style)
            )

    def _footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(_MUTED)
        canvas.drawRightString(letter[0] - 0.6 * inch, 0.4 * inch, f"Page {doc.page}")
        canvas.drawString(0.6 * inch, 0.4 * inch, "LLM Toolkit : Quality Score Report")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Quality Score Report",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
