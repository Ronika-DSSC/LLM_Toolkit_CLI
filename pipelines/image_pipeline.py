"""
Image pipeline: extract text from images using LLM
"""

from typing import Any

from PIL import Image

from utils.llm_client import llm_client
from utils.markdown_writer import write_markdown
from utils.confidence_score import score_extraction



def process(file_path: str, run_id: str) -> dict[str, Any]:
    
    extraction = llm_client.extract_text_from_image(file_path)
    extracted_text = extraction["text"]
    model = extraction["model"]

    metadata = _image_metadata(file_path)
    confidence = score_extraction(extracted_text, llm_client=llm_client)
    confidence_score = confidence.score

    body = extracted_text if extracted_text.strip() else "*(no text extracted)*"
    markdown_path = write_markdown(
        source_path=file_path,
        metadata={
            "pipeline": "image",
            "model": model,
            "confidence_score": confidence_score,
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "format": metadata.get("format"),
        },
        body=body,
        run_id=run_id,
    )

    return {
        "extracted_text": extracted_text,
        "confidence_score": confidence_score,
        "confidence_metrics": confidence.as_dict()["metrics"],
        "markdown_path": markdown_path,
        "model": model,
        "metadata": metadata,
    }



def _image_metadata(file_path: str) -> dict[str, Any]:
    try:
        with Image.open(file_path) as img:
            return {"width": img.width, "height": img.height, "format": img.format}
    except Exception:
        return {"width": None, "height": None, "format": None}
