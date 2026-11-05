"""
LLM call for :
  - OCR-style text extraction from images
  - LLM-assisted text cleaning + confidence scoring for extracted document text
"""

import base64
import mimetypes
import os
import re
from typing import Any
from urllib import response

from gradio.mcp import prompt
import requests

from dotenv import load_dotenv

load_dotenv()

# LLM_URL = os.getenv("LLM_URL", "").rstrip("/")
# LLM_MODEL = os.getenv("LLM_MODEL", "")
# LLM_VISION_MODEL = os.getenv("LLM_VISION_MODEL", "")
# LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MARKDOWN_FENCE_RE = re.compile(r"^```(?:markdown)?\s*\n(.*)\n```\s*$", re.DOTALL)

VISION_EXTRACTION_PROMPT = (
    "Extract all readable text from this image exactly as it appears. "
    "Preserve reading order and paragraph breaks. Respond with the "
    "extracted text only, no commentary, no markdown fences."
)

TABLE_CLEANING_PROMPT_TEMPLATE = (
    "Below is a markdown table extracted from a CSV file. Clean up only "
    "the FORMATTING of the cell contents:\n"
    "  - trim leading/trailing whitespace inside cells\n"
    "  - collapse repeated internal spaces\n"
    "  - fix obvious character-encoding artifacts (e.g. 'Ã©' -> 'e')\n"
    "  - normalize placeholder empty values (NULL, N/A, NaN, --) to empty\n\n"
    "STRICT RULES — these matter more than any cleanup:\n"
    "  - Do NOT change, correct, reformat, round, or reinterpret any data "
    "value, number, date, name, code, or identifier.\n"
    "  - Do NOT add, remove, reorder, merge, or split any row or column.\n"
    "  - Do NOT change the header row's column names.\n"
    "  - Return EXACTLY the same number of rows and columns.\n"
    "  - If a cell is already clean, return it byte-for-byte unchanged.\n\n"
    "Respond with ONLY the markdown table. No commentary, no code fences.\n\n"
    "{table}"
)

TEXT_CLEANING_PROMPT_TEMPLATE = (
    "The following text was extracted from a {source_type} document, "
    "possibly via OCR, and may contain spacing errors, broken words, or "
    "scanning noise. Clean it up: fix spacing, rejoin words that were "
    "incorrectly split across lines, and remove obvious OCR artifacts, "
    "*without* changing the actual wording or meaning. Respond with the "
    "cleaned text only, no commentary, no markdown fences.\n\n"
    "---\n{raw_text}\n---"
)


class LLMClient:
    def __init__(
        self,
        llm_url: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ):
        self.llm_url = (
            llm_url
            if llm_url is not None
            else os.getenv("LLM_URL", "")
        ).rstrip("/")

        self.model = (
            model
            if model is not None
            else os.getenv("LLM_MODEL", "")
        )

        self.vision_model = (
            vision_model
            if vision_model is not None
            else os.getenv("LLM_VISION_MODEL", "")
        )

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("LLM_API_KEY", "")
        )

        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
        )

        self.temperature = temperature
        self.max_tokens = max_tokens

        if not self.llm_url:
            raise RuntimeError("LLM_URL is not set.")

        if not self.model:
            raise RuntimeError("LLM model is not set.")
        

    def extract_text_from_image(self, image_path: str) -> dict[str, Any]:
        
        if not self.vision_model:
            raise RuntimeError(
                "Image text extraction requires LLM_VISION_MODEL to be set "
                "to a vision-capable model."
            )
       
        b64 = self._encode_image_base64(image_path)
        mime = mimetypes.guess_type(image_path)[0] or "image/png"

        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
        }
        response = self._call_chat_completions(payload)
        text = self._extract_message_content(response)
        confidence = self._estimate_confidence(response, text)
        return {"text": text, "confidence": confidence, "model": self.vision_model}

    def clean_and_score_text(self, raw_text: str, source_type: str) -> dict[str, Any]:
        
            prompt = TEXT_CLEANING_PROMPT_TEMPLATE.format(
                source_type=source_type, raw_text=raw_text
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            # Add max_tokens only when it was provided
            if self.max_tokens is not None:
                payload["max_tokens"] = self.max_tokens
            response = self._call_chat_completions(payload)
            cleaned = self._extract_message_content(response)
            confidence = self._estimate_confidence(response, cleaned)
            return {"cleaned_text": cleaned, "confidence": confidence, "model": self.model}

    def clean_table(self, table_markdown: str) -> dict[str, Any]:
        
        prompt = TABLE_CLEANING_PROMPT_TEMPLATE.format(table=table_markdown)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        # Add max_tokens only when it was provided
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        response = self._call_chat_completions(payload)
        cleaned = self._extract_message_content(response)
        return {"cleaned_table": cleaned, "model": self.model}
    
    def assess_quality(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        # Add max_tokens only when it was provided
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        response = self._call_chat_completions(payload)
        return self._extract_message_content(response)
    
    def _call_chat_completions(self, payload: dict) -> dict:
        url = self.llm_url
        headers = {"Content-Type": "application/json",}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_seconds,)
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for url {url}, "
                f"response body: {resp.text[:2000]}",
                response=resp,
            )
        return resp.json()

    @staticmethod
    def _extract_message_content(response: dict) -> str:
        try:
            content = (response["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected LLM response shape: {response}") from e

        content = _THINK_TAG_RE.sub("", content).strip()
        fence_match = _MARKDOWN_FENCE_RE.match(content)
        if fence_match:
            content = fence_match.group(1).strip()
        return content
        

    @staticmethod
    def _estimate_confidence(response: dict, text: str) -> float:
        if not text or not text.strip():
                return 0.0
        try:
                finish_reason = response["choices"][0].get("finish_reason")

        except (KeyError, IndexError, TypeError):
                finish_reason = None
            
        if finish_reason == "length":
                return 0.4
        return 0.8

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

