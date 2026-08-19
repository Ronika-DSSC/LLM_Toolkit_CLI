"""
LLM interaction module.

Responsibilities
----------------
- Format retrieved context
- Send prompts to the LLM
- Return model output
- Optionally parse JSON responses
"""

import os
import json
from typing import Optional, Any

from embedding_api import http_post_json

from dotenv import load_dotenv

load_dotenv()


LLM_API_URL = os.getenv( "LLM_URL","")
LLM_API_KEY = os.getenv("LLM_API_KEY","")


def call_chat_llm(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2,) -> str:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    body = {"model": model, "temperature": float(temperature), "stream": False, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],}
    resp = http_post_json(LLM_API_URL, headers, body,)
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> Optional[Any]:
    """
    Extract JSON object from LLM response.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    return None



