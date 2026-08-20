import logging
import os
import re
import json
import time
import argparse
import logging
import hashlib
import requests
import numpy as np
from typing import List

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# ----------------------------------- Defaults / Endpoints ---------------------------------
EMBEDDING_API_URL = os.getenv("EMBEDDING_URL")
LLM_API_URL = os.getenv("LLM_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# ----------------------------------------- Logging ----------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",)


# -------------------------------------- HTTP helpers --------------------------------------
def http_post_json(url: str, headers: dict, body: dict, timeout: int = 1200) -> requests.Response:
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code} at {url}: {resp.text[:500]}")
    return resp


# -------------------------------------- Embeddings ----------------------------------------
def embed_texts(texts: List[str], embedding_model: str, batch_size: int = 16) -> np.ndarray:
    headers = {"Content-Type": "application/json"}
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

    all_embs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        body = {"model": embedding_model, "input": batch}
        resp = http_post_json(EMBEDDING_API_URL, headers, body)
        data = resp.json()

        if "embeddings" in data:
            all_embs.extend(data["embeddings"])
        elif "data" in data:
            all_embs.extend([d["embedding"] for d in data["data"]])
        else:
            raise RuntimeError(f"Unexpected embedding response schema. Keys: {list(data.keys())}")

    return np.array(all_embs, dtype=np.float32)
