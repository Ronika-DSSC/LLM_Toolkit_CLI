import re
import os
from typing import List, Dict


# --------------------------------- Prompt-set parsing ---------------------------------
PROMPTSET_RE = re.compile(r"^\s*#+\s*PROMPT_SET\s*:\s*(?P<name>.+?)\s*$", re.M)


def safe_stem(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")


def parse_promptset_text(text: str) -> List[Dict[str, str]]:
    """Parse delimiter-based prompt-set files.

    Expected structure per set:
      #### PROMPT_SET: <name>
      ---RAG---
      ...
      ---USER---
      ...

    Returns list of dicts: {name, rag, user}
    """
    if not text:
        return []

    # Find all set headers
    headers = list(PROMPTSET_RE.finditer(text))
    if not headers:
        # allow files without header -> treat whole file as one set if contains delimiters
        headers = [re.match(r"^", text)]

    sets: List[Dict[str, str]] = []

    # If we used the dummy header, name becomes stem later.
    if headers and headers[0] is not None and hasattr(headers[0], 'groupdict') and 'name' not in headers[0].groupdict():
        # This shouldn't happen with our dummy match; handled below.
        pass

    # Build segments between headers
    for i, h in enumerate(headers):
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        # name
        mname = PROMPTSET_RE.search(block)
        name = mname.group('name').strip() if mname else "prompt_set"
        # split on delimiters
        rag_marker = re.search(r"---\s*RAG\s*---", block, re.I)
        user_marker = re.search(r"---\s*USER\s*---", block, re.I)
        if not rag_marker or not user_marker:
            continue
        rag_text = block[rag_marker.end():user_marker.start()].strip()
        user_text = block[user_marker.end():].strip()
        if rag_text and user_text:
            sets.append({"name": name, "rag": rag_text, "user": user_text})

    return sets


def load_promptsets_from_file(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    sets = parse_promptset_text(text)
    if not sets:
        raise RuntimeError(f"No PROMPT_SET blocks found in {path}. Expected '#### PROMPT_SET:' plus ---RAG--- and ---USER---.")

    # Ensure unique names by prefixing file stem if needed
    stem = safe_stem(path)
    out = []
    seen = set()
    for s in sets:
        name = safe_stem(s['name'])
        full_name = f"{stem}__{name}" if len(sets) > 1 else stem
        # If multiple sets exist, keep file__setname; if single set, use file stem
        if len(sets) > 1:
            full_name = f"{stem}__{name}"
        else:
            full_name = stem
        # Avoid collisions
        if full_name in seen:
            j = 2
            while f"{full_name}_{j}" in seen:
                j += 1
            full_name = f"{full_name}_{j}"
        seen.add(full_name)
        out.append({"id": full_name, "rag": s['rag'], "user": s['user']})
    return out




