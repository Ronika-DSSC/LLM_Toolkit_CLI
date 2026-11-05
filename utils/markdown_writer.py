"""
Writes pipeline output as markdown files into the project's data
directory
"""

import datetime
import os

DATA_DIR = os.environ.get("DATA_DIR", "data")
MARKDOWN_DIR = os.path.join(DATA_DIR, "markdown")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")


def ensure_data_dirs() -> None:
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

def run_markdown_dir(run_id: str) -> str:
    ensure_data_dirs()
    path = os.path.join(MARKDOWN_DIR, run_id)
    os.makedirs(path, exist_ok=True)
    return path

def output_markdown_path(source_path: str, run_id: str) -> str:
    run_dir = run_markdown_dir(run_id)
    base = os.path.basename(source_path)
    stem, ext = os.path.splitext(base)
    ext_tag = ext.lstrip(".").lower() or "file"
    return os.path.join(run_dir, f"{stem}.{ext_tag}.md")


def write_markdown(source_path: str, metadata: dict, body: str, run_id: str) -> str:
 
    out_path = output_markdown_path(source_path, run_id)

    frontmatter = ["---", f"source: {os.path.basename(source_path)}", f"run_id: {run_id}"]
    frontmatter.append(f"processed_at: {datetime.datetime.utcnow().isoformat()}Z")
    for key, value in metadata.items():
        frontmatter.append(f"{key}: {value}")
    frontmatter.append("---\n")

    content = "\n".join(frontmatter) + "\n" + body.strip() + "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    return out_path
