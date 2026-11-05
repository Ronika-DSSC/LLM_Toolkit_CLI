"""
Content-aware file type detection.
Classifies an uploaded file into one of: "image", "pdf", "docx", "txt", "csv", "unknown".

"""

import mimetypes
import os

try:
    import magic  
    _HAS_MAGIC = True
except ImportError: 
    _HAS_MAGIC = False


EXTENSION_MAP = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".gif": "image",
    ".webp": "image",
    "bmp":"image",
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".csv":"csv"
}

# Mapping of MIME type (or prefix) -> internal file type label.
MIME_PREFIX_MAP = {
    "image/": "image",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/csv": "csv",
    "text/plain": "txt",
}


def _mime_to_type(mime: str) -> str | None:
    if not mime:
        return None
    for prefix, ftype in MIME_PREFIX_MAP.items():
        if mime.startswith(prefix):
            return ftype
    return None


def detect_file_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    ext_guess = EXTENSION_MAP.get(ext)

    content_guess = None

    if _HAS_MAGIC:
        try:
            mime = magic.from_file(file_path, mime=True)
            content_guess = _mime_to_type(mime)

            if content_guess is None and mime == "application/zip" and ext_guess == "docx":
                content_guess = "docx"
            
            if content_guess == "txt" and ext_guess == "csv":
                content_guess = "csv"
        
        except Exception:
            content_guess = None

    if content_guess is None:
        mime_type, _ = mimetypes.guess_type(file_path)
        content_guess = _mime_to_type(mime_type or "")

    return content_guess or ext_guess or "unknown"
