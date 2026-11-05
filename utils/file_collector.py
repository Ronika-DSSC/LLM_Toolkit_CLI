"""
Recursively flatten a list of uploaded paths (files and/or directories) into a deduplicated, sorted list of individual file paths.
"""

import os
from typing import Optional

HIDDEN_PREFIXES = (".",)  # skip dotfiles like .DS_Store, .gitkeep, etc.
IGNORED_NAMES = {"Thumbs.db", "desktop.ini"}
IGNORED_DIR_PARTS = {"__MACOSX"}

MAX_ZIP_ENTRIES = 5_000
MAX_ZIP_UNCOMPRESSED_BYTES = 2 * 1024**3  # 2 GB

def collect_files(paths: list[str]) -> list[str]:
    collected: set[str] = set()

    for path in paths:
        if not path:
            continue

        if os.path.isdir(path):
            _walk_into(path, collected)
        elif os.path.isfile(path):
            if not _is_ignored(os.path.basename(path)):
                collected.add(path)
        # silently skip anything else (e.g. a broken symlink)

    return sorted(collected)

def _walk_into(directory: str, collected: set[str]) -> None:
    for root, dirs, files in os.walk(directory):
        # Prune ignored directories in place so os.walk doesn't descend
        # into them at all.
        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIR_PARTS and not d.startswith(HIDDEN_PREFIXES)
        ]
        for fname in files:
            if _is_ignored(fname):
                continue
            collected.add(os.path.join(root, fname))

    
def _is_ignored(filename: str) -> bool:
    return filename.startswith(HIDDEN_PREFIXES) or filename in IGNORED_NAMES