#!/usr/bin/env python3
"""Export the reviewable repository source; never recurse into data locations."""
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT_FILES = {"README.md", "requirements.txt", "pyproject.toml", ".gitignore"}
ROOT_DIRS = {"crest_pipeline", "configs", "docs", "tests", "scripts", "legacy"}
SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".sh", ".sbatch", ".ini", ".json", ".ipynb"}
NAMES = {"makefile", "Makefile", "LICENSE", "LICENSE.txt", ".gitignore",
         "requirements.txt", "requirements-conversion-only.txt"}


def source_payloads(root):
    root = Path(root)
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in sorted(ROOT_DIRS):
        paths.extend((root / name).rglob("*"))
    total = 0
    for path in sorted(paths):
        if not path.is_file() or path.is_symlink() or any(p.is_symlink() for p in path.parents):
            continue
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts or path.name.startswith("local") and rel.parts[0] == "configs":
            continue
        if path.suffix not in SUFFIXES and path.name not in NAMES:
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"Unexpected oversized source: {rel}")
        payload = path.read_bytes()
        if path.suffix == ".ipynb":
            notebook = json.loads(payload)
            dirty = bool(notebook.get("metadata")) or any(
                c.get("outputs") or c.get("attachments") or c.get("metadata") or c.get("execution_count") is not None
                for c in notebook["cells"])
            # Preserve bytes (and historical hashes) if already sanitized.
            if dirty:
                cells = []
                for cell in notebook["cells"]:
                    clean = {"cell_type": cell["cell_type"], "source": cell["source"], "metadata": {}}
                    if clean["cell_type"] == "code":
                        clean.update(outputs=[], execution_count=None)
                    cells.append(clean)
                payload = json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 0}).encode()
        payload.decode("utf-8")
        if b"\0" in payload:
            raise ValueError(f"Unexpected binary content: {rel}")
        total += len(payload)
        if total > 25 * 1024 * 1024:
            raise ValueError("Source export exceeds 25 MiB")
        yield rel, payload


def export(root, destination):
    # Validate before opening an output archive, so invalid files do not leave a partial export.
    files = list(source_payloads(root))
    with tarfile.open(destination, "x:gz") as archive:
        for rel, payload in files:
            entry = tarfile.TarInfo("crest-reconstruction/" + rel.as_posix())
            entry.size = len(payload)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(payload))
    return len(files)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/export_code.py /path/to/new-code-archive.tar.gz")
    count = export(Path(__file__).resolve().parents[1], Path(sys.argv[1]).expanduser())
    print(f"Exported {count} source files to {sys.argv[1]}")
