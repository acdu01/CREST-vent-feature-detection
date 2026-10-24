import csv
import hashlib
import json
from pathlib import Path
import cv2

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda obj: getattr(obj, "name", str(obj))) + "\n")
    tmp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")


def image_files(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sequences(cfg):
    """One flat folder or one level of sequence folders; no recursive data crawl."""
    root = Path(cfg["input"]["path"])
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    if cfg["input"]["kind"] == "videos":
        videos = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
        items = [(p.stem, p) for p in videos]
    else:
        folders = sorted(p for p in root.iterdir() if p.is_dir() and image_files(p))
        direct = image_files(root)
        if direct and folders:
            raise ValueError("Use either a flat image folder or sequence subfolders, not both")
        items = [(p.name, p) for p in folders] if folders else [(root.name, root)] if direct else []
    if not items:
        raise ValueError(f"No {cfg['input']['kind']} found in {root}")
    if len({name for name, _ in items}) != len(items):
        raise ValueError("Duplicate sequence names; rename videos with colliding stems")
    for name, _ in items:
        if any(c.isspace() for c in name) or name in {".", ".."}:
            raise ValueError("Sequence names must not contain whitespace")
    return items
