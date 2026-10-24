"""Run orchestration with bounded image working sets and explicit provenance."""
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata, util
from pathlib import Path
import os
import platform
import re
import shutil
import subprocess
import uuid
import cv2
import numpy as np
from .files import image_files, read_image, save_image, sequences, sha256, write_csv, write_json
from .metrics import MODEL_FIELDS, AGGREGATE_FIELDS, aggregate, model_metrics
from .preprocess import prepare_frames, depth_transform, image_quality, quality_tier


def needs_depth(cfg):
    return any("_depth_" in v for v in cfg["variants"])


def git_info(root):
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                                           text=True, timeout=10).strip()
        return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short")}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None}


def preflight(cfg, until="reconstruct"):
    problems = []
    try:
        sequences(cfg)
    except ValueError as exc:
        problems.append(str(exc))
    source, output = Path(cfg["input"]["path"]), Path(cfg["output"])
    if source == output or source in output.parents or output in source.parents:
        problems.append("Input and output directories must not contain each other")
    if cfg["input"]["kind"] == "videos" and not shutil.which("ffmpeg"):
        problems.append("ffmpeg is not installed")
    if needs_depth(cfg):
        mask = cfg["depth"]["mask_root"]
        if mask:
            if not Path(mask).is_dir():
                problems.append(f"Depth mask directory missing: {mask}")
        else:
            if not Path(cfg["depth"]["model_path"]).is_dir():
                problems.append("Set depth.model_path to a local Hugging Face Depth Anything V1 model directory")
            for name in ["torch", "transformers", "PIL"]:
                if util.find_spec(name) is None:
                    problems.append(f"Depth inference dependency missing: {name}")
    if until == "reconstruct":
        try:
            from .sfm import get_colmap
            get_colmap()
        except (ImportError, RuntimeError) as exc:
            problems.append(f"COLMAP unavailable: {exc}")
        if any(v.startswith("rdd_") for v in cfg["variants"]):
            for name in ["torch", "torchvision", "yacs", "timm", "kornia", "einops"]:
                if util.find_spec(name) is None:
                    problems.append(f"RDD dependency missing: {name}")
            for key in ["checkpoint", "config"]:
                if not Path(cfg["rdd"][key]).is_file():
                    problems.append(f"RDD {key} missing: {cfg['rdd'][key]}")
            if not (Path(cfg["rdd"]["repo"]) / "src/RDD.py").is_file():
                problems.append("Set rdd.repo to the official xtcpete/rdd checkout")
    if cfg["device"] == "cuda" and util.find_spec("torch"):
        import torch
        if not torch.cuda.is_available():
            problems.append("device=cuda requested, but torch cannot access CUDA")
    return problems


def provenance(cfg, until):
    root = Path(__file__).resolve().parent.parent
    versions = {}
    for name in ["numpy", "opencv-python", "opencv-python-headless", "torch", "transformers", "pycolmap", "PyYAML"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    result = {"python": platform.python_version(), "packages": versions,
              "source_git": git_info(root),
              "source_sha256": {p.name: sha256(p) for p in sorted((root / "crest_pipeline").glob("*.py"))}}
    if needs_depth(cfg) and not cfg["depth"]["mask_root"]:
        model = Path(cfg["depth"]["model_path"])
        result["depth_files_sha256"] = {p.relative_to(model).as_posix(): sha256(p)
                                        for p in sorted(model.rglob("*")) if p.is_file()}
    if until == "reconstruct" and any(v.startswith("rdd_") for v in cfg["variants"]):
        result["rdd_git"] = git_info(cfg["rdd"]["repo"])
        result["rdd_checkpoint_sha256"] = sha256(cfg["rdd"]["checkpoint"])
        result["rdd_config_sha256"] = sha256(cfg["rdd"]["config"])
    return result


def external_mask(path, shape, higher_is_near):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Missing depth mask: {path}")
    if mask.ndim == 3:
        if mask.shape[2] != 3 or not (np.array_equal(mask[..., 0], mask[..., 1]) and
                                      np.array_equal(mask[..., 0], mask[..., 2])):
            raise ValueError(f"Depth masks must be grayscale, not colorized: {path}")
        mask = mask[..., 0]
    if mask.shape != shape or mask.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Depth mask must be uint8/uint16 and align with prepared image dimensions: {path}")
    mask = mask.astype(np.float32) / np.iinfo(mask.dtype).max
    return mask if higher_is_near else 1-mask


def quality_groups(images, values, tiers, settings, destination):
    """Partition measured frame tiers, or explicitly choose whole-sequence analysis."""
    paths = image_files(images)
    if settings["grouping"] == "sequence_mean":
        mean = float(np.mean(values))
        return [(quality_tier(mean, settings["low_max"], settings["high_min"]),
                 Path(images), len(paths), mean)]
    groups = []
    for tier in ("low", "medium", "high"):
        selected = [(p, v) for p, v, t in zip(paths, values, tiers) if t == tier]
        if not selected:
            continue
        folder = Path(destination) / tier
        folder.mkdir(parents=True)
        for path, _ in selected:
            # Only link files we generated inside this run; originals are never linked.
            os.link(path, folder / path.name)
        groups.append((tier, folder, len(selected), float(np.mean([v for _, v in selected]))))
    return groups


def run(cfg, run_id=None, until="reconstruct"):
    if until not in {"prepare", "reconstruct"}:
        raise ValueError("until must be prepare or reconstruct")
    problems = preflight(cfg, until)
    if problems:
        raise ValueError("Preflight failed:\n- " + "\n- ".join(problems))
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a simple directory name")
    output = Path(cfg["output"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "kind": "reconstructed_method_not_historical_reproduction",
                "started_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                "until": until, "sequences": []}
    write_json(output / "run.json", manifest)
    quality_rows, results = [], []
    try:
        manifest["provenance"] = provenance(cfg, until)
        snapshot = output / "source"
        snapshot.mkdir()
        for source in Path(__file__).parent.glob("*.py"):
            shutil.copy2(source, snapshot / source.name)
        write_json(output / "run.json", manifest)
        predictor = extractor = None
        if needs_depth(cfg) and not cfg["depth"]["mask_root"]:
            from .models import DepthPredictor
            predictor = DepthPredictor(cfg["depth"]["model_path"], cfg["device"])
        if until == "reconstruct" and any(v.startswith("rdd_") for v in cfg["variants"]):
            from .models import RDDExtractor
            extractor = RDDExtractor(cfg["rdd"], cfg["device"])
        for sequence, source in sequences(cfg):
            print(f"Preparing {sequence}", flush=True)
            seq_dir = output / "sequences" / sequence
            frames = seq_dir / "frames"
            prepare_frames(source, frames, cfg)
            frame_paths = image_files(frames)
            record = {"name": sequence, "source": str(source), "images": len(frame_paths)}
            if cfg["input"]["kind"] == "images":
                record["frame_sources"] = [
                    {"frame": target.name, "source": str(original),
                     "source_size": original.stat().st_size,
                     "source_mtime_ns": original.stat().st_mtime_ns,
                     "prepared_sha256": sha256(target)}
                    for original, target in zip(image_files(source), frame_paths)]
            else:
                record.update(source_size=source.stat().st_size, source_mtime_ns=source.stat().st_mtime_ns)
                record["frame_sources"] = [{"frame": p.name, "prepared_sha256": sha256(p)} for p in frame_paths]
            if needs_depth(cfg):
                depth_dir = seq_dir / "depth"
                depth_dir.mkdir()
                mask_hashes = {}
                for frame in frame_paths:
                    image = read_image(frame)
                    if predictor:
                        near = predictor(image)
                        if not cfg["depth"]["higher_is_near"]:
                            near = 1-near
                    else:
                        path = Path(cfg["depth"]["mask_root"]) / sequence / frame.name
                        near = external_mask(path, image.shape[:2], cfg["depth"]["higher_is_near"])
                        mask_hashes[str(path)] = sha256(path)
                    np.save(depth_dir / (frame.stem + ".npy"), near, allow_pickle=False)
                record["external_mask_sha256"] = mask_hashes
            manifest["sequences"].append(record)
            write_json(output / "run.json", manifest)
            # Share prepared images across SIFT and RDD; process one frame at a time.
            processed = {"raw": frames}
            for variant in cfg["variants"]:
                method, transform = variant.split("_", 1)
                if transform not in processed:
                    dest = seq_dir / transform
                    dest.mkdir()
                    settings = cfg["preprocess"]
                    sigma = settings["low_sigma"] if transform == "depth_low" else settings["high_sigma"]
                    for frame in frame_paths:
                        near = np.load(seq_dir / "depth" / (frame.stem + ".npy"), allow_pickle=False)
                        image = depth_transform(read_image(frame), near, sigma, settings["gamma"],
                                                settings["levels"], settings["method"])
                        save_image(dest / frame.name, image)
                    processed[transform] = dest
                images = processed[transform]
                values, tiers = [], []
                for frame in image_files(images):
                    value, tier = image_quality(read_image(frame), cfg["quality"])
                    values.append(value)
                    tiers.append(tier)
                    quality_rows.append({"sequence": sequence, "variant": variant, "frame": frame.name,
                                         "laplacian_variance": value, "quality": tier})
                print(f"  {variant}: {len(values)} images; quality counts {dict(Counter(tiers))}", flush=True)
                if until == "reconstruct":
                    from .sfm import reconstruct
                    groups = quality_groups(images, values, tiers, cfg["quality"],
                                            seq_dir / (variant + "_quality_inputs"))
                    for tier, group_images, count, mean_quality in groups:
                        models = reconstruct(group_images, seq_dir / variant / tier, method, cfg, extractor) if count >= 2 else {}
                        rows = model_metrics(models, sequence, variant, count, tier, mean_quality)
                        if count < 2:
                            rows[0]["status"] = "insufficient_images"
                        results.extend(rows)
                write_csv(output / "image-quality.csv", quality_rows,
                          ["sequence", "variant", "frame", "laplacian_variance", "quality"])
                write_csv(output / "models.csv", results, MODEL_FIELDS)
                write_csv(output / "summary.csv", aggregate(results), AGGREGATE_FIELDS)
        manifest["status"] = "prepared" if until == "prepare" else (
            "completed_with_no_model" if any(r["status"] != "success" for r in results) else "completed")
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", manifest)
    return output
