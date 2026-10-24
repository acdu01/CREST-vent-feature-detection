"""Strict, portable configuration. Relative paths are relative to the YAML file."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

DEFAULTS = {
    "input": {"kind": "images", "path": "../data/images", "fps": 1.0,
              "crop_top_fraction": 1.0, "max_long_side": None},
    "output": "../runs",
    "device": "cpu",
    "variants": ["sift_raw", "sift_depth_low", "sift_depth_high",
                 "rdd_raw", "rdd_depth_low", "rdd_depth_high"],
    "depth": {"model_path": "../weights/depth-anything-v1", "mask_root": None,
              "higher_is_near": True},
    "preprocess": {"method": "depth_sigma", "low_sigma": 4.0,
                   "high_sigma": 12.0, "levels": 8, "gamma": 1.4},
    "rdd": {"repo": "../external/rdd", "checkpoint": "../weights/rdd.pth",
            "config": "rdd_cvpr.yaml", "top_k": 4096, "threshold": 0.01,
            "ratio": 0.8, "max_distance": 0.7, "block_size": 256},
    "quality": {"low_max": 20.0, "high_min": 100.0, "grouping": "per_image"},
}
VARIANTS = tuple(DEFAULTS["variants"])


def load_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a mapping")
    cfg = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key not in cfg:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(cfg[key], dict):
            if not isinstance(value, dict) or set(value) - set(cfg[key]):
                raise ValueError(f"Invalid keys in {key}")
            cfg[key].update(value)
        else:
            cfg[key] = value
    if cfg["input"]["kind"] not in {"images", "videos"}:
        raise ValueError("input.kind must be images or videos")
    if cfg["device"] not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not isinstance(cfg["variants"], list) or not cfg["variants"]:
        raise ValueError("variants must be a nonempty list")
    if any(v not in VARIANTS for v in cfg["variants"]) or len(set(cfg["variants"])) != len(cfg["variants"]):
        raise ValueError(f"variants must be unique members of {VARIANTS}")
    for section, key in [("input", "fps"), ("preprocess", "gamma"),
                         ("preprocess", "low_sigma"), ("preprocess", "high_sigma"),
                         ("rdd", "max_distance")]:
        x = cfg[section][key]
        if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise ValueError(f"{section}.{key} must be finite and positive")
    for section, key, minimum in [("preprocess", "levels", 2), ("rdd", "top_k", 2),
                                  ("rdd", "block_size", 1)]:
        x = cfg[section][key]
        if type(x) is not int or x < minimum:
            raise ValueError(f"{section}.{key} must be an integer >= {minimum}")
    if not 0 < cfg["input"]["crop_top_fraction"] <= 1:
        raise ValueError("crop_top_fraction must be in (0, 1]")
    size = cfg["input"]["max_long_side"]
    if size is not None and (type(size) is not int or size < 32):
        raise ValueError("max_long_side must be null or an integer >= 32")
    if cfg["preprocess"]["method"] not in {"depth_sigma", "legacy_blend"}:
        raise ValueError("Unknown preprocessing method")
    if cfg["preprocess"]["high_sigma"] < cfg["preprocess"]["low_sigma"]:
        raise ValueError("high_sigma must be >= low_sigma")
    if not 0 < cfg["rdd"]["ratio"] < 1 or not 0 <= cfg["rdd"]["threshold"] <= 1:
        raise ValueError("RDD ratio/threshold out of range")
    if not 0 <= cfg["quality"]["low_max"] < cfg["quality"]["high_min"]:
        raise ValueError("Quality thresholds must satisfy 0 <= low_max < high_min")
    if cfg["quality"]["grouping"] not in {"per_image", "sequence_mean"}:
        raise ValueError("quality.grouping must be per_image or sequence_mean")
    if type(cfg["depth"]["higher_is_near"]) is not bool:
        raise ValueError("depth.higher_is_near must be boolean")
    for section, key in [("input", "path"), ("depth", "model_path"),
                         ("depth", "mask_root"), ("rdd", "repo"),
                         ("rdd", "checkpoint"), ("rdd", "config")]:
        value = cfg[section][key]
        if value is None and (section, key) != ("depth", "mask_root"):
            raise ValueError(f"{section}.{key} must be a path")
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section}.{key} must be a nonempty path string")
            p = Path(value).expanduser()
            cfg[section][key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    if not isinstance(cfg["output"], str) or not cfg["output"].strip():
        raise ValueError("output must be a nonempty path string")
    out = Path(cfg["output"]).expanduser()
    cfg["output"] = str((path.parent / out).resolve() if not out.is_absolute() else out.resolve())
    return cfg
