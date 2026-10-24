"""Frame preparation, relative depth modulation, and measured image quality."""
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from .files import image_files, read_image, save_image


def prepare_frames(source, dest, cfg):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    settings = cfg["input"]
    if settings["kind"] == "videos":
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for video input")
        filters = [f"fps={settings['fps']}"]
        fraction = settings["crop_top_fraction"]
        if fraction != 1:
            filters.append(f"crop=iw:trunc(ih*{fraction}):0:0")
        size = settings["max_long_side"]
        if size:
            filters.append(f"scale=w='min(iw,{size})':h='min(ih,{size})':force_original_aspect_ratio=decrease")
        command = ["ffmpeg", "-nostdin", "-n", "-i", str(source), "-vf", ",".join(filters),
                   str(dest / "frame_%06d.png")]
        with (dest.parent / "ffmpeg.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    else:
        for index, path in enumerate(image_files(source), 1):
            image = read_image(path)
            height = max(1, int(image.shape[0] * settings["crop_top_fraction"]))
            image = image[:height]
            size = settings["max_long_side"]
            if size and max(image.shape[:2]) > size:
                scale = size / max(image.shape[:2])
                image = cv2.resize(image, (max(1, round(image.shape[1]*scale)),
                                           max(1, round(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
            # Stable names also make COLMAP pair lists safe for source paths with spaces.
            save_image(dest / f"frame_{index:06d}.png", image)
    if not image_files(dest):
        raise ValueError(f"Frame extraction produced no images for {source}")


def normalize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("Depth must be a finite H×W array")
    span = float(np.ptp(depth))
    if span <= 1e-8:
        raise ValueError("Constant depth prediction: cannot infer relative distance")
    return (depth - depth.min()) / span


def depth_transform(image, near, sigma, gamma=1.4, levels=8, method="depth_sigma"):
    """near=1 preserves foreground; near=0 applies maximum background blur.

    depth_sigma interpolates between Gaussian scales at each pixel. legacy_blend
    recovers the soft blend structure from 4_depth_model/overlay_raw_grey.py,
    using OpenCV Gaussian smoothing rather than Pillow's approximation.
    """
    if near.shape != image.shape[:2] or not np.isfinite(near).all():
        raise ValueError("Depth mask must be finite and match image dimensions")
    if np.min(near) < 0 or np.max(near) > 1 or sigma <= 0 or gamma <= 0 or levels < 2:
        raise ValueError("Invalid depth/blur parameters")
    far = 1.0 - np.power(near, gamma)
    src = image.astype(np.float32)
    if method == "legacy_blend":
        blurred = cv2.GaussianBlur(src, (0, 0), sigma)
        result = src * (1-far[..., None]) + blurred * far[..., None]
    elif method == "depth_sigma":
        # Accumulate one scale at a time instead of allocating a full H×W×levels stack.
        position = far * (levels-1)
        result = np.zeros_like(src)
        for level in range(levels):
            weight = np.maximum(0, 1-np.abs(position-level))[..., None]
            blurred = src if level == 0 else cv2.GaussianBlur(src, (0, 0), sigma*level/(levels-1))
            result += blurred * weight
    else:
        raise ValueError(f"Unknown blur method: {method}")
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def quality_tier(value, low_max=20.0, high_min=100.0):
    if not np.isfinite(value) or value < 0:
        raise ValueError("Laplacian variance must be finite and nonnegative")
    return "low" if value <= low_max else "high" if value > high_min else "medium"


def image_quality(image, settings):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())
    return value, quality_tier(value, settings["low_max"], settings["high_min"])
