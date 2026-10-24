"""Adapters for pretrained models; weights must already exist on the run host.

Interfaces: https://github.com/xtcpete/rdd and
https://huggingface.co/docs/transformers/model_doc/depth_anything .
No model weights or training datasets are bundled or automatically fetched.
"""
import importlib
from pathlib import Path
import sys
import cv2
import numpy as np
from .preprocess import normalize_depth


class DepthPredictor:
    def __init__(self, model_path, device):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=True).to(device)
        if self.model.config.model_type != "depth_anything":
            raise ValueError("Expected a Depth Anything checkpoint")
        self.model.eval()

    def __call__(self, image):
        inputs = self.processor(images=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            depth = self.model(**inputs).predicted_depth
            depth = self.torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=image.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        return normalize_depth(depth.cpu().numpy())


class RDDExtractor:
    def __init__(self, settings, device):
        import torch
        self.torch = torch
        self.device = device
        repo = Path(settings["repo"]).resolve()
        sys.path.insert(0, str(repo))
        module = importlib.import_module("src.RDD")
        if repo not in Path(module.__file__).resolve().parents:
            raise RuntimeError("Imported src.RDD from another checkout; use a fresh Python process")
        from src.config.default import get_cfg_defaults
        from src.utils import lower_config
        cfg = get_cfg_defaults()
        cfg.merge_from_file(settings["config"])
        config = lower_config(cfg)["rdd"]
        config["top_k"] = settings["top_k"]
        config["detection_thr"] = settings["threshold"]
        # Full checkpoint supplies the backbone; do not fetch ImageNet weights.
        config["descriptor"]["pretrain_backbone"] = False
        self.model = module.build(config=config, weights=settings["checkpoint"]).to(device)
        # Upstream RDD.eval() does not return self in all revisions.
        self.model.eval()

    def __call__(self, image):
        if min(image.shape[:2]) < 32:
            raise ValueError("RDD images must be at least 32 pixels on each side")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(self.device) / 255.0
        with self.torch.inference_mode():
            output = self.model.extract(tensor)[0]
        keypoints = output["keypoints"].detach().cpu().numpy().astype(np.float32)
        descriptors = output["descriptors"].detach().cpu().numpy().astype(np.float32)
        if keypoints.shape != (len(descriptors), 2) or descriptors.ndim != 2 or descriptors.shape[1] != 256:
            raise ValueError("RDD must return aligned N×2 keypoints and N×256 descriptors")
        if not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all():
            raise ValueError("RDD produced non-finite features")
        if len(keypoints) and (np.any(keypoints < 0) or
                              np.any(keypoints[:, 0] >= image.shape[1]) or
                              np.any(keypoints[:, 1] >= image.shape[0])):
            raise ValueError("RDD keypoints are outside the input image")
        return keypoints, descriptors
