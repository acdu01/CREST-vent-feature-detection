"""Small synthetic geometry tests; no research datasets or learned weights."""
from copy import deepcopy
import json
import cv2
import numpy as np
import pytest

colmap = pytest.importorskip("pycolmap")
pytestmark = pytest.mark.skipif(colmap.__version__ != "3.13.0", reason="requires paper COLMAP version")

from crest_pipeline.config import DEFAULTS
from crest_pipeline.files import save_image
from crest_pipeline.sfm import reconstruct
from crest_pipeline.pipeline import run


def test_rdd_bridge_and_geometric_reconstruction(tmp_path):
    rng = np.random.default_rng(21)
    xyz = rng.uniform([-1, -0.7, 5], [1, 0.7, 8], (200, 3))
    descriptors = rng.normal(size=(200, 256)).astype(np.float32)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    keypoints = {}
    for i in range(5):
        camera = np.array([(i-2)*0.35, (i % 2)*0.05, 0.0])
        relative = xyz-camera
        # ImageReader's unknown-EXIF focal length prior: 1.2 * max(width,height).
        xy = relative[:, :2] / relative[:, 2:] * (1.2*256) + [128, 96]
        keypoints[i] = xy.astype(np.float32)-0.5
        save_image(image_dir / f"frame_{i:06d}.png", np.full((192, 256, 3), i, dtype=np.uint8))
    def extractor(image):
        return keypoints[int(image[0, 0, 0])], descriptors.copy()
    cfg = deepcopy(DEFAULTS)
    models = reconstruct(image_dir, tmp_path / "sfm", "rdd", cfg, extractor)
    assert models
    best = max(models.values(), key=lambda model: model.num_reg_images())
    assert best.num_reg_images() == 5
    assert best.num_points3D() >= 180
    assert best.compute_mean_reprojection_error() < 0.1
    assert len((tmp_path / "sfm/pairs.txt").read_text().splitlines()) == 10
    db = colmap.Database.open(str(tmp_path / "sfm/database.db"))
    try:
        ids = sorted(image.image_id for image in db.read_all_images())
        assert db.read_matches(ids[0], ids[1]).shape == (200, 2)
        assert db.read_keypoints(ids[0]).shape == (200, 2)
    finally:
        db.close()


def test_sift_native_path(tmp_path):
    rng = np.random.default_rng(22)
    base = rng.integers(0, 256, (128, 160, 3), dtype=np.uint8)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(3):
        image = cv2.warpAffine(base, np.float32([[1, 0, 3*i], [0, 1, i]]), (160, 128))
        save_image(image_dir / f"frame_{i:06d}.png", image)
    reconstruct(image_dir, tmp_path / "sfm", "sift", deepcopy(DEFAULTS))
    db = colmap.Database.open(str(tmp_path / "sfm/database.db"))
    try:
        images = db.read_all_images()
        assert len(images) == 3
        assert all(db.read_descriptors(image.image_id).shape[1] == 128 for image in images)
    finally:
        db.close()
    assert json.loads((tmp_path / "sfm/colmap-options.json").read_text())["version"] == "3.13.0"


def test_full_sift_run_records_unsuccessful_geometry(tmp_path):
    source = tmp_path / "input" / "blank"
    source.mkdir(parents=True)
    for i in range(2):
        save_image(source / f"{i}.png", np.full((64, 64, 3), 100+i, dtype=np.uint8))
    cfg = deepcopy(DEFAULTS)
    cfg["input"]["path"] = str(source.parent)
    cfg["output"] = str(tmp_path / "runs")
    cfg["variants"] = ["sift_raw"]
    result = run(cfg, "blank-scene")
    manifest = json.loads((result / "run.json").read_text())
    assert manifest["status"] == "completed_with_no_model"
    import csv
    rows = list(csv.DictReader((result / "models.csv").open()))
    assert len(rows) == 1
    assert rows[0]["status"] == "no_model"
    assert rows[0]["mean_reprojection_error"] == ""
    summary = list(csv.DictReader((result / "summary.csv").open()))[0]
    assert summary["failed_runs"] == "1"
    assert summary["mean_points"] == ""
