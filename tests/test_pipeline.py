from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import cv2
import pytest
import yaml

from crest_pipeline.config import DEFAULTS, load_config
from crest_pipeline.files import save_image
from crest_pipeline.matching import match_descriptors
from crest_pipeline.metrics import aggregate, model_metrics
from crest_pipeline.pipeline import external_mask, quality_groups, run
from crest_pipeline.preprocess import depth_transform, normalize_depth, prepare_frames, quality_tier


@pytest.mark.parametrize("value,tier", [(0, "low"), (20, "low"), (20.01, "medium"),
                                        (100, "medium"), (100.01, "high")])
def test_quality_boundaries(value, tier):
    assert quality_tier(value) == tier


@pytest.mark.parametrize("method", ["depth_sigma", "legacy_blend"])
def test_depth_blur_preserves_near_and_smooths_far(method):
    checker = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
    image = np.repeat(checker[..., None], 3, axis=2)
    near = np.ones((64, 64), dtype=np.float32)
    near[:, 32:] = 0
    result = depth_transform(image, near, sigma=4, method=method)
    np.testing.assert_array_equal(result[:, :32], image[:, :32])
    assert result[:, 40:].var() < image[:, 40:].var() / 100


def test_depth_invalid_and_orientation(tmp_path):
    with pytest.raises(ValueError, match="Constant"):
        normalize_depth(np.ones((10, 10)))
    with pytest.raises(ValueError, match="finite"):
        normalize_depth(np.array([[float("nan")]]))
    mask = np.array([[0, 65535]], dtype=np.uint16)
    save_image(tmp_path / "mask.png", mask)
    np.testing.assert_array_equal(external_mask(tmp_path / "mask.png", (1, 2), False), [[1, 0]])
    with pytest.raises(ValueError, match="align"):
        external_mask(tmp_path / "mask.png", (2, 2), True)
    save_image(tmp_path / "color.png", np.array([[[0, 100, 255]]], dtype=np.uint8))
    with pytest.raises(ValueError, match="grayscale"):
        external_mask(tmp_path / "color.png", (1, 1), True)


def test_matching_order_and_chunking():
    rng = np.random.default_rng(3)
    descriptors = rng.normal(size=(40, 256)).astype(np.float32)
    perm = rng.permutation(len(descriptors))
    small = match_descriptors(descriptors, descriptors[perm], block_size=3)
    large = match_descriptors(descriptors, descriptors[perm], block_size=100)
    np.testing.assert_array_equal(small, large)
    np.testing.assert_array_equal(perm[small[:, 1]], small[:, 0])
    assert len(small) == 40


def test_ambiguous_descriptors_are_rejected():
    duplicate = np.array([[1, 0], [1, 0]], dtype=np.float32)
    assert match_descriptors(duplicate, duplicate).shape == (0, 2)
    assert match_descriptors(np.empty((0, 2)), np.eye(2)).shape == (0, 2)
    with pytest.raises(ValueError, match="Zero"):
        match_descriptors(np.zeros((2, 2)), np.eye(2))


def test_config_paths_and_typos(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("input:\n  path: images\nvariants: [sift_raw]\n")
    parsed = load_config(config)
    assert parsed["input"]["path"] == str(tmp_path / "images")
    config.write_text("preprocess:\n  low_sgima: 7\n")
    with pytest.raises(ValueError, match="Invalid keys"):
        load_config(config)


def fixture_config(tmp_path):
    source = tmp_path / "input" / "sequence"
    source.mkdir(parents=True)
    rng = np.random.default_rng(4)
    for name in ["image one.png", "image two.png"]:
        save_image(source / name, rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    masks = tmp_path / "masks" / "sequence"
    masks.mkdir(parents=True)
    for i in range(1, 3):
        save_image(masks / f"frame_{i:06d}.png", np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (48, 1)))
    cfg = deepcopy(DEFAULTS)
    cfg["input"]["path"] = str(source.parent)
    cfg["output"] = str(tmp_path / "runs")
    cfg["depth"]["mask_root"] = str(masks.parent)
    cfg["variants"] = ["sift_raw", "sift_depth_low", "rdd_depth_high"]
    return cfg


def test_prepare_pipeline_and_no_overwrite(tmp_path):
    cfg = fixture_config(tmp_path)
    before = {p.name: p.read_bytes() for p in (tmp_path / "input/sequence").iterdir()}
    result = run(cfg, "test", until="prepare")
    manifest = json.loads((result / "run.json").read_text())
    assert manifest["status"] == "prepared"
    assert len(manifest["sequences"][0]["frame_sources"]) == 2
    assert (result / "sequences/sequence/depth_high/frame_000001.png").is_file()
    assert len((result / "image-quality.csv").read_text().splitlines()) == 7
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "input/sequence").iterdir()}
    with pytest.raises(FileExistsError):
        run(cfg, "test", until="prepare")


def test_failure_is_recorded(tmp_path):
    cfg = fixture_config(tmp_path)
    (tmp_path / "masks/sequence/frame_000002.png").unlink()
    with pytest.raises(ValueError, match="Missing depth"):
        run(cfg, "failed", until="prepare")
    manifest = json.loads((tmp_path / "runs/failed/run.json").read_text())
    assert manifest["status"] == "failed"
    assert "Missing depth" in manifest["error"]


def test_input_output_overlap_rejected(tmp_path):
    cfg = fixture_config(tmp_path)
    cfg["output"] = cfg["input"]["path"] + "/runs"
    with pytest.raises(ValueError, match="contain each other"):
        run(cfg, "bad", until="prepare")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
def test_video_sampling_preserves_multiple_videos(tmp_path):
    cfg = deepcopy(DEFAULTS)
    cfg["input"]["kind"] = "videos"
    for name in ["one", "two"]:
        video = tmp_path / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc=size=64x48:rate=30:duration=2", "-c:v", "mpeg4", str(video)], check=True)
        prepare_frames(video, tmp_path / name, cfg)
    assert len(list((tmp_path / "one").glob("*.png"))) == 2
    assert len(list((tmp_path / "two").glob("*.png"))) == 2


class Model:
    def __init__(self, images, points):
        self.images, self.points = images, points
    def num_reg_images(self): return self.images
    def num_points3D(self): return self.points
    def compute_mean_track_length(self): return 4.0
    def compute_mean_reprojection_error(self): return 0.5


def test_aggregation_does_not_count_fragments_as_separate_runs():
    rows = model_metrics({0: Model(10, 100), 1: Model(3, 20)}, "a", "rdd_raw", 12, "high", 200)
    rows += model_metrics({}, "b", "rdd_raw", 12, "high", 200)
    summary = aggregate(rows)[0]
    assert summary["runs"] == 2
    assert summary["successful_runs"] == 1
    assert summary["failed_runs"] == 1
    assert summary["mean_points"] == 100
    assert summary["mean_registration_rate"] == 10 / 12


def test_cli_plan_needs_no_models(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("variants: [sift_raw]\n")
    from crest_pipeline.cli import main
    main(["plan", "--config", str(path)])


def test_quality_groups_use_measurements_not_variant_names(tmp_path):
    images = tmp_path / "depth_high"
    images.mkdir()
    for i in range(3):
        save_image(images / f"{i}.png", np.zeros((4, 4, 3), dtype=np.uint8))
    groups = quality_groups(images, [10, 30, 200], ["low", "medium", "high"],
                            DEFAULTS["quality"], tmp_path / "groups")
    assert [(tier, n, value) for tier, _, n, value in groups] == [
        ("low", 1, 10), ("medium", 1, 30), ("high", 1, 200)]
    assert [p.name for p in groups[2][1].iterdir()] == ["2.png"]
