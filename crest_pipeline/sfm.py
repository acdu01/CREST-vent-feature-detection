"""COLMAP 3.13 API boundary; native SIFT or imported RDD correspondences."""
from itertools import combinations
from pathlib import Path
import numpy as np
from .files import image_files, read_image, write_json
from .matching import match_descriptors


def get_colmap():
    import pycolmap
    if pycolmap.__version__ != "3.13.0":
        raise RuntimeError(f"Expected pycolmap 3.13.0; found {pycolmap.__version__}")
    return pycolmap


def reconstruct(images, output, method, cfg, extractor=None):
    colmap = get_colmap()
    images, output = Path(images), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    database_path = output / "database.db"
    sparse = output / "sparse"
    sparse.mkdir()
    names = [p.name for p in image_files(images)]
    if len(names) < 2:
        raise ValueError("At least two images are required for reconstruction")
    device = colmap.Device.cpu if cfg["device"] == "cpu" else colmap.Device.cuda
    # Save actual COLMAP defaults, not just our explicitly supplied options.
    write_json(output / "colmap-options.json", {
        "version": colmap.__version__, "camera_mode": "AUTO", "method": method,
        "extraction": colmap.FeatureExtractionOptions().todict(),
        "matching": colmap.FeatureMatchingOptions().todict(),
        "verification": colmap.TwoViewGeometryOptions().todict(),
        "mapping": colmap.IncrementalPipelineOptions().todict(),
    })
    if method == "sift":
        colmap.extract_features(str(database_path), str(images), image_names=names, device=device)
        colmap.match_exhaustive(str(database_path), device=device)
    elif method == "rdd":
        if extractor is None:
            raise ValueError("An RDD extractor is required")
        db = colmap.Database.open(str(database_path))
        db.close()
        colmap.import_images(str(database_path), str(images), image_names=names)
        feature_dir = output / "features"
        feature_dir.mkdir()
        db = colmap.Database.open(str(database_path))
        try:
            ids = {image.name: image.image_id for image in db.read_all_images()}
            if set(ids) != set(names):
                raise RuntimeError("COLMAP did not import every image")
            for name in names:
                keypoints, descriptors = extractor(read_image(images / name))
                if keypoints.shape != (len(descriptors), 2) or descriptors.ndim != 2:
                    raise ValueError("Keypoint/descriptor count mismatch; refusing to truncate")
                # Pixel-center convention: upstream RDD uses center (0,0), COLMAP (0.5,0.5).
                db.write_keypoints(ids[name], np.ascontiguousarray(keypoints + 0.5, dtype=np.float32))
                np.save(feature_dir / (name + ".npy"), descriptors, allow_pickle=False)
            pairs_path = output / "pairs.txt"
            settings = cfg["rdd"]
            with pairs_path.open("w") as pairs:
                for first, second in combinations(names, 2):
                    matches = match_descriptors(
                        np.load(feature_dir / (first + ".npy"), allow_pickle=False),
                        np.load(feature_dir / (second + ".npy"), allow_pickle=False),
                        ratio=settings["ratio"], max_distance=settings["max_distance"],
                        block_size=settings["block_size"])
                    db.write_matches(ids[first], ids[second], matches)
                    pairs.write(f"{first} {second}\n")
        finally:
            db.close()
        colmap.verify_matches(str(database_path), str(pairs_path))
    else:
        raise ValueError(f"Unknown feature method: {method}")
    return colmap.incremental_mapping(str(database_path), str(images), str(sparse))
