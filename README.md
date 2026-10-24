# CREST underwater reconstruction

An organized reconstruction of the workflow in **Depth-Informed Transformer
Keypoint Selection for 3D Reconstruction of Natural Underwater Structures**:

**1 FPS frames → Depth Anything → depth-guided blur → SIFT or pretrained RDD →
exhaustive image pairing → COLMAP geometry and sparse reconstruction → metrics.**

This implementation combines recovered project code and newly implemented missing
stages. It is **not a recovered snapshot of the exact final experiment** and has
not reproduced the paper's reported numbers. The available notes establish a
working SuperPoint subset experiment, while the paper describes RDD. The new
pipeline implements the latter; both historical versions are preserved.

## Organization

```text
crest_pipeline/        Canonical Python package and command-line interface
configs/               Paths, model selection, preprocessing and experiment settings
docs/                  Method, provenance, setup and validation records
tests/                 Synthetic processing and COLMAP integration tests
scripts/               Cluster launcher and code-only export
legacy/local/          Original local scripts, README and dependency list
legacy/cluster/        Transferred cluster source, notebooks and notes
legacy/source-sha256.json  Checksums of preserved historical files
data/                  Your images/videos (ignored by Git)
weights/               Local pretrained checkpoints (ignored by Git)
external/              Upstream model checkouts (ignored by Git)
runs/                  Generated artifacts and reports (ignored by Git)
```

Historical data and model folders still present at the old numbered paths were
left untouched. The numbered Python scripts moved into `legacy/local/`; do not
use them as the canonical entry points. Their relative-path assumptions remain
historical. The original archive is also retained outside this repository.

## Install

Use a new Python 3.10 environment on the cluster rather than the old TensorFlow
1.12 environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[sfm,test]'
```

This installs COLMAP's Python package pinned to **3.13.0**. Video extraction also
requires `ffmpeg`. For learned models, install the matching PyTorch/CUDA build,
the depth extra and RDD dependencies as described in [setup](docs/setup.md).
Weights and datasets are never downloaded by the pipeline.

## Start with SIFT

Place each sequence in its own directory, or point the configuration at a flat
image directory:

```text
data/images/
  sequence_01/     image1.png, image2.png, ...
  sequence_02/     image1.png, image2.png, ...
```

Edit `configs/sift.yaml` to point at your images. **Relative paths are resolved
relative to the YAML file**, so `../data/images` is relative to `configs/`.

```bash
python -m crest_pipeline plan --config configs/sift.yaml
python -m crest_pipeline doctor --config configs/sift.yaml
python -m crest_pipeline run --config configs/sift.yaml
```

`plan` loads no models and writes nothing. `doctor` checks prerequisites. Every
`run` creates a new output directory; existing runs are never overwritten.

## Run the paper-method experiment suite

Edit `configs/paper.yaml`: set the input path, `device`, local Depth Anything
model path, RDD checkout and checkpoint. Select the desired variants:

| Variant | Features | Input |
| --- | --- | --- |
| `sift_raw` | Native COLMAP SIFT | Unmodified prepared frames |
| `sift_depth_low` | SIFT | Lower-strength depth blur |
| `sift_depth_high` | SIFT | Higher-strength depth blur |
| `rdd_raw` | Pretrained RDD | Unmodified prepared frames |
| `rdd_depth_low` | Pretrained RDD | Lower-strength depth blur |
| `rdd_depth_high` | Pretrained RDD | Higher-strength depth blur |

```bash
python -m crest_pipeline doctor --config configs/paper.yaml
python -m crest_pipeline run --config configs/paper.yaml
```

To inspect only frame preparation, depth and blur first:

```bash
python -m crest_pipeline run --config configs/paper.yaml --until prepare
```

That run does not require RDD or COLMAP. It still requires a local Depth Anything
checkpoint, unless you provide prepared grayscale masks using `depth.mask_root`.
A later full run creates a fresh directory; automatic resume is intentionally not
implemented, so stale results cannot silently be mixed with new settings.

For video input, set `input.kind: videos` and point `input.path` at a flat folder
of videos. Each video is processed separately at the configured FPS; extraction
never deletes another video's frames. The default is 1 FPS with no crop or resize.

## Read the results

Each run contains:

- `run.json`: resolved settings, source/checkpoint hashes, installed versions,
  source image mapping, timestamps and completion/failure status.
- `source/`: a copy of the pipeline Python source used for that run.
- `image-quality.csv`: actual Laplacian variance and tier for every input to each variant.
- `models.csv`: metrics for every reconstructed component, including failures.
- `summary.csv`: arithmetic means from one primary component per sequence/variant/tier,
  with successful and failed run counts.
- `sequences/`: prepared frames, depth masks, transformed images, feature caches,
  COLMAP databases and sparse reconstructions.

Quality is measured, not inferred from a folder called `high_blur`. By default,
images are partitioned into high (>100), medium (>20 and ≤100), and low (≤20)
Laplacian-variance tiers **after preprocessing**, and each populated tier is
reconstructed separately. A tier with fewer than two frames is recorded as
`insufficient_images`; failed geometry produces `no_model`. Neither is silently
reported as a successful zero-error reconstruction.

For whole-sequence reconstructions, set `quality.grouping: sequence_mean`.
This explicitly labels each complete sequence by its mean image sharpness;
it is a different evaluation policy and is recorded in the run configuration.
The primary component is the one with most registered images, then most 3D points.
All other components remain in `models.csv` and on disk.

## Reconstruction choices that are not historical facts

The exact final depth checkpoint, RDD checkpoint, blur strengths and feature
import bridge were not recovered. Defaults are documented starting values,
not the original experiment settings. In particular, native COLMAP 3.13 matches
128-D uint8 SIFT descriptors; RDD produces 256-D floats. The new RDD path performs
exhaustive mutual-nearest-neighbor matching on the full float descriptors,
then imports matches for COLMAP geometric verification and mapping. It does not
silently reinterpret RDD vectors as SIFT or use a new PCA fit. This bridge is an
explicit reconstruction choice beyond what the paper specifies.

See [method and assumptions](docs/method.md), [historical evidence](docs/paper-provenance-review.md)
and [validation](docs/validation.md). Table I contains unresolved discrepancies
with the available local summary data. This implementation does not adjust outputs
to force agreement with published values.

## Test and share code

```bash
python -m pytest -q
python scripts/export_code.py /path/to/crest-code.tar.gz
```

The exporter includes only the package, configurations, documentation, tests,
launcher scripts and preserved historical source. It excludes model weights,
images, datasets, generated reports and reconstruction artifacts. Do not upload
`data/`, `weights/`, `runs/`, `external/` or old numbered data directories.
