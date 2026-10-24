# Depth-Informed Transformer Keypoints for Underwater 3D Reconstruction

A research pipeline for reconstructing natural underwater structures from images
and video. It combines depth-guided image preprocessing, feature detection and
matching, and COLMAP structure-from-motion (SfM), which estimates camera poses
and 3D points from overlapping views.

Underwater imagery often contains haze, suspended particles, uneven lighting,
and limited overlap between views. This project explores using estimated depth
to suppress unreliable background detail before selecting and matching image
features. The associated study examines hydrothermal vent imagery collected by
HOV *Alvin* at the East Pacific Rise.

**Paper:** Anna Du, Yunzhu Chen, and Victoria Preston.
[*Depth-Informed Transformer Keypoint Selection for 3D Reconstruction of Natural Underwater Structures*](https://doi.org/10.1109/OCEANS66983.2026.11616893).
OCEANS 2026 Sanya.
[Full text on ResearchGate](https://www.researchgate.net/publication/411909635_Depth-Informed_Transformer_Keypoint_Selection_for_3D_Reconstruction_of_Natural_Underwater_Structures).

## What the pipeline does

```text
Images or video frames
  → Relative depth estimation with Depth Anything V1
  → Depth-guided Gaussian blur
  → SIFT or pretrained RDD feature extraction
  → Feature matching and COLMAP geometric verification
  → Sparse 3D reconstruction and evaluation reports
```

The depth-guided variants apply stronger smoothing to distant regions while
preserving nearby structure. RDD provides learned transformer-based features;
SIFT provides a conventional baseline. Raw-image variants skip depth estimation
and blur, allowing comparisons with the same reconstruction backend.

The package produces **sparse reconstructions**, camera poses, and metrics.
Dense reconstruction and mesh generation are separate downstream steps.
Predicted depth is used as a relative weighting field, not as a measurement of
distance in meters.

## Quick start: SIFT baseline

Start with the CPU baseline, which requires no neural model checkpoints.
Use Python 3.10 for compatibility with the optional learned-model dependencies.

```bash
git clone https://github.com/acdu01/CREST-vent-feature-detection.git
cd CREST-vent-feature-detection
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[sfm]'
```

This installs the Python package and PyCOLMAP 3.13.0. Install `ffmpeg` separately
if you plan to process video.

Provide overlapping images of your scene, either in a single flat folder or in
one subfolder per sequence:

```text
data/images/
  sequence_01/
    image001.jpg
    image002.jpg
    ...
  sequence_02/
    image001.jpg
    image002.jpg
    ...
```

Use sequence names without spaces. Do not mix images and sequence subfolders at
the input root. Datasets and pretrained weights are not included or downloaded
automatically.

Set `input.path` in [`configs/sift.yaml`](configs/sift.yaml) to your image folder.
The default is `../data/images`. **Relative paths are resolved from the YAML
file's directory**, not your shell's working directory.

From the repository root, run:

```bash
python -m crest_pipeline plan --config configs/sift.yaml
python -m crest_pipeline doctor --config configs/sift.yaml
python -m crest_pipeline run --config configs/sift.yaml
```

`plan` displays the resolved configuration without loading models or writing
files. `doctor` checks input paths and dependencies. `run` saves results to a new
directory under `runs/` and prints its location. Existing runs are never overwritten.

By default, frames are grouped by measured sharpness and each group is
reconstructed separately. To reconstruct each complete sequence together, add
this setting to your configuration:

```yaml
quality:
  grouping: sequence_mean
```

## Depth-guided and transformer experiments

[`configs/paper.yaml`](configs/paper.yaml) defines six comparison variants:

| Variant | Feature detector | Image preprocessing |
| --- | --- | --- |
| `sift_raw` | SIFT | None |
| `sift_depth_low` | SIFT | Lower-strength depth-guided blur |
| `sift_depth_high` | SIFT | Higher-strength depth-guided blur |
| `rdd_raw` | Pretrained RDD | None |
| `rdd_depth_low` | Pretrained RDD | Lower-strength depth-guided blur |
| `rdd_depth_high` | Pretrained RDD | Higher-strength depth-guided blur |

Follow the [model setup guide](docs/setup.md) to install the additional
dependencies and provide a local Depth Anything V1 model and compatible RDD
checkout and checkpoint. Then edit `configs/paper.yaml` to set `input.path`,
`depth.model_path`, `rdd.repo`, `rdd.checkpoint`, and `device` (`cpu` or `cuda`).
Select the variants you want to run in `variants`.

```bash
python -m crest_pipeline doctor --config configs/paper.yaml
python -m crest_pipeline run --config configs/paper.yaml
```

To inspect prepared frames, depth maps, and transformed images before
reconstruction:

```bash
python -m crest_pipeline run --config configs/paper.yaml --until prepare
```

Preparation does not require RDD or COLMAP. Depth-guided variants still need a
local depth model, or aligned grayscale masks supplied through `depth.mask_root`
as described in the setup guide. Each invocation starts a fresh run; a full run
does not resume a preparation-only run.

For video input, set `input.kind: videos` and point `input.path` at a flat folder
of videos. Each video becomes a separate sequence. The default sampling rate is
1 frame per second, with no cropping or resizing.

## Outputs and evaluation

Each completed reconstruction run contains:

| Output | Contents |
| --- | --- |
| `run.json` | Resolved settings, input mapping, source and checkpoint hashes, dependency versions, and run status |
| `source/` | Snapshot of the pipeline code used for the run |
| `image-quality.csv` | Measured sharpness and quality tier for each variant's images |
| `models.csv` | Per-component reconstruction metrics and failure records |
| `summary.csv` | Aggregated metrics with successful and failed run counts |
| `sequences/` | Prepared images, depth and feature artifacts where applicable, COLMAP databases, and sparse models |

Metrics include registered image count, 3D point count, mean track length, and
mean reprojection error. Summary statistics use one primary component per
sequence, variant, and quality tier, selected by registered image count and then
3D point count. All reconstructed components remain available.

With the default `quality.grouping: per_image`, sharpness is measured after
preprocessing using Laplacian variance: low ≤20, medium >20 and ≤100, and high
>100. Each populated tier is reconstructed independently. Groups with fewer
than two frames and unsuccessful reconstructions are recorded as failures;
their unknown errors are excluded from successful-run averages.

## Research status

This repository implements the paper's workflow using recovered project code
and newly implemented stages. It has **not reproduced the published numerical
results**. Exact historical checkpoints and some experiment settings are
unavailable; the supplied blur strengths, RDD matching bridge, and aggregation
policy are documented implementation choices.

Synthetic processing and COLMAP integration checks are documented in the
[validation record](docs/validation.md). Full Depth Anything/RDD inference and
an end-to-end experiment on the study dataset have not yet been validated in
this implementation. See [method and assumptions](docs/method.md) for technical
details and the [provenance review](docs/paper-provenance-review.md) for the
historical evidence and unresolved result discrepancies.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`crest_pipeline/`](crest_pipeline/) | Pipeline implementation and command-line interface |
| [`configs/`](configs/) | Input paths, model settings, and experiment variants |
| [`docs/`](docs/) | Setup, method details, and validation records |
| [`tests/`](tests/) | Processing and COLMAP integration tests |
| [`scripts/`](scripts/) | Slurm launcher and code export utility |

Local `data/`, `weights/`, `external/`, and `runs/` directories are excluded from
Git. Historical files referenced in the provenance documentation are local
archives and are not included in a standard clone.

To run the tests:

```bash
python -m pip install -e '.[sfm,test]'
python -m pytest -q
```

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{du2026depth,
  author    = {Du, Anna and Chen, Yunzhu and Preston, Victoria},
  title     = {Depth-Informed Transformer Keypoint Selection for {3D} Reconstruction of Natural Underwater Structures},
  booktitle = {OCEANS 2026 Sanya},
  year      = {2026},
  doi       = {10.1109/OCEANS66983.2026.11616893},
  url       = {https://doi.org/10.1109/OCEANS66983.2026.11616893}
}
```
