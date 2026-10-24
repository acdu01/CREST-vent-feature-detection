# Validation record

Validated locally on 2026-10-23 using Python 3.13.11, NumPy 2.4.2, OpenCV 4.13.0,
pytest 9.1.1, FFmpeg, and PyCOLMAP **3.13.0** in an isolated validation environment.
The recommended cluster environment remains Python 3.10 for upstream RDD.

`python -m pytest -q`: **22 tests passed**.

The tests cover:

- Exact quality-tier boundaries (20 and 100).
- Foreground preservation and background smoothing for both blur methods.
- Depth-mask polarity, shape, grayscale validation, constant/non-finite depth.
- Full-dimensional descriptor matching, order preservation, chunk-size invariance,
  ambiguous-match rejection and empty features.
- Configuration typos and relative path resolution.
- A complete preparation run with synthetic images and masks, source preservation,
  output non-overwrite, and failure status recording.
- FFmpeg extraction of two synthetic 30-FPS videos at 1 FPS, confirming that the
  second video's extraction does not remove the first one's frames.
- Per-image quality grouping based on measured values rather than blur names.
- Primary-model aggregation, fragment exclusion and failure counts.
- Real COLMAP geometric verification and incremental reconstruction from five
  synthetic cameras and 200 known 3D correspondences, using the same keypoint and
  match import path as RDD. All five images registered, at least 180 points were
  recovered, and mean reprojection error was below 0.1 px.
- Real native COLMAP SIFT extraction/matching and database descriptor shape.
- A complete blank-scene SIFT run that records failed geometry without inventing
  zero reprojection error or including failures in successful-run averages.
- Source-only export and notebook output removal.

The package also installed successfully in editable mode, and its installed
`crest plan` command resolved the supplied configuration without loading models.

## Not yet validated

No project images, depth maps or pretrained neural checkpoints were downloaded
or processed. Actual Depth Anything and RDD inference, CUDA extensions and a
complete real-data experiment have **not** been run here. The synthetic geometry
test validates the correspondence-to-COLMAP boundary; it is not an RDD model
accuracy test. The neural adapters were checked against upstream interfaces.

The paper's exact final numbers remain unreproduced. Historical checkpoint
identity, preprocessing settings, grouping policy and Table I provenance remain
uncertain and are explicitly documented in `method.md`.

Integration tests skip if PyCOLMAP 3.13.0 is not installed; FFmpeg sampling skips
if FFmpeg is absent. The local validation above included both dependencies, so
the reported run did not skip these checks.
