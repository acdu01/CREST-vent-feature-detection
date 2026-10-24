# Reconstructed method and provenance

The supplied paper is the method specification. `legacy/cluster/GEMINI.md` and
`CONCLUSION.md` describe the earlier SuperPoint pipeline and successful subset
test. `legacy/local/4_depth_model/overlay_raw_grey.py` is the available depth
modulation prototype. These sources do not establish the final RDD checkpoint,
depth checkpoint, experiment parameters or the complete published results.

## What was retained and what was rebuilt

| Component | Evidence | Canonical implementation |
| --- | --- | --- |
| Sampling | Paper says 1 FPS; old script uses 10 FPS and clears output repeatedly | Newly implemented isolated per-video extraction at 1 FPS, optional explicit crop/resize |
| Depth | Paper names Depth Anything | New local-checkpoint inference adapter; model size/checkpoint remains an explicit choice |
| Depth modulation | Local prototype normalizes a grayscale mask, raises it to gamma and blends blurred/original images | Recovered mask convention and gamma idea; two documented processing implementations |
| Learned features | Paper names pretrained RDD with no domain-specific training | New adapter to official RDD; old custom transformer is preserved only in legacy |
| SIFT geometry | Available scripts call native exhaustive matching and incremental mapping | Rebuilt around COLMAP 3.13.0 defaults, isolated outputs and reporting |
| RDD geometry | Historical feature bridge missing | New exhaustive float-descriptor matcher and correspondence import |
| Quality | Paper defines Laplacian thresholds | New measured frame quality and explicit grouping policy |
| Metrics | Old inspection scripts and paper report point count, track length, reprojection error | New per-model reports, explicit primary-model selection and actual group averages |

## Depth transformation

Normalize relative depth per image to [0,1], interpret it as `near`, invert if
the configured polarity requires it, and set `far = 1 - near**gamma`. Constant
inference predictions are rejected because their depth ordering is undefined.

- `depth_sigma` (default): approximate a spatially varying Gaussian sigma by
  linearly interpolating among `levels` Gaussian scales, from zero to the
  configured maximum sigma. Each output pixel's requested scale is `far*sigma`.
  The implementation accumulates one scale at a time, avoiding a large pyramid
  allocation. This is a newly implemented approximation to the paper's equation.
- `legacy_blend`: blend the original and a single globally blurred image with
  weight `far`. This follows the structure of the recovered prototype's blur
  mode. OpenCV Gaussian smoothing differs numerically from Pillow's approximation;
  the original Pillow code is preserved. Dimming is excluded from canonical
  defaults because the paper's main transformation specifies blur.

Maximum sigmas **4 and 12**, **8 levels**, and gamma **1.4** are explicit starting
settings. The first three are new choices; gamma comes from the prototype but
is not confirmed as a final experiment value. No crop or resize is the default.
Changing resolution changes both sharpness measurements and effective blur.

## RDD/COLMAP boundary

[COLMAP 3.13's database format](https://colmap.github.io/legacy/3.13/database.html)
documents 128-dimensional uint8 descriptors for its native matcher. RDD yields
256-dimensional float descriptors. Consequently, passing the latter unchanged
to the native SIFT matcher is not a valid implementation of the paper.

The new bridge preserves full RDD vectors, normalizes them, enumerates every
image pair, and performs mutual nearest-neighbor matching. It uses angular
descriptor distances with ratio 0.8 and maximum angle 0.7 radians. These are
explicit choices consistent with the numerical default SIFT thresholds, not a
claim of identical native matching behavior or recovered historical settings.
There is no LightGlue, PCA conversion or retraining in this path.

Matches and keypoints are imported using the public PyCOLMAP database API.
RDD coordinates receive the half-pixel offset for COLMAP's pixel-center
convention. Correspondence counts must agree exactly; mismatches are errors,
never silently truncated. COLMAP performs geometric verification with default
settings and incremental mapping with default settings. All reconstructed
components remain on disk. The SIFT branch uses native COLMAP extraction and
exhaustive matching directly.

References: [RDD inference implementation](https://github.com/xtcpete/rdd/blob/02ff3724279332eca395befd8a8ba5198f3484f0/src/RDD.py),
[PyCOLMAP 3.13 API](https://colmap.github.io/legacy/3.13/pycolmap/pycolmap.html).

## Quality and evaluation

Compute grayscale OpenCV Laplacian variance with kernel size 1 on each actual
prepared/transformed uint8 image. Thresholds follow the paper exactly:
low ≤20, medium >20 and ≤100, high >100. Frame memberships can change after
preprocessing; a blur setting is not assumed to imply a quality tier.

`per_image` groups measured frames by tier before reconstruction. Within each
sequence and preprocessing variant, all populated tiers are retained; tiers with
too few frames are recorded explicitly. `sequence_mean` instead reconstructs the
complete sequence and labels it using mean variance. The paper does not fully
resolve the historical grouping/aggregation implementation, so the policy is
recorded rather than inferred from names of old result folders.

Report every component. Choose the primary component by registered image count,
then point count, then smallest model ID. Summary rows are unweighted arithmetic
means of successful primary components, grouped by variant and measured quality.
Failures are counted separately; their unknown reprojection errors are not
treated as zero. Registration rate uses the number of images actually supplied
to that reconstruction. This primary-component/averaging policy is newly explicit.

This reconstructs a runnable method, not the final paper numbers. See the
preserved [evidence review](paper-provenance-review.md) for Figure 6 directory
matches and unresolved Table I discrepancies. Paths in that review refer to
the pre-reorganization workspace; originals now live under `legacy/`.
