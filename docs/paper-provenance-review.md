# Identifying the implementation used for the paper

Status: the final implementation has **not yet been identified in the available
source**. Preserve the existing versions; do not promote the transferred
SuperPoint experiment to the paper implementation.

Paper reviewed: `CREST_paper-B2XPkcYY (1).pdf`, “Depth-Informed Transformer
Keypoint Selection for 3D Reconstruction of Natural Underwater Structures.”
Page numbers below refer to PDF pages, starting at 1. Method text was extracted
locally, and Table I on page 8 was also checked visually.

## Procedure to recover

| Stage | Paper procedure | Available implementation and evidence |
| --- | --- | --- |
| Frame extraction | 1 FPS from 30 FPS videos (pp. 6–7) | Both local and transferred `1_video_processing/video2img.py` specify 10 FPS and crop the top two-thirds. Local README says 6 FPS. None establishes the final sampling command. |
| Depth estimation | Depth Anything, grayscale relative weighting maps scaled to [0,1] (§III-B, p. 4) | No Depth Anything inference implementation found in the supplied source. Exact model variant/checkpoint and preprocessing still need recovery. |
| Depth-guided transformation | Continuous depth-guided Gaussian smoothing; stronger smoothing in distant regions (§III-C, pp. 4–5) | Local `4_depth_model/overlay_raw_grey.py` blends a globally blurred image with the original using a depth mask. Its example uses a test image, mode `both`, strength 0.4, radius 8 and gamma 1.4. This is a related prototype, not evidence of final run parameters. It is not literally a spatially varying Gaussian radius as expressed in the paper. |
| Learned features | Pretrained RDD, no domain-specific fine-tuning (§III-D, p. 5) | Transferred scripts use SuperPoint. Local `5_transformer_model/model.py` defines `KeypointTransformer`; `train.py` trains it. Neither is the RDD implementation described in the paper. |
| Reconstruction | COLMAP 3.13.0, default exhaustive matching and incremental mapping (§III-E, pp. 5–6) | The successful subset runner in the cluster notes uses sequential matching with overlap 20, resized 640×480 inputs and converted SuperPoint descriptors. Other available scripts use exhaustive SIFT reconstruction, but no recovered RDD import/run script ties them to the final experiments. |
| Quality tiers | Laplacian variance: high >100, medium >20 and ≤100, low ≤20 (§IV-A, p. 6) | No Laplacian measurement implementation found in the supplied source. `table/analysis.py` assigns tiers from `low_blur`/`high_blur` folder names, so it cannot establish the per-image measurements. |
| Evaluation | 3D point count, mean track length, mean reprojection error (§IV-A; Table I) | Local summary CSVs provide useful output-directory clues, but the available analysis script and summaries do not reproduce all of Table I. |

The transferred `CONCLUSION.md` reports a successful 50-frame SuperPoint test
producing two partial models. That is evidence of a historical experiment, not
evidence that this version produced the paper's final results. `GEMINI.md`
also describes the SuperPoint pipeline. The archived root README contains our
transfer instructions and cannot establish experiment provenance.

No RDD, Depth Anything or Laplacian implementation was found in current local
project code or by a targeted content search of eight distinct locally available
branch, remote-tracking and stash tips. This does not establish what is currently
on the cluster or on an unfetched remote branch.

## Strongest result-location clues: Figure 6

The paper's Figure 6 on page 7 lists three exact point counts. All match rows
in the existing local `table/colmap_analysis.csv` for the same sequence:
`AT5007_AL5144_4K-arm_2023_01_17_20_40_20_03`.

| Figure | Method in paper | Points | Relative output path in local summary |
| --- | --- | ---: | --- |
| 6(a) | SIFT | 56,011 | `1_sparse/AT5007_AL5144_4K-arm_2023_01_17_20_40_20_03/sparse/0` |
| 6(b) | RDD | 23,759 | `5_sparse/AT5007_AL5144_4K-arm_2023_01_17_20_40_20_03` |
| 6(c) | Depth + RDD | 20,377 | `45_sparse_high_blur/AT5007_AL5144_4K-arm_2023_01_17_20_40_20_03` |

These exact matches are strong evidence about the result locations to search.
They do not identify a source commit, checkpoint or command. None of these
output directories was in the supplied cluster tree.

## Table I discrepancies to resolve before declaring a final version

- The 36,919 / 10.15 / 0.85 SIFT row matches an individual summary run for
  sequence `...20_31_26_01`. The 22,138 / 10.52 / 0.57 depth-SIFT row and
  12,658 / 13.07 / 1.43 RDD row likewise match individual runs for that sequence.
  They do not match the means across the corresponding folder groups in the
  available CSV, despite the table's “Average” caption.
- The low-quality depth-SIFT row has 17,490 points and track length 10.54,
  matching that sequence's `4_sparse_high_blur` row, but the CSV reports
  reprojection error **0.575854**, whereas the paper prints **6.58**.
- No run in `table/colmap_analysis.csv` has the depth-RDD table point counts
  **1,848** or **2,334**. Available depth-RDD group averages also differ.

Possible explanations include later runs, a different aggregation/subset or
transcription differences. The current files do not distinguish these. Do not
change the paper, results, or code to force agreement; recover the final run
records and analysis script.

## Preservation and next collection

The original transferred archive and its extracted contents remain unchanged.
A separate `local-source-before-paper-selection.tar.gz` preserves the currently
available local project source/configuration, including uncommitted depth and
transformer code, without datasets, checkpoints or images. `preservation-sha256.txt`
records the two archives' hashes. Neither snapshot is labeled as the final
paper implementation.

Locate the parent cluster project containing the output directories above.
Collect source and configuration for Depth Anything inference, depth blur
(including high/low settings and mask orientation), RDD extraction, COLMAP
feature import/matching/mapping, Laplacian measurement, and table generation.
Keep scheduler scripts and explicit run configuration, dependency/version
records, commit IDs and checkpoint names/hashes. Do not download checkpoints,
images, depth maps, descriptors or reconstruction outputs. Existing small local
summary CSVs were read to trace provenance; no cluster data were downloaded.

Directory-only locator, run on the cluster with the project parent substituted:

```bash
find /path/to/cluster/project-parent \
  -type d \( -name .git -o -name .venv -o -name venv -o -name .cache -o -name node_modules \) -prune -o \
  -type d \( -iname '*rdd*' -o -iname '*depth*anything*' \
    -o -name 1_sparse -o -name 5_sparse \
    -o -name '4_sparse_*' -o -name '45_sparse_*' \) -print -prune \
  > paper-project-locations.txt
```

This lists matching directory paths, does not open file contents, and does not
descend into matched reconstruction output directories. It can still traverse
other directories and may take time on a large shared filesystem; scope it to
your project parent. The resulting text file is sufficient for the next step.
