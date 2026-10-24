# Environment and external models

Run reconstruction on the cluster. The local review workflow needs source only.
The pipeline never fetches images, datasets or weights. Its model adapters require
files already present on the host where inference runs.

## CPU baseline and validation

Use Python 3.10 (also recommended by upstream RDD), a new virtual environment,
`ffmpeg` for videos, and:

```bash
python -m pip install -e '.[sfm,test]'
```

The canonical requirements replace the old TensorFlow environment. Its original
dependency list is retained in `legacy/local/requirements.txt`. Historical
SuperPoint code has different dependencies and must not be mixed into this venv.

## Depth Anything V1

Install an appropriate PyTorch build for the cluster CUDA drivers, then:

```bash
python -m pip install -e '.[depth]'
```

Provide a local Hugging Face-format Depth Anything V1 model directory containing
the model configuration, pretrained weights and image processor configuration.
For example, the V1 large model is published as
`LiheYoung/depth-anything-large-hf`; the original CREST model size is unknown.
Use the checkpoint already available on the cluster if possible. The adapter
uses `local_files_only=True` for both model and processor, requires a
`depth_anything` model, runs inference only, resizes predicted relative depth to
the prepared image size, and normalizes it per image.

Sources: [Depth Anything V1](https://github.com/LiheYoung/Depth-Anything),
[Transformers Depth Anything adapter](https://huggingface.co/docs/transformers/model_doc/depth_anything).

Alternatively, set `depth.mask_root` to existing grayscale masks organized as:

```text
mask_root/sequence_name/frame_000001.png
mask_root/sequence_name/frame_000002.png
```

These must align with **prepared** image dimensions and order, including any
configured cropping/resizing. Use 8- or 16-bit grayscale PNGs, not colored depth
visualizations. For image input, prepared names follow lexicographically sorted
source filenames. The exact mapping is saved in `run.json`. The mask field is
scaled by 255 or 65535; it is not re-normalized independently. Set
`depth.higher_is_near: false` for masks where bright pixels mean farther away.

## RDD

Use the official [xtcpete/rdd](https://github.com/xtcpete/rdd) implementation.
The adapter was written against source revision
`02ff3724279332eca395befd8a8ba5198f3484f0` (2026-06-15), using its explicit
CVPR/ResNet configuration. This is an interface reference for the reconstructed
code, **not the recovered historical CREST revision**. Upstream changed its
backbone and weights after the original CVPR release; do not treat an arbitrary
new default checkpoint as interchangeable with the original model.

On the cluster, source-only installation:

```bash
git clone https://github.com/xtcpete/rdd external/rdd
git -C external/rdd checkout 02ff3724279332eca395befd8a8ba5198f3484f0
git -C external/rdd submodule update --init --recursive
python -m pip install -r external/rdd/requirements.txt
python -m pip install -e '.[sfm,depth,test]'
```

Follow upstream instructions for PyTorch/CUDA and the optional deformable
attention extension. The final command restores the pinned COLMAP version if
upstream requirements change it. Supply a matching **CVPR ResNet RDD checkpoint**
at the path configured in `rdd.checkpoint`. A ConvNeXt checkpoint requires its
corresponding configuration instead of `configs/rdd_cvpr.yaml`; such a change
must be reported as a different model choice. No LightGlue model is needed.

The adapter passes explicit configuration and weights to `src.RDD.build`,
disables pretrained backbone downloads, and uses `.extract()` in inference
mode. Missing or incompatible weights cause an error; random weights are never
used as a fallback. The selected checkpoint hash, config hash and checkout
commit/status are recorded per run.

## Scheduler

`scripts/run_cluster.sbatch` is a basic Slurm launcher. Set any site-specific
partition, account, GPU and resource requests before submission. Set `device:
cuda` in your YAML only when a GPU is allocated. From the repository root:

```bash
sbatch scripts/run_cluster.sbatch configs/paper.yaml
```

No jobs were submitted during reconstruction of this repository.
