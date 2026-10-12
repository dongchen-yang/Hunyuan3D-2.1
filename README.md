# Hunyuan3D-Paint for emission maps

This repository adapts the multiview texture model of **Hunyuan3D 2.1** (Hunyuan3D-Paint) from PBR
materials to **emission**. Given six views of a mesh's ground-truth PBR materials and geometry, and an
image of the object, it generates the emission of the same six views; projecting them onto the mesh's
UV map is a separate step. It is the Hunyuan3D baseline of LightgenBench
([Hugging Face `3dlg-hcvc/LightgenBench`](https://huggingface.co/datasets/3dlg-hcvc/LightgenBench)).

Hunyuan3D 2.1 and most of this code are the work of the Tencent Hunyuan3D team:

> **Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material.**
> Tencent Hunyuan3D Team, 2025. [Report](https://arxiv.org/abs/2506.15442) ·
> [Code](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) ·
> [Models](https://huggingface.co/tencent/Hunyuan3D-2.1)

The code and weights are under the Tencent Hunyuan 3D Community License (`LICENSE`, `Notice.txt`). If
you use this code, please cite Hunyuan3D ([below](#citation)). Upstream's own README, for its shape
model, texture demo, Gradio app and API (all kept here, unchanged), is
[`README_hunyuan3d.md`](README_hunyuan3d.md).

## What differs from Hunyuan3D-Paint 2.1

- **Input.** Upstream's UNet takes the noisy latent with the normal and position maps' latents (12
  channels). Here the ground-truth albedo, metallic-roughness and alpha (glTF opacity) views are
  encoded the same way and appended: `[noisy 0-3 | normal 4-7 | position 8-11 | albedo 12-15 |
  metallic-roughness 16-19 | alpha 20-23]`. `train.py` widens the pretrained `conv_in` to 24 channels,
  zero-initializing the new ones.
- **Output.** One generation stream, the emission, instead of upstream's two (albedo,
  metallic-roughness). Its learned token and its attention weights start from upstream's albedo
  stream. The model is `HunyuanPaint` in `hy3dpaint/hunyuanpaintpbr/unet/model.py`; its docstring
  explains the mechanics (the UNet rebuilt from this package after the pipeline loads, the one
  material slot, the frozen leftover parameters).
- **Reference image.** The object's thumbnail (the TexVerse preview image shipped with LightgenBench),
  through upstream's reference path and its augmentation during training.
- **Sampling.** The model's `validation_step`: 30 Euler-ancestral steps, no classifier-free guidance.
  Upstream's `HunyuanPaintPipeline.__call__` does not sample this model.
- **Data.** The loader (`TextureDataset` in `hy3dpaint/src/data/dataloader/objaverse_loader_forTexturePBR.py`)
  reads six fixed views per shape. Upstream's PBR training path does not run in this repository.

The upstream files we changed are edited in place; `main` is upstream's code, so

```bash
git fetch origin main && git diff origin/main -- hy3dpaint/
```

shows every change.

## Setup

```bash
conda create -n hunyuanpaint python=3.10 -y && conda activate hunyuanpaint
pip install -r requirements-emission.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

`requirements-emission.txt` lists what training and inference import, at the versions that produced
our results: torch 2.11 + CUDA 12.8 (RTX 4090, L40S, A40, H100, RTX PRO 6000), diffusers 0.30.0 and
pytorch-lightning 1.9.5 (the checkpoint format). Upstream's `requirements.txt` covers its demo and app.

The model starts from Hunyuan3D-2.1's paint weights at the revision every run used, and loads
`facebook/dinov2-giant` for the reference image:

```bash
huggingface-cli download tencent/Hunyuan3D-2.1 --revision 0b94677654c57bb9a6b6845cd7b704ccf551d327 \
    --include "hunyuan3d-paintpbr-v2-1/*"
huggingface-cli download facebook/dinov2-giant
```

On a machine without internet, set `HF_HUB_OFFLINE=1` after these downloads. Run every command
below from `hy3dpaint/`.

## Data

LightgenBench is gated: accept its terms on the dataset page, then log in (`huggingface-cli login`).

```bash
huggingface-cli download 3dlg-hcvc/LightgenBench --repo-type dataset --local-dir <download_dir> \
    --include "splits.json" "checksums.sha256" "data/*/multiview/*.tar" "data/*/thumbnail/*.tar"
bash scripts/lightgenbench/unpack_hf_dataset.sh <download_dir> <data_dir>
```

The script checks every tar against `checksums.sha256` and unpacks the views and thumbnails into
`<data_dir>/shapes/<uuid>/` (`render_tex/` holds each view's albedo, metallic-roughness, alpha,
normal, position and emission maps) and `<data_dir>/thumbs/<uuid>.png`. It writes the example lists
from `splits.json`, checking that the splits are disjoint and that every shape in them has its files
on disk:
`release_train.json` (36,426 shapes), `release_val64.json` (the 64 validation shapes training
validates on) and `release_test.json` (200). Last, it writes the training config
`<data_dir>/lightgen-emission-release-alpha-thumb.yaml`: `cfgs/lightgen-emission-release-alpha-thumb.yaml`
with the weights, the data and the data-loader workers filled in (`SNAPSHOT=<dir>` and
`NUM_WORKERS=<n>` override the defaults). That is about 37 GB to download and 40 GB unpacked.

## Training

```bash
WANDB_MODE=offline python train.py --base <data_dir>/lightgen-emission-release-alpha-thumb.yaml \
    --name release --logdir logs/ --gpus 0,1,2,3
```

This is the recipe of our release run: 4 GPUs × batch 4 = global batch 16, learning rate 7e-5, 50,000
steps, bf16, gradient checkpointing. The learning rate restarts every 10,000 steps (1,000 warm-up +
9,000 cosine, each cycle's peak 0.9 times the last). On 4 × H100 80 GB a step took 3.4–3.8 s, about
two days in all; a 24 GB GPU cannot train this model. Training logs to Weights & Biases (project
`LightGen`; log in instead of `WANDB_MODE=offline` to log online).

Outputs go to `logs/lightgen-emission-release-alpha-thumb-release/`: the config as run
(`configs/project.yaml`) and, in `checkpoints/`, `last.ckpt`, the three best by `val/emission_mse`,
and `troughs/trough-step=000N0000.ckpt` at the end of each learning-rate cycle (17.9 GB each). The
same command resumes from `last.ckpt`. `train.py` takes no `key=value` overrides (it ignores them):
to change the recipe, edit the config. With fewer GPUs, keep the global batch at 16 with
`lightning.trainer.accumulate_grad_batches`, and multiply `val_check_interval` by the same factor: it
counts batches, while `max_steps` and the checkpoint interval count optimizer steps.

Choose among the trough checkpoints with your own metric on the validation split; do not trust
`val/emission_mse` or the three "best" checkpoints. Validation samples 64 shapes once each, so the
value is noisy, and from the second checkpoint on each is ranked by the validation one interval
earlier (the checkpoint callback runs before the validation of the same step; the first checkpoint
comes before any validation). We chose step 40,000 by the point-sampled emission IoU on the 200
validation shapes.

## Inference on the test split

```bash
python predict_emission_views.py \
    --config logs/lightgen-emission-release-alpha-thumb-release/configs/project.yaml \
    --ckpt logs/lightgen-emission-release-alpha-thumb-release/checkpoints/troughs/trough-step=00040000.ckpt \
    --examples_json <data_dir>/release_test.json --thumbnail_dir <data_dir>/thumbs --out <out_dir>
```

This writes `<out_dir>/<uuid>/seed00/00{0..5}_emission.png`, the six predicted views (512 × 512, in the
camera order of the shape's `render_tex/`), and `<out_dir>/manifest.json`. `--seeds N` samples N seeds
per shape. Each shape's noise comes from its uuid and the seed index, not from its place in the list;
on one GPU type a rerun writes the same bytes, and another GPU type differs slightly (a mean of 0.1/255
between an RTX 4090 and an RTX PRO 6000). A 24 GB GPU is enough, at about 4 s per shape on an RTX
4090. Projecting the views onto the UV map and the benchmark's scoring are not part of this
repository.

## Tests

```bash
python -m pytest scripts/lightgen/ -q
```

The loader, the example-list builder, the release re-layout and the inference script's seed and
reference-image rules (CPU only).

## Repository

| path (under `hy3dpaint/`) | contents |
|---|---|
| `train.py`, `predict_emission_views.py` | training; inference on chosen shapes |
| `hunyuanpaintpbr/` | the model (`unet/model.py`) and the UNet (`unet/modules.py`), upstream's files edited in place |
| `src/data/dataloader/objaverse_loader_forTexturePBR.py` | the loader, upstream's file with our class |
| `cfgs/` | the release config template; the paper run's config (`-fir-xg`) |
| `scripts/lightgenbench/` | `unpack_hf_dataset.sh`: the Hugging Face dataset to the data folder and config |
| `scripts/lightgen/` | what the data script runs (re-layout, example lists) and the tests |
| `scripts/internal/` | our cluster launchers and regression checks; not needed to use the code |

Upstream's shape model (`hy3dshape/`), texture demo (`demo.py`, `textureGenPipeline.py`), Gradio app
and API are unchanged.

## Citation

```bibtex
@misc{hunyuan3d2025hunyuan3d,
    title={Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2506.15442},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
