"""Predict the six emission views of each shape with a trained emission checkpoint.

    cd hy3dpaint
    python predict_emission_views.py \
        --config logs/<run>/configs/project.yaml \
        --ckpt   logs/<run>/checkpoints/troughs/trough-step=00040000.ckpt \
        --examples_json <data_dir>/release_test.json \
        --thumbnail_dir <data_dir>/thumbs \
        --out <out_dir>

`<data_dir>` is what `scripts/lightgenbench/unpack_hf_dataset.sh` wrote; `--config` is the config the
checkpoint trained with (train.py saves it in the run's `configs/`). The output is

    <out_dir>/<uuid>/seed<NN>/00{0..5}_emission.png    one PNG per view, 512 x 512, unmasked
    <out_dir>/manifest.json                              what was loaded and the seed of each shape

in the camera order of the shape's `render_tex/` views. The sampling is the model's own
`validation_step` (30 Euler-ancestral steps, no guidance, bf16 autocast). Each shape's noise is seeded
from its uuid and the seed index, not from its place in the list. On one GPU type a rerun writes the
same bytes; another GPU type gives slightly different ones. A 24 GB GPU is enough.

A config that names the old class paths listed in scripts/convert_config.py is converted on load.
Projecting the views onto the UV map and scoring them is done by the LightGen evaluation code.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HY3DPAINT_ROOT = Path(__file__).resolve().parent
REF_SOURCES = ("frontal_albedo", "thumbnail")


def sample_seed(sha: str, seed: int) -> int:
    """Per-shape RNG seed for `(sha, seed)`, independent of order, batching and skipped shapes.

    The LightGen evaluation seeds every baseline with this function. The full unsigned 64-bit prefix
    of the digest: masking it would give another RNG stream for the same seed index.
    """
    return int.from_bytes(hashlib.sha256(f"{sha}:{seed}".encode()).digest()[:8], "big")


def load_config(path):
    """The config at `path` as an OmegaConf object, its older class paths converted
    (scripts/convert_config.py, loaded by file path so no other `scripts` package can shadow it)."""
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "convert_config", HY3DPAINT_ROOT / "scripts" / "convert_config.py")
    convert_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert_config)
    with open(path) as f:
        text, _ = convert_config.convert_text(f.read())
    return OmegaConf.create(text)


def resolve_ref_source(config, thumbnail_dir):
    """The training config decides the reference image at inference.

    Reads `data.params.train[0].params.ref_source` (absent -> "frontal_albedo") and checks
    --thumbnail_dir against it, so a thumbnail-trained checkpoint is never sampled with frontal-albedo
    references, or the reverse. Returns (ref_source, thumbnail_dir or None); raises ValueError.
    """
    params = config.data.params.train[0].params
    ref_source = str(params.get("ref_source", "frontal_albedo"))
    if ref_source not in REF_SOURCES:
        raise ValueError(f"config ref_source {ref_source!r} not in {REF_SOURCES}")
    if ref_source == "thumbnail":
        if not thumbnail_dir:
            raise ValueError("this config trained with ref_source=thumbnail: pass --thumbnail_dir "
                             "(<dir>/<uuid>.png)")
        if not os.path.isdir(thumbnail_dir):
            raise ValueError(f"--thumbnail_dir {thumbnail_dir} is not a directory")
        return ref_source, str(thumbnail_dir)
    if thumbnail_dir:
        raise ValueError("this config trained with frontal-albedo references; --thumbnail_dir would "
                         "sample it under a condition it never saw")
    return ref_source, None


def expand_conv_in(model, config):
    """train.py's `noise_in_channels` block: widen the pretrained 12-channel `conv_in` to the config's
    width (24 with alpha), zero-filling the new channels, before the checkpoint's wider weights load."""
    import torch

    noise_in_channels = config.model.params.get("noise_in_channels", None)
    if noise_in_channels is None:
        return None
    inner = model.unet.unet
    if inner.conv_in.in_channels == noise_in_channels:
        return inner.conv_in.in_channels
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            noise_in_channels,
            inner.conv_in.out_channels,
            inner.conv_in.kernel_size,
            inner.conv_in.stride,
            inner.conv_in.padding,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, : inner.conv_in.in_channels, :, :].copy_(inner.conv_in.weight)
        new_conv_in.bias.zero_()
        new_conv_in.bias[: inner.conv_in.bias.size(0)].copy_(inner.conv_in.bias)
        inner.conv_in = new_conv_in
    return noise_in_channels


def load_model(config_path, ckpt_path, snapshot=None, val_steps=30, device="cuda"):
    """Build the model from its training config and load the checkpoint's whole state dict, failing on
    any missing or unexpected key (a half-loaded model looks like a badly trained one).
    Returns (model, info), info being what the manifest records."""
    import torch

    from src.utils.train_util import instantiate_from_config

    config = load_config(config_path)
    sd_config = config.model.params.stable_diffusion_config
    if snapshot is not None:
        # A config holds the absolute snapshot path of the machine it trained on.
        sd_config.pretrained_model_name_or_path = snapshot
    # `custom_pipeline: ./hunyuanpaintpbr` is relative to the working directory, which main() has set
    # to hy3dpaint/, so the pipeline code loaded is the one training used.
    snapshot_used = str(sd_config.pretrained_model_name_or_path)

    # Gradient checkpointing does nothing without gradients; the step count is set below, explicitly.
    config.model.params.gradient_checkpointing = False
    model = instantiate_from_config(config.model)
    expand_conv_in(model, config)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint {ckpt_path} does not match the model built from {config_path}: "
            f"{len(missing)} missing key(s) {missing[:5]}, {len(unexpected)} unexpected key(s) "
            f"{unexpected[:5]}; pass the config the checkpoint trained with")
    info = dict(config=str(config_path), snapshot=snapshot_used, ckpt=str(ckpt_path),
                val_steps=int(val_steps), finetuned=True,
                global_step=int(state.get("global_step", -1)) if isinstance(state, dict) else -1)

    model.val_num_inference_steps = int(val_steps)
    model.eval()
    model.to(device)
    # The pipeline (VAE, text encoder) is not a child module, so model.to() does not move it; training
    # moves it in on_fit_start.
    model.pipeline.to(device)
    return model, info


def _split_panel(panel, num_view):
    """`validation_step`'s panel (GT above prediction, views side by side) -> the predicted views,
    (B, N, 3, H, W) float in [0, 1], in camera order."""
    B, C, H2, WN = panel.shape
    H = H2 // 2
    W = WN // num_view
    pred = panel[:, :, H:, :]
    return pred.reshape(B, C, H, num_view, W).permute(0, 3, 1, 2, 4)


def sample_shape(model, batch, rng_seed):
    """One shape through the model's own `validation_step`, under the global seed `rng_seed`.

    The seed is global because the noise is drawn inside the model with no generator: the six
    conditioning-latent `posterior.sample()` calls, the initial noise and the 30 ancestral steps.
    """
    import torch

    batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.manual_seed(rng_seed)   # CPU and every CUDA device
    model.validation_step_outputs.clear()
    # bf16 as in training; the model keeps its VAE, text encoder and DINO in bf16 and its unet in fp32,
    # so without autocast the cross-attention gets mixed dtypes and raises.
    device_type = "cuda" if str(model.device).startswith("cuda") else "cpu"
    with torch.autocast(device_type, dtype=torch.bfloat16), torch.no_grad():
        model.validation_step(batch, 0)
    panel = model.validation_step_outputs[-1]
    model.validation_step_outputs.clear()
    return _split_panel(panel.float().cpu(), model.num_view)


def write_views(views, out_dir, stems):
    """(N, 3, H, W) float [0, 1] -> `<out_dir>/<stem>_emission.png`, unmasked."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for stem, v in zip(stems, views):
        arr = np.clip(v.permute(1, 2, 0).numpy() * 255.0, 0, 255).round().astype(np.uint8)
        path = os.path.join(out_dir, f"{stem}_emission.png")
        Image.fromarray(arr).save(path)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a checkpoint train.py wrote (.ckpt)")
    ap.add_argument("--config", required=True, help="the config the checkpoint trained with")
    ap.add_argument("--examples_json", required=True,
                    help="JSON list of shape directories (each with render_tex/ and render_cond/)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--thumbnail_dir", default=None,
                    help="directory of <uuid>.png thumbnails; required if the config trained with "
                         "ref_source: thumbnail, refused otherwise")
    ap.add_argument("--snapshot", default=None,
                    help="the Hunyuan3D-2.1 hunyuan3d-paintpbr-v2-1 snapshot directory, if not the "
                         "config's pretrained_model_name_or_path")
    ap.add_argument("--seeds", type=int, default=1, help="how many seeds per shape")
    ap.add_argument("--seed_base", type=int, default=0, help="first seed index")
    ap.add_argument("--val_steps", type=int, default=30, help="sampling steps (training used 30)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N shapes")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Absolute before the chdir below.
    config_path = str(Path(args.config).resolve())
    examples_json = str(Path(args.examples_json).resolve())
    out_root = str(Path(args.out).resolve())
    ckpt_path = str(Path(args.ckpt).resolve())
    thumbnail_dir = str(Path(args.thumbnail_dir).resolve()) if args.thumbnail_dir else None
    ref_source, thumbnail_dir = resolve_ref_source(load_config(config_path), thumbnail_dir)
    print(f"[ref] source={ref_source} thumbnail_dir={thumbnail_dir}", flush=True)
    # The model and loader are imported as top-level packages of hy3dpaint/, and the config's
    # custom_pipeline is a path relative to it.
    sys.path.insert(0, str(HY3DPAINT_ROOT))
    os.chdir(HY3DPAINT_ROOT)

    from torch.utils.data import DataLoader

    from src.data.dataloader.objaverse_loader_forTexturePBR import TextureDataset

    model, info = load_model(config_path, ckpt_path, snapshot=args.snapshot,
                             val_steps=args.val_steps, device=args.device)
    info["ref_source"] = ref_source
    info["thumbnail_dir"] = thumbnail_dir

    dataset = TextureDataset(examples_json, num_view=model.num_view, image_size=model.view_size,
                             ref_source=ref_source, thumbnail_dir=thumbnail_dir)
    if args.limit is not None:
        dataset.dirs = dataset.dirs[: args.limit]
    # One shape per batch: it fits a 24 GB GPU and keeps each shape's draw independent of batching.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # The loader stacks views f"{v:03d}" for v in range(num_view); the same stems name the outputs.
    stems = [f"{v:03d}" for v in range(model.num_view)]
    seeds = [args.seed_base + i for i in range(args.seeds)]

    os.makedirs(out_root, exist_ok=True)
    shapes = []
    t_start = time.time()
    for i, batch in enumerate(loader):
        example_dir = batch["name"][0]
        sha = Path(example_dir.rstrip("/")).name
        entry = dict(sha=sha, example_dir=example_dir, seeds={})
        for seed in seeds:
            t0 = time.time()
            rng_seed = sample_seed(sha, seed)
            views = sample_shape(model, batch, rng_seed)[0]  # (N, 3, H, W)
            seed_dir = os.path.join(out_root, sha, f"seed{seed:02d}")
            write_views(views, seed_dir, stems)
            entry["seeds"][str(seed)] = {"dir": os.path.relpath(seed_dir, out_root),
                                         "rng_seed": rng_seed}
            print(f"[{i + 1}/{len(dataset)}] {sha} seed={seed} rng={rng_seed} "
                  f"{time.time() - t0:.1f}s  mean={float(views.mean()):.4f} "
                  f"max={float(views.max()):.4f}", flush=True)
        shapes.append(entry)

    manifest = dict(
        **info,
        seeds=seeds,
        examples_json=examples_json,
        num_view=int(model.num_view),
        view_size=int(model.view_size),
        n_shapes=len(shapes),
        wall_seconds=round(time.time() - t_start, 1),
        shapes=shapes,
    )
    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done: {len(shapes)} shapes x {len(seeds)} seed(s) -> {out_root}\n"
          f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
