#!/usr/bin/env python
"""Smoke test for the lightgen pbr->emission adaptation: one real train step, one sample.

What this proves (each one is an `assert`, not a print):
  1. `cfgs/lightgen-emission-overfit10.yaml` instantiates through `instantiate_from_config`.
  2. The unet running inside the pipeline is THIS fork's `UNet2p5DConditionModel`, not the
     `diffusers_modules.local` class diffusers loads from the snapshot's own bundled
     `unet/modules.py`. Without the swap the embeds_albedo/embeds_mr concat silently no-ops
     and the new conv_in channels would never see data -- so this is the load-bearing check.
  3. `train.py`'s conv_in expansion block (copied verbatim below) widens 12 -> 20 channels.
  4. The inherited `set_learned_parameters` freeze recipe holds: conv_in trainable,
     `unet_dual` fully frozen, `learned_text_clip_emission` trainable.
  5. One `training_step` forward+backward produces a finite loss AND a nonzero gradient in
     `conv_in.weight.grad[:, 12:20]` -- i.e. the albedo/mr conditioning channels are actually
     wired to the loss.
  6. The conditioning-dropout branch runs.
  7. One `validation_step` completes the sampling loop and the VAE decode.

Run (from the repo root, `Hunyuan3DPaint-emissive/`):

    conda run -n hunyuanpaint bash -c 'PYTHONNOUSERSITE=1 \
      python hy3dpaint/scripts/lightgen/smoke_train_step.py 2>&1 \
      | tee ../log/$(date +%Y%m%d-%H%M%S)-smoke-train-step.log'

`PYTHONNOUSERSITE=1` is mandatory on this workstation: a stray
~/.local/lib/python3.10/site-packages shadows the conda env (see Task 2's env notes). It must
be set by the launching shell -- setting it inside this file would be too late.
"""

import argparse
import os
import sys

# Config paths (`./hunyuanpaintpbr`, `scripts/lightgen/pilot10_local.json`) resolve against
# hy3dpaint/, which is also where train.py runs from.
HY3DPAINT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(HY3DPAINT_DIR)
sys.path.insert(0, HY3DPAINT_DIR)

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402
from torchvision.utils import save_image  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="cfgs/lightgen-emission-overfit10.yaml")
    p.add_argument(
        "--view-size",
        type=int,
        default=512,
        help="override model.params.view_size; drop to 256 if 512 does not fit in VRAM",
    )
    p.add_argument("--val-steps", type=int, default=5, help="scheduler steps for the validation_step")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        default=None,
        help="force on; default is whatever the config says (the overfit config sets it true)",
    )
    p.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", help="force off"
    )
    p.add_argument("--save-panel", default=None, help="optional path for the validation GT/pred panel PNG")
    return p.parse_args()


def expand_conv_in(model, noise_in_channels):
    """Verbatim copy of train.py's conv_in expansion (train.py lines 217-263, 2026-07-30).

    Kept as a copy on purpose: this smoke must exercise what train.py does, and train.py is
    upstream code this task does not modify. If train.py ever changes, re-sync this.
    """
    model_unet = model.unet.unet
    if hasattr(model_unet, "unet"):
        model_unet = model_unet.unet

    if noise_in_channels is not None:
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                noise_in_channels,
                model_unet.conv_in.out_channels,
                model_unet.conv_in.kernel_size,
                model_unet.conv_in.stride,
                model_unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, : model_unet.conv_in.in_channels, :, :].copy_(model_unet.conv_in.weight)

            new_conv_in.bias.zero_()
            new_conv_in.bias[: model_unet.conv_in.bias.size(0)].copy_(model_unet.conv_in.bias)

            model_unet.conv_in = new_conv_in
    return model_unet


def main():
    args = parse_args()
    # Read by HunyuanPaintEmission.__init__, so it must be set before the model is built.
    os.environ["LIGHTGEN_VAL_STEPS"] = str(args.val_steps)

    from src.utils.train_util import instantiate_from_config

    print(f"torch: {torch.__version__}  cuda_available: {torch.cuda.is_available()}")
    print(f"cwd: {os.getcwd()}")

    config = OmegaConf.load(args.config)
    config.model.params.view_size = args.view_size
    if args.gradient_checkpointing is not None:
        config.model.params.gradient_checkpointing = args.gradient_checkpointing
    noise_in_channels = config.model.params.get("noise_in_channels", None)
    print(
        f"config: {args.config}  view_size={args.view_size}  noise_in_channels={noise_in_channels}  "
        f"gradient_checkpointing={config.model.params.get('gradient_checkpointing', False)}"
    )

    # ---------------------------------------------------------------- 1. build
    model = instantiate_from_config(config.model)
    print(f"model class: {type(model).__name__}  pbr_settings={model.pbr_settings}")

    # ------------------------------------------------- 2. the unet class swap
    unet_module = type(model.unet).__module__
    pipe_unet_module = type(model.pipeline.unet).__module__
    print(f"model.unet class:    {type(model.unet).__name__}  (module: {unet_module})")
    print(f"pipeline.unet class: {type(model.pipeline.unet).__name__}  (module: {pipe_unet_module})")
    assert unet_module.startswith("hunyuanpaintpbr"), f"unet came from {unet_module}, not the fork package"
    assert pipe_unet_module.startswith("hunyuanpaintpbr"), f"pipeline.unet came from {pipe_unet_module}"
    assert not unet_module.startswith("diffusers_modules"), "still using the snapshot's bundled unet class"
    assert model.pipeline.unet is model.unet, "pipeline.unet and model.unet diverged"
    # Diffusers' component registry must agree with the live object, or save_pretrained would
    # write the wrong (or a null) class back into model_index.json.
    # (diffusers records the full module path, not just the top-level package, for classes that
    # are not part of a library it knows how to load.)
    registered = tuple(model.pipeline.config["unet"])
    print(f"pipeline.config['unet']: {registered}")
    assert registered[0] is not None and registered[0].startswith("hunyuanpaintpbr"), registered
    assert registered[1] == "UNet2p5DConditionModel", registered

    # --------------------------------------------- 3. conv_in expansion 12->20
    print(f"conv_in before expansion: {tuple(model.unet.unet.conv_in.weight.shape)}")
    inner = expand_conv_in(model, noise_in_channels)
    print(f"conv_in after  expansion: {tuple(inner.conv_in.weight.shape)}")
    assert tuple(inner.conv_in.weight.shape) == (320, noise_in_channels, 3, 3)
    assert torch.count_nonzero(inner.conv_in.weight[:, 12:20]) == 0, "new channels must be zero-init"
    assert torch.count_nonzero(inner.conv_in.weight[:, :12]) > 0, "pretrained channels must be preserved"

    # ------------------------------------- 4. set_learned_parameters interaction
    named = list(model.unet.named_parameters())
    conv_in_trainable = [n for n, p in named if "conv_in" in n and p.requires_grad]
    dual_trainable = [n for n, p in named if "unet_dual" in n and p.requires_grad]
    emission_token = [n for n, p in named if "learned_text_clip_emission" in n and p.requires_grad]
    unreachable_trainable = [
        n for n, p in named if p.requires_grad and ("_mr" in n or n.endswith("learned_text_clip_albedo"))
    ]
    print(f"trainable *conv_in* params: {len(conv_in_trainable)} e.g. {conv_in_trainable[:2]}")
    print(f"trainable *unet_dual* params: {len(dual_trainable)}")
    print(f"trainable *learned_text_clip_emission* params: {emission_token}")
    print(f"frozen unreachable elements: {model.num_frozen_unreachable_params:,}")
    assert conv_in_trainable, "expected at least one trainable conv_in parameter"
    assert inner.conv_in.weight.requires_grad, "the expanded conv_in must be trainable"
    assert not dual_trainable, f"unet_dual must be fully frozen, but {len(dual_trainable)} params are trainable"
    assert emission_token == ["unet.learned_text_clip_emission"], f"unexpected emission token params: {emission_token}"
    assert not unreachable_trainable, f"unreachable params still trainable: {unreachable_trainable[:5]}"
    assert model.num_frozen_unreachable_params > 0, "froze nothing -- the naming assumption broke"

    total = sum(p.numel() for p in model.unet.parameters())
    trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    print(f"unet params: total={total:,}  trainable={trainable:,}  ({100.0 * trainable / total:.1f}%)")
    assert 0 < trainable < total, "freeze recipe did not freeze anything"

    # ------------------------------------------------------------- 5. one batch
    ds_cfg = config.data.params.train[0]
    dataset = instantiate_from_config(ds_cfg)
    batch = default_collate([dataset[0]])
    print(f"batch: {dataset[0]['name']}")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    device = torch.device(args.device)
    model = model.to(device)
    model.pipeline.to(device)  # what HunyuanPaint.on_fit_start does
    # Upstream's prepare_batch_data only `.to(device)`s images_cond and the targets; the
    # normal/position stacks rely on the Trainer having already moved the batch. Do the same
    # here, via the very hook the Trainer uses, rather than papering over it in model code.
    batch = model.transfer_batch_to_device(batch, device, 0)
    gc_on = any(getattr(m, "gradient_checkpointing", False) for m in model.unet.unet.modules())
    print(f"gradient checkpointing active on the main unet: {gc_on}")
    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    # -------------------------------------------- 6. one training forward/backward
    # drop_cond_prob is forced to 0 for the graded step so the conv_in[:, 12:20] gradient
    # assertion is deterministic -- with the configured 0.1 the albedo condition would be
    # zeroed ~10% of runs and the check would flake. The dropout branch is exercised below.
    configured_drop_cond_prob = model.drop_cond_prob
    model.drop_cond_prob = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model.training_step(batch, 0)
    print(f"loss = {loss.item():.6f}  (dtype {loss.dtype})")
    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()

    grad = inner.conv_in.weight.grad
    assert grad is not None, "conv_in.weight.grad is None -- conv_in is not on the graph"
    new_grad = grad[:, 12:20]
    old_grad = grad[:, :12]
    print(
        f"conv_in.weight.grad: |new 12:20| max={new_grad.abs().max().item():.3e} "
        f"nonzero={torch.count_nonzero(new_grad).item()}/{new_grad.numel()}  "
        f"|old 0:12| max={old_grad.abs().max().item():.3e}"
    )
    print(
        f"  albedo channels 12:16 max={grad[:, 12:16].abs().max().item():.3e}  "
        f"mr channels 16:20 max={grad[:, 16:20].abs().max().item():.3e}"
    )
    assert torch.count_nonzero(new_grad) > 0, "no gradient reached the new albedo/mr conv_in channels"
    assert torch.isfinite(grad).all(), "conv_in gradient has non-finite entries"

    emission_param = model.unet.unet.learned_text_clip_emission
    assert emission_param.requires_grad, "learned_text_clip_emission is not trainable"
    print(
        f"learned_text_clip_emission: shape={tuple(emission_param.shape)} "
        f"grad_is_none={emission_param.grad is None} "
        f"max|grad|={'n/a' if emission_param.grad is None else f'{emission_param.grad.abs().max().item():.3e}'}"
    )

    # Every trainable parameter must actually receive a gradient: train.py runs
    # DDPStrategy(find_unused_parameters=False), which hard-errors on any that does not.
    ungraded = [(n, p.numel()) for n, p in model.unet.named_parameters() if p.requires_grad and p.grad is None]
    print(f"trainable params that received NO gradient: {len(ungraded)} tensors, {sum(n for _, n in ungraded):,} elements")
    for name, numel in ungraded[:20]:
        print(f"   {name}  ({numel:,})")
    assert not ungraded, (
        f"{len(ungraded)} trainable parameters never receive a gradient; "
        f"DDPStrategy(find_unused_parameters=False) in train.py would abort on these"
    )

    train_peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"peak VRAM after training_step: {train_peak:.2f} GiB")

    model.zero_grad(set_to_none=True)

    # ------------------------------------------ 7. the conditioning-dropout branch
    model.drop_cond_prob = 1.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        drop_loss = model.training_step(batch, 1)
    print(f"loss with drop_cond_prob=1.0 = {drop_loss.item():.6f}")
    assert torch.isfinite(drop_loss), "dropout-path loss is not finite"
    model.drop_cond_prob = configured_drop_cond_prob

    # ---------------------------------------------------- 8. one validation step
    model.eval()
    print(f"validation_step: {model.val_num_inference_steps} scheduler steps")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.validation_step(batch, 0)
    assert len(model.validation_step_outputs) == 1
    panel = model.validation_step_outputs[0]
    print(f"validation panel: {tuple(panel.shape)} range=[{panel.min().item():.3f}, {panel.max().item():.3f}]")
    assert torch.isfinite(panel).all(), "validation panel has non-finite entries"

    panel_path = args.save_panel or os.path.join(HY3DPAINT_DIR, "..", "log", "smoke_val_panel.png")
    os.makedirs(os.path.dirname(os.path.abspath(panel_path)), exist_ok=True)
    save_image(panel.float(), panel_path)
    print(f"wrote validation panel: {os.path.abspath(panel_path)}")
    # on_validation_epoch_end is inherited but needs an attached Trainer (self.all_gather), so
    # the panel is written directly here instead.

    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"peak VRAM overall: {peak:.2f} GiB   (view_size={args.view_size}, batch=1, 6 views, bf16 autocast)")
    print(f"trainable unet params: {trainable:,}")
    print("SMOKE_TRAIN_STEP_OK")


if __name__ == "__main__":
    main()
