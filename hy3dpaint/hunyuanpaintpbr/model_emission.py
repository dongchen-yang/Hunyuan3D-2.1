"""pbr->emission (multiview): HunyuanPaint adapted per the 2026-07-30 lightgen spec (revised).

Design (user-decided 2026-07-30): SINGLE generation stream. GT albedo + metallic-
roughness latents are channel-concatenated onto the geometry conditioning (normal +
CCM) -- the exact mechanism upstream uses for geometry -- and the denoised/supervised
output is the emission multiview alone (pbr_settings=["emission"], learned token
warm-started from the pretrained albedo token). Single reference (frontal albedo
render); upstream's dual-ref consistency loss dropped. No teacher-forcing, no
per-slot timestep mixing.

Three mechanics in here are non-obvious and are the reason this file exists at all:

1. THE UNET CLASS SWAP. `DiffusionPipeline.from_pretrained(<snapshot>, custom_pipeline=
   ./hunyuanpaintpbr)` does NOT build `pipe.unet` from this package. The snapshot's
   `model_index.json` declares the unet component as `["modules", "UNet2p5DConditionModel"]`,
   which makes diffusers load `<snapshot>/unet/modules.py` as a dynamic module
   (`diffusers_modules.local.modules`). That bundled copy has no knowledge of the
   `embeds_albedo` / `embeds_mr` concat this design depends on. So after the pipeline
   loads we throw that unet away and rebuild it from THIS package's
   `UNet2p5DConditionModel.from_pretrained` (same weights, same strict 12-channel load) --
   see `_swap_in_fork_unet`. Nothing downstream works without this.

2. ONE MATERIAL SLOT. Upstream runs two generation slots (albedo, mr) through every
   attention block. Emission needs exactly one. `_retarget_material_slots` sets N_pbr=1
   everywhere and points the per-material attention processors at the pretrained
   *albedo* projections, so the emission slot inherits albedo's attention weights the
   same way its learned text token does.

3. THE LEFTOVER MATERIAL PARAMETERS. With one slot, every `to_{q,k,v,out}_mr` / `to_v_mr` /
   `learned_text_clip_mr` parameter -- plus `learned_text_clip_albedo`, superseded by the
   emission token -- becomes unreachable. Upstream's `set_learned_parameters` marks anything
   named `*albedo*`/`*mr*` trainable, which would hand the optimizer ~74.5M parameters that
   can never receive a gradient, a hard error under `DDPStrategy(find_unused_parameters=
   False)`. `_freeze_unreachable_parameters` freezes them; the smoke test asserts that the
   set of gradient-less trainable parameters is then empty.

DO NOT sample from this model with `HunyuanPaintPipeline.__call__`. That path prepares only
`images_normal` / `images_position` into latents and replicates only those for classifier-free
guidance, so it would feed a 12-channel stack to the 20-channel conv_in and (if that were
patched) mis-size the guidance batch. `validation_step` below is the reference sampling loop
for this model; any inference tooling should follow it or extend the pipeline first.
"""

import gc
import os

import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import v2
from diffusers import EulerAncestralDiscreteScheduler

from .unet.model import HunyuanPaint
from .unet.modules import Basic2p5DTransformerBlock, UNet2p5DConditionModel


class HunyuanPaintEmission(HunyuanPaint):
    """Single-stream emission generator: PBR + geometry in, emission multiview out."""

    #: the generation slot this model produces (drives `prepare_batch_data` and the learned token)
    EMISSION_TOKEN = "emission"
    #: the pretrained material slot whose attention projections the emission slot reuses
    ATTENTION_SLOT = "albedo"

    def __init__(
        self,
        stable_diffusion_config,
        *args,
        val_num_inference_steps=30,
        gradient_checkpointing=False,
        **kwargs,
    ):
        # `HunyuanPaint.__init__` strict-loads the pretrained ["albedo", "mr"] checkpoint; the
        # switch to a single emission slot happens afterwards, on top of the loaded weights.
        super().__init__(stable_diffusion_config, *args, **kwargs)

        # LIGHTGEN_VAL_STEPS lets the smoke test cut the sampling loop short without a config edit.
        self.val_num_inference_steps = int(os.environ.get("LIGHTGEN_VAL_STEPS", val_num_inference_steps))

        self.pbr_settings = [self.EMISSION_TOKEN]
        self.pipeline.set_pbr_settings(self.pbr_settings)

        self._swap_in_fork_unet(stable_diffusion_config)
        self._add_emission_token()
        self._retarget_material_slots()

        # Re-run the freeze recipe: it was applied in super().__init__ to the unet we just
        # discarded, and the emission token did not exist yet when it ran.
        self.pipeline.set_learned_parameters()
        self.num_frozen_unreachable_params = self._freeze_unreachable_parameters()

        # A full 512^2 / 6-view / batch-1 step does not fit in 24 GB without this (measured:
        # OOM without, 17.4 GiB peak with, on an RTX 4090). Only the main unet needs it --
        # unet_dual is entirely frozen, so its reference pass builds no autograd graph.
        if gradient_checkpointing:
            self.unet.unet.enable_gradient_checkpointing()

    # ------------------------------------------------------------------ build

    def _swap_in_fork_unet(self, stable_diffusion_config):
        """Replace the snapshot-bundled `UNet2p5DConditionModel` with this package's class.

        Same directory, same weights, same `strict=True` load -- only the *code* changes,
        which is the entire point (see module docstring, item 1).
        """
        snapshot = str(stable_diffusion_config["pretrained_model_name_or_path"])
        unet_dir = os.path.join(snapshot, "unet")
        config_path = os.path.join(unet_dir, "config.json")
        weights_path = os.path.join(unet_dir, "diffusion_pytorch_model.bin")
        for path in (config_path, weights_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{path} is missing -- HunyuanPaintEmission rebuilds the unet from the "
                    f"snapshot's own unet/ directory. If the weights ship under a different "
                    f"filename (e.g. .safetensors), UNet2p5DConditionModel.from_pretrained "
                    f"needs adapting too."
                )

        # Drop the snapshot-class unet before building the replacement, so only one full
        # copy of a ~1.9B-parameter fp32 model is resident at a time.
        del self.unet
        self.pipeline.unet = None
        gc.collect()

        fork_unet = UNet2p5DConditionModel.from_pretrained(unet_dir)
        # `UNet2p5DConditionModel.from_pretrained` builds the wrapper with no schedulers;
        # `HunyuanPaint` gives the wrapper it builds both, so match that.
        fork_unet.train_sched = self.train_scheduler
        fork_unet.val_sched = self.pipeline.scheduler
        # `register_modules`, not plain assignment: `DiffusionPipeline.__setattr__` only
        # recomputes the (library, class) pair in `pipeline.config` when the existing entry is
        # non-null, and the `= None` above nulled it -- so a plain assignment here would leave
        # `config["unet"] == (None, None)` and a later `save_pretrained` would emit that.
        self.pipeline.register_modules(unet=fork_unet)
        self.unet = fork_unet

        actual = type(self.unet).__module__
        if not actual.startswith(__package__):
            raise RuntimeError(
                f"unet class swap did not take: pipe.unet comes from {actual!r}, expected a "
                f"module under {__package__!r}. The embeds_albedo/embeds_mr concat lives in this "
                f"package's unet/modules.py and would silently not run."
            )

    def _add_emission_token(self):
        """Register `learned_text_clip_emission`, warm-started from the pretrained albedo token.

        Registered on the inner `UNet2DConditionModel` (not the 2.5D wrapper) so it sits
        beside `learned_text_clip_{albedo,mr,ref}` under the `unet.unet.` state-dict prefix
        that `train.py`'s `resume_from` path expects.
        """
        inner = self.unet.unet
        if not hasattr(inner, f"learned_text_clip_{self.EMISSION_TOKEN}"):
            warm_start = getattr(inner, f"learned_text_clip_{self.ATTENTION_SLOT}")
            inner.register_parameter(
                f"learned_text_clip_{self.EMISSION_TOKEN}",
                torch.nn.Parameter(warm_start.detach().clone()),
            )

    def _retarget_material_slots(self):
        """Collapse the two upstream material slots down to one emission slot.

        Three different `pbr_setting` attributes are read for two different reasons:
          * the 2.5D wrapper's  -- read by `HunyuanPaintPipeline` to look up
            `learned_text_clip_<token>` and to size the latent batch, so it must say "emission";
          * each transformer block's -- only `len()` is read (that's N_pbr), so the name is
            cosmetic but is kept as "emission" for readability;
          * each attention processor's -- the names SELECT weights (`to_q` vs `to_q_mr`), so
            these must name a slot the pretrained checkpoint actually has. "albedo" points the
            emission slot at the pretrained albedo projections, matching how its learned text
            token is warm-started.
        """
        self.unet.pbr_setting = [self.EMISSION_TOKEN]
        # Only the main unet is retargeted: `unet_dual`'s blocks were built with pbr_setting=None
        # (N_pbr=1 already) and carry no material-specific projections.
        for block in self.unet.unet.modules():
            if not isinstance(block, Basic2p5DTransformerBlock):
                continue
            block.pbr_setting = [self.EMISSION_TOKEN]
            for attn_name in ("attn1", "attn_refview"):
                attn = getattr(block, attn_name, None)
                processor = getattr(attn, "processor", None) if attn is not None else None
                if processor is None:
                    continue
                if hasattr(processor, "pbr_setting"):
                    processor.pbr_setting = [self.ATTENTION_SLOT]
                if hasattr(processor, "pbr_settings"):  # RefAttnProcessor2_0's alias
                    processor.pbr_settings = [self.ATTENTION_SLOT]
                if hasattr(processor, "n_pbr_tokens"):
                    processor.n_pbr_tokens = 1

    def _freeze_unreachable_parameters(self):
        """Freeze trainable parameters that a single emission slot can never reach.

        Two families, both left trainable by the inherited `set_learned_parameters` purely
        because their names contain "mr" or "albedo":
          * every `to_{q,k,v,out}_mr` / `to_v_mr` projection plus `learned_text_clip_mr` --
            the second material slot no longer exists (~74.4M elements);
          * `learned_text_clip_albedo` -- the emission slot carries its own token, and
            albedo's is read exactly once, as the warm start (78,848 elements).

        Left trainable these receive no gradient, which `DDPStrategy(find_unused_parameters=
        False)` -- what train.py uses -- treats as a hard error. Returns the number of frozen
        elements; 0 would mean the naming assumption broke, not that there was nothing to do.
        """
        unreachable_suffixes = (f"learned_text_clip_{self.ATTENTION_SLOT}",)
        frozen = 0
        for name, param in self.unet.named_parameters():
            if not param.requires_grad:
                continue
            if "_mr" in name or name.endswith(unreachable_suffixes):
                param.requires_grad = False
                frozen += param.numel()
        return frozen

    # ------------------------------------------------------------- conditioning

    def _resize01(self, images):
        return v2.functional.resize(
            images.to(self.device), self.view_size, interpolation=3, antialias=True
        ).clamp(0, 1)

    def _encode_conditions(self, batch):
        """Encode everything except the emission target.

        `prepare_batch_data` reads `images_emission` into `target_imgs` because
        `self.pbr_settings == ["emission"]`; albedo and mr are conditions here, not targets,
        so they are encoded separately and ride in on the channel-concat path.
        """
        cond_imgs, _, target_imgs, normal_imgs, position_imgs = self.prepare_batch_data(batch)

        cached = {
            "embeds_albedo": self.encode_images(self._resize01(batch["images_albedo"])),
            "embeds_mr": self.encode_images(self._resize01(batch["images_mr"])),
        }
        if normal_imgs is not None:
            cached["embeds_normal"] = self.encode_images(normal_imgs[0])
        if position_imgs is not None:
            cached["embeds_position"] = self.encode_images(position_imgs[0])
            cached["position_maps"] = position_imgs[0]

        B = cond_imgs.shape[0]
        emission_token = getattr(self.unet, f"learned_text_clip_{self.EMISSION_TOKEN}")
        cached["shading_embeds"] = emission_token.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # (B, 1, 77, 1024)
        cached["ref_latents"] = self.encode_images(cond_imgs)
        if self.unet.use_dino:
            cached["dino_hidden_states"] = self.dino_v2(cond_imgs[:, :1, ...])
        cached["mva_scale"] = 1.0
        cached["ref_scale"] = 1.0
        return target_imgs, cached

    def _apply_condition_dropout(self, cached, B):
        """Upstream's classifier-free-guidance dropout recipe, extended to albedo/mr.

        Draw structure follows `HunyuanPaint.training_step` exactly -- independent per-sample
        draws for the map embeddings, a separate draw for `position_maps`, a separate draw for
        DINO, and the same three-way mva/ref draw -- because inference-time guidance in
        `HunyuanPaintPipeline` relies on that having been trained.

        One deliberate divergence: upstream guards the embedding drop with
        `if "normal_imgs" in cached_condition` / `if "position_imgs" in cached_condition`, but
        the keys it actually sets are `embeds_normal` / `embeds_position` -- so upstream never
        drops those two at all. The keys are corrected here, which means normal/position
        conditioning really is dropped, as the lightgen spec calls for.
        """
        for b in range(B):
            if np.random.rand() < self.drop_cond_prob:
                for key in ("embeds_normal", "embeds_position", "embeds_albedo", "embeds_mr"):
                    if key in cached:
                        cached[key][b, ...] = torch.zeros_like(cached[key][b, ...])
            if np.random.rand() < self.drop_cond_prob:
                if "position_maps" in cached:
                    cached["position_maps"][b, ...] = torch.zeros_like(cached["position_maps"][b, ...])
            if self.unet.use_dino and np.random.rand() < self.drop_cond_prob:
                cached["dino_hidden_states"][b, ...] = torch.zeros_like(cached["dino_hidden_states"][b, ...])

        prob = np.random.rand()
        if prob < self.drop_cond_prob:
            cached["mva_scale"] = 0.0
            cached["ref_scale"] = 0.0
        elif prob > 1.0 - self.drop_cond_prob:
            if np.random.rand() < 0.5:
                cached["mva_scale"] = 0.0
            else:
                cached["ref_scale"] = 0.0

    # ------------------------------------------------------------------- steps

    def training_step(self, batch, batch_idx):
        target_imgs, cached = self._encode_conditions(batch)
        emission_lat = self.encode_images(target_imgs[self.EMISSION_TOKEN])  # (B, N, 4, h, w)
        B, N = emission_lat.shape[:2]

        t = torch.randint(0, self.num_timesteps, size=(B,), device=self.device).long()
        t_flat = rearrange(t.unsqueeze(-1).repeat(1, N), "b n -> (b n)")
        flat = rearrange(emission_lat, "b n c h w -> (b n) c h w")
        noise = torch.randn_like(flat)
        noisy = rearrange(
            self.train_scheduler.add_noise(flat, noise, t_flat), "(b n) c h w -> b n c h w", b=B
        )

        self._apply_condition_dropout(cached, B)

        assert (
            self.train_scheduler.config.prediction_type == "v_prediction"
        ), f"expected v_prediction, got {self.train_scheduler.config.prediction_type}"
        v_pred = self.forward_unet(noisy.unsqueeze(1), t_flat, **cached)  # slot axis N_pbr=1
        v_pred = rearrange(v_pred, "(b p n) c h w -> b p n c h w", p=1, n=N)[:, 0]
        v_target = rearrange(self.get_v(flat, noise, t_flat), "(b n) c h w -> b n c h w", b=B)

        loss = torch.nn.functional.mse_loss(v_pred, v_target.to(v_pred.dtype))
        self.log_dict({"train/emission_loss": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if getattr(self, "_trainer", None) is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Standard single-stream sampling; conditions ride along at every step."""
        target_imgs, cached = self._encode_conditions(batch)
        gt_emission = target_imgs[self.EMISSION_TOKEN]  # (B, N, 3, H, W)
        B, N = gt_emission.shape[:2]
        h, w = cached["embeds_albedo"].shape[-2:]

        # A shared cache dict makes the reference (unet_dual) pass, the DINO projection and the
        # RoPE voxel indices run once for the whole loop instead of once per step -- the same
        # trick `HunyuanPaintPipeline.denoise` uses.
        cached["cache"] = {}

        sched = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing="trailing"
        )
        sched.set_timesteps(self.val_num_inference_steps, device=self.device)
        z = torch.randn(B * N, 4, h, w, device=self.device) * sched.init_noise_sigma
        for t in sched.timesteps:
            z_in = sched.scale_model_input(z, t)
            z_in = rearrange(z_in, "(b n) c h w -> b n c h w", b=B).unsqueeze(1)  # slot axis
            t_step = torch.full((B * N,), int(t.item()), device=self.device, dtype=torch.long)
            v = self.forward_unet(z_in, t_step, **cached)  # (B*N, 4, h, w) back
            z = sched.step(v.float(), t, z).prev_sample

        vae_dtype = next(self.pipeline.vae.parameters()).dtype
        img = self.pipeline.vae.decode(
            z.to(vae_dtype) / self.pipeline.vae.config.scaling_factor, return_dict=False
        )[0]
        pred = (img * 0.5 + 0.5).clamp(0, 1)  # (B*N, 3, H, W)
        gt = rearrange(gt_emission.to(pred.device, pred.dtype), "b n c h w -> (b n) c h w")
        panel = torch.cat([gt, pred], dim=-2)
        self.validation_step_outputs.append(rearrange(panel, "(b n) c h w -> b c h (n w)", b=B))
