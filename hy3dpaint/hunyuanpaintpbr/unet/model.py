# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import gc
import os

# import ipdb
import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_info
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from einops import rearrange

from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    ControlNetModel,
)

from .modules import Basic2p5DTransformerBlock, Dino_v2, UNet2p5DConditionModel
import math


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class HunyuanPaint(pl.LightningModule):
    """lightgen: pbr->emission (multiview), our model edited into upstream's HunyuanPaint in place (refactor
        step 8b; configs from before the refactor name it `hunyuanpaintpbr.model_emission.HunyuanPaintEmission`,
        which scripts/convert_config.py maps here). Adapted per the 2026-07-30 lightgen spec (revised).

    Design (user-decided 2026-07-30): SINGLE generation stream. GT albedo + metallic-
    roughness latents are channel-concatenated onto the geometry conditioning (normal +
    CCM) -- the exact mechanism upstream uses for geometry -- and the denoised/supervised
    output is the emission multiview alone (pbr_settings=["emission"], learned token
    warm-started from the pretrained albedo token). Single reference image (the frontal
    albedo render or the TexVerse thumbnail: the loader's `ref_source`); upstream's dual-ref
    consistency loss dropped. No teacher-forcing, no per-slot timestep mixing.

    Three mechanics in here are non-obvious and are the reason this file exists at all:

    1. THE UNET CLASS SWAP. `DiffusionPipeline.from_pretrained(<snapshot>, custom_pipeline=
       ./hunyuanpaintpbr)` does NOT build `pipe.unet` from this package. The snapshot's
       `model_index.json` declares the unet component as `["modules", "UNet2p5DConditionModel"]`,
       which makes diffusers load `<snapshot>/unet/modules.py` as a dynamic module
       (`diffusers_modules.local.modules`). That bundled copy has no knowledge of the
       `embeds_albedo` / `embeds_mr` concat this design depends on, so it would build a
       12-channel input stack and hand it to the expanded conv_in (24 channels, 20 without alpha) -- a loud shape error on
       the first step, not a silent wrong-answer. So after the pipeline loads we throw that unet
       away and rebuild it from THIS package's `UNet2p5DConditionModel.from_pretrained` (same
       weights, same strict 12-channel load) -- see `_swap_in_fork_unet`.

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
    guidance, so it would feed a 12-channel stack to the expanded conv_in and (if that were
    patched) mis-size the guidance batch. `validation_step` below is the reference sampling loop
    for this model; any inference tooling should follow it or extend the pipeline first.
    """

    #: the generation slot this model produces (drives `prepare_batch_data` and the learned token)
    EMISSION_TOKEN = "emission"
    #: the pretrained material slot whose attention projections the emission slot reuses
    ATTENTION_SLOT = "albedo"

    def __init__(
        self,
        stable_diffusion_config,
        control_net_config=None,
        num_view=6,
        view_size=320,
        drop_cond_prob=0.1,
        with_normal_map=None,
        with_position_map=None,
        pbr_settings=["albedo", "mr"],
        val_num_inference_steps=30,
        gradient_checkpointing=False,
        use_alpha=True,
        **kwargs,
    ):
        """Initializes the HunyuanPaint Lightning Module.

        Args:
            stable_diffusion_config: Configuration for loading the Stable Diffusion pipeline
            control_net_config: Configuration for ControlNet (optional)
            num_view: Number of views to process
            view_size: Size of input views (height/width)
            drop_cond_prob: Probability of dropping conditioning input during training
            with_normal_map: Flag indicating whether normal maps are used
            with_position_map: Flag indicating whether position maps are used
            pbr_settings: List of PBR materials to generate (e.g., albedo, metallic-roughness)
            val_num_inference_steps: lightgen: sampling steps of validation_step (LIGHTGEN_VAL_STEPS overrides)
            gradient_checkpointing: lightgen: checkpoint the main unet's activations
            use_alpha: lightgen: the glTF opacity map is a condition (conv_in channels 20-23)
            **kwargs: Additional keyword arguments
        """
        super(HunyuanPaint, self).__init__()

        self.num_view = num_view
        self.view_size = view_size
        self.drop_cond_prob = drop_cond_prob
        self.pbr_settings = pbr_settings

        # init modules
        pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)
        pipeline.set_pbr_settings(self.pbr_settings)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )

        self.with_normal_map = with_normal_map
        self.with_position_map = with_position_map

        self.pipeline = pipeline

        self.pipeline.vae.use_slicing = True

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)

        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = UNet2p5DConditionModel(
                self.pipeline.unet, train_sched, self.pipeline.scheduler, self.pbr_settings
            )
        self.train_scheduler = train_sched  # use ddpm scheduler during training

        self.register_schedule()

        pipeline.set_learned_parameters()

        if control_net_config is not None:
            pipeline.unet = pipeline.unet.bfloat16().requires_grad_(control_net_config.train_unet)
            self.pipeline.add_controlnet(
                ControlNetModel.from_pretrained(control_net_config.pretrained_model_name_or_path),
                conditioning_scale=0.75,
            )

        self.unet = pipeline.unet

        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.vae = self.pipeline.vae.bfloat16()
        self.pipeline.text_encoder = self.pipeline.text_encoder.bfloat16()

        if self.unet.use_dino:
            self.dino_v2 = Dino_v2("facebook/dinov2-giant")
            self.dino_v2 = self.dino_v2.bfloat16()

        self.validation_step_outputs = []

        # lightgen: the emission model. Upstream's build above strict-loads the pretrained
        # ["albedo", "mr"] checkpoint; the switch to a single emission slot happens afterwards, on
        # top of the loaded weights.
        # Whether the per-texel glTF opacity map is a condition (conv_in channels 20-23).
        # True is the current contract -- it is what gives this baseline the same input as
        # TEXGen's 13ch variant, TRELLIS.2 and SegviGen. The flag exists so the archived
        # 20-channel checkpoint (`step=00045000`, the "without alpha" eval row) stays loadable
        # for re-scoring without checking out an older commit; set it false ONLY together with
        # noise_in_channels 20 and the dataset's own use_alpha=false. Mismatches are caught per
        # batch by `_assert_condition_channel_layout`, which checks the real conv_in width
        # rather than trusting these three flags to agree.
        self.use_alpha = use_alpha

        # LIGHTGEN_VAL_STEPS lets the smoke test cut the sampling loop short without a config edit.
        self.val_num_inference_steps = int(os.environ.get("LIGHTGEN_VAL_STEPS", val_num_inference_steps))

        self.pbr_settings = [self.EMISSION_TOKEN]
        self.pipeline.set_pbr_settings(self.pbr_settings)

        self._swap_in_fork_unet(stable_diffusion_config)
        self._add_emission_token()
        self._retarget_material_slots()

        # Re-run the freeze recipe: it was applied above to the unet we just
        # discarded, and the emission token did not exist yet when it ran.
        self.pipeline.set_learned_parameters()
        self.num_frozen_unreachable_params = self._freeze_unreachable_parameters()

        # A full 512^2 / 6-view / batch-1 step does not fit in 24 GB without this (measured:
        # OOM without, 17.4 GiB peak with, on an RTX 4090).
        #
        # This checkpoints the MAIN unet only. Note that `unet_dual`'s reference pass is NOT
        # graph-free, despite every `unet_dual.*` parameter being frozen: its
        # `encoder_hidden_states_ref` is `learned_text_clip_ref`, which the inherited
        # `set_learned_parameters` leaves trainable (the name contains neither "attn1" nor
        # "unet_dual"). That one leaf makes the dual pass require grad, so its activations are
        # retained. The smoke test's "no trainable parameter is gradient-less" assertion is the
        # direct evidence -- `learned_text_clip_ref` does receive a gradient.
        #
        # Two further levers exist if the reference pass ever needs to be cheaper, neither
        # applied here (venus has the headroom, and both change what trains):
        #   * `self.unet.unet_dual.enable_gradient_checkpointing()` -- keeps the token trainable;
        #   * freezing `learned_text_clip_ref` -- then the dual pass really is graph-free.
        if gradient_checkpointing:
            self.unet.unet.enable_gradient_checkpointing()

    def register_schedule(self):

        self.num_timesteps = self.train_scheduler.config.num_train_timesteps

        betas = self.train_scheduler.betas.detach().cpu()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float())

        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1).float())

    def on_fit_start(self):
        device = torch.device(f"cuda:{self.local_rank}")
        self.pipeline.to(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, "images_val"), exist_ok=True)

    def prepare_batch_data(self, batch):
        """Preprocesses a batch of input data for training/inference.

        Args:
            batch: Raw input batch dictionary

        Returns:
            tuple: Contains:
                - cond_imgs: Primary conditioning images (B, 1, C, H, W)
                - cond_imgs_another: Secondary conditioning images (B, 1, C, H, W)
                - target_imgs: Dictionary of target PBR images resized and clamped
                - images_normal: Preprocessed normal maps (if available)
                - images_position: Preprocessed position maps (if available)
        """

        images_cond = batch["images_cond"].to(self.device)  # (B, M, C, H, W), where M is the number of reference images
        cond_imgs, cond_imgs_another = images_cond[:, 0:1, ...], images_cond[:, 1:2, ...]

        cond_size = self.view_size
        cond_imgs = v2.functional.resize(cond_imgs, cond_size, interpolation=3, antialias=True).clamp(0, 1)
        cond_imgs_another = v2.functional.resize(cond_imgs_another, cond_size, interpolation=3, antialias=True).clamp(
            0, 1
        )

        target_imgs = {}
        for pbr_token in self.pbr_settings:
            target_imgs[pbr_token] = batch[f"images_{pbr_token}"].to(self.device)
            target_imgs[pbr_token] = v2.functional.resize(
                target_imgs[pbr_token], self.view_size, interpolation=3, antialias=True
            ).clamp(0, 1)

        images_normal = None
        if "images_normal" in batch:
            images_normal = batch["images_normal"]  # (B, N, C, H, W)
            images_normal = v2.functional.resize(images_normal, self.view_size, interpolation=3, antialias=True).clamp(
                0, 1
            )
            images_normal = [images_normal]

        images_position = None
        if "images_position" in batch:
            images_position = batch["images_position"]  # (B, N, C, H, W)
            images_position = v2.functional.resize(
                images_position, self.view_size, interpolation=3, antialias=True
            ).clamp(0, 1)
            images_position = [images_position]

        return cond_imgs, cond_imgs_another, target_imgs, images_normal, images_position

    @torch.no_grad()
    def forward_text_encoder(self, prompts):
        device = next(self.pipeline.vae.parameters()).device
        text_embeds = self.pipeline.encode_prompt(prompts, device, 1, False)[0]
        return text_embeds

    @torch.no_grad()
    def encode_images(self, images):
        """Encodes input images into latent representations using the VAE.

        Handles both standard input (B, N, C, H, W) and PBR input (B, N_pbrs, N, C, H, W)
        Maintains original batch structure in output latents.

        Args:
            images: Input images tensor

        Returns:
            torch.Tensor: Latent representations with original batch dimensions preserved
        """

        B = images.shape[0]
        image_ndims = images.ndim
        if image_ndims != 5:
            N_pbrs, N = images.shape[1:3]
        images = (
            rearrange(images, "b n c h w -> (b n) c h w")
            if image_ndims == 5
            else rearrange(images, "b n_pbrs n c h w -> (b n_pbrs n) c h w")
        )
        dtype = next(self.pipeline.vae.parameters()).dtype

        images = (images - 0.5) * 2.0
        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor

        latents = (
            rearrange(latents, "(b n) c h w -> b n c h w", b=B)
            if image_ndims == 5
            else rearrange(latents, "(b n_pbrs n) c h w -> b n_pbrs n c h w", b=B, n_pbrs=N_pbrs)
        )

        return latents

    def forward_unet(self, latents, t, **cached_condition):
        """Runs the UNet model to predict noise/latent residuals.

        Args:
            latents: Noisy latent representations (B, C, H, W)
            t: Timestep tensor (B,)
            **cached_condition: Dictionary of conditioning inputs (text embeds, reference images, etc)

        Returns:
            torch.Tensor: UNet output (predicted noise or velocity)
        """

        dtype = next(self.unet.parameters()).dtype
        latents = latents.to(dtype)
        shading_embeds = cached_condition["shading_embeds"]
        pred_noise = self.pipeline.unet(latents, t, encoder_hidden_states=shading_embeds, **cached_condition)
        return pred_noise[0]

    def predict_start_from_z_and_v(self, x_t, t, v):
        """
        Predicts clean image (x0) from noisy latents (x_t) and
        velocity prediction (v) using the v-prediction formula.

        Args:
            x_t: Noisy latents at timestep t
            t: Current timestep
            v: Predicted velocity (v) from UNet

        Returns:
            torch.Tensor: Predicted clean image (x0)
        """

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v(self, x, noise, t):
        """Computes the target velocity (v) for v-prediction training.

        Args:
            x: Clean latents (x0)
            noise: Added noise
            t: Current timestep

        Returns:
            torch.Tensor: Target velocity
        """

        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

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
                    f"{path} is missing -- HunyuanPaint rebuilds the unet from the "
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
                f"package's unet/modules.py; without it the input stack is 12 channels wide and "
                f"the expanded conv_in raises on the first forward."
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
        if self.use_alpha:
            # Per-texel glTF opacity. Encoded exactly like the other PBR conditions (the map
            # stores the scalar replicated across RGB, so the 3-channel VAE takes it as-is) and
            # appended LAST in the conv_in stack -- see modules.py's `_PBR_STACK`.
            cached["embeds_alpha"] = self.encode_images(self._resize01(batch["images_alpha"]))
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
        self._assert_condition_channel_layout(cached)
        return target_imgs, cached

    def _assert_condition_channel_layout(self, cached):
        """Fail loudly if the conv_in channel stack would come out mis-aligned.

        Each condition's conv_in channel offset is nothing but its position in the concat
        inside `UNet2p5DConditionModel.forward`, so an absent key does not remove a condition --
        it slides every later one down four channels. Dropping albedo would feed mr into the
        channels the model learned as albedo; dropping a geometry map would feed albedo into
        pretrained normal/position channels. Neither raises on its own: the widths still work
        out if the conv_in was expanded to match.

        Two separate failures are checked:

        1. a HOLE in the stack -- a key missing from the middle, which silently shifts every
           later condition onto channels that mean something else;
        2. a WIDTH disagreement between the stack this model is about to build and the conv_in
           it will feed. `use_alpha` (here), the dataset's own `use_alpha` and the config's
           `noise_in_channels` are three knobs that must agree, and nothing else forces them to.
           Checking the real `conv_in.in_channels` catches every combination, including the
           quiet one: noise_in_channels 24 with use_alpha false, which trains happily with four
           permanently-zero input channels and is not recoverable after the fact.

        `modules.py` carries the same offset check (it is the code that owns the offsets, and
        the pipeline can reach it without passing through here). This one fires earlier, with
        the conditioning still in hand. A raise rather than `assert` so `python -O` cannot
        drop it.
        """
        expected = ["embeds_normal", "embeds_position", "embeds_albedo", "embeds_mr"]
        if self.use_alpha:
            expected.append("embeds_alpha")
        missing = [key for key in expected if key not in cached]
        if missing:
            raise RuntimeError(
                f"conditioning is missing {missing}; conv_in expects the full positional stack "
                f"{self._stack_description()}. To drop a condition (e.g. for guidance) zero "
                f"its tensor -- never remove the key."
            )
        if "embeds_alpha" in cached and not self.use_alpha:
            raise RuntimeError(
                "use_alpha is false but embeds_alpha was built; it would be concatenated onto "
                "conv_in channels the model has no width for."
            )

        # +1 for the noisy emission latent itself, which occupies channels 0-3.
        want = 4 * (len(expected) + 1)
        got = self.unet.unet.conv_in.in_channels
        if want != got:
            raise RuntimeError(
                f"conv_in has {got} input channels but the condition stack "
                f"{self._stack_description()} needs {want}. Set the config's "
                f"noise_in_channels to {want} (train.py expands conv_in from the pretrained 12 "
                f"to whatever it says), or flip use_alpha to match the checkpoint you are "
                f"loading -- the 20-channel run is the one without alpha."
            )

    def _stack_description(self):
        return ("[noisy 0-3 | normal 4-7 | position 8-11 | albedo 12-15 | mr 16-19"
                + (" | alpha 20-23]" if self.use_alpha else "]"))

    def _apply_condition_dropout(self, cached, B):
        """Upstream's classifier-free-guidance dropout recipe, extended to albedo/mr/alpha.

        Draw structure follows upstream's `training_step` (tag pre-refactor) exactly -- independent per-sample
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
                # Zeroing (not removing) is what `_assert_condition_channel_layout` and
                # modules.py's `_PBR_STACK` prefix rule require -- the key must stay so the
                # positional channel offsets hold. alpha drops with the same draw as the rest
                # of the stack, so guidance sees a coherent all-conditions-off sample.
                for key in ("embeds_normal", "embeds_position", "embeds_albedo", "embeds_mr",
                            "embeds_alpha"):
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

        # fp32 loss: under bf16 autocast `v_pred` comes back bf16, and rounding the fp32 target
        # down to match it would quantise the regression target itself.
        loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float())
        self.log_dict({"train/emission_loss": loss}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if getattr(self, "_trainer", None) is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)
        prefix = "train"
        loss_dict = {}
        loss_dict.update({f"{prefix}/loss": loss})
        return loss, loss_dict

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

        # A monitorable scalar so ModelCheckpoint can select on sampling quality
        # (monitor: 'val/emission_mse', mode: min). Note this is NOT the training objective --
        # training regresses v in latent space, this is pixel MSE between the fully sampled
        # emission views and the GT views, both in [0, 1]. fp32 for the same reason the training
        # loss is fp32: under bf16 autocast `pred` comes back bf16. sync_dist=True averages the
        # per-rank shards under DDP, so the monitored value is the same on every rank.
        val_mse = torch.nn.functional.mse_loss(pred.float(), gt.float())
        self.log(
            "val/emission_mse",
            val_mse,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )

        panel = torch.cat([gt, pred], dim=-2)
        self.validation_step_outputs.append(rearrange(panel, "(b n) c h w -> b c h (n w)", b=B))

    @torch.no_grad()
    def on_validation_epoch_end(self):
        """Aggregates validation results at epoch end.

        Gathers outputs from all GPUs (if distributed training),
        creates a unified visualization grid, and saves to disk.
        Only rank 0 process performs saving.
        """
        # [Result aggregation and visualization...]
        # Key steps:
        # 1. Gather validation outputs from all processes
        # 2. Create image grid combining ground truths and predictions
        # 3. Save visualization with step-numbered filename
        # 4. Clear memory for next validation cycle

        # lightgen: a RESUMED run gets one validation with ZERO batches before training
        # restarts -- pytorch-lightning 1.9 calls _run_validation() from the training epoch
        # loop's on_advance_end() while the restored val dataloader still yields nothing
        # ("Validation: 0it"). Upstream then cats an empty list and the whole run dies:
        #   ValueError: torch.cat(): expected a non-empty list of Tensors
        # That is what killed segments 2, 3 and 4 of campaign 20260816-013134 within minutes
        # each, after segment 1 had banked 26,150 steps -- a chain cannot survive its own
        # first hand-off without this guard.
        #
        # Returning early is collective-safe HERE, and only because the emptiness is
        # all-or-nothing: the val DistributedSampler pads to an equal per-rank length, so
        # either every rank has batches or none does (all four ranks raised this identically).
        # If that ever stops holding, this needs an all_reduce on a has-output flag instead --
        # a partial early return would deadlock the all_gather below, which is worse than the
        # crash it replaces.
        if not self.validation_step_outputs:
            rank_zero_info("validation produced no batches (expected once on resume) -- "
                           "skipping the grid for this cycle")
            return

        images = torch.cat(self.validation_step_outputs, dim=0)
        all_images = self.all_gather(images)
        all_images = rearrange(all_images, "r b c h w -> (r b) c h w")

        if self.global_rank == 0:
            grid = make_grid(all_images, nrow=8, normalize=True, value_range=(0, 1))
            save_image(grid, os.path.join(self.logdir, "images_val", f"val_{self.global_step:07d}.png"))

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)

        def lr_lambda(step):
            warm_up_step = 1000
            T_step = 9000
            gamma = 0.9
            min_lr = 0.1 if step >= warm_up_step else 0.0
            max_lr = 1.0
            normalized_step = step % (warm_up_step + T_step)
            current_max_lr = max_lr * gamma ** (step // (warm_up_step + T_step))
            if current_max_lr < min_lr:
                current_max_lr = min_lr
            if normalized_step < warm_up_step:
                lr_step = min_lr + (normalized_step / warm_up_step) * (current_max_lr - min_lr)
            else:
                step_wc_wp = normalized_step - warm_up_step
                ratio = step_wc_wp / T_step
                lr_step = min_lr + 0.5 * (current_max_lr - min_lr) * (1 + math.cos(math.pi * ratio))
            return lr_step

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": False,
            "name": None,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
