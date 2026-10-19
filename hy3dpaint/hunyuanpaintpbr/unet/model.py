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
    """Emission model: one generation stream, the emission views, conditioned on the normal and
    position maps and on the albedo, metallic-roughness and alpha latents, which are concatenated
    onto conv_in the way upstream concatenates the geometry maps. One reference image; the
    emission token and attention slot start from albedo's.

    Sample with `validation_step`, not `HunyuanPaintPipeline.__call__`: the pipeline prepares only
    the geometry conditions, so conv_in would get a 12-channel stack.
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
            val_num_inference_steps: Sampling steps of validation_step (LIGHTGEN_VAL_STEPS overrides)
            gradient_checkpointing: Flag indicating whether the main unet checkpoints its activations
            use_alpha: Flag indicating whether the glTF opacity map is a condition (conv_in channels 20-23)
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

        # The emission slot, built on top of the pretrained ["albedo", "mr"] weights loaded above.
        # use_alpha false goes with noise_in_channels 20 and the dataset's use_alpha false;
        # _assert_condition_channel_layout checks them against the real conv_in width.
        self.use_alpha = use_alpha

        self.val_num_inference_steps = int(os.environ.get("LIGHTGEN_VAL_STEPS", val_num_inference_steps))

        self.pbr_settings = [self.EMISSION_TOKEN]
        self.pipeline.set_pbr_settings(self.pbr_settings)

        self._swap_in_fork_unet(stable_diffusion_config)
        self._add_emission_token()
        self._retarget_material_slots()

        # Run again: the first call set up the replaced unet, before the emission token existed.
        self.pipeline.set_learned_parameters()
        self.num_frozen_unreachable_params = self._freeze_unreachable_parameters()

        # Main unet only; a 512x512, 6-view, batch-1 step needs it to fit in 24 GB.
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

    def _swap_in_fork_unet(self, stable_diffusion_config):
        """Rebuild the unet with this package's `UNet2p5DConditionModel` (same weights, strict load).

        The snapshot's model_index.json makes diffusers build it from the snapshot's own copy of
        modules.py, which lacks the albedo, mr and alpha concat.
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

        # Free the old unet first: one copy of the ~1.9B parameters at a time.
        del self.unet
        self.pipeline.unet = None
        gc.collect()

        fork_unet = UNet2p5DConditionModel.from_pretrained(unet_dir)
        # from_pretrained builds the wrapper without schedulers.
        fork_unet.train_sched = self.train_scheduler
        fork_unet.val_sched = self.pipeline.scheduler
        # register_modules, not assignment: after `= None`, assignment leaves config["unet"] = (None, None).
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
        """Register `learned_text_clip_emission` on the inner unet, beside the albedo, mr and ref
        tokens, warm-started from the albedo token."""
        inner = self.unet.unet
        if not hasattr(inner, f"learned_text_clip_{self.EMISSION_TOKEN}"):
            warm_start = getattr(inner, f"learned_text_clip_{self.ATTENTION_SLOT}")
            inner.register_parameter(
                f"learned_text_clip_{self.EMISSION_TOKEN}",
                torch.nn.Parameter(warm_start.detach().clone()),
            )

    def _retarget_material_slots(self):
        """Collapse upstream's two material slots to one emission slot.

        The wrapper and the blocks name the generated slot; the attention processors' names
        select weights (`to_q` vs `to_q_mr`), so they name albedo, whose projections the emission
        slot uses.
        """
        self.unet.pbr_setting = [self.EMISSION_TOKEN]
        # unet_dual's blocks have one slot already and no material projections.
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
        """Freeze what one emission slot never reaches: the `*_mr` projections and the mr and albedo
        tokens. Left trainable they get no gradient, which DDP with find_unused_parameters=False
        rejects. Returns the number of frozen elements.
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

    def _resize01(self, images):
        return v2.functional.resize(
            images.to(self.device), self.view_size, interpolation=3, antialias=True
        ).clamp(0, 1)

    def _encode_conditions(self, batch):
        """Encode the conditions; albedo and mr are conditions here, emission the only target."""
        cond_imgs, _, target_imgs, normal_imgs, position_imgs = self.prepare_batch_data(batch)

        # Each encode draws from the global RNG (VAE sampling): this order (albedo, mr, alpha,
        # normal, position, reference) is part of every seeded result.
        cached = {
            "embeds_albedo": self.encode_images(self._resize01(batch["images_albedo"])),
            "embeds_mr": self.encode_images(self._resize01(batch["images_mr"])),
        }
        if self.use_alpha:
            # glTF opacity, stored as RGB; last in the conv_in stack (modules.py).
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
        """Raise if the conv_in stack would be misaligned: a missing condition shifts every later one
        onto other channels, and the stack's width must equal conv_in's (use_alpha, the dataset's
        use_alpha and noise_in_channels have to agree).
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

        # +1: the noisy latent, channels 0-3.
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
        """Upstream's condition dropout with the same draws, extended to albedo, mr and alpha.

        Upstream tests the keys "normal_imgs" and "position_imgs", which it never sets, so it never
        drops normal and position; here the keys are the ones set.
        """
        for b in range(B):
            if np.random.rand() < self.drop_cond_prob:
                # Zero, never remove: the channel offsets are positional.
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

        # fp32: under bf16 autocast v_pred is bf16.
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
        """Samples the emission views (Euler ancestral, no guidance) and stores the GT/prediction panel."""
        target_imgs, cached = self._encode_conditions(batch)
        gt_emission = target_imgs[self.EMISSION_TOKEN]  # (B, N, 3, H, W)
        B, N = gt_emission.shape[:2]
        h, w = cached["embeds_albedo"].shape[-2:]

        # One cache for the loop, as in HunyuanPaintPipeline.denoise: the reference pass runs once.
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

        # Pixel MSE of the sampled views against GT in [0, 1], averaged over ranks.
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

        # A resumed run starts with one validation of zero batches (pytorch-lightning 1.9), and
        # torch.cat fails on an empty list. The sampler pads every rank to the same length, so all
        # ranks return here together and the all_gather below cannot hang.
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
