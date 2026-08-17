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

import torch
import os, sys
import argparse
import shutil
import subprocess
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment, SLURMEnvironment
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

from src.utils.train_util import instantiate_from_config
import warnings

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="only resume model weights",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="base_config.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="experiment name",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes to use",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,",
        help="gpu ids to use",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging data",
    )
    return parser


class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "project.yaml"))


class CodeSnapshot(Callback):
    """
    Modified from https://github.com/threestudio-project/threestudio/blob/main/threestudio/utils/callbacks.py#L60
    """

    def __init__(self, savedir):
        self.savedir = savedir

    def get_file_list(self):
        return [
            b.decode()
            for b in set(subprocess.check_output('git ls-files -- ":!:configs/*"', shell=True).splitlines())
            | set(  # hard code, TODO: use config to exclude folders or files
                subprocess.check_output("git ls-files --others --exclude-standard", shell=True).splitlines()
            )
        ]

    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)

    #       for f in self.get_file_list():
    #           if not os.path.exists(f) or os.path.isdir(f):
    #               continue
    #           os.makedirs(os.path.join(self.savedir, os.path.dirname(f)), exist_ok=True)
    #           shutil.copyfile(f, os.path.join(self.savedir, f))

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except:
            rank_zero_warn(
                "Code snapshot is not saved. Please make sure you have git installed and are in a git repository."
            )


if __name__ == "__main__":
    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())
    torch.set_float32_matmul_precision("medium")

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    cfg_fname = os.path.split(opt.base)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    exp_name = "-" + opt.name if opt.name != "" else ""
    # LightGen: optional per-submission run stamp appended to the logdir, so two runs of the same
    # config do not share a logdir (and therefore do not silently auto-resume each other via the
    # last.ckpt lookup just below). It comes from the environment -- NOT from datetime.now() here --
    # because pytorch-lightning 1.9's DDP launcher re-execs train.py once per rank, so an
    # in-process timestamp would differ between ranks and each rank would write its own logdir.
    # The sbatch launcher exports LIGHTGEN_RUN_TS=${LIGHTGEN_RUN_TS:-$(date +%Y%m%d-%H%M%S)} once,
    # before python, so every re-exec inherits the same value. Unset => upstream behaviour.
    run_ts = os.environ.get("LIGHTGEN_RUN_TS", "")
    logdir = os.path.join(opt.logdir, cfg_name + exp_name + ("-" + run_ts if run_ts else ""))

    # assert not os.path.exists(logdir) or 'test' in logdir, logdir
    if os.path.exists(logdir) and opt.resume is None:
        auto_resume_path = os.path.join(logdir, "checkpoints", "last.ckpt")
        if os.path.exists(auto_resume_path):
            opt.resume = auto_resume_path
            print(f"Auto set resume ckpt {opt.resume}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    codedir = os.path.join(logdir, "code")

    node_rank = int(os.environ.get("NODE_RANK", 0))  # 当前节点的编号
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 当前节点上的 GPU 编号
    num_gpus_per_node = torch.cuda.device_count()  # 每个节点上的 GPU 数量

    global_rank = node_rank * num_gpus_per_node + local_rank
    seed_everything(opt.seed + global_rank)

    # init configs
    config = OmegaConf.load(opt.base)
    lightning_config = config.lightning
    trainer_config = lightning_config.trainer

    trainer_config["accelerator"] = "gpu"
    rank_zero_print(f"Running on GPUs {opt.gpus}")
    try:
        ngpu = int(opt.gpus)
    except:
        ngpu = len(opt.gpus.strip(",").split(","))
    trainer_config["devices"] = ngpu

    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)

    model_unet = model.unet.unet
    model_unet_prefix = "unet.unet."
    if hasattr(model_unet, "unet"):
        model_unet = model_unet.unet
        model_unet_prefix += "unet."

    if getattr(config, "init_unet_from", None):
        unet_ckpt_path = config.init_unet_from
        sd = torch.load(unet_ckpt_path, map_location="cpu")
        model_unet.load_state_dict(sd, strict=True)

    if getattr(config, "init_vae_from", None):
        vae_ckpt_path = config.init_vae_from
        sd_vae = torch.load(vae_ckpt_path, map_location="cpu")

        def replace_key(key_str):
            replace_pairs = [("key", "to_k"), ("query", "to_q"), ("value", "to_v"), ("proj_attn", "to_out.0")]
            for replace_pair in replace_pairs:
                key_str = key_str.replace(replace_pair[0], replace_pair[1])
            return key_str

        sd_vae = {replace_key(k): v for k, v in sd_vae.items()}
        model.pipeline.vae.load_state_dict(sd_vae, strict=True)

    if hasattr(model.unet, "controlnet"):
        if getattr(config, "init_control_from", None):
            unet_ckpt_path = config.init_control_from
            sd_control = torch.load(unet_ckpt_path, map_location="cpu")
            model.unet.controlnet.load(sd_control, strict=True)

    noise_in_channels = config.model.params.get("noise_in_channels", None)
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

    if hasattr(model.unet, "controlnet"):
        if config.model.params.get("control_in_channels", None):
            control_in_channels = config.model.params.control_in_channels
            model.unet.controlnet.config["conditioning_channels"] = control_in_channels
            condition_conv_in = model.unet.controlnet.controlnet_cond_embedding.conv_in

            new_condition_conv_in = torch.nn.Conv2d(
                control_in_channels,
                condition_conv_in.out_channels,
                kernel_size=condition_conv_in.kernel_size,
                stride=condition_conv_in.stride,
                padding=condition_conv_in.padding,
            )

            with torch.no_grad():
                new_condition_conv_in.weight[:, : condition_conv_in.in_channels, :, :] = condition_conv_in.weight
                if condition_conv_in.bias is not None:
                    new_condition_conv_in.bias = condition_conv_in.bias

            model.unet.controlnet.controlnet_cond_embedding.conv_in = new_condition_conv_in

    rank_zero_print(f"Loaded Init ...")

    if getattr(config, "resume_from", None):
        cnet_ckpt_path = config.resume_from
        sds = torch.load(cnet_ckpt_path, map_location="cpu")["state_dict"]
        sd0 = {k[len(model_unet_prefix) :]: v for k, v in sds.items() if model_unet_prefix in k}
        # model.unet.unet.unet.load_state_dict(sd0, strict=True)
        model_unet.load_state_dict(sd0, strict=True)
        if hasattr(model.unet, "controlnet"):
            sd1 = {k[16:]: v for k, v in sds.items() if "unet.controlnet." in k}
            model.unet.controlnet.load_state_dict(sd1, strict=True)
        rank_zero_print(f"Loaded {cnet_ckpt_path} ...")

    if opt.resume and opt.resume_weights_only:
        model = model.__class__.load_from_checkpoint(opt.resume, **config.model.params)

    model.logdir = logdir

    # trainer and callbacks
    trainer_kwargs = dict()

    # logger
    # LightGen: wandb, not TensorBoard -- all training runs log to the single wandb
    # project "LightGen" (see AGENTS.md naming conventions). Run name defaults to the
    # upstream cfg_name+exp_name but is overridable via LIGHTGEN_WANDB_NAME (set by the
    # sbatch launch script, e.g. "mvpaint_pbr->emission_overfit10_v1"). WANDB_MODE=offline
    # in the environment routes this to an offline run for later `wandb sync`.
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "project": "LightGen",
            "name": os.environ.get("LIGHTGEN_WANDB_NAME", cfg_name + exp_name),
            "save_dir": logdir,
            "offline": os.environ.get("WANDB_MODE", "") == "offline",
        },
    }
    logger_cfg = OmegaConf.merge(default_logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # model checkpoint
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:08}",
            "verbose": True,
            "save_last": True,
            "every_n_train_steps": 5000,
            "save_top_k": -1,  # save all checkpoints
        },
    }

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # callbacks
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train.SetupCallback",
            "params": {
                "resume": opt.resume,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
            },
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            },
        },
        "code_snapshot": {
            "target": "train.CodeSnapshot",
            "params": {
                "savedir": codedir,
            },
        },
    }
    # lightgen: a SECOND, unconditional checkpointer for the LR-cycle troughs.
    #
    # The main callback runs save_top_k=3 on val/emission_mse, and ModelCheckpoint decides
    # BEFORE writing whether a checkpoint would enter the top-k -- a step that would not is
    # never written at all, not written-then-deleted. So no external watcher can preserve a
    # specific step: campaign 20260816-013134's step 20,000 simply never existed on disk.
    #
    # That matters because val/emission_mse is noise-dominated here (64 shapes through 30
    # stochastic sampling steps), so the checkpoints it keeps are partly luck, while the ones
    # actually worth scoring are the LR-cycle troughs -- configure_optimizers uses
    # warm_up_step=1000 + T_step=9000, i.e. a 10,000-step cosine-with-restarts cycle, and the
    # trough is the converged end of each. Saving those unconditionally is what lets the final
    # model be chosen with the real point-sampled evaluator instead of a noisy proxy.
    #
    # Opt-in: set lightning.trough_checkpoint.every_n_train_steps in the config. Writes to a
    # `troughs/` subdirectory so it cannot compete with the main callback's top-k bookkeeping.
    if "trough_checkpoint" in lightning_config:
        trough_cfg = OmegaConf.merge(
            OmegaConf.create({
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "troughs"),
                    "filename": "trough-{step:08}",
                    "save_top_k": -1,      # unconditional: no monitor, nothing to lose to
                    "save_last": False,    # last.ckpt belongs to the main callback alone
                    "verbose": True,
                },
            }),
            OmegaConf.create({"params": lightning_config.trough_checkpoint}),
        )
        default_callbacks_cfg["trough_checkpoint"] = trough_cfg
        rank_zero_print(
            f"++++ trough checkpointer: every "
            f"{lightning_config.trough_checkpoint.get('every_n_train_steps')} steps -> "
            f"{os.path.join(ckptdir, 'troughs')} ++++"
        )

    default_callbacks_cfg["checkpoint_callback"] = modelckpt_cfg

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    trainer_kwargs["precision"] = "bf16"
    # LightGen: upstream's find_unused_parameters=False is kept as the default -- it is the fast
    # path, and the 2-GPU smoke on the emission model confirmed our freeze pattern (unet_dual
    # frozen, learned_text_clip_ref trainable) does not leave parameters unused. The escape hatch
    # exists because that is exactly the failure mode a freeze pattern can produce, and it shows
    # up only under DDP: set LIGHTGEN_DDP_FIND_UNUSED=1 if a run dies with DDP's "expected to
    # have finished reduction in the prior iteration" error. It costs a graph traversal per
    # backward -- measure before leaving it on.
    ddp_find_unused = os.environ.get("LIGHTGEN_DDP_FIND_UNUSED", "0") == "1"
    if ddp_find_unused:
        rank_zero_print("++++ DDP find_unused_parameters=True (LIGHTGEN_DDP_FIND_UNUSED=1) ++++")

    ddp_kwargs = {"find_unused_parameters": ddp_find_unused}
    # LightGen: inside an sbatch, lightning 1.9 auto-detects SLURMEnvironment (any job whose name
    # is not "bash"/"interactive") and then takes the world size from SLURM_NTASKS *regardless of
    # --gpus*, assuming the ranks were launched by `srun`. Our launch scripts are one Slurm task
    # + lightning's own subprocess launcher (star2 policy), so SLURM_NTASKS is 1 and a `--gpus
    # 0,1` run silently trains on ONE rank while holding two GPUs -- observed on cs-venus-05,
    # job 238262: "Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1". Pinning
    # LightningEnvironment restores the subprocess launcher (creates_processes_externally is
    # False there, so DDPStrategy spawns the missing ranks itself).
    #
    # Single node only, and only when Slurm has fewer tasks than we asked for GPUs. Two launch
    # shapes are deliberately left alone:
    #   * a genuine `srun --ntasks=<ranks> python train.py` (SLURM_NTASKS already matches) keeps
    #     the SLURM environment and its rank assignment;
    #   * anything with --num_nodes > 1 keeps it too, because LightningEnvironment cannot replace
    #     it there: its subprocess launcher only spawns *local* children and rendezvouses on
    #     127.0.0.1, so a multi-node job would sit waiting for ranks that never appear. Multi-node
    #     must be srun-launched with one task per rank.
    slurm_ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if ngpu > 1 and opt.num_nodes == 1 and SLURMEnvironment.detect() and slurm_ntasks < ngpu:
        rank_zero_print(
            f"++++ SLURM_NTASKS={slurm_ntasks} < requested ranks ({ngpu}); "
            "using LightningEnvironment so DDP spawns its own ranks ++++"
        )
        ddp_kwargs["cluster_environment"] = LightningEnvironment()
    trainer_kwargs["strategy"] = DDPStrategy(**ddp_kwargs)

    # trainer
    trainer = Trainer(**trainer_config, **trainer_kwargs, num_nodes=opt.num_nodes, inference_mode=False)
    trainer.logdir = logdir

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup("fit")

    # configure learning rate
    base_lr = config.model.base_learning_rate
    if "accumulate_grad_batches" in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    model.learning_rate = base_lr
    rank_zero_print("++++ NOT USING LR SCALING ++++")
    rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")

    # run training loop
    if opt.resume and not opt.resume_weights_only:
        trainer.fit(model, data, ckpt_path=opt.resume)
    else:
        trainer.fit(model, data)
