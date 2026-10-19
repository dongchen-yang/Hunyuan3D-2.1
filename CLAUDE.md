# CLAUDE.md

LightGen's emission fork of Hunyuan3D-2.1 Paint (`pbr→emission (multiview)` in the paper). `README.md`
is the users' doc (what differs from upstream, setup, LightgenBench data, training, inference); this
file is for working on the code. `README_hunyuan3d.md` is upstream's README; upstream's shape model,
demo, app and API are kept unchanged (D12 in `REFACTOR_TODO.md`, the refactor's plan of record).
Campaign history and evaluation: the parent's `docs/baselines/hunyuan3d-paint.md`.

## Branches

`main` is upstream's code, so `git diff main` shows exactly our changes. `lightgen` is frozen at tag
`pre-refactor` (`413dc28`): the code the paper and release runs trained on, which the campaign
`.commit` files name. The refactored code lands on `release` (D11). Commit inside the fork, then bump
the parent's gitlink.

## Runs behind the results

| | paper run | release run |
|---|---|---|
| jupiter | `outputs/mvpaint_74k_alpha_thumb_agentic` | `outputs/mvpaint_alpha_thumb_lightgenbench_v1` |
| data | 74k v2 agentic | LightgenBench: 36,426 / 200 / 200 |
| config | `cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir-xg.yaml` (= the run's `config.yaml`, converted) | `cfgs/hunyuan-paint-emission.yaml`, filled (named `lightgen-emission-release-alpha-thumb.yaml` when it ran) |
| scored | `ckpts/troughs/trough-step=00050000.ckpt` | `ckpts/troughs/trough-step=00040000.ckpt` |

Both: 4 × H100 80 GB, global batch 16, LR 7e-5, 50,000 steps, 24-channel `conv_in`, thumbnail
reference. Checkpoints are 17.9 GB and pickle no class of ours: the contract is the 2,482 state-dict
keys and the `target:` strings of stored configs, which name `hunyuanpaintpbr.model_emission.HunyuanPaintEmission`
and `src.data.dataloader.lightgen_emission_loader.LightgenEmissionDataset`, modules this code no longer
has. `hy3dpaint/scripts/convert_config.py` is the one place for every such conversion (configs, and
checkpoints with `--ckpt`, which ours do not need): `predict_emission_views.py`, the training check and
the parent's `evaluation/mvpaint_bridge/predict_emission_views.py` call it.

## The public tree

The published repository (D16 in `REFACTOR_TODO.md`) is this branch without its internal paths:
`REFACTOR_TODO.md`, `CLAUDE.md`, `hy3dpaint/scripts/internal/` (the checks, the cluster launchers, the
env record, the tests) and `hy3dpaint/cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir-xg.yaml` (the
paper run's config). Everything else is public: `README.md`, comments and `--help` texts name only
public files and no cluster, user, account, jupiter path, parent-repo path or plan step (the scan in
`REFACTOR_TODO.md` § Part 2). A change lands here first, is checked here, then goes out as a new export
commit (step 23).

Tests (CPU): `python -m pytest scripts/internal/tests/ -q` from `hy3dpaint/`: the loader, the example
lists, the re-layout, the predictor's seed and reference rules, the converter. The loader's first test
reads its fixture shapes from jupiter.

## Environments

- Workstation: conda `hunyuanpaint` (`/localhome/dya78/software/miniconda3/envs/hunyuanpaint`, Python
  3.10, torch 2.11.0 + CUDA 12.8); `requirements-emission.txt` pins its versions. sp4090 has a copy at
  the same path (activate with `PATH`, no conda there). Solar: `~/envs/hunyuanpaint` (conda-pack).
- Alliance clusters: a venv from `scripts/internal/stage_alliance.sh env` (Python 3.12, the
  wheelhouse torch 2.11.0, transformers 4.46.2, the `opencv/4.12.0` module).
- The Hunyuan3D-2.1 snapshot is pinned to revision `0b94677654c57bb9a6b6845cd7b704ccf551d327`.

## Commands (from `hy3dpaint/`)

```bash
python train.py --base <cfg> --name <name> --logdir logs/ --gpus 0,1,2,3      # README.md: data and config
bash scripts/internal/submit_release.sh                                        # the release recipe on Alliance (see below)
python predict_emission_views.py --config <run>/configs/project.yaml --ckpt <ckpt> \
    --examples_json <json> --thumbnail_dir <thumbs> --out <out> [--snapshot <dir>] [--seeds N]
python -m pytest scripts/internal/tests/ -q
```

The release launchers check out and require branch `release` (`submit_release.sh:94`,
`train_release_body.sh:86`); a stamp submitted earlier stays on the commit its `.commit` file names.

## Checking a code change

`hy3dpaint/scripts/internal/gate/`, run from the fork root; data and outputs in
`C=/local-scratch/localhome/dya78/lightgen_eval_local/hunyuan_refactor_checks`, outside the repo.
`python3 hy3dpaint/scripts/internal/gate/run_checks.py --rev <commit> --checks paper,release[,fresh,resume] [--full]`
prints one verdict (PASS, FAIL, INCOMPLETE). It runs the frozen parent predictor (or, with
`--entry fork`, the tree's own) on fixed inputs, seed 0, over every GPU free right now, and compares
each GPU type with its own baselines (`baselines.json`); outputs differ across GPU types. The pass
rule, the machines and the training check: `REFACTOR_TODO.md`, How we work and Runs and the gate.

## Traps

- diffusers 0.30 copies `./hunyuanpaintpbr` into `HF_MODULES_CACHE` on every load, in every process.
  On shared scratch a rank imported a half-written file and the others hung at DDP init
  (killarney, 2026-09-21). The release body gives each node a node-local cache, where the copy takes
  microseconds and has not failed since; the 4 ranks of a node still share it, so a cache per
  process is the complete fix. Each check run has its own.
- The release's `checksums.sha256` changed on 2026-09-22 (a dataset-card image added, every data
  line the same): the pin in `train_release_body.sh` and `stage_alliance.sh` (`16ad1851…`) fails on a
  download of the current revision (`a2d81c1f…`); pin the download to revision `8bd0d71f` or update
  the pin.
- `lr_abs=0.000` in a resumed segment is a display artifact; read the learning rate from a checkpoint
  (`optimizer_states[0]['param_groups'][0]['lr']`).
- `val/emission_mse` is too noisy to select on, and from the second checkpoint on each top-3 save is
  ranked by the validation one interval earlier (the first, at step 1,250, comes before any
  validation and is not ranked); choose among the trough checkpoints with the point-sampled
  evaluator.
- fir's shared-GPU `bygpu` partition: two inference jobs died with CUDA out of memory at
  `.to(device)` after a memory check passed; run inference elsewhere.
- `train.py` ignores `key=value` arguments: edit the config.
