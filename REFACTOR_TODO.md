# Hunyuan3D-Emission refactor — plan

Turn our Hunyuan3D-2.1 fork into a clean Hunyuan3D-Emission (display name `pbr→emission
(multiview)`; directory `Hunyuan3D-Emission/` and branch `release` from step 15): only the code behind
the paper and release runs, our changes as edits on upstream's original modules, generic entry points
for LightgenBench users. Modeled on TEXGen-Emission's finished plan (`../TEXGen-Emission/REFACTOR_TODO.md`)
and TRELLIS2's (`../TRELLIS2/.claude/worktrees/trellis2-emission-refactor/REFACTOR_TODO.md`); the rules
below are TEXGen's unless a line says otherwise.

A September attempt (a trim into `deprecated/2026-09-16/`, no renames, a deterministic-mode inference
gate) was specified and approved on 2026-09-17, then called off on 2026-09-19 before any file moved.
Its spec and plan are in parent history (`git show eb97a89^:docs/superpowers/specs/2026-09-16-hunyuan3d-paint-refactor-design.md`,
same for `plans/2026-09-16-hunyuan3d-paint-refactor.md`); use them as a source of facts, not of rulings.

## Progress

- [x] 1. Housekeeping
- [x] 2. Check setup: inference gate, tests, training check, noise envelopes (no code change)
- [x] 3. Dead one-off scripts
- [x] 4. Dropped: upstream's demo, app and API layer stays (D12)
- [x] 5. Launchers (per D14)
- [ ] 5b. Move the kept cluster scripts into `hy3dpaint/scripts/internal/`
- [ ] 6. Configs (per D14)
- [ ] 7. Coverage run (no code change)
- [ ] 8. Fold our model and loader into upstream's originals (per D13)
- [ ] 9. Cleanups inside the folded code
- [ ] 10. Our other edits inside upstream's files
- [ ] 11. The JSON builders and the release data scripts
- [ ] 12. Environment files
- [ ] 13. Release layer: generic entry points for LightgenBench users (per D15)
- [ ] 14. Docs
- [ ] 15. Branch `release` (per D11) and the rename to `Hunyuan3D-Emission/` (per D4) (needs Dongchen's go)
- [ ] 16. Cut-over (per D8)
- [ ] 17. Release cut: our internal files leave the published tree (per D10)

The check setup is step 2, not step 7 as in TEXGen: TEXGen's step 1 moved an existing gate into its
repo, while this fork has no gate yet, and every later step (deletions included) runs one.

## How we work

- Steps run automatically, in order (Dongchen 2026-10-10: "you just run it automatically. after 1
  step, you start the next one"): each step is verified by its checks, then committed, its **Result**
  filled in and reported in one line, and the next one starts. Dongchen can add **Instructions** to
  any step at any time.
- **Result** is at most three lines (what changed, commits, verdict); details live in the commit
  messages, the fork's docs and the step's report.
- One step = one commit with one purpose. Removals and moves never share a commit with a behavior
  change. Cleanups change no behavior; fixes go under "Later", one step each, only if wanted.
- Stop and ask only for: (1) an open decision the step needs; (2) a check that fails,
  unless the cause is an obvious slip in the step; (3) something the plan does not cover (a needed
  fix, a behavior change, a file outside the step's list); (4) an action outside the worktree and our
  compute: a push, a parent-repo commit, the merge, a write to jupiter's shared tree. Jobs on our
  machines for the checks need no go; reading checkpoints from jupiter needs none either.
- After each step that touches files the inference gate must pass for both runs (subset or full, as
  the step says) and every test must pass with no skip. Steps that touch training code (`train.py`,
  the model, the loader, `src/data/objaverse_hunyuan.py`, `hunyuanpaintpbr/unet/*`) also need the
  training check. `missing` must be 0; an empty input never passes.
- Pass rule (TEXGen's, Dongchen 2026-10-06 and 2026-10-08): a view that differs from its baseline by
  at most 1/255 per pixel, on at most 0.5% of its channel values, is noise and passes; any other view
  must lie within the envelope, the largest difference between any two of at least 4 runs of
  `pre-refactor` on the same GPU type. Training check: every recorded value within its envelope, the
  structure (step order, batch names, keys) exact. A result outside is rerun once on another GPU of
  the type, never the one it ran on, before it counts as failed (by hand: `run_checks.py` does not
  rerun; `--checks <check>` with a new label). Every run is reported.
- Run checks in parallel across our machines (the compute-access skill): the gates on the
  workstation 4090 and on any lab 4090 that step 2 qualifies; the training checks on a Solar node
  with at least 48 GB (a 4090 cannot train this model at any batch size); prepare the next step while
  checks run, commit it only after the step before passes. Skip a machine that needs a 2FA login.
- Never wait for a GPU: right before dispatch, probe every candidate (free GPU memory and free host
  RAM: a checkpoint load takes about 20 GB, so 28 GB `MemAvailable`) and send work only to the ones
  free now; a busy GPU or a held lock is skipped, not waited on. A Solar job is submitted without
  `-w` (a pinned job on our own nodes evicts another lab's) and must start at once or be canceled.
  If nothing is free, report it and stop. TEXGen's `scripts/internal/gate/fanout.py` does this for
  TEXGen; step 2 adapts it.
- Compare only within one GPU type and environment: outputs differ between GPU types. Never gate
  against the published views of the paper run: they came from Solar (L40S and A40-class), and on
  the same checkpoint and seed xg4090 and an A40 agree on 0 of 34 shapes (per-view max |Δ| median
  14–16/255).
- A run on the workstation 4090 holds both lock files: `/local-scratch/localhome/dya78/lightgen_4090.lock`
  (the Hunyuan drivers, `../evaluation/newdata_eval/sweeps/common.sh:5`, the compute-access
  `hostprobe.py`) and `/local-scratch/localhome/dya78/.locks/gpu0-rtx4090.lock` (the TEXGen and
  TRELLIS.2 gates). Each lock alone does not exclude the other lanes.
- Every gate and training run gets its own `HF_MODULES_CACHE`: diffusers 0.30 copies
  `./hunyuanpaintpbr` into that cache on every load, so a shared one races and can import another
  tree's copy (`scripts/lightgen/train_release_body.sh:101-108` is the fix for training).
- Upstream is branch `main` (`82920d6`, Tencent's 2025-10-17 commit the fork branched from on
  2026-07-30); Tencent's repo is not a remote. `git diff main -- <file>` shows what we changed. Our
  changes are edits to upstream's original modules (D13: our model and loader fold into them);
  upstream code we do not use stays untouched, even inside files we edit (D9).
- A doc or comment in a kept file that names a removed file is fixed in the same step. Exception:
  the paper run's config `cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir-xg.yaml` keeps its
  bytes (it is the run's own config; D14).
- Each step updates the fork's own docs it affects (`README.md`, and from step 14 the doc of record)
  in the same commit. Parent docs (`AGENTS.md`, `docs/baselines/hunyuan3d-paint.md`,
  `docs/status/checkpoints.md`, `docs/evaluation.md`) describe `lightgen` and change only at step 15:
  a step that makes one stale adds an item to step 15's parent-docs list.
- The evaluation lane's code in the parent (`evaluation/mvpaint_bridge/`, `evaluation/newdata_eval/`)
  is not edited here; it changes only where this work breaks it, at step 15.
- All edits happen in the worktree `.claude/worktrees/hunyuan3d-emission-refactor` (branch
  `hunyuan3d-emission-refactor`, from tag `pre-refactor` = `413dc28`); `git diff pre-refactor` shows
  every edit. The main checkout stays on `lightgen`.
- In the main checkout: never `git add -A` (the worktree is only excluded locally, through
  `info/exclude`, until step 1's ignore rule reaches the branch of record at step 15) and never
  `git clean -x`/`-X` (its ignored `log/` holds the 2026-07-30 env-install and smoke logs).
- The parent pointer is bumped once, at step 15, path-limited: the parent carries other sessions'
  uncommitted work (stage only this work's hunks, never `git add -A`). Nothing is pushed without
  Dongchen's go.

## Runs and the gate

| run | used for | config | checkpoint |
|---|---|---|---|
| paper, jupiter `outputs/mvpaint_74k_alpha_thumb_agentic` | the user study and the qualitative figures; the test-set row until 2026-10-02 (`../docs/evaluation.md:428-431`) | `cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir-xg.yaml` (= the run's jupiter `config.yaml`; fir, 4 × H100, segment 1 at `b04eebf`, segment 2 at `cdcae12` with the `-fir` twin) | `ckpts/troughs/trough-step=00050000.ckpt`, sha256 `cbbc902b3ede9a87d8bf32b42f84db02406387a7f8ac4b559d31343d140f2d06` |
| release, jupiter `outputs/mvpaint_alpha_thumb_lightgenbench_v1` | the paper's tables since 2026-10-02 (`paper_draft/tables/quantitative_eval.tex:12`, `supp_quant_mask.tex:12`, `supp_quant_color.tex:14`) and `../docs/evaluation.md` § Released dataset | template `cfgs/lightgen-emission-release-alpha-thumb.yaml`; inference used the filled copy `/local-scratch/localhome/dya78/lightgen_release_val200/cfg/lightgen-emission-release-alpha-thumb.yaml` (fir at `8a9cb8d` to step 40,859, killarney at `413dc28` from 40k) | `ckpts/troughs/trough-step=00040000.ckpt` (written on fir), sha256 `88260ff0a0e49ba049c904e791cdba0df3e629ffbd3cbf9da2699033b1fbe144` |

Model, loader and `train.py` are byte-identical from `7363928` (2026-08-23) to `pre-refactor`.
Neither checkpoint is on the workstation (the old local copies are gone; only `.sha256` sidecars are
left in `../evaluation/mvpaint_bridge/ckpts/thumb_agentic/`); step 2 pulls both. Each is 17.9 GB and
pickles no class of ours (checked on a local `last.ckpt`: only `collections` and `torch` globals, no
`hyper_parameters`); what code depends on is the state-dict keys and the `target:` strings of the
configs (Keep stable 2, 4).

The checks (built in step 2, `hy3dpaint/scripts/internal/gate/`; data and outputs in
`C=/local-scratch/localhome/dya78/lightgen_eval_local/hunyuan_refactor_checks`, outside the repo):

```bash
G=hy3dpaint/scripts/internal/gate
python3 $G/run_checks.py --rev <commit> --checks paper,release[,fresh,resume] [--full]   # one verdict: PASS, FAIL, INCOMPLETE
bash $G/gate.sh <tree> <out> paper|release [subset20|<shas>]    # one gate by hand (GATE_* overrides: Solar)
bash $G/train-check.sh <tree> <out> fresh|resume                # one training check (>= 48 GB)
```

- The gate runs the parent's predictor, frozen into `C/tooling/repo/` (from parent `6ffabdc`), against
  `git archive <commit> hy3dpaint`, seed 0, 30 Euler-ancestral steps, no guidance, bf16 autocast:
  the sampling of every published row. It loads a weights-only copy of each checkpoint
  (`slim_ckpt.py`: `state_dict` and `global_step`, the only keys the predictor reads; 9.43 GiB, about
  20 GB of host RAM less per process) and the run's own jupiter `config.yaml`. Inputs are fixed
  copies (`make_inputs.py`): the paper run's 204 test shapes and the release run's 200, each with
  `subset20.txt` (every 10th shape) and a `MANIFEST.sha256`. It compares the six views
  `<sha>/seed00/00{0..5}_emission.png` (`compare_views.py`) under the pass rule.
- Where: the workstation 4090 (both lock files; `run_checks.py` uses it only when it is free right
  now), else Solar's RTX PRO 6000 Blackwell Server Edition (cs-venus-19, `star2/checks.sbatch`,
  subset20 baselines only). Training checks run on that Blackwell type only (a 4090 cannot train this
  model). Baselines and envelopes per type: `baselines.json`.
- Routine steps: subset20 (about 2 min on the 4090 plus the model load). Full (`--full`): 204 and 200
  shapes, about 15 min each, for steps that touch the model, the loader, `train.py` or the configs,
  and before step 15.
- The training check uses the math attention kernel (`HUNYUAN_GATE_MATH_SDP=1`): with the fused
  kernels, which deterministic mode does not flag, two runs drifted (loss up to 3e-4); with it, every
  pair of runs is identical.
- Measured on tag `pre-refactor` (2026-10-10): every repeat is byte-identical. 4090: paper 204 twice,
  release 200 twice and against the 2026-09-22 views (made with the full checkpoint, 5 seeds in one
  process). Blackwell: each subset20 gate 4 times, fresh and resume 4 times each. Across types the
  views differ (Blackwell vs 4090: every view, mean 0.1/255, single pixels up to 104/255).

## Keep stable (other code depends on these)

1. The `--hy3dpaint_root` layout: `hy3dpaint/train.py` exists (`../evaluation/mvpaint_bridge/predict_emission_views.py:121`);
   after a `chdir` into `hy3dpaint/`, `src` and `hunyuanpaintpbr` import as top-level packages
   (`:111-132`); `custom_pipeline: ./hunyuanpaintpbr` resolves against that cwd, with `pipeline.py`
   and `unet/{modules,attn_processor}.py`, which diffusers copies into the modules cache. Our edits
   in `unet/` run only because `HunyuanPaintEmission._swap_in_fork_unet` (`model_emission.py:124-168`)
   replaces the unet that diffusers builds from the snapshot's own copies of those files (identical
   to upstream `main`). Callers: the predictor (default root `:70`), `../evaluation/newdata_eval/sweeps/job_hunyuan.sh:13,21`,
   `sweeps/star2/hunyuan_worker.sbatch:104,180,214,376`, `sweeps/stage_hunyuan_host.sh:99-100,146-150`,
   `sweeps/hosts.json:13`, the off-repo release drivers `/local-scratch/localhome/dya78/lightgen_release_{val200,test200}/*.sh`.
2. The `target:` paths in every stored config (jupiter `outputs/*/config.yaml`, each logdir's
   `configs/project.yaml`, the filled release config, `../evaluation/mvpaint_bridge/test_predict_ref_source.py:14`):
   `hunyuanpaintpbr.model_emission.HunyuanPaintEmission`, `src.data.objaverse_hunyuan.DataModuleFromConfig`,
   `src.data.dataloader.lightgen_emission_loader.LightgenEmissionDataset`, `train.SetupCallback`,
   `train.CodeSnapshot`; and `src.utils.train_util.instantiate_from_config` (predictor `:181`). The
   predictor also imports the loader by module path (`:356`). After the fold (D13) our two modules
   stay as one-line re-exports until step 15; from then on `scripts/convert_config.py` maps the old
   paths in stored configs.
3. Config files named from outside: `cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir-xg.yaml`
   (`job_hunyuan.sh:20`, `hunyuan_worker.sbatch:104`, `stage_hunyuan_host.sh:146`,
   `../evaluation/render_emissive/out_blocked_case/02fcc43854994fb0b4c5b4bd847fbd3a/scripts/run_xg.sh:6`)
   and `cfgs/lightgen-emission-release-alpha-thumb.yaml` (the release launchers). Keys read from
   outside: `data.params.train[0].params.ref_source` (predictor `:91-92`),
   `model.params.{noise_in_channels, gradient_checkpointing, stable_diffusion_config.pretrained_model_name_or_path}`.
   Every key of the stored configs stays accepted.
4. State-dict keys: 2,482 in all, `unet.unet.*` (1,062, among them `conv_in.weight` [320, 24, 3, 3]
   and `learned_text_clip_{albedo,mr,ref,emission}`), `unet.unet_dual.*` (686), `dino_v2.dino_v2.*`
   (727), and seven schedule buffers. The predictor fails on any missing or unexpected key
   (`:209-217`), so attribute and module names inside the model (`HunyuanPaintEmission`, after step 8
   upstream's `HunyuanPaint`), the UNets and the attention processors stay, and so do the frozen, unreachable `*_mr*` tensors. The conv_in
   widening runs before the load (`train.py:256-272`, copied by the predictor `:137-168`).
5. The model surface the predictor uses: `num_view`, `view_size`, `device`, `val_num_inference_steps`
   (and its `LIGHTGEN_VAL_STEPS` override, `model_emission.py:90`), `pipeline` (and `.to`),
   `unet.unet.conv_in`, `validation_step(batch, 0)` appending the `cat([gt, pred], -2)` panel to
   `validation_step_outputs` (split back by the predictor's `_split_panel`, `:233-246`).
6. The global-RNG draw order, with no `generator=` anywhere: six `posterior.sample` calls
   (`unet/model.py:234`), one per encoded map in the order albedo, mr, alpha, normal, position, ref
   (`model_emission.py:258-276`); then `torch.randn` (`:432`); then 30 ancestral scheduler steps
   (`:438`). Any reorder changes the output of every published seed.
7. The loader: `LightgenEmissionDataset(json_path, num_view, image_size, use_alpha=True,
   ref_source="frontal_albedo", thumbnail_dir=None, augment_ref=False)`, `REF_SOURCES`, `.dirs`
   (rebound by the predictor's `--limit`, `:377-379`), `batch["name"]` = the example dir whose
   basename is the sha (`:397-398`); the fixture layout `render_tex/00{0..5}_{albedo,mr,alpha,normal,pos,emission}.png`,
   `render_tex/transforms.json`, `render_cond/000.png` (the shape dirs, written by
   `../data_processing/data_preparation/{multiview.py,run.py}`), and `<thumbnail_dir>/<sha>.png`
   (staged separately); the star2 worker's preflight checks `render_tex/000_albedo.png` and the
   thumbnail (`hunyuan_worker.sbatch:303-320`).
8. `train.py`'s outputs and switches: the logdir `logs/<cfg stem>-<name>[-$LIGHTGEN_RUN_TS]/` (re-derived
   by the launchers, `train_release_body.sh:223`), `checkpoints/{last.ckpt, last-v*.ckpt, step=XXXXXXXX.ckpt, troughs/trough-step=XXXXXXXX.ckpt}`,
   `configs/project.yaml`; `--resume`, `--seed` and the auto-resume from `<logdir>/checkpoints/last.ckpt`
   (`:190-194`); the environment variables `LIGHTGEN_RUN_TS`, `LIGHTGEN_WANDB_NAME`, `WANDB_MODE`,
   `LIGHTGEN_DDP_FIND_UNUSED`; the wandb project `LightGen` (`:325`).
9. Upstream data the parent reads: `hy3dpaint/train_examples/001/render_tex/transforms.json`
   (`../data_processing/data_preparation/hunyuan_cameras.py:38-41`; its parity test
   `tests/test_multiview_cameras.py:153-167` skips silently when the file is missing). Parent
   docstrings cite `src/data/dataloader/objaverse_loader_forTexturePBR.py:53` (the 127 gray),
   `DifferentiableRenderer/MeshRender.py` and `hy3dshape/tools/render/render.py`.
10. The release scripts' interface: `submit_release.sh` and its knobs (`STAMP`, `SEGMENTS`,
    `SEG_START`, `DEPEND`, `SMOKE`, `CLUSTER`, `WALLTIME`, `EXCLUDE`, `--dry-run`),
    `train_release_body.sh`, `stage_alliance.sh`, `relayout_release.py`, `val64_release_shas.txt`,
    `$ROOT/release_hf.verified`, `$ROOT/log/campaign_<name>_<stamp>.commit` (a job runs the commit it
    was submitted on); the release run's stamp `LIGHTGEN_RUN_TS=20260919-212302`.
11. The pretrained inputs and the env: snapshot rev `0b94677654c57bb9a6b6845cd7b704ccf551d327`,
    DINOv2-giant built from the hub id `facebook/dinov2-giant` (`unet/model.py:120`), the
    `hunyuanpaint` env (Python 3.10, torch 2.11.0+cu128, diffusers 0.30.0, transformers 4.46.0,
    pytorch-lightning 1.9.5), and the Alliance venv `stage_alliance.sh` builds.
12. The predictor's output layout `<out>/<sha>/seed<NN>/00{0..5}_emission.png` and `manifest.json`
    (`n_shapes`, per-seed `rng_seed`): read by `../evaluation/mvpaint_bridge/bridge_batch.py:102`,
    `job_hunyuan.sh:36`, `hunyuan_worker.sbatch:384-398`, `../paper_figures/hunyuan_extra_light/config.py:12`.
    Any fork-side inference entry (D15) writes the same layout.
13. The directory `Hunyuan3DPaint-emissive/` until step 15 renames it `Hunyuan3D-Emission/` (D4)
    (`../.gitmodules:15-18`; Keep stable 1; `hunyuan_cameras.py:38-41`; jupiter
    `trellis2_bw/eval_inputs/solar_hunyuan_stage/Hunyuan3DPaint-emissive.tar`, which unpacks to that
    name). The Alliance clones are named `Hunyuan3D-2.1-emissive`.
14. Branch `lightgen` and tag `pre-refactor`: the parent gitlink is `413dc28`, the cluster clones
    track `lightgen`, and the release run's segments ran `8a9cb8d` and `413dc28`. From step 15
    `lightgen` stays frozen there and `release` is the branch of record (D11).

## Decisions

Carried over from TEXGen:

- D1 Removed files are deleted, not moved to `deprecated/`; git history and the tag `pre-refactor`
  keep them. (The September plan moved them into `deprecated/2026-09-16/` instead.)
- D2 Checks cover both the paper and the release checkpoints (the paper's tables now come from the
  release one, its user study and figures from the paper one).
- D3 Cleanup steps change no behavior.
- D5 Steps that touch training code get the training check: a short run of the real `train.py`,
  per-step loss compared before and after, plus one resume from a published checkpoint (the release
  40k trough). `train.py` already seeds once (`seed_everything(opt.seed + global_rank)`, `:205`,
  before the model is built); the check adds deterministic mode, a reseed right before
  `trainer.fit` and a recording callback through a `sitecustomize.py` on `PYTHONPATH`, not by
  editing `train.py`, and sets `trainer.benchmark: false` in its own config.
- D6 Every config- or env-switched option in kept code stays (`use_alpha: false` with 20 channels,
  `ref_source: frontal_albedo`, `augment_ref`, `lightning.trough_checkpoint`, `LIGHTGEN_VAL_STEPS`,
  `LIGHTGEN_DDP_FIND_UNUSED`, the single-node SLURM pin); only their configs go.
- D7 Work happens on branch `hunyuan3d-emission-refactor`; the parent pointer is bumped once, at
  step 15.
- D9 Upstream code we do not use stays untouched, files and blocks alike: the demo, app and API layer
  (D12), `hy3dshape/`, `textureGenPipeline.py`, `DifferentiableRenderer/`, `custom_rasterizer/`,
  `utils/`, `convert_utils.py`, `train_examples/`, `cfgs/hunyuan-paint-pbr.yaml` (its target
  `hunyuanpaintpbr.model.HunyuanPaint` does not resolve on `main` either), the dead
  `multiple_devices` branch in `unet/attn_processor.py:692-695`, `CodeSnapshot.get_file_list` in
  `train.py:139-146`, and the upstream methods our folded code does not replace (D13).
- [ ] D10 (open, decided at step 17) Internal files in the published code. TEXGen's ruling (Dongchen
  2026-10-09): the release drops the check tooling, and with it the other internal files. Here
  `scripts/internal/` will hold the gate and our cluster launchers (accounts `rrg-msavva_gpu`,
  `def-msavva_gpu`, `aip-msavva`, `def-msavva`; `$HOME/scratch` defaults; a mail address,
  `submit_release.sh:117`). With one branch of record (D11), step 17 either deletes them from
  `release` and tags the last commit that has them (`checks-tooling`, as TEXGen's step 19), or keeps
  `release` internal and publishes a separate cut of it (TEXGen's `main` plus `release`).
  Recommended: the first.

Decided by Dongchen on 2026-10-10:

- D4 The directory becomes `Hunyuan3D-Emission/` at step 15, the paper's name for the lane. The
  GitHub repository's name (`dongchen-yang/Hunyuan3D-2.1`) is decided at step 15 (TEXGen kept its own).
- D8 Cut-over (step 16): the Solar stage tar, the xg4090 staged tree and the Alliance clones move to
  `release`. Past campaigns keep their pinned commits.
- D11 The result lands on a new branch `release`, created at step 15 from this branch's head:
  `.gitmodules` and the parent gitlink follow it, `lightgen` stays frozen at `pre-refactor` (the code
  the paper and release runs trained on, which the campaign `.commit` files name), and `main` stays
  upstream's. (`lightgen` exists because the fork's `main` is upstream's code: our work went on a
  separate branch so `git diff main` shows exactly our changes.)
- D12 Upstream's demo, app and API layer stays; step 4 is dropped.
- D13 Fold, as TEXGen's step 8: `HunyuanPaintEmission` into upstream's `HunyuanPaint`
  (`hunyuanpaintpbr/unet/model.py:47`) and `LightgenEmissionDataset` into upstream's `TextureDataset`
  (`src/data/dataloader/objaverse_loader_forTexturePBR.py:25`), our methods replacing upstream's in
  place. Until step 15 our two files stay as one-line re-exports, because the parent predictor
  imports the loader by module path (`../evaluation/mvpaint_bridge/predict_emission_views.py:356`) and
  the gate runs a frozen copy of it; at step 15 the parent moves to the new paths, the re-exports are
  deleted, and a `scripts/convert_config.py` maps the old class paths in stored configs (no checkpoint
  needs converting: none pickles a class of ours). With the fold, upstream's own two-slot PBR training
  path no longer exists in the fork (its config already fails to import on `main`); the kept demo
  stack uses neither class.
- D14 Keep only the release launchers and config (`submit_release.sh`, `train_release_body.sh`,
  `stage_alliance.sh`, the release config template) and the paper run's inference config
  `-fir-xg.yaml` (four parent scripts name it; it equals the run's jupiter `config.yaml`); delete the
  other 10 configs and the 74k-era launchers. The paper run's launchers stay readable at the commits
  each segment ran (`b04eebf`, `cdcae12`); both carry a silent exit (`ls -1t last.ckpt last-v*.ckpt |
  head -1` under `set -euo pipefail` exits 2 when either name is missing; `train_release_body.sh:229-231`
  has the `|| true` form).

Open:

- [ ] D15 An inference script for LightgenBench users: people outside the lab who download the dataset
  from Hugging Face and run this baseline from the published fork, without our parent repo. Today the
  fork can train, but the only script that turns a checkpoint into views is the parent's
  `evaluation/mvpaint_bridge/predict_emission_views.py` (with `evaluation/newdata_eval/seedutil.py`),
  and the views-to-UV bridge (`bridge_views_to_uv.py`) is the parent's too. TEXGen-Emission ships its
  inference script (`inference_specific_samples.py`) with a README command. Recommended: the same
  here, a fork-side script that writes the same views (Keep stable 12), gated byte-identical against
  the frozen parent predictor; the parent predictor stays, the bridge stays in the parent, and the
  README says where it is.

## Done

1. (part) Tag `pre-refactor` = `413dc28` (the `lightgen` head, equal to the parent gitlink and to
   `origin/lightgen`) and this worktree (branch `hunyuan3d-emission-refactor`), 2026-10-06. The tag
   is local and lightweight. `.claude/worktrees/` is excluded through the submodule's local
   `info/exclude`; nothing was committed on `lightgen`.
2. Decisions D4, D8, D11, D12, D13, D14 answered by Dongchen, 2026-10-10 (§ Decisions); D10 and D15
   open.

## Steps

### 1. Housekeeping

- [x] Tag `pre-refactor` and the worktree (Done 1).
- [x] This plan on the branch (`REFACTOR_TODO.md`).
- [x] `.claude/` in `.gitignore` (the September spec's decision 9; it reaches the branch of record at step 15).
- [x] Leave the dangling annotated tag object `e4d3a55` (the September `pre-trim-2026-09-16`, deleted
  by name) alone; `git gc` may prune it.
- [x] `../docs/status/refactor.md` has no Hunyuan entry, not even for the September attempt: added
  to step 15's list.

**Instructions:**

**Result:** `.gitignore` ignores `.claude/` (`.gitignore:191`, `git check-ignore`); the plan on the
branch (`b14e382`, `483155f`). Tests 18/18 passed, no skip. PASS.

### 2. Check setup: inference gate, tests, training check, noise envelopes (no code change)

All runs on tag `pre-refactor`. Tooling in `hy3dpaint/scripts/internal/gate/`; data and run outputs in
`C` (§ Runs and the gate), outside the repo.

- [x] Both checkpoints pulled (sha256 against § Runs) into the parent-ignored
  `../evaluation/mvpaint_bridge/ckpts/<run>/` (`C/ckpts` links there), with weights-only copies.
- [x] The predictor frozen into `C/tooling/repo/` (parent `6ffabdc`).
- [x] Fixed inputs (`make_inputs.py`): paper 204, release 200, the training-check root (16 train and 2
  val release shapes), each with a `MANIFEST.sha256`.
- [x] `gate.sh`, `compare_views.py` (TEXGen's tolerance and envelope), `train-check.sh` with
  `traincheck/` (Lightning 1.9.5 hooks, `summarize_ckpt.py`), `compare_traincheck.py`,
  `star2/checks.sbatch`, `run_checks.py` (the one command), `baselines.json`.
- [x] Envelopes: 4090 paper 2 runs, release 3 (with the 2026-09-22 views); Blackwell subset20 gates 4
  runs each, fresh and resume 4 each. Every repeat byte-identical.
- [x] Lab 4090s: xg4090 was down and sp4090 has no `hunyuanpaint` env; neither is used.
- [x] Deterministic mode for the gates: not needed (normal mode repeats byte for byte, as every
  published view was made). The training check needs it plus the math attention kernel.
- [x] Tests: the fork's 18 (no skip); the parent's `evaluation/mvpaint_bridge/test_predict_ref_source.py`
  (7, `hunyuanpaint`); camera parity against the tree under test,
  `hunyuan_cameras.regenerate("<tree>/hy3dpaint/train_examples/001/render_tex/transforms.json")` from
  the parent root (the parent's own test calls `regenerate()` without a path, which prefers the main
  checkout's file over `HUNYUAN_TRANSFORMS_JSON`, `hunyuan_cameras.py:139-142`).

**Instructions:**

**Result:** The check tooling (above), reviewed by Codex (8 of 10 findings fixed: stale-label reuse,
richer checkpoint summary, envelopes need 4 runs, Solar device mapping and collection). Self-test on
`pre-refactor` PASS (all IDENTICAL); tests 18 + 7 + camera parity. Training repeats only with math attention.

### 3. Dead one-off scripts

Not used by the paper or the release run; nothing calls them.

- [x] `hy3dpaint/scripts/lightgen/smoke_train_step.py`: broken since `f4dd311` (it calls
  `_assert_condition_channel_layout` on the class, assumes 20 channels and `/localscratch` paths).
  Fix the comment that names it: `env_setup.sh:126`.
- [x] `hy3dpaint/scripts/lightgen/make_examples_json.py`: superseded by `make_74k_jsons.py`. Fix the
  comments that name it: `make_74k_jsons.py:4`, `make_val_subset.py:17`. The comment in
  `cfgs/lightgen-emission-74k.yaml:14` goes with that config in step 6. The parent predictor's
  `--examples_json` help names it: step 15's list.

Checks: the subset gates, the tests.

**Instructions:**

**Result:** Both deleted, the three comments that named them reworded. Subset gates IDENTICAL
(Blackwell, job 296022); tests 18/18. PASS.

### 4. Dropped: upstream's demo, app and API layer stays (D12)

### 5. Launchers (per D14)

Kept: `submit_release.sh`, `train_release_body.sh`, `stage_alliance.sh`, `relayout_release.py` and its
test, `make_74k_jsons.py` and its test, `make_val_subset.py`, `val64_release_shas.txt`, `env_setup.sh`
and `smoke_load_pretrained.py` (step 12), `test_lightgen_emission_loader.py` and `pilot10_local.json`
(the loader test's fixture list).

- [x] Deleted (D14): `train_74k.sbatch`, `train_74k_alpha.sbatch`,
  `train_74k_alpha_fir.sbatch`, `train_74k_alpha_nonzero_nocopy_fir.sbatch`,
  `train_74k_alpha_nonzero_nocopy_venus05.sbatch`, `train_74k_alpha_thumb_agentic_fir.sbatch`,
  `train_74k_alpha_thumb_agentic_fir_xg.sbatch`, `train_74k_alpha_thumb_agentic_venus05.sbatch` (never
  trained), `smoke_fir.sbatch`, `ddp_smoke.sbatch`, `overfit_pilot10.sbatch`,
  `infer_troughs_fir.sbatch`, `infer_t50k_seeds_fir.sbatch` (one-off `_v2 + alpha` inference that
  copies the parent predictor), `submit_chain_fir.sh`, `stage_fir.sh` and `env_setup_fir.sh` (both
  replaced by `stage_alliance.sh`), `val64_v2_shas.txt` (read only by the 74k-era launchers).
- [x] Fix the comments in kept files that name them: `train_release_body.sh:12`,
  `stage_alliance.sh:12,46`, the release config `:43`, `make_74k_jsons.py:46`, `make_val_subset.py:25`.
  The `-fir-xg` config's comment `:50` stays (D14). `../AGENTS.md:66` names `submit_chain_fir.sh` as
  the training command: step 15's list.

Checks: `submit_release.sh --dry-run` for each cluster writes the same job file as on
`pre-refactor` (it prints only the header and writes the whole script to disk, `:120-126`; compare
the files with the date and `COMMIT=` lines normalized, the edited comments being the only other
difference); the subset gates; the tests.

**Instructions:**

**Result:** 17 files deleted, 5 comment lines in kept files point at `pre-refactor` or `cdcae12`. Subset gates
IDENTICAL (job 296027); tests 18/18; `--dry-run` job files equal on all six clusters but the reworded comment. PASS.

### 5b. Move the kept cluster scripts into `hy3dpaint/scripts/internal/`

- [ ] `submit_release.sh`, `train_release_body.sh`, `stage_alliance.sh` (they hold the accounts, the
  `$HOME/scratch` defaults, the mail address; D10), next to `gate/`. Update the paths they use to
  call each other and the tools left in `scripts/lightgen/` (`make_74k_jsons.py`,
  `relayout_release.py`, `val64_release_shas.txt`, the tests `stage_alliance.sh:123` runs) in the same
  commit; nothing else changes.

Checks: `--dry-run` as in step 5, identical apart from the moved paths; the tests.

**Instructions:**

**Result:**

### 6. Configs (per D14)

- [ ] Kept: `cfgs/lightgen-emission-release-alpha-thumb.yaml`,
  `cfgs/lightgen-emission-74k-alpha-thumb-agentic-fir-xg.yaml` (bytes unchanged), upstream's
  `cfgs/hunyuan-paint-pbr.yaml` (D9).
- [ ] Deleted, with D14 as recommended: `lightgen-emission-74k.yaml` (the `_v2` 20-channel row),
  `-74k-alpha.yaml` (never trained), `-74k-alpha-fir.yaml` (the `_v2 + alpha` row),
  `-74k-alpha-nonzero-nocopy.yaml` and `-nonzero-nocopy-venus05.yaml` (the heuristic-filtering row),
  `-74k-alpha-thumb-agentic-fir.yaml` (the paper run's second segment; differs from `-fir-xg` in four
  data paths), `-74k-alpha-thumb-agentic-venus05.yaml` (never trained), `-ddp-smoke.yaml`,
  `-fir-smoke.yaml`, `-overfit10.yaml`; and `scripts/lightgen/pilot10_venus.json` (read only by
  `-ddp-smoke.yaml` and `-overfit10.yaml`). Their switches stay in code (D6); published manifests that
  record their names are history (`git show pre-refactor:hy3dpaint/cfgs/<file>`).
- [ ] Fix the release config's header (`:4-6`), which sends readers to the deleted `-fir.yaml` and its
  venus-05 twin for the reasoning: point at the doc of record or the `-fir-xg` config.

Checks: every kept config loads with OmegaConf and its `target:` strings import, except upstream's
`hunyuan-paint-pbr.yaml` (D9; its target does not resolve on `main` either); the full gates (the
configs are gate inputs); the tests.

**Instructions:**

**Result:**

### 7. Coverage run (no code change)

The training check config under a line-coverage tracer with its multiprocessing support, one GPU,
one worker (the train loader always passes `prefetch_factor=2`, `src/data/objaverse_hunyuan.py:62-68`,
which PyTorch rejects with zero workers), so that training, validation, both checkpoint saves and a
resume run; plus the frozen predictor on a few shapes of each
gate. Output: per file of the import closure (`train.py`, `model_emission.py`, the loader,
`objaverse_hunyuan.py`, `unet/{model,modules,attn_processor}.py`, `pipeline.py`, `loader_util.py`), the
lines that never ran. "Never ran" is not "dead" for the switches of D6, DDP paths, the SLURM pin and
upstream code (D9).

**Instructions:**

**Result:**

### 8. Fold our model and loader into upstream's originals (per D13)

| ours | upstream original |
|---|---|
| `src/data/dataloader/lightgen_emission_loader.py` `LightgenEmissionDataset` (subclass of `BaseDataset`) | `src/data/dataloader/objaverse_loader_forTexturePBR.py:25` `TextureDataset` |
| `hunyuanpaintpbr/model_emission.py` `HunyuanPaintEmission` (subclass; overrides `__init__`, `training_step`, `validation_step`; adds nine helpers) | `hunyuanpaintpbr/unet/model.py:47` `HunyuanPaint` |

Each fold moves our code into the original file, editing upstream's class in place: our methods
replace upstream's, our helpers join them, and upstream methods ours do not replace stay as they are
(D9). Our file becomes a one-line re-export until step 15 (D13); the release config template names
the new path in the same commit (the `-fir-xg` config keeps its bytes, D14). One commit per fold, in
this order. The commit message names the upstream behavior each edit replaces (`TextureDataset`'s
glob-picked random views and lighting-suffix reference pairs; `HunyuanPaint`'s two-slot
`training_step` with its dual-reference loss, its `validation_step` through `pipeline.__call__`) and
what the fold changes observably (the class name in logs and in a new logdir's `configs/project.yaml`).

- [ ] 8a. Loader: `LightgenEmissionDataset` → `TextureDataset`, with our constructor (Keep stable 7).
  The loader tests move to the new path. Full gates, the training check, the tests.
- [ ] 8b. Model: `HunyuanPaintEmission` → `HunyuanPaint`. Unchanged: Keep stable 4, 5 and 6 (attribute
  names, the panel, the RNG draw order); `_swap_in_fork_unet` (building the repo class directly is
  under Later); our condition-dropout fix (`model_emission.py:341-379`; upstream's
  `unet/model.py:363,367` test a key that never exists, so upstream never drops normal and position).
  Full gates, the training check fresh and resume, the tests.

**Instructions:**

**Result:**

### 9. Cleanups inside the folded code

No behavior change; one commit each where code changes. Line numbers are from before the folds.

- [ ] Stale docstrings: `model_emission.py:13-22` and `:38-41` (a 12-channel stack into a 20-channel
  conv_in; the default is 24).
- [ ] The conv_in channel layout `[noisy 0-3 | normal 4-7 | position 8-11 | albedo 12-15 | mr 16-19 |
  alpha 20-23]` is spelled out three times (the guard in `unet/modules.py:965-990`,
  `_assert_condition_channel_layout` and `_stack_description` at `model_emission.py:283-339`, the
  docstring): one definition, the others refer to it.
- [ ] Branches step 7 shows never run and no switch reaches: `if normal_imgs/position_imgs is not
  None` (`model_emission.py:266-270`), `self.unet.use_dino` (`:276`, `:368`; hardcoded True at
  `unet/modules.py:788`), `getattr(self, "_trainer")` (`:410`; false only in the smoke script deleted
  in step 3), the alpha check at `:319-323`. Remove only with the coverage record behind it.
- [ ] `_resize01` (`model_emission.py:243-246`) duplicates the resize in `prepare_batch_data`
  (`unet/model.py:179-181`), now in the same class: reuse it only if the gates stay within tolerance.
- [ ] The loader's `MAPS` comment (`lightgen_emission_loader.py:60-62`) says the loader order is the
  conv_in order; it is not, and the model reads the maps by key. Stale citations: `:57` names
  `data_processing/multiview_render_pipeline/append_alpha_views.py` (now
  `data_processing/archive/multiview_render_pipeline_74k/`); `:64` cites upstream
  `objaverse_loader_forTexturePBR.py:52-54` for the background colors (they are at `:51-53`, in the
  same file after the fold). The loader test's unused `FIXTURE_ROOT` (`test_lightgen_emission_loader.py:25-27`).
- Unchanged: `images_cond` stacks the reference twice (slot 1 is never used, but it is the model's
  input shape); the decode guard; `LIGHTGEN_VAL_STEPS` (Keep stable 5).

Checks: the full gates; the training check, fresh and resume; the tests.

**Instructions:**

**Result:**

### 10. Our other edits inside upstream's files

Each stays a minimal edit in place; `git diff main -- <file>` is its record. One commit per file,
only where something changes.

- [ ] `hy3dpaint/train.py` (93 lines added, 5 removed): `:25` imports; `:179-187` the `LIGHTGEN_RUN_TS` logdir suffix
  (Keep stable 8); `:316-330` the wandb logger in place of TensorBoard (project `LightGen`); `:378-413`
  the trough checkpointer; `:426-437` `LIGHTGEN_DDP_FIND_UNUSED`; `:438-462` the single-node SLURM
  pin; `:256-272` the conv_in expansion (the predictor copies it). Look for stale comments and for
  switches no launcher sets any more (they stay, D6, but the doc says so). `--resume_weights_only`
  (`:308-309`) skips the conv_in expansion and cannot load our checkpoints: under Later.
- [ ] `hunyuanpaintpbr/unet/model.py:20,580-598` (the empty-validation resume guard).
- [ ] `hunyuanpaintpbr/unet/modules.py:965-1008` (the PBR stack appended to the conv_in input, and its
  guard).
- [ ] `hunyuanpaintpbr/unet/attn_processor.py:689-691,753-756` (the device fix; the comment names a
  `process_pbr_hidden_states` that does not exist, the call site is `__call__` at `:717`).
- [ ] `src/data/objaverse_hunyuan.py:74-84` (validation `DistributedSampler(shuffle=False)` and the
  configured batch size instead of 4): these change upstream's behavior for upstream's own config;
  they stay, and the doc says so.

Checks after each commit: the full gates; the training check, fresh and resume; the tests.

**Instructions:**

**Result:**

### 11. The JSON builders and the release data scripts

- [ ] `make_74k_jsons.py`: `:48` an invalid escape `\ ` in the docstring (a SyntaxWarning under the
  Alliance Python 3.12); `:71` the stale `append_alpha_views.py` path; `/localscratch` examples in the
  docstring (`:43-50`).
- [ ] `make_val_subset.py`: `:24` the stale path, `:29` a `/localscratch` example.
- [ ] `relayout_release.py`: read; nothing known.

Checks: the tests; rebuilding the training check root's JSONs with the edited builder gives identical
files; `--dry-run` unchanged.

**Instructions:**

**Result:**

### 12. Environment files

- [ ] `requirements.txt` is upstream's (demo extras such as gradio, open3d, pymeshlab, cupy, rembg,
  deepspeed, `bpy==4.0`; no torch line); `env_setup.sh:108` installs it minus `bpy`. Add one file for
  what our training and inference import (torch, torchvision, pytorch-lightning 1.9.5, omegaconf,
  diffusers 0.30.0, transformers, huggingface_hub, einops, numpy, Pillow, opencv, tqdm, wandb) with
  the versions that ran: the conda `hunyuanpaint` (Python 3.10.20, torch 2.11.0+cu128, transformers
  4.46.0, numpy 1.24.4, opencv-python 4.10.0.84, huggingface-hub 0.30.2) and the Alliance venv
  (`stage_alliance.sh:35-56`: Python 3.12, torch 2.11.0 wheelhouse, transformers 4.46.2, the opencv
  4.12.0 module). Upstream's file stays for upstream's code (D9).
- [ ] `env_setup.sh` (`:35` defaults to `/local-scratch/localhome/dya78/tmp`) and its final check
  `smoke_load_pretrained.py` (needs a GPU and the hub cache).
- [ ] Gaps: `stage_alliance.sh:91` uses the `hf` CLI, which the conda env lacks (huggingface-hub
  0.30.2 has `huggingface-cli`).
- [ ] Invariants the doc states: the snapshot rev, diffusers 0.30.0 (whose per-load copy of the custom
  pipeline is why `HF_MODULES_CACHE` must be node-local), pytorch-lightning 1.9.5 (the checkpoints'
  format).

Checks: the gate env does not change. A throwaway env built from the new file runs the tests and the
subset gates; its result is recorded (a version difference may move bytes, which is information here,
not a failure).

**Instructions:**

**Result:**

### 13. Release layer: generic entry points for LightgenBench users (per D15)

The code will be published for LightgenBench users to train and run. No user or cluster paths in this
layer. Needs steps 5b and 11; its README commands are written in step 14.

- [ ] `scripts/lightgenbench/prepare_lightgenbench.sh <download_dir> <data_root>`: the generic half of
  `stage_alliance.sh` and the data part of `train_release_body.sh`: check the Hugging Face tars
  (`data/{train,val,test}/{multiview,thumbnail}/*.tar`) against `checksums.sha256`, untar,
  `relayout_release.py` into `render_tex/`, `render_cond/000.png` and `<thumbs>/<uuid>.png`, then
  the example JSONs with `make_74k_jsons.py` and `val64_release_shas.txt`.
- [ ] Filling the release template's tokens (`__ROOT__`, `__STAGE__`, `__NUM_WORKERS__`) without
  hand edits; the config states the recipe (global batch 16 = 4 GPUs × 4, LR 7e-5, 50,000 steps,
  cosine with warm restarts every 10,000, troughs every 10,000, 40k selected on the validation split).
- [ ] The inference entry per D15, writing Keep stable 12's layout; gated byte-identical against the
  frozen parent predictor on both runs.
- [ ] `train.py` requires wandb (`:316-330`): document `WANDB_MODE=offline` (making it optional is
  under Later).
- [ ] The internal launchers stay as they are.

**Instructions:**

**Result:**

### 14. Docs

One doc of record inside the fork, keeping upstream's attribution, license and citation (`LICENSE`,
`Notice.txt`, upstream's README citation block).

- [ ] Root `README.md` (upstream's, about the demo, the API and docker) becomes the doc of record, or a
  separate `HUNYUAN3D_EMISSION.md` with upstream's README kept: decide here. Contents: what the fork
  is (one emission stream generated from six ground-truth PBR views, geometry and the TexVerse
  thumbnail; the conv_in layout; the unet swap; sampling without guidance); the two runs and their
  checkpoints; training and inference for LightgenBench (step 13); the data contract (Keep stable 7,
  with a link to `../data_processing/data_preparation/README.md`); the tests; the traps carried over
  from `../docs/baselines/hunyuan3d-paint.md` (`val_check_interval` equal to the checkpoint cadence
  ranks each checkpoint by the previous validation; `save_top_k` deletes the troughs, hence the trough
  checkpointer; `val/emission_mse` is too noisy to select on; the 10,000-step warm restarts;
  `lr_abs=0.000` after a resume is a display artifact; the `HF_MODULES_CACHE` race; fir `bygpu` OOM at
  `.to(device)`); the upstream delta (`git diff --stat main`).
- [ ] `hy3dpaint/README.md` (upstream's): keep, or fold into the doc of record.
- [ ] A short `CLAUDE.md` at the fork root pointing at the doc of record (the sibling submodules have
  one; this fork has none).

**Instructions:**

**Result:**

### 15. Branch `release` (per D11) and the rename to `Hunyuan3D-Emission/` (per D4) (needs Dongchen's go)

- [ ] The full gates and the training check on the branch head against the `pre-refactor`
  baselines; the tests.
- [ ] Create `release` at this branch's head (D11); `lightgen` stays at `pre-refactor`.
- [ ] Delete the two re-exports of D13 and add `scripts/convert_config.py` (old class paths → new,
  in a stored config); the parent predictor converts `--config` on the fly and imports the loader
  from its new path, as TEXGen-Emission's `run_texgen_infer.py` does; `test_predict_ref_source.py:14`
  moves too. Smoke: the paper and release checkpoints through the converted path, 2 shapes each,
  against their 4090 baselines.
- [ ] In the parent, one commit: the gitlink bump, `.gitmodules` `branch = release`, and the rename
  per D4 (`git mv Hunyuan3DPaint-emissive Hunyuan3D-Emission`, the submodule's name stays, then
  `git -C Hunyuan3D-Emission worktree repair`; every parent reference: search again; on 2026-10-06 the predictor's default root `:70`,
  its error message and the init command it suggests `:123-125`, its docstring and help `:45,53,321`, `../evaluation/newdata_eval/sweeps/{job_hunyuan.sh,
  stage_hunyuan_host.sh, star2/hunyuan_worker.sbatch, hosts.json}`, `../data_processing/data_preparation/hunyuan_cameras.py:38-41`
  and its test, `../paper_figures/baseline_pipelines/{gen_panel_hunyuan.py:1-7, README.md:58-70}`
  (also stale on the reference image and the channel count), the `run_xg.sh` of Keep stable 3,
  and outside the repo the release drivers of Keep stable 1). Stage only this work's hunks: other
  sessions leave uncommitted edits in the parent.
- [ ] Parent docs: `../AGENTS.md:34,56,66,176` (`:66` names `submit_chain_fir.sh`; the release run
  used `submit_release.sh`), a new `../.claude/rules/hunyuan3d.md` (path-scoped like `texgen.md` and
  `trellis2.md`; listed in `AGENTS.md`), `../docs/baselines/hunyuan3d-paint.md:5-7,28-35,50,162-190,279,290`,
  `../docs/status/checkpoints.md:34,52,318-319` (local copies that no longer exist),
  `../docs/evaluation.md:346` (says the paper inference ran on the local 4090 and omits
  `--blacken_bg`; it ran on Solar with it), `../docs/status/refactor.md` (a Hunyuan entry; its
  duplicated `### Data processing` heading).
- [ ] Evaluation-lane follow-ups, recorded in `../docs/status/refactor.md`, not fixed here: the bridge
  subtracts a 128/255 background while the renderer fills 127 (`bridge_views_to_uv.py:418`,
  `multiview.py:68`); `rescore_prf.py:112-133` maps the Hunyuan rows to the pre-fix directories; the
  star2 worker reuses the fork tree of its first bootstrap until `BOOTSTRAP_OK` and `.fork_ok` are
  deleted, and shares `HF_MODULES_CACHE` (`hunyuan_worker.sbatch:153,250`); `stage_hunyuan_host.sh:46,68`
  needs a local `trough-step=00020000.ckpt` that is gone; `test_bridge.py:402,417` pin their cwd to
  the main checkout; the predictor's docstring (`:259-260`) counts 5 VAE `posterior.sample()` calls at
  `unet/model.py:233` (there are six, at `:234`); the two 4090 lock files.
- [ ] Decide here: rename the GitHub repository `dongchen-yang/Hunyuan3D-2.1` too (TEXGen kept its
  name)? Push `release` only on Dongchen's go.

**Instructions:**

**Result:**

### 16. Cut-over (per D8)

- [ ] Solar: rebuild jupiter `trellis2_bw/eval_inputs/solar_hunyuan_stage/Hunyuan3DPaint-emissive.tar`
  from the merged tree (it unpacks to the directory name), update `CFG_REL` in
  `hunyuan_worker.sbatch`, and delete the nodes' `BOOTSTRAP_OK` and `.fork_ok` so they unpack it
  again.
- [ ] xg4090: restage `lightgen_sweep_stage/Hunyuan3DPaint-emissive` (`hosts.json:13`).
- [ ] The Alliance clones (`Hunyuan3D-2.1-emissive` on fir, killarney, rorqual, nibi): check out
  `release` (the release launchers' `.commit` pinning keeps past campaigns on their commits).

**Instructions:**

**Result:**

### 17. Release cut: our internal files leave the published tree (per D10)

Runs after step 15's final check, which needs the tooling.

- [ ] Tag the last commit that has it (`checks-tooling`), so a later check can restore it.
- [ ] Per D10: delete `scripts/internal/` (the gate and our cluster launchers) and the doc's section
  on checking a code change (one line pointing at the tag stays); decide `REFACTOR_TODO.md` and
  `CLAUDE.md` with it.

**Instructions:**

**Result:**

## Later, only if wanted (behavior changes, one step each)

- [ ] **Build the repo's unet directly** instead of building the snapshot's and swapping it
  (`model_emission.py:124-168`; the unet is loaded twice per rank today).
- [ ] **Drop the frozen, unreachable parameters** (`*_mr*` and `learned_text_clip_albedo`, 129
  tensors, 74.5 M elements, `model_emission.py:217-239`): they sit in every checkpoint and the
  optimizer covers them. Needs a key-mapping shim for every published checkpoint (Keep stable 4).
- [ ] **wandb optional** in `train.py` (`:316-330`), with a CSV or TensorBoard fallback.
- [ ] **`--resume_weights_only`** (`train.py:308-309`) skips the conv_in expansion: fix it or remove it.
- [ ] **A node-local `HF_MODULES_CACHE` set by `train.py` itself** (today only
  `train_release_body.sh:101-108` sets it).

## Out of scope

- The evaluation lane: the predictor, the bridge, the scorers and the sweeps change only where this
  work breaks them (step 15); their drift is recorded as follow-ups.
- The data format and the renderer (the fixture layout, the 127 background, the cameras), the bake.
- Sampling changes (the model trains with condition dropout 0.1 and samples without guidance), any
  retraining, any published number.
- The paper (read-only).
