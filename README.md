# Hunyuan3D-Emission

Hunyuan3D-Emission generates the emission of a 3D shape in six views from its PBR materials, its geometry and a reference image. It is a LightGenBench baseline built on the multiview texture model of [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1).



## Environment Setup
From the root of the repository, create the environment and download the pretrained weights.
```bash
# Create and activate the environment
conda create -n hunyuanpaint python=3.10 -y
conda activate hunyuanpaint
pip install -r requirements-emission.txt --extra-index-url https://download.pytorch.org/whl/cu128

# The Hunyuan3D 2.1 paint weights the model starts from, at a pinned revision
huggingface-cli download tencent/Hunyuan3D-2.1 --revision 0b94677654c57bb9a6b6845cd7b704ccf551d327 \
    --include "hunyuan3d-paintpbr-v2-1/*"

# The image encoder for the reference image
huggingface-cli download facebook/dinov2-giant
```
Run every command below from `hy3dpaint/`.



## Data Setup
The dataset download is described at https://github.com/3dlg-hcvc/lightgenbench.
This baseline needs the multiview and thumbnail files of the released dataset.
```bash
# Download the files this baseline reads
huggingface-cli download 3dlg-hcvc/LightgenBench --repo-type dataset --local-dir <download_dir> \
    --include "splits.json" "checksums.sha256" "data/*/multiview/*.tar" "data/*/thumbnail/*.tar"

# Unpack them into <data_dir> and write the example lists and the training config
bash scripts/lightgenbench/unpack_hf_dataset.sh <download_dir> <data_dir>
```



## Training
Train on 4 GPUs with the config the data setup wrote.
```bash
# Writes logs/hunyuan-paint-emission-release/ (rerun to resume); set HF_MODULES_CACHE to a local disk
HF_MODULES_CACHE=<local_dir> WANDB_MODE=offline python train.py --base <data_dir>/hunyuan-paint-emission.yaml \
    --name release --logdir logs/ --gpus 0,1,2,3
```
A checkpoint is kept at each learning-rate restart, in `logs/hunyuan-paint-emission-release/checkpoints/troughs/`; choose one on the validation split.



## Prediction
Predict the six emission views of each test shape.
```bash
# <checkpoint>: a checkpoint from logs/hunyuan-paint-emission-release/checkpoints/troughs/
python predict_emission_views.py \
    --config logs/hunyuan-paint-emission-release/configs/project.yaml \
    --ckpt <checkpoint> \
    --examples_json <data_dir>/release_test.json --thumbnail_dir <data_dir>/thumbs \
    --out <out_dir>
```
Score the predictions with the LightGenBench evaluation, which first turns the six views into atlases: https://github.com/3dlg-hcvc/lightgenbench

### With our checkpoint
Our checkpoint trained with the recipe in `<data_dir>/hunyuan-paint-emission.yaml`, the config the data setup wrote. Put the checkpoint and that config where training writes its own, and the prediction command above runs with only `--ckpt` filled in.
```bash
# Our checkpoint (coming soon): the Checkpoint column of https://github.com/3dlg-hcvc/lightgenbench#baselines
mkdir -p logs/hunyuan-paint-emission-release/checkpoints/troughs logs/hunyuan-paint-emission-release/configs
# Download <our checkpoint>.ckpt into logs/hunyuan-paint-emission-release/checkpoints/troughs/

# The config, where training saves it
cp <data_dir>/hunyuan-paint-emission.yaml logs/hunyuan-paint-emission-release/configs/project.yaml

python predict_emission_views.py \
    --config logs/hunyuan-paint-emission-release/configs/project.yaml \
    --ckpt logs/hunyuan-paint-emission-release/checkpoints/troughs/<our checkpoint>.ckpt \
    --examples_json <data_dir>/release_test.json --thumbnail_dir <data_dir>/thumbs \
    --out <out_dir>
```



## License
The code and weights are under the Tencent Hunyuan 3D 2.1 Community License (`LICENSE`, `Notice.txt`), which grants no rights in the European Union, the United Kingdom and South Korea.



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
