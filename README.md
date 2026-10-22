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

# Check and unpack them into <data_dir>, write the example lists and the training config
# <data_dir>/hunyuan-paint-emission.yaml (SNAPSHOT=<dir> points it at the weights outside the Hugging Face cache)
bash scripts/lightgenbench/unpack_hf_dataset.sh <download_dir> <data_dir>
```



## Training
Train on 4 GPUs with the config the data setup wrote.
```bash
# Writes logs/hunyuan-paint-emission-release/; the same command resumes from its last.ckpt
# HF_MODULES_CACHE on a local disk: every GPU process copies ./hunyuanpaintpbr into it when it loads
# (log in to Weights & Biases instead of WANDB_MODE=offline to log online)
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
This writes the predictions to `<out_dir>`.
The six views per shape are scored after the LightGenBench evaluation turns them into atlases.
Score the predictions with the LightGenBench evaluation: https://github.com/3dlg-hcvc/lightgenbench



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
Hunyuan3D-Paint 2.1 builds on RomanTex and MaterialMVP, so please cite them as well.
```bibtex
@article{feng2025romantex,
  title={RomanTex: Decoupling 3D-aware Rotary Positional Embedded Multi-Attention Network for Texture Synthesis},
  author={Feng, Yifei and Yang, Mingxin and Yang, Shuhui and Zhang, Sheng and Yu, Jiaao and Zhao, Zibo and Liu, Yuhong and Jiang, Jie and Guo, Chunchao},
  journal={arXiv preprint arXiv:2503.19011},
  year={2025}
}

@article{he2025materialmvp,
  title={MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion},
  author={He, Zebin and Yang, Mingxin and Yang, Shuhui and Tang, Yixuan and Wang, Tao and Zhang, Kaihao and Chen, Guanying and Liu, Yuhong and Jiang, Jie and Guo, Chunchao and Luo, Wenhan},
  journal={arXiv preprint arXiv:2503.10289},
  year={2025}
}
```
