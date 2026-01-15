# ArrowGEV: Grounding Events in Video via Learning the Arrow of Time

We study the temporal directionality problem in Grounding Events in Videos. Specifically, we enable Vision-Language Models to capture the intrinsic temporal structure of events by distinguishing between time-sensitive and time-insensitive semantics. In this work, we utilize a reinforcement learning framework to optimize the model's policy and design a temporal directionality reward to ensure the effective discrimination of event validity across forward and reversed videos.

<div style='display:flex; gap: 0.25rem; '>
  <a href='https://arxiv.org/pdf/2601.06559v1'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
  <a href='ParadiseYu/ArrowGEV-7B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ArrowGEV_7B-blue'></a>
  <a href='https://huggingface.co/datasets/ParadiseYu/ArrowGEV-Data/tree/main/Arrow-R1-Eval/Training'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ArrowGEV_dataset-blue'></a>
</div>

<p align="center" width="100%">
<a target="_blank"><img src="assets/main_arch.png" alt="Paradigm Comparisons on VideoQA" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Setup

### Install
see [docs/INSTALL.md](./docs/INSTALL.md)

### Dataset

Download the video data following [docs/DATA.md](./docs/DATA.md), and use our data annotation [ArrowGEV-Data](https://huggingface.co/datasets/ParadiseYu/ArrowGEV-Data/tree/main/Arrow-R1-Eval/Training)

## Training

**1) Download this GitHub**
```
git clone https://github.com/Yu-Fangxu/ArrowGEV.git
```

**2) Setup Environment**

We recommend creating a new environment:
```bash
conda create -n ArrowGEV python==3.10
conda activate ArrowGEV
```

Then install all the dependencies:
```
pip install -r requirements.txt
```

**3) Run Command for EACL**

```
bash scripts/posttrain/train_rl_SF.sh
```

## Acknowledgements

We thank the following projects: [Time-R1](https://github.com/xiaomi-research/time-r1), [Video-R1](https://github.com/tulerfeng/Video-R1).

## Citation

If you find our work useful, please consider cite our paper：

```bibtex
@article{yu2026arrowgev,
  title={ArrowGEV: Grounding Events in Video via Learning the Arrow of Time},
  author={Yu, Fangxu and Lu, Ziyao and Niu, Liqiang and Meng, Fandong and Zhou, Jie},
  journal={arXiv preprint arXiv:2601.06559},
  year={2026}
}
```
