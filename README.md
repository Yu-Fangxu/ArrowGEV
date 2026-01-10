# ArrowGEV: Grounding Events in Video via Learning the Arrow of Time

<div style='display:flex; gap: 0.25rem; '>
  <a href='https://arxiv.org/pdf/2503.13377v2'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
  <a href='ParadiseYu/ArrowGEV-7B'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ArrowGEV_7B-blue'></a>
  <a href='https://huggingface.co/datasets/ParadiseYu/ArrowGEV-Data/tree/main/Arrow-R1-Eval/Training'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ArrowGEV_dataset-blue'></a>
</div>

<p align="center" width="100%">
<a target="_blank"><img src="assets/main_arch.png" alt="Paradigm Comparisons on VideoQA" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Overview

### Contributions of Time-R1

*   **Time-R1: RL-based framework for temporal video grounding**. We introduce a reasoning-guided post-training framework via RL with verifiable reward to enhance the capabilities of LVLMs on the TVG task. 
*   **TimeRFT: Time-aware reinforcement fine-tuning.** We explore data-efficient post-training strategies on our curated RL-friendly dataset, which trains the model to progressively comprehend difficult samples, leading to better generalization.
*   **TVGBench: Comprehensive benchmark for LVLMs on TVG.** We carefully construct a small yet comprehensive benchmark for LVLM evaluation, assessing 11 types of queries and featuring balanced distributions across both videos and queries.
*   **State-of-the-Art results and generalization.** Extensive experiments demonstrate that Time-R1 achieves state-of-the-art performance across multiple downstream datasets using only 2.5K training data, while improving its general video understanding capabilities. 

### Key Features of this codebase:

*   **A codebase that supports training LVLM with RL.** 
*   **Speedup inference time for temporal video grounding and video QA by vllm library.**<img src="./assets/vLLM-Full-Logo.png" alt="vLLM Logo" width="40px">
*   **Experiment toolkits**: Support training on our TimeRFT, Charades, and ActivityNet; support vLLM inference on TVGBench, Charades, ActivityNet, MVBench,s TempCompass, VideoMME, EgoSchema.

## Setup

### Install
see [docs/INSTALL.md](./docs/INSTALL.md)

### Dataset

Download the video data following [Time-R1](https://github.com/xiaomi-research/time-r1), and use our data annotation [ArrowGEV-Data](https://huggingface.co/datasets/ParadiseYu/ArrowGEV-Data/tree/main/Arrow-R1-Eval/Training)

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

If you find our work useful, please consider cite our paper :)


```bibtex
@article{wang2025timer1,
      title={Time-R1: Post-Training Large Vision Language Model for Temporal Video Grounding}, 
      author={Wang, Ye and Wang, Ziheng and Xu, Boshen and Du, Yang and Lin, Kejun and Xiao, Zihan and Yue, Zihao and Ju, Jianzhong and Zhang, Liang and Yang, Dingyi and Fang, Xiangnan and He, Zewen and Luo, Zhenbo and Wang, Wenxuan and Lin, Junqi and Luan, Jian and Jin, Qin},
      journal={arXiv preprint arXiv:2503.13377},
      year={2025},
}
```

```bibtex
@article{wang2025timezero,
      title={TimeZero: Temporal Video Grounding with Reasoning-Guided LVLM},
      author={Wang, Ye and Xu, Boshen and Yue, Zihao and Xiao, Zihan and Wang, Ziheng and Zhang, Liang and Yang, Dingyi and Wang, Wenxuan and Jin, Qin},
      journal={arXiv preprint arXiv:2503.13377},
      year={2025}
}
```
