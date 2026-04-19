# Installation

Create a fresh environment and install the pinned dependencies:

```bash
conda create -n ArrowGEV python=3.10.12 -y
conda activate ArrowGEV
pip install -r requirements.txt
```

The following versions must match for bug-free training and vLLM inference:

| Package | Version |
| --- | --- |
| CUDA | 12.4 |
| torch | 2.6.0 |
| transformers | 4.51.1 |
| vllm | 0.8.4 |
| trl | 0.17.0 |
| numba | 0.61.2 |

See `requirements.txt` for the complete list.
