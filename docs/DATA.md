# Dataset Preparation

All code paths default to `./dataset`. Override with `ARROWGEV_DATASET_ROOT`
if you mount the data elsewhere.

## Training data

- **Annotations.** We ship the filtered training split at
  `dataset/ArrowGEV/annotations/train_2k5.json`. Each item has
  `{video, video_reverse_path, timestamp, sentence, duration, sensitive, qid}`.
- **Videos.** Download the organized version from
  [ParadiseYu/ArrowGEV-Data](https://huggingface.co/datasets/ParadiseYu/ArrowGEV-Data),
  or reassemble them from the original sources:
  [VTG-IT](https://huggingface.co/datasets/Yongxin-Guo/VTG-IT),
  [TimeIT](https://huggingface.co/datasets/ShuhuaiRen/TimeIT),
  [HTStep](https://openreview.net/pdf?id=vv3cocNsEK),
  [LongVid](https://huggingface.co/datasets/OpenGVLab/LongVid).
- **Reversed videos.** After downloading, run

  ```bash
  python reverse_video.py \
      --input_folder  dataset/ArrowGEV/videos/arrowgev_data \
      --output_folder dataset/ArrowGEV/videos/arrowgev_data
  ```

  to emit a `<stem>_reverse.mp4` next to every source video.

## Expected layout

```
dataset
└── ArrowGEV
    ├── annotations
    │   └── train_2k5.json
    └── videos
        └── arrowgev_data/*.mp4
```
