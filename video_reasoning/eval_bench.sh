#!/bin/bash
# run_models.sh

model_paths=(
    # "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/Video-R1/Video-R1-7B"
    # "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/Video-R1/log/Qwen2.5-VL-7B-GRPO/checkpoint-500"
    # /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/ckpts/beta0_5epoch
    # /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/ckpts/beta0_5epoch_using_exp
    # "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/ckpts/Qwen2.5-VL-7B-Instruct"
    # "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/ckpts/tiou_5epoch"
    "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/ckpts/beta0_5epoch"
)

file_names=(
    # "Video-R1"
    # "GRPO-1K"
    # "GRPO-10K"
    # "GRPO-0.5K"
    # "Reverse_time"
    # "Reverse_time_using_exp"
    # "Qwen2.5-VL-7B"
    # "tiou_5epoch"
    "beta0_5epoch"
)

export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/Video-R1/src/eval_bench.py --model_path "$model" --file_name "$file_name"
done
