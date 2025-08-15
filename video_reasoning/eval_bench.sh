#!/bin/bash
# run_models.sh

model_paths=(
    "./Qwen2.5-VL-7B-Instruct"
)

file_names=(
    "Qwen2.5-VL-7B"
)

export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/Video-R1/src/eval_bench.py --model_path "$model" --file_name "$file_name"
done
