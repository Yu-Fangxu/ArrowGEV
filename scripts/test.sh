#!/bin/bash
# testing MODEL_NAME on $EVAL_DATASET dataset using vLLM inference
# $EVAL_DATASET filepath: ./dataset/$EVAL_DATASET
export VLLM_WORKER_MULTIPROC_METHOD=spawn
GPU_LIST="0,1,2,3,4,5,6,7"
BASE_PATH="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/ckpts"
MODEL_NAME="ours"
# specify the dataset you want to use, choose from: ["charades", "activitynet", "tvgbench", "mvbench", "videomme", "egoschema", "tempcompass"]
EVAL_DATASET="charades"  
# for tempcompass, default is "multi-choice"
# for egoschema, default is full-set 
SPLIT="test"  

IFS=',' read -ra gpus <<< "$GPU_LIST"
num_gpus=${#gpus[@]}
# 执行推理任务
for ((i=0; i<num_gpus; i++)); do
    gpu=${gpus[i]}
    CUDA_VISIBLE_DEVICES=$gpu python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/evaluate.py \
        --model_base "$BASE_PATH/$MODEL_NAME" \
        --batch_size 4 \
        --curr_idx $i \
        --total_idx $num_gpus \
        --max_new_tokens 1024 \
        --split $SPLIT \
        --datasets $EVAL_DATASET \
        --output_dir "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/logs/eval/$MODEL_NAME/$EVAL_DATASET" \
        --use_r1_thinking_prompt \
        --use_vllm_inference \
        --use_nothink & # uncomment this line to use no-think prompt, especially for VQA tasks
done
wait

# calculate metrics
# default inference code:
python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/src/vllm_inference/eval_all.py --model_name $MODEL_NAME --split $SPLIT --dataset $EVAL_DATASET
