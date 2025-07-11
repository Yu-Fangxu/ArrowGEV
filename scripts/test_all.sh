#!/bin/bash
# testing MODEL_NAME on $EVAL_DATASET dataset using vLLM inference
# $EVAL_DATASET filepath: ./dataset/$EVAL_DATASET
export VLLM_WORKER_MULTIPROC_METHOD=spawn
GPU_LIST="0,1,2,3,4,5,6,7"
BASE_PATH="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/ckpts"

MODEL_NAMES=(
    # "ours"
    # "ours_7B"
    # "ours_7B_2epoch"
    "Time-R1-7B"
    # "Qwen2.5-VL-3B-Instruct"
    # "Qwen2.5-VL-7B-Instruct"  # 添加第三个模型
)
# specify the dataset you want to use, choose from: ["charades", "activitynet", "tvgbench", "mvbench", "videomme", "egoschema", "tempcompass"]
EVAL_DATASETS=(
    "charades"
    "activitynet"
    "tvgbench"
    # "mvbench"
    # "videomme"
    # "tempcompass"
    # "egoschema" # 如果需要也可以加入
)
# for tempcompass, default is "multi-choice"
# for egoschema, default is full-set 
SPLIT="test"  

IFS=',' read -ra gpus <<< "$GPU_LIST"
num_gpus=${#gpus[@]}
# 执行推理任务
for model_name in "${MODEL_NAMES[@]}"; do
    # 内层循环：遍历该模型需要测试的每一个数据集
    for dataset_name in "${EVAL_DATASETS[@]}"; do
        for ((i=0; i<num_gpus; i++)); do
            gpu=${gpus[i]}
            CUDA_VISIBLE_DEVICES=$gpu python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/evaluate.py \
                --model_base "$BASE_PATH/$model_name" \
                --batch_size 4 \
                --curr_idx $i \
                --total_idx $num_gpus \
                --max_new_tokens 1024 \
                --split $SPLIT \
                --datasets $dataset_name \
                --output_dir "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/logs/eval/$model_name/${dataset_name}" \
                --use_r1_thinking_prompt \
                --use_vllm_inference \
                --use_nothink & # uncomment this line to use no-think prompt, especially for VQA tasks
            done
            
            wait
            
            echo "Finished evaluation for dataset: $dataset_name. Starting aggregation."
            # 在所有任务完成后，再执行结果汇总脚本
            # 注意：这个汇总脚本可能只需要运行一次，而不是在GPU循环内运行
            python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/src/vllm_inference/eval_all.py --model_name $model_name --split $SPLIT --dataset $dataset_name
        
    done
done


# calculate metrics
# default inference code:
# python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/src/vllm_inference/eval_all.py --model_name $MODEL_NAME --split $SPLIT --dataset $EVAL_DATASET
