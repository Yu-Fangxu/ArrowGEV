#!/bin/bash
# training model with sample filtering per epoch

GPU_LIST="0,1,2,3,4,5,6,7"
export CUDA_VISIBLE_DEVICES=$GPU_LIST
export EXP_NAME=ArrowGEV
export WANDB_PROJECT=Video-GRPO
export PYTHONPATH=".:$PYTHONPATH"
export DEBUG_MODE="true"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# 初始路径设置
INIT_DATA_PATH="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/dataset/ArrowGEV/annotations/train_2k5.json"
INIT_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

for FILTER_INDEX in {0..0}; do
    # 设置动态变量
    export WANDB_NAME="${EXP_NAME}_0071_filter${FILTER_INDEX}"
    
    if [ $FILTER_INDEX -eq 0 ]; then
        DATA_PATH=$INIT_DATA_PATH
        LOAD_MODEL_PATH=$INIT_MODEL_PATH
    else
        PREV_WANDB_NAME="${EXP_NAME}_0071_filter$((FILTER_INDEX-1))"
        DATA_PATH=/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/logs/$EXP_NAME/$PREV_WANDB_NAME/filtering_epoch$((FILTER_INDEX-1))/train_v4_cloud_0071_all.json
        LOAD_MODEL_PATH="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/logs/$EXP_NAME/${PREV_WANDB_NAME}/train_epoch$((FILTER_INDEX-1))/$min_checkpoint_name"
    fi

    # DATA_PATH=$INIT_DATA_PATH
    # LOAD_MODEL_PATH=$INIT_MODEL_PATH

    # 设置路径变量
    OUTDIR=/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/logs/$EXP_NAME/${WANDB_NAME}/train_epoch$FILTER_INDEX
    export LOG_PATH="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/${OUTDIR}/log.txt"
    mkdir -p $OUTDIR

    # 训练阶段
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun --nproc_per_node="8" \
        --nnodes="1" \
        --node_rank="0" \
        --master_addr="127.0.0.1" \
        --master_port="12371" \
        /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/main.py \
        --deepspeed /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/scripts/zero3_offload.json \
        --output_dir $OUTDIR \
        --model_name_or_path $LOAD_MODEL_PATH \
        --train_data_path $DATA_PATH \
        --video_folder /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/dataset/ArrowGEV/videos/timerft_data \
        --dataset_name timerft \
        --max_prompt_length 8192 \
        --max_completion_length 20 \
        --num_generations 8 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --logging_steps 1 \
        --bf16 \
        --torch_dtype bfloat16 \
        --data_seed 42 \
        --gradient_checkpointing true \
        --attn_implementation flash_attention_2 \
        --fix_vit true \
        --slide_window false \
        --run_name $WANDB_NAME \
        --report_to "tensorboard" \
        --reward_funcs iou_ct format \
        --temperature 1.0 \
        --prompt_type v1 \
        --is_curriculum_learning false \
        --logging_dir "${OUTDIR}/${WANDB_NAME}" \
        --save_strategy epoch \
        --is_early_stopping true \
        --save_only_model false \
        --report_to none \
        --num_train_epochs 3 \
        # $([ $FILTER_INDEX -eq 0 ] && echo "5" || echo "1")
    # 推理阶段
    # vllm_output_dir=/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/logs/$EXP_NAME/$WANDB_NAME/filtering_epoch$FILTER_INDEX
    # mkdir -p "$vllm_output_dir"

    # # # 查找最小checkpoint
    # checkpoints_parent_path="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/logs/${EXP_NAME}/${WANDB_NAME}/train_epoch$FILTER_INDEX"
    # min_checkpoint_name=$(find "$checkpoints_parent_path" -maxdepth 1 -type d -name "checkpoint-*" -print0 | \
    #                     xargs -0 -n1 basename | \
    #                     awk -F'-' '{print $2, $0}' | \
    #                     sort -n | \
    #                     head -n 1 | \
    #                     awk '{print $2}')
    # model_base="${checkpoints_parent_path}/${min_checkpoint_name}"
    # echo $model_base
    # # 并行推理
    # IFS=',' read -ra gpus <<< "$GPU_LIST"
    # num_gpus=${#gpus[@]}
    
    # # 执行推理任务
    # for ((i=0; i<num_gpus; i++)); do
    #     gpu=${gpus[i]}
    #     CUDA_VISIBLE_DEVICES=$gpu python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/evaluate.py \
    #         --model_base "$model_base" \
    #         --datatype tg \
    #         --batch_size 4 \
    #         --curr_idx $i \
    #         --total_idx $num_gpus \
    #         --max_new_tokens 1024 \
    #         --split $DATA_PATH \
    #         --datasets tvgbench_filter \
    #         --output_dir "$vllm_output_dir" \
    #         --use_r1_thinking_prompt \
    #         --use_vllm_inference &
    # done
    # wait
    # 数据处理
    # python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/src/vllm_inference/calc_difficulty.py --input $vllm_output_dir \
    #     --split $DATA_PATH \
    #     --output_dir "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/"
    # # 生成过滤后的数据
    # python /mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/ArrowGEV/src/utils/process_data.py --input_json $vllm_output_dir/train_v4_cloud.json --task 0071_all

done
