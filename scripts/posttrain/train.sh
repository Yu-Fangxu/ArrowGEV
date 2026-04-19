#!/bin/bash
# RL post-training with the dynamic sample-filtering curriculum from the paper.
#
# Each FILTER_INDEX iteration:
#   (1) trains one epoch of GRPO on the current working set,
#   (2) runs vLLM inference on that working set with the new checkpoint,
#   (3) scores every sample by IoU, drops "mastered" samples (IoU > eta),
#       and emits the next working set.
# FILTER_INDEX=0 starts from the full annotation at INIT_DATA_PATH; later
# iterations resume from the previous iteration's filtered JSON + checkpoint.
#
# Paper's dynamic difficulty filter:
#   keep a sample iff 0 < IoU <= FILTER_THRESHOLD (default 0.70).
set -euo pipefail

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export EXP_NAME="${EXP_NAME:-ArrowGEV}"
export WANDB_PROJECT="${WANDB_PROJECT:-Video-GRPO}"
export PYTHONPATH=".:${PYTHONPATH:-}"
export DEBUG_MODE="${DEBUG_MODE:-true}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Paths / hyper-parameters
INIT_DATA_PATH="${INIT_DATA_PATH:-./dataset/ArrowGEV/annotations/train_2k5.json}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
VIDEO_FOLDER="${VIDEO_FOLDER:-./dataset/ArrowGEV/videos/arrowgev_data}"

FILTER_THRESHOLD="${FILTER_THRESHOLD:-0.70}"   # keep samples with 0 < IoU <= threshold
FILTER_K="${FILTER_K:-2500}"                   # max samples retained per iteration
NUM_FILTER_ITERS="${NUM_FILTER_ITERS:-5}"      # paper trains for 5 epochs
NUM_EPOCHS_PER_ITER="${NUM_EPOCHS_PER_ITER:-1}"

IFS=',' read -ra gpus <<< "$GPU_LIST"
num_gpus=${#gpus[@]}

# Pick the earliest (lowest-step) checkpoint inside an output dir.
find_earliest_checkpoint() {
    local parent="$1"
    find "$parent" -maxdepth 1 -type d -name "checkpoint-*" -print0 \
        | xargs -0 -n1 basename \
        | awk -F'-' '{print $2, $0}' \
        | sort -n | head -n 1 | awk '{print $2}'
}

for (( FILTER_INDEX=0; FILTER_INDEX<NUM_FILTER_ITERS; FILTER_INDEX++ )); do
    export WANDB_NAME="${EXP_NAME}_filter${FILTER_INDEX}"

    if [ "$FILTER_INDEX" -eq 0 ]; then
        DATA_PATH="$INIT_DATA_PATH"
        LOAD_MODEL_PATH="$INIT_MODEL_PATH"
    else
        PREV_WANDB_NAME="${EXP_NAME}_filter$((FILTER_INDEX-1))"
        PREV_LOGDIR="./logs/$EXP_NAME/$PREV_WANDB_NAME"
        DATA_PATH="$PREV_LOGDIR/filtering_epoch$((FILTER_INDEX-1))/train_filtered.json"
        CKPT_PARENT="$PREV_LOGDIR/train_epoch$((FILTER_INDEX-1))"
        MIN_CKPT_NAME=$(find_earliest_checkpoint "$CKPT_PARENT")
        if [ -z "$MIN_CKPT_NAME" ]; then
            echo "No checkpoint found under $CKPT_PARENT — aborting." >&2
            exit 1
        fi
        LOAD_MODEL_PATH="$CKPT_PARENT/$MIN_CKPT_NAME"
    fi

    OUTDIR="./logs/$EXP_NAME/${WANDB_NAME}/train_epoch$FILTER_INDEX"
    export LOG_PATH="$OUTDIR/log.txt"
    mkdir -p "$OUTDIR"

    # --------------------------- Training stage ---------------------------
    CUDA_VISIBLE_DEVICES="$GPU_LIST" torchrun --nproc_per_node="$num_gpus" \
        --nnodes="1" \
        --node_rank="0" \
        --master_addr="127.0.0.1" \
        --master_port="12371" \
        main.py \
        --deepspeed scripts/zero3_offload.json \
        --output_dir "$OUTDIR" \
        --model_name_or_path "$LOAD_MODEL_PATH" \
        --train_data_path "$DATA_PATH" \
        --video_folder "$VIDEO_FOLDER" \
        --dataset_name arrowgev \
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
        --run_name "$WANDB_NAME" \
        --reward_funcs directionality format \
        --alpha_coeff 0.5 \
        --temperature 1.0 \
        --is_curriculum_learning false \
        --logging_dir "$OUTDIR/${WANDB_NAME}" \
        --save_strategy epoch \
        --is_early_stopping true \
        --save_only_model false \
        --report_to none \
        --num_train_epochs "$NUM_EPOCHS_PER_ITER"

    # Skip the filter stage on the last iteration — nothing would consume it.
    if [ "$((FILTER_INDEX + 1))" -ge "$NUM_FILTER_ITERS" ]; then
        continue
    fi

    # --------------------------- Inference stage --------------------------
    CKPT_PARENT="$OUTDIR"
    MIN_CKPT_NAME=$(find_earliest_checkpoint "$CKPT_PARENT")
    if [ -z "$MIN_CKPT_NAME" ]; then
        echo "No checkpoint produced by training under $CKPT_PARENT — aborting." >&2
        exit 1
    fi
    MODEL_BASE="$CKPT_PARENT/$MIN_CKPT_NAME"
    VLLM_OUT="./logs/$EXP_NAME/$WANDB_NAME/filtering_epoch$FILTER_INDEX"
    mkdir -p "$VLLM_OUT"

    for ((i=0; i<num_gpus; i++)); do
        gpu=${gpus[i]}
        CUDA_VISIBLE_DEVICES=$gpu python evaluate.py \
            --model_base "$MODEL_BASE" \
            --datatype tg \
            --batch_size 4 \
            --curr_idx "$i" \
            --total_idx "$num_gpus" \
            --max_new_tokens 1024 \
            --split "$DATA_PATH" \
            --datasets tvgbench_filter \
            --output_dir "$VLLM_OUT" \
            --use_vllm_inference &
    done
    wait

    # ----------------------- Scoring + filter stage -----------------------
    python src/vllm_inference/calc_difficulty.py \
        --input "$VLLM_OUT" \
        --split "$DATA_PATH" \
        --output_dir "./"

    python src/utils/process_data.py \
        --input_json  "$VLLM_OUT/train_v4_cloud.json" \
        --output_json "$VLLM_OUT/train_filtered.json" \
        --threshold "$FILTER_THRESHOLD" \
        -k "$FILTER_K"
done
