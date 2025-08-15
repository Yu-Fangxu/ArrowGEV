export VLLM_WORKER_MULTIPROC_METHOD=spawn
GPU_LIST="0,1,2,3,4,5,6,7"
BASE_PATH="/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/ckpts"

MODEL_NAMES=(
    "Qwen2.5-VL-72B-Instruct"
)

python your_script_name.py \
  --datatype sensitivity \
  --model_base "$BASE_PATH/$model_name" \
  --input_json "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/VLLMs/MCQ_1_TS_InternVL3-8B.jsonl" \
  --output_json "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/VLLMs/statistics.json" \
  --batch_size 16 \
  --use_vllm_inference