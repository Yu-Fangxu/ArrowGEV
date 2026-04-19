import base64
import concurrent.futures
import itertools
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from string import Template
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from openai import OpenAI
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from src.arrowgev import ArrowGEV_Trainer

torch.serialization.add_safe_globals([ZeroStageEnum])
torch.serialization.add_safe_globals([LossScaler])


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

RM_PROMPT = """
**You are an AI assistant specializing in the analysis of temporal properties of events.**
You will be given a sentence describing an event.

**Your task is to:**
1.  Analyze the event described in the sentence.
2.  Determine if the event is temporally **sensitive** or **insensitive**.
3.  Output the results in a strict JSON format without any additional text or explanations.

---
### **Input**
**Event Sentence:** ${sentence}

---
### **Evaluation Criteria**
* **Time-Sensitive (sensitive: yes):** The event has a clear forward direction. If played in reverse, it describes a different, often nonsensical or opposite, event. This indicates temporal asymmetry.
    * *Example:* "A person puts a picture on the wall." (Reversed: "A person takes a picture off the wall.")
    * *Example:* "A glass shatters." (Reversed: "Shards of glass assemble into a whole glass.")

* **Time-Insensitive (sensitive: no):** The event is a continuous state or a cyclical action. If played in reverse, the fundamental nature of the event does not change. This indicates temporal symmetry.
    * *Example:* "A person is playing with a light switch." (Reversed: Still looks like a person playing with a light switch.)
    * *Example:* "A ball is bouncing in place." (Reversed: Still a ball bouncing in place.)
---
### **Output Format**
Now, please output your result below in a JSON format by filling in the placeholders in [] without any explanations:

{
  "reason": "[Briefly explain why the event is reversible or irreversible, describing the forward and reverse action.]",
  "sensitive": "[yes/no]"
}
"""

RM_SYSTEM = "You are a helpful assistant."

# Reward-model endpoints (OpenAI-compatible). Configure via environment variables:
#   ARROWGEV_RM_BASE_URLS  comma-separated list of base URLs, e.g. "http://host:8000/v1,http://host:8001/v1"
#   ARROWGEV_RM_MODEL      model id / local path served by those endpoints
#   ARROWGEV_RM_API_KEY    API key (defaults to "EMPTY" for self-hosted vLLM/SGLang servers)
_RM_BASE_URLS = [
    u.strip()
    for u in os.environ.get("ARROWGEV_RM_BASE_URLS", "").split(",")
    if u.strip()
]
_RM_API_KEY = os.environ.get("ARROWGEV_RM_API_KEY", "EMPTY")
_RM_MODEL = os.environ.get("ARROWGEV_RM_MODEL", "")
clients = [OpenAI(api_key=_RM_API_KEY, base_url=url) for url in _RM_BASE_URLS]

def calculate_iou(interval1_start, interval1_end, interval2_start, interval2_end, duration):
    """
    Calculates the Intersection over Union (IoU) of two 1D intervals.

    Args:
        interval1_start (float or int): Start of the first interval.
        interval1_end (float or int): End of the first interval.
        interval2_start (float or int): Start of the second interval.
        interval2_end (float or int): End of the second interval.

    Returns:
        float: The IoU value (between 0.0 and 1.0), or 0.0 if the union is zero.
    """
    # Ensure intervals are correctly ordered (start <= end)
    # This might not be strictly necessary if inputs are always ordered,
    # but it adds robustness.
    i1_s, i1_e = min(interval1_start, interval1_end), max(interval1_start, interval1_end)
    i2_s, i2_e = min(interval2_start, interval2_end), max(interval2_start, interval2_end)

    # Calculate the intersection
    # The intersection starts at the maximum of the two start points
    # The intersection ends at the minimum of the two end points
    intersection_start = max(i1_s, i2_s)
    intersection_end = min(i1_e, i2_e)

    # Intersection length: if intersection_end < intersection_start, there's no overlap, so intersection is 0
    intersection = max(0, intersection_end - intersection_start)

    # Calculate the union
    # The union starts at the minimum of the two start points
    # The union ends at the maximum of the two end points
    union_start = min(i1_s, i2_s)
    union_end = max(i1_e, i2_e)

    # Union length
    union = union_end - union_start

    # Calculate IoU
    if union > 0:
        iou = intersection / union
    else:
        # If union is 0 (e.g., both intervals are single points and identical, or invalid input),
        # IoU is typically considered 0 or undefined. We return 0.0 here.
        iou = 0.0

    gt_start_norm = 1.0 * interval2_start / duration
    gt_end_norm = 1.0 * interval2_end / duration
    pred_start_norm = 1.0 * interval1_start / duration
    pred_end_norm = 1.0 * interval1_end / duration
    iou = (
        iou
        * (1 - abs(gt_start_norm - pred_start_norm))
        * (1 - abs(gt_end_norm - pred_end_norm))
    )

    return iou

def llm_reward(sentence, completions, solutions, durations, alpha_coeff, sensitivity=None):
    def get_rm_result(client, sentence, f_period, f_ground, r_period, r_ground):
        prompt = Template(RM_PROMPT).substitute(sentence=sentence)
        user_content = [
            {
                "type": "text",
                "text": prompt
            }
        ]

        msgs = [
            {
                "role": "system",
                "content": RM_SYSTEM,
                # "name": "string"
            },
            {
                "role": "user",
                "content": user_content,
                # "name": "string"
            },
        ]

        response = client.chat.completions.create(
            messages=msgs,
            model=_RM_MODEL,
            max_tokens=4096,
            temperature=0.0,
            top_p=1.0,
            n=1,
            seed=42,
            extra_body={"repetition_penalty": 1.05},
        )
        response_content = response.choices[0].message.content
        cleaned_content = response_content.strip()

        json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
        if json_match:
            cleaned_content = json_match.group(1).strip()

        return cleaned_content

    contents = completions
    num_rollout = len(contents) // 2
    reverse_contents = contents[num_rollout:]
    contents = contents[:num_rollout]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    tempo_pattern = r'<answer>(.*?)</answer>'
    cot_pattern = r'<think>(.*?)</think>'
    client_cycle = itertools.cycle(clients)

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        futures = []
        content_list = []
        rev_content_list = []
        sol_list = []
        reasoning_list = []
        for content, rev_content, sol, duration in zip(contents, reverse_contents, solutions, durations):
            client = next(client_cycle) if clients else None
            parsed_times = parse_timestamp_output(content)
            parsed_times_rev = parse_timestamp_output(rev_content)

            start_time, end_time = 0, 0
            start_time_rev, end_time_rev = 0, 0
            gt_start, gt_end = sol  # forward temporal interval
            reverse_start = duration - gt_end  # reverse temporal interval
            reverse_end = duration - gt_start
            if parsed_times:
                start_time, end_time = parsed_times
            if parsed_times_rev:
                start_time_rev, end_time_rev = parsed_times_rev

            sol_list.append(sol)

            reasoning = "Invalid Reason."
            cot_match = re.search(cot_pattern, content, re.DOTALL)
            if cot_match:
                reasoning = cot_match.group(1).strip()
            reasoning_list.append(reasoning)

            f_period = str([start_time, end_time])
            f_ground = str([gt_start, gt_end])
            r_period = str([start_time_rev, end_time_rev])
            r_ground = str([reverse_start, reverse_end])
            if sensitivity is None and client is not None:
                future = executor.submit(
                    get_rm_result,
                    client,
                    sentence=sentence,
                    f_period=f_period,
                    f_ground=f_ground,
                    r_period=r_period,
                    r_ground=r_ground,
                )
                futures.append(future)

            content_list.append([start_time, end_time])
            rev_content_list.append([start_time_rev, end_time_rev])

        rm_results = []
        if sensitivity is None and futures:
            for future in concurrent.futures.as_completed(futures):
                rm_results.append(future.result())
        else:
            default = f'{{"sensitive": "{sensitivity or "yes"}"}}'
            rm_results = [default] * len(content_list)

        assert len(rm_results) == len(content_list)
        assert len(rm_results) == len(sol_list)

        ious = []

        for rm_result, content, rev_content, sol, reasoning, duration in zip(
            rm_results, content_list, rev_content_list, sol_list, reasoning_list, durations
        ):
            try:
                rm_res = json.loads(rm_result)
            except json.JSONDecodeError:
                rm_res = {"sensitive": "yes"}
            sensitivity = rm_res["sensitive"]

            gt_start, gt_end = sol
            reverse_start = duration - gt_end # reverse temporal interval
            reverse_end = duration - gt_start

            r_s = duration - content[1]
            r_e = duration - content[0]

            iou_pos = calculate_iou(content[0], content[1], gt_start, gt_end, duration)
            iou_neg = calculate_iou(rev_content[0], rev_content[1], r_s, r_e, duration)
            reward = 0.0
            if sensitivity.lower() == "yes":
                reward += iou_pos + alpha_coeff * (1 - iou_neg)
                second_term = 1 - iou_neg
            else:
                reward += iou_pos + alpha_coeff * iou_neg
                second_term = iou_neg
            
            rewards.append(reward)
            ious.append(iou_pos)

            log_path = os.environ.get("LOG_PATH")
            if log_path and os.environ.get("DEBUG_MODE") == "true":
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"------- {current_time} Total reward: {reward}, sensitivity: {sensitivity}, "
                        f"iou_pos: {iou_pos}, iou_neg: {iou_neg} -------\n"
                    )
                    f.write(f"Candidate Interval: {content}\n")
                    f.write(f"Reverse Candidate Interval: {rev_content}\n")
                    f.write(f"Reference: {sol}\n")
                    f.write(f"Reverse Reference: {(reverse_start, reverse_end)}\n")
                    f.write(f"Forward IoU: {iou_pos}\n")
                    f.write(f"Reverse Term: {second_term}\n")
    return rewards, sensitivity, ious


@dataclass
class MY_GRPOConfig(GRPOConfig):
    fix_vit: bool = field(
        default=False,
        metadata={"help": "Whether to fix the ViT model"},
    )

    slide_window: bool = field(
        default=False,
        metadata={"help": "Whether to use slide window"},
    )
    max_window_layers: int = field(
        default=2, metadata={"help": "sliding window layers bottom"}
    )
    sliding_window_length: int = field(
        default=4096, metadata={"help": "sliding window length"}
    )

    use_grpo: bool = field(
        default=True,
        metadata={"help": "Whether to use GRPO"},
    )
    alpha_coeff: float = field(
        default=0.5,
        metadata={"help": "Weight of the reverse-video term in llm_reward."},
    )
    local_search: bool = field(
        default=False,
        metadata={"help": "Whether to use local search."},
    )
    adv_adjust: bool = field(
        default=False,
        metadata={"help": "Adjust the advantage weight."},
    )
    adv_adjust_miou: str = field(
        default="exp",
        metadata={"help": "How to adjust the advantage weight."},
    )
    tau: float = field(
        default=2.0,
        metadata={"help": "Temperature used by the advantage adjustment."},
    )

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["iou", "format"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="./dataset/finetune/charades/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )

    eval_data_path: str = field(
        default="./dataset/finetune/charades/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="./dataset/finetune/charades/Charades/Charades_v1",
        metadata={"help": "Path to the folder containing video files."},
    )

    is_curriculum_learning: bool = field(
        default=False,
        metadata={"help": "Whether to use curriculum learning."},
    )

    is_early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether to use early stopping"},
    )

def parse_timestamp_output(output_string):
    """Parse the last ``<answer>...</answer>`` block into (start, end) floats."""
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)
    if not answer_matches:
        return None

    last_answer_content = answer_matches[-1]
    matches = re.findall(
        r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", last_answer_content, re.IGNORECASE
    )
    if not matches:
        return None
    last_match = matches[-1]
    return float(last_match[0]), float(last_match[2])


def iou_timestamp_reward(completions, solution, **kwargs):
    """Reward function that returns the IoU between predicted and GT timestamps."""
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(completions, solution):

        reward = 0.0
        parsed_times = parse_timestamp_output(content)
        start_time, end_time = 0, 0
        gt_start, gt_end = sol
        s, e = gt_start, gt_end
        if parsed_times:
            start_time, end_time = parsed_times
            from_number = start_time
            to_number = end_time

            intersection = max(0, min(to_number, e) - max(from_number, s))
            union = max(to_number, e) - min(from_number, s)
            if union > 0:
                iou = intersection / union

            reward = iou
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"Content: {content}\n")
                    f.write(f"pred second: {start_time}, {end_time}\n")
                    f.write(f"gt second: {gt_start}, {gt_end}\n")
                    f.write(
                        f"------------- {current_time} IoU reward: {reward} -------------\n"
                    )

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    return [1.0 if match else 0.0 for match in matches]


def extract_think_content(completion: str) -> Optional[str]:
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = think_pattern.findall(completion)
    if matches:
        return matches[-1].strip()
    return None


def reward_timestep_pair(
    completions: List[str],
    weight: float = 0.2,
    max_count: int = 1,
    **kwargs,
) -> List[float]:
    rewards = []
    pair_pattern = re.compile(
        r"<timestep>\s*(\d+\.?\d*)\s+to\s+(\d+\.?\d*)\s*</timestep>",
        re.IGNORECASE | re.DOTALL,
    )

    for completion in completions:
        score = 0.0
        think_content = extract_think_content(completion)

        if think_content:
            pair_matches = pair_pattern.findall(think_content)
            pair_count = len(pair_matches)
            capped_count = min(pair_count, max_count)
            score = weight * capped_count
        else:
            score = 0.0

        rewards.append(max(0.0, score))

    return rewards


def reward_think_length(
    completions: List[str],
    weight: float = 0.001,
    max_length: int = 500,
    **kwargs,
) -> List[float]:
    rewards = []
    for completion in completions:
        score = 0.0
        think_content = extract_think_content(completion)

        if think_content:
            think_length = len(think_content)
            capped_length = min(think_length, max_length)
            score = weight * capped_length
        else:
            score = 0.0

        rewards.append(max(0.0, score))

    return rewards


DEFAULT_STRUCTURE_KEYWORDS = [
    "analyze",
    "compare",
    "deduce",
    "however",
    "therefore",
    "because",
    "step",
    "observe",
    "notice",
    "identify",
    "wait",
]


def reward_keyword_usage(
    completions: List[str],
    keywords: Optional[List[str]] = None,
    weight: float = 0.1,
    max_count: int = 2,
    **kwargs,
) -> List[float]:
    if keywords is None:
        keywords = DEFAULT_STRUCTURE_KEYWORDS
    rewards = []

    for completion in completions:
        score = 0.0
        think_content = extract_think_content(completion)

        if think_content:
            content_lower = think_content.lower()
            keyword_count = sum(1 for word in keywords if word in content_lower)
            capped_count = min(keyword_count, max_count)
            score = weight * capped_count
        else:
            score = 0.0

        rewards.append(max(0.0, score))

    return rewards


def reward_paragraph_structure(
    completions: List[str],
    weight: float = 0.05,
    max_paragraphs: int = 2,
    **kwargs,
) -> List[float]:
    rewards = []
    for completion in completions:
        score = 0.0
        think_content = extract_think_content(completion)

        if think_content:
            paragraphs = [p for p in think_content.split("\n") if p.strip()]
            capped_paragraphs = min(len(paragraphs), max_paragraphs)
            score = weight * capped_paragraphs
        else:
            score = 0.0

        rewards.append(max(0.0, score))

    return rewards


def diversity_reward_func(completions, num_generations=8, **kwargs):
    if not completions:
        return []

    batch_size = len(completions) // num_generations
    diversity_rewards = []
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=True
    )

    for i in range(batch_size):
        group_start_idx = i * num_generations
        group_end_idx = (i + 1) * num_generations
        current_group_completions = completions[group_start_idx:group_end_idx]

        group_rewards = np.zeros(num_generations)
        for j in range(num_generations):
            total_dissimilarity = 0
            count = 0
            for k in range(num_generations):
                if j == k:
                    continue
                try:
                    # rouge_score expects strings, handle potential non-string content if necessary
                    score = scorer.score(
                        str(current_group_completions[j]),
                        str(current_group_completions[k]),
                    )["rougeL"].fmeasure
                    total_dissimilarity += 1.0 - score
                    count += 1
                except Exception as e:
                    print(f"Warning: Error calculating ROUGE score: {e}. Skipping pair.")

            if count > 0:
                group_rewards[j] = total_dissimilarity / count
            else:
                group_rewards[j] = 0.0

        diversity_rewards.extend(group_rewards.tolist())

    return diversity_rewards


reward_funcs_registry = {
    "iou": iou_timestamp_reward,
    "format": format_reward,
    "llm_reward": llm_reward,
}

metric_funcs_registry = {
    "reward_timestep_pair": reward_timestep_pair,
    "reward_think_length": reward_think_length,
    "reward_keyword_usage": reward_keyword_usage,
    "reward_paragraph_structure": reward_paragraph_structure,
}


def load_json_dataset_tg(
    train_data_path, is_curriculum_learning=False, preprocessed_data_path=None
):
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = []
        for item in tqdm(data, desc=f"Processing {split_name} items"):
            video_path = item.get("video")
            video_reverse_path = item.get("video_reverse_path")
            timestamps = item.get("timestamp")
            sentence = item.get("sentence")
            duration = item.get("duration")
            video_start = item.get("video_start")
            video_end = item.get("video_end")
            qid = item.get("qid")
            sensitive = item.get("sensitive")
            sentence = sentence.strip().lower()
            if sentence.endswith("."):
                sentence = sentence[:-1]

            if not os.path.isfile(video_path):
                continue

            example = {
                "task_type": "tg",
                "qid": qid,
                "problem": sentence,
                "choices": "",
                "solution": (float(timestamps[0]), float(timestamps[1])),
                "video_path": video_path,
                "video_reverse_path": video_reverse_path,
                "sensitive": sensitive,
                "durations": duration,
                "video_start": video_start,
                "video_end": video_end,
                "preprocessed_path": "",
            }
            examples.append(example)

        if not examples:
            return None

        print("is_curriculum_learning:", is_curriculum_learning)
        if not is_curriculum_learning:
            random.shuffle(examples)

        for i, ex in enumerate(examples[:5]):
            print(f"  sample: {i+1}: {ex}")

        dataset = Dataset.from_list(examples)

        def __getitem__(self, idx):
            example = dataset[idx]
            return example

        from types import MethodType

        dataset.__getitem__ = MethodType(__getitem__, dataset)
        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")

    return train_dataset


class SaveEpochEndCallback(TrainerCallback):
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            trainer = kwargs.get("trainer")
            if trainer is None:
                return

            epoch_checkpoint_dir = os.path.join(
                args.output_dir, f"epoch-{int(state.epoch)}"
            )

            print(
                f"\n{'='*20} Callback: Saving model checkpoint at end of epoch {int(state.epoch)} to {epoch_checkpoint_dir} {'='*20}\n"
            )
            trainer.save_model(epoch_checkpoint_dir)


class StopAfterNEpochsCallback(TrainerCallback):
    def __init__(self, num_epochs_to_train=1):
        super().__init__()
        self.num_epochs_to_train = num_epochs_to_train
        print(
            f"Callback initialized: Training will stop after {self.num_epochs_to_train} completed epoch(s)."
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.epoch >= self.num_epochs_to_train:
            print(
                f"Epoch {state.epoch:.0f} completed. Stopping training as per StopAfterNEpochsCallback (target: {self.num_epochs_to_train} epoch(s))."
            )
            control.should_training_stop = True


def set_global_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main(script_args, training_args, model_args):

    set_global_seed(42)

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    metric_funcs = list(metric_funcs_registry.values())

    dataset = load_json_dataset_tg(
        script_args.train_data_path,
        script_args.is_curriculum_learning,
    )

    trainer_cls = ArrowGEV_Trainer

    callbacks_list = []
    if script_args.is_early_stopping:
        callbacks_list.append(
            StopAfterNEpochsCallback(num_epochs_to_train=training_args.num_train_epochs)
        )

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        metric_funcs=metric_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        callbacks=callbacks_list,
    )
    if torch.cuda.is_available():
        trainer.model = trainer.model.to("cuda")

    if training_args.resume_from_checkpoint is not None:
        trainer_state_path = os.path.join(
            training_args.resume_from_checkpoint, "trainer_state.json"
        )
        if os.path.exists(trainer_state_path):
            print(f"Loading trainer state from: {trainer_state_path}")
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            resumed_global_step = trainer_state.get("global_step", 0)

        num_micro_batches_per_epoch_per_gpu = len(trainer.get_train_dataloader())
        max_step = math.ceil(
            trainer.args.num_train_epochs
            * num_micro_batches_per_epoch_per_gpu
            / trainer.args.gradient_accumulation_steps
        )
        trainer.args.max_steps = resumed_global_step + max_step

        if hasattr(trainer, "state") and hasattr(trainer.state, "max_steps"):
            trainer.state.max_steps = max_step

        print(f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, MY_GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
