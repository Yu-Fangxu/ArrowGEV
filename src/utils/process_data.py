"""Filter a per-sample difficulty JSON produced by :mod:`calc_difficulty`.

The paper's dynamic difficulty filter keeps only samples the policy has *not*
yet mastered — i.e. samples with ``0 < IoU <= threshold``. Two alternative
sampling modes (``gaussian``, ``random``) are kept for ablations.

Example::

    python src/utils/process_data.py \\
        --input_json  logs/exp/filtering_epoch0/train_v4_cloud.json \\
        --output_json logs/exp/filtering_epoch0/train_filtered.json \\
        --threshold 0.70 \\
        -k 2500
"""

import argparse
import json
import math
import os
import random

import numpy as np
import torch


def get_difficulty_safe(item):
    """Return ``item['difficulty']`` as a finite float, else ``None``."""
    difficulty = item.get("difficulty")
    if difficulty is None:
        return None
    try:
        difficulty_float = float(difficulty)
    except (ValueError, TypeError):
        return None
    if math.isnan(difficulty_float) or math.isinf(difficulty_float):
        return None
    return difficulty_float


def save_json(data_list, output_path, description):
    if data_list and isinstance(data_list[0], dict) and "data" in data_list[0]:
        data_to_save = [item["data"] for item in data_list]
    else:
        data_to_save = data_list
    if not data_to_save:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
    print(f"[{description}] wrote {len(data_to_save)} items -> {output_path}")


def random_sample(data_list, k, output_path, description):
    if not isinstance(data_list, list):
        return
    n = len(data_list)
    k = min(n, k)
    sampled = data_list if k >= n else random.sample(data_list, k)
    save_json(sampled, output_path, f"{description} (random: {len(sampled)})")


def difficulty_sorted_sample(data_list, k, output_path, description):
    """Sort by descending difficulty and pick ``k`` items evenly along the sorted list."""
    if not data_list or k <= 0:
        return
    n = len(data_list)
    actual_k = min(n, k)
    sorted_list = sorted(data_list, key=lambda x: x["difficulty_float"], reverse=True)
    if actual_k >= n:
        sampled = sorted_list
    else:
        indices = torch.linspace(0, n - 1, steps=actual_k).round().long()
        indices = torch.clamp(indices, 0, n - 1)
        unique_indices = torch.unique(indices)
        sampled = [sorted_list[i] for i in unique_indices]
    save_json(sampled, output_path, description)


def gaussian_sample(data_list, k, output_path, description, center=0.3, std_dev=0.2):
    """Importance-sample ``k`` items with weights from ``N(center, std_dev)``."""
    if not data_list or k <= 0:
        return
    n = len(data_list)
    actual_k = min(n, k)
    if actual_k == 0:
        return

    difficulties = np.array([item["difficulty_float"] / 100.0 for item in data_list])
    probs = np.exp(-((difficulties - center) ** 2) / (2 * std_dev**2))
    probs /= probs.sum()

    try:
        sampled = [data_list[i] for i in np.random.choice(n, actual_k, False, p=probs)]
    except ValueError as e:
        print(e)
        return
    save_json(
        sampled,
        output_path,
        f"{description} (gaussian, mean={center}, std={std_dev})",
    )


def process_ddata(input_json_path, output_path, threshold=0.7, mode="threshold", k=2500):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_items = []
    for item in data:
        d = get_difficulty_safe(item)
        if isinstance(item, dict) and d is not None:
            valid_items.append(
                {"difficulty_float": d, "p_value": d / 100.0, "data": item}
            )
    if not valid_items:
        print("No valid items found.")
        return
    print(f"valid items: {len(valid_items)} (original: {len(data)})")

    if mode == "threshold":
        subset = [item for item in valid_items if 0 < item["p_value"] <= threshold]
        difficulty_sorted_sample(
            subset, k, output_path, f"threshold filter (0 < p <= {threshold:.2f})"
        )
    elif mode == "gaussian":
        subset = [item for item in valid_items if item["p_value"] > 0]
        gaussian_sample(subset, k, output_path, "gaussian filter")
    elif mode == "random":
        random_sample(valid_items, k, output_path, "random filter")
    else:
        raise ValueError(f"Unknown --mode: {mode!r}")

    print("Done.")


def _default_output(input_json):
    stem = input_json[:-5] if input_json.endswith(".json") else input_json
    return f"{stem}_filtered.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_json", required=True, help="Scored input JSON (from calc_difficulty).")
    parser.add_argument(
        "--output_json",
        default=None,
        help="Output JSON path (default: <input>_filtered.json).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Keep samples with 0 < IoU <= threshold (default: 0.7).",
    )
    parser.add_argument(
        "--mode",
        choices=["threshold", "gaussian", "random"],
        default="threshold",
    )
    parser.add_argument(
        "-k",
        "--max_samples",
        type=int,
        default=2500,
        help="Maximum samples retained after filtering.",
    )
    args = parser.parse_args()

    output_path = args.output_json or _default_output(args.input_json)
    process_ddata(
        args.input_json,
        output_path,
        threshold=args.threshold,
        mode=args.mode,
        k=args.max_samples,
    )
