"""Rewrite the ``video`` field of an annotation JSON so it points to the
matching ``*_reverse.*`` clip produced by :mod:`reverse_video`.

Example:
    python label_sensitivity.py \\
        --input  dataset/ArrowGEV/annotations/train_2k5.json \\
        --output dataset/ArrowGEV/annotations/train_2k5_reverse.json
"""

import argparse
import json
import os


def rewrite_paths(input_json: str, output_json: str) -> None:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "video" not in item:
            continue
        name, ext = os.path.splitext(item["video"])
        item["video"] = f"{name}_reverse{ext}"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Wrote reversed-video annotations to {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source annotation JSON.")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination annotation JSON (may equal --input to overwrite).",
    )
    args = parser.parse_args()
    rewrite_paths(args.input, args.output)


if __name__ == "__main__":
    main()
