"""Download an ArrowGEV dataset snapshot from the HuggingFace Hub.

Example:
    python download_data.py --repo_id ParadiseYu/ArrowGEV-Data --local_dir ./dataset/ArrowGEV
"""

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", default="ParadiseYu/ArrowGEV-Data")
    parser.add_argument("--local_dir", default="./dataset/ArrowGEV")
    parser.add_argument("--repo_type", default="dataset", choices=["dataset", "model"])
    args = parser.parse_args()

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        repo_type=args.repo_type,
    )


if __name__ == "__main__":
    main()
