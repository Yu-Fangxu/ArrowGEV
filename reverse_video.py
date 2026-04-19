"""Produce a time-reversed copy of every video in a folder.

The reversed clips are used as negatives for the temporal directionality
reward during training. Output files are named ``<stem>_reverse.mp4``.

Example:
    python reverse_video.py \\
        --input_folder  dataset/ArrowGEV/videos/arrowgev_data \\
        --output_folder dataset/ArrowGEV/videos/arrowgev_data
"""

import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def reverse_video(input_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in reversed(frames):
        writer.write(frame)
    writer.release()


def process_folder(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name in tqdm(os.listdir(input_dir)):
        src = os.path.join(input_dir, name)
        if Path(name).suffix.lower() not in VIDEO_EXTS:
            continue
        dst = os.path.join(output_dir, f"{Path(name).stem}_reverse.mp4")
        try:
            reverse_video(src, dst)
        except Exception as err:
            print(f"Failed on {name}: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_folder", required=True, help="Folder of source videos.")
    parser.add_argument(
        "--output_folder",
        default=None,
        help="Where to write reversed clips (defaults to --input_folder).",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder or input_folder
    if not os.path.isdir(input_folder):
        raise SystemExit(f"Input folder does not exist: {input_folder}")

    process_folder(input_folder, output_folder)
    print("Done.")


if __name__ == "__main__":
    main()
