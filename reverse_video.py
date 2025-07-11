import os
import cv2
from pathlib import Path
from tqdm import tqdm
def reverse_and_concat(input_path, output_path):
    """
    1. 倒放视频
    2. 将倒放视频拼接到原视频末尾
    3. 保存为 *_reverse.mp4
    """
    # 读取原视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {input_path}")

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 读取所有帧
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame)
    cap.release()

    # 倒放帧
    reversed_frames = original_frames[::-1]

    # 拼接原视频和倒放视频
    # combined_frames = original_frames + reversed_frames
    combined_frames = reversed_frames # only reversed

    # 写入新视频
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in combined_frames:
        out.write(frame)
    out.release()

def process_folder(input_dir, output_dir=None):
    """批量处理文件夹中的所有视频"""
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    video_exts = ['.mp4', '.avi', '.mov', '.mkv']

    for file in tqdm(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, file)
        ext = Path(file).suffix.lower()
        if ext not in video_exts:
            continue

        try:
            # 生成输出路径
            output_name = f"{Path(file).stem}_reverse_only.mp4"  # 强制使用.mp4后缀
            output_path = os.path.join(output_dir, output_name)

            # 处理视频
            reverse_and_concat(filepath, output_path)
            print(f"成功处理: {file} -> {output_name}")

        except Exception as e:
            print(f"处理失败 {file}: {str(e)}")

if __name__ == "__main__":
    input_folder = "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/timer1/videos/timerft_data"
    output_folder =  "/mnt/gemininjceph3/geminicephfs/pr-others-prctrans/fangxuyu/time-r1/dataset/timer1/videos/timerft_data"
    if not os.path.isdir(input_folder):
        print("错误: 路径无效")
    else:
        process_folder(input_folder, output_folder)
        print("全部处理完成！")