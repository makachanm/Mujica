from pathlib import Path

from common import run_command


def extract_frames(video_path, chunk, work_dir):
    """윈도우마다 한가운데 시점의 프레임을 한 장씩 뽑아 frame_path 에 적어 둔다."""
    for window in chunk.windows:
        frame_path = str(Path(work_dir) / f"chunk{chunk.number:02d}_window{window.index:04d}.jpg")
        middle_time = (window.start + window.end) / 2

        try:
            save_frame(video_path, middle_time, frame_path)
        except RuntimeError:
            # 소리가 영상보다 조금 길면, 마지막 윈도우 한가운데에는 영상 프레임이 없을 수 있다.
            save_frame(video_path, window.start, frame_path)
        window.frame_path = frame_path


def save_frame(video_path, time, frame_path):
    run_command([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{time:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "scale=640:-2",
        "-q:v", "4",
        frame_path,
    ])
