from pathlib import Path

from common import run_command


def cut_clips(video_path, segments, output_dir):
    """구간마다 mp4 클립을 잘라 저장하고 clip_path 에 적어 둔다."""
    for order, segment in enumerate(segments, start=1):
        clip_path = str(Path(output_dir) / f"candidate_{order:02d}.mp4")

        # 재인코딩해야 키프레임 위치와 상관없이 정확한 시각에서 잘린다.
        # +faststart 는 브라우저가 파일을 다 받기 전에 재생을 시작하게 해 준다.
        run_command([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{segment.start:.3f}",
            "-t", f"{segment.end - segment.start:.3f}",
            "-i", video_path,
            "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            clip_path,
        ])
        segment.clip_path = clip_path
