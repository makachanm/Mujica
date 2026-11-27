"""
Mujica 파이프라인. 영상과 요청 문장을 받아 요청한 장면을 찾고, 후보 클립으로 잘라 돌려준다.
사용자가 적은 짧은 영상 설명(video_description)은 에이전트와 비전 모델에게 영상 배경으로 알려 준다. 비어 있어도 된다.

settings
    ollama_url          Ollama 서버 주소
    agent_model         서브에이전트 모델 (도구 호출 지원). claude- 로 시작하면 Claude API 를 쓴다
    vision_model        프레임을 보고 답하는 모델 (이미지 입력 지원). claude- 로 시작하면 Claude API 를 쓴다
    chunk_count         영상을 나눌 조각 수 = 서브에이전트 수
    window_seconds      대표 프레임 한 장이 맡는 시간(초)
    frames_per_look     에이전트가 한 번에 볼 수 있는 프레임 수
    whisper_server_url  OpenAI 호환 받아쓰기 서버 주소
    whisper_model       받아쓰기 서버의 모델 이름
    language            받아쓰기 언어 코드
    detect_songs        노래·음악 구간을 찾아 받아쓰기에서 빼고 윈도우에 표시할지

고유명사 사전은 glossary.json 에서 읽어 영상 배경에 같이 넣는다. 없으면 쓰지 않는다 (step6 describe_glossary 참고).

report_progress(문장, 끝났는지) 는 진행 상황을 받는 비동기 함수다.
"""

import asyncio
import tempfile
from pathlib import Path

from step.step1_probe_video import probe_video
from step.step2_split_timeline import split_timeline
from step.step4_extract_frames import extract_frames
from step.step5_transcribe_audio import WhisperServer, transcribe_chunk
from step.step6_chunk_agent import create_model, describe_background, run_chunk_agent
from step.step7_merge_results import merge_segments
from step.step8_cut_clips import cut_clips


GLOSSARY_PATH = Path(__file__).parent / "glossary.json"


async def find_songs(video_path, has_audio, settings, report_progress):
    no_songs = {"songs": [], "mute": []}
    if not has_audio:
        return no_songs
    if not settings["detect_songs"]:
        return no_songs

    # torch 는 불러오는 데만 몇 초 걸려서, 노래 감지를 켰을 때만 불러온다.
    from step.step3_detect_songs import detect_songs

    await report_progress("노래 구간 찾는 중", False)
    return await asyncio.to_thread(detect_songs, video_path)


async def find_moments(user_request, video_description, video_path, settings, output_dir, report_progress):
    chunk_count = settings["chunk_count"]

    await report_progress("영상 읽는 중", False)
    duration, has_audio = await asyncio.to_thread(probe_video, video_path)
    chunks = split_timeline(duration, chunk_count, settings["window_seconds"])

    song_ranges = await find_songs(video_path, has_audio, settings, report_progress)
    background_text = describe_background(video_description, GLOSSARY_PATH)

    whisper_server = WhisperServer(settings["whisper_server_url"], settings["whisper_model"], settings["language"])
    agent_model = create_model(settings["agent_model"], settings["ollama_url"])
    vision_model = create_model(settings["vision_model"], settings["ollama_url"])

    with tempfile.TemporaryDirectory(prefix="mujica_") as work_dir:
        # 조각을 하나 준비하는 대로 그 조각의 에이전트를 출발시켜서,
        # 다음 조각을 준비하는 동안 앞 조각의 에이전트가 같이 일하게 한다.
        agent_tasks = []
        for chunk in chunks:
            await report_progress(f"조각 {chunk.number}/{chunk_count} 준비 중", False)
            await asyncio.to_thread(extract_frames, video_path, chunk, work_dir)
            if has_audio:
                await asyncio.to_thread(transcribe_chunk, video_path, chunk, work_dir, whisper_server, song_ranges)

            agent_run = run_chunk_agent(
                agent_model, vision_model, chunk, chunk_count, user_request, settings["frames_per_look"],
                background_text,
            )
            agent_tasks.append(asyncio.create_task(agent_run))

        await report_progress(f"에이전트 {chunk_count}개가 찾는 중", False)
        segments_per_chunk = await asyncio.gather(*agent_tasks)

    segments = merge_segments(segments_per_chunk, settings["window_seconds"])

    await report_progress(f"클립 {len(segments)}개 자르는 중", False)
    await asyncio.to_thread(cut_clips, video_path, segments, output_dir)

    await report_progress("완료", True)
    return segments
