# 평가 결과 (2026-11-28 17:49)

- 에이전트 모델: `claude-sonnet-5-5`, 비전 모델: `claude-sonnet-5-5`
- 조각 1개, 윈도우 10.0초, 한 번에 8장
- Whisper: `deepdml/faster-whisper-large-v3-turbo-ct2`, 노래 감지: True
- 덮어쓴 값: {'agent_model': 'claude-sonnet-5-5', 'vision_model': 'claude-sonnet-5-5'}

| 사례 | 성공/실행 | tIoU | recall@0.5 | precision@0.5 | 찾은 구간 수 | 시간(초) |
|---|---|---|---|---|---|---|
| anime_ending_song | 1/1 | 0.99 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 39.71 ± 0.00 |
| anime_dinosaur | 1/1 | 0.65 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 29.85 ± 0.00 |
| anime_rain_puddle | 1/1 | 0.85 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 38.41 ± 0.00 |
| ohtani_home_run | 1/1 | 0.54 ± 0.00 | 0.50 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 54.62 ± 0.00 |
| ohtani_interview_question | 1/1 | 0.84 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 25.18 ± 0.00 |
| ohtani_first_run_answer | 1/1 | 0.90 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 43.50 ± 0.00 |
