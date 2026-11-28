# 평가 결과 (2026-11-28 18:49)

- 코드: `1dad02c`
- 에이전트 모델: `claude-sonnet-5-5`, 비전 모델: `claude-sonnet-5-5`
- 조각 1개, 윈도우 10.0초, 한 번에 8장
- Whisper: `deepdml/faster-whisper-large-v3-turbo-ct2`, 노래 감지: True
- 덮어쓴 값: {'agent_model': 'claude-sonnet-5-5', 'vision_model': 'claude-sonnet-5-5'}

| 사례 | 성공/실행 | tIoU | recall@0.5 | precision@0.5 | 찾은 구간 수 | 시간(초) |
|---|---|---|---|---|---|---|
| anime_ending_song | 5/5 | 0.99 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 36.47 ± 1.59 |
| anime_dinosaur | 5/5 | 0.75 ± 0.18 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 36.39 ± 6.23 |
| anime_rain_puddle | 5/5 | 0.87 ± 0.14 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 34.59 ± 5.19 |
| ohtani_home_run | 5/5 | 0.30 ± 0.22 | 0.30 ± 0.27 | 0.50 ± 0.50 | 1.40 ± 0.55 | 66.29 ± 11.52 |
| ohtani_interview_question | 5/5 | 0.84 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 22.83 ± 0.54 |
| ohtani_first_run_answer | 5/5 | 0.68 ± 0.39 | 0.80 ± 0.45 | 0.80 ± 0.45 | 0.80 ± 0.45 | 37.78 ± 8.86 |
