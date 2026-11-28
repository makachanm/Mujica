# 평가 결과 (2026-11-28 17:26)

- 에이전트 모델: `qwen3.5:9b`, 비전 모델: `qwen3.5:9b`
- 조각 1개, 윈도우 2.0초, 한 번에 8장
- Whisper: `deepdml/faster-whisper-large-v3-turbo-ct2`, 노래 감지: True
- 덮어쓴 값: {'window_seconds': 2.0}

| 사례 | 성공/실행 | tIoU | recall@0.5 | precision@0.5 | 찾은 구간 수 | 시간(초) |
|---|---|---|---|---|---|---|
| anime_rain_puddle | 1/1 | 0.12 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 149.52 ± 0.00 |
