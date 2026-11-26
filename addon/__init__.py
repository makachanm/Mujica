"""받아쓰기 품질 기능.

song_detect       노래·음악 구간 감지 (PANNs Cnn14). 받아쓰기에서 빼고 윈도우에 표시
hallucination     Whisper 할루시네이션·가사 찌꺼기 사후 필터
glossary          고유명사 사전 → 에이전트 프롬프트 (Whisper hotwords 문자열도 만들 수 있다)
whisper_segments  세그먼트 단위 받아쓰기(verbose_json) + 위 셋을 묶어 Mujica 윈도우에 채움
"""
