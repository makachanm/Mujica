# 벤치마크

영상에서 요청한 장면을 얼마나 정확히 찾는지 잰다. Mujica와 범용 에이전트(Claude Code) 기준선을 같은 사례로 비교한다.

```
benchmark/
  cases.json      사례와 정답 구간
  scoring.py      채점, 요약표, 기록 저장 (두 스크립트가 같이 씀)
  run_mujica.py   Mujica 평가
  run_agent.py    범용 에이전트 기준선
  videos/         테스트 영상 (git 제외)
  results/        실험 기록
```

## 실행

레포 루트에서 모듈로 실행한다. API 키와 서버 주소는 `options.json`(화면에서 저장한 설정)을 쓴다.

```
.venv/bin/python -m benchmark.run_mujica --runs 5 --set agent_model=claude-sonnet-5-5 --set vision_model=claude-sonnet-5-5
.venv/bin/python -m benchmark.run_mujica --case bbb_bow_making --set chunk_count=4
.venv/bin/python -m benchmark.run_agent --runs 5
.venv/bin/python -m benchmark.run_agent --case bbb_spike_trap
```

- `--case` 를 주지 않으면 `cases.json` 의 사례를 전부 돌린다 (여러 번 줄 수 있음).
- `--set 이름=값` 은 `options.json` 값을 덮어쓴다 (run_mujica 만).
- 결과는 `results/<시각>_<mujica|agent>_<모델>/` 에 `results.json`(실행별 찾은 구간과 근거)과 `summary.md`(사례별 평균 ± 표준편차)로 남는다. `--name` 으로 폴더 이름을 정할 수 있다. API 키는 남기지 않는다.

범용 에이전트는 빈 임시 폴더에 영상(`video.<확장자>`)과 받아쓰기 도구(`transcribe.py`, Mujica와 같은 Whisper 서버·모델 사용)만 넣고 그 안에서 `claude -p --bare` 로 돌린다. 정답이 있는 레포는 보이지 않는다.

## 영상

| 파일 | 내용 |
|---|---|
| `videos/anime_ending.webm` | 애니메이션 엔딩 (일본어, 약 90초) |
| `videos/ohtani_nlcs.webm` | 오타니 쇼헤이 NLCS 4차전 홈런·인터뷰 (영어) |
| `videos/blender/big_buck_bunny.webm` | Big Buck Bunny (블렌더 재단, 약 10분) |
| `videos/blender/spring.webm` | Spring (블렌더 재단, 약 7분 44초, 대사 없음) |
| `videos/blender/sintel.webm` | Sintel (블렌더 재단, 약 14분 48초, 영어 대사) |
| `videos/blender/tears_of_steel.webm` | Tears of Steel (아직 사례 없음) |

## 사례

| 이름 | 영상 | 요청 | 정답(초) |
|---|---|---|---|
| anime_ending_song | anime_ending | 애니메이션 엔딩곡이 나오는 장면 | 0–89 |
| anime_dinosaur | anime_ending | 캐릭터가 공룡 옷을 입고 있는 장면 | 17–27 |
| anime_rain_puddle | anime_ending | 캐릭터가 비 오는 웅덩이를 내려다보는 장면 | 47–57 |
| ohtani_home_run | ohtani_nlcs | 오타니 쇼헤이가 홈런을 치는 장면 | 65–77, 93–107 |
| ohtani_interview_question | ohtani_nlcs | 기자가 오타니에게 4차전 몸 상태를 묻는 장면 | 1.8–17.8 |
| ohtani_first_run_answer | ohtani_nlcs | 오타니가 첫 점수를 주지 않는 게 중요하다고 말하는 장면 | 36–45 |
| bbb_bow_making | big_buck_bunny | 토끼가 나뭇가지를 깎아 활과 화살을 만드는 장면 | 251–265 |
| bbb_spike_trap | big_buck_bunny | 날다람쥐가 뾰족한 말뚝 함정 위의 나뭇가지에 매달려 있는 장면 | 445–452 |
| bbb_flying_squirrel | big_buck_bunny | 날다람쥐가 다리를 펴고 하늘을 활공하는 장면 | 424–426, 429–431, 434–440, 482–484, 580.5–596.5 |
| spring_flame | spring | 소녀가 두 손에 주황색 불꽃을 들고 바라보는 장면 | 229.5–234.5 |
| spring_overlook | spring | 소녀와 강아지가 바위 위에서 햇살 비치는 산맥을 바라보는 장면 | 384.5–394.5 |
| spring_title | spring | SPRING 제목이 화면에 나오는 장면 | 418–425 |
| sintel_logo | sintel | SINTEL 타이틀 로고가 화면에 나오는 장면 | 90.5–98 |
| sintel_shaman_question | sintel | 주술사가 신텔에게 이곳에 무슨 일로 왔는지 묻고, 신텔이 누군가를 찾고 있다고 답하는 장면 | 127–140 |
| sintel_dragon_grab | sintel | 어른 용이 아기 용을 붙잡아 데려가는 장면 | 324–336.5 |
| sintel_fool_unprepared | sintel | 주술사가 신텔에게 아무 준비 없이 혼자 다니다니 바보 같다고 말하는 장면 | 117.5–124 |
| sintel_good_night | sintel | 신텔이 아기 용 스케일스에게 잘 자라고 인사하는 장면 | 227.5–230 |
| sintel_dragon_lands | sintel | 주술사가 신텔에게 이곳은 용의 땅이고 생각보다 가까이 왔다고 말하는 장면 | 457.5–462.5 |

## 지표

| 지표 | 뜻 |
|---|---|
| tIoU | 찾은 구간 전체와 정답 구간 전체가 겹친 길이 / 합친 길이 |
| 정답률 (recall@0.5) | 정답 구간 가운데 IoU 0.5 이상으로 겹치는 찾은 구간이 있는 비율 |
| precision@0.5 | 찾은 구간 가운데 IoU 0.5 이상으로 겹치는 정답 구간이 있는 비율 |
| 시간 | 요청부터 답까지 걸린 시간 |
| 입력·출력 토큰 | Claude 토큰 (입력은 캐시 쓰기·읽기 포함). Ollama 모델은 세지 않음 |
| 비용, 턴 수 | 범용 에이전트만 (Claude Code가 알려 주는 값) |

## 실험 기록 (`results/`)

공통 조건: 윈도우 10초, 한 번에 8장, Whisper `deepdml/faster-whisper-large-v3-turbo-ct2`, 노래 감지 켬. 에이전트와 비전 모델은 같은 모델.
01–04는 예전 스크립트(`evaluate.py`)로 돌려서 summary.md 형식이 조금 다르다.

| 폴더 | 코드 | 대상 | 실행 수 | 내용 |
|---|---|---|---|---|
| 01_baseline | look_closely 이전 | Mujica: qwen3.5:9b, Haiku, Sonnet | 1 | 대표 프레임(윈도우 가운데 한 장)만 보는 방식 |
| 02_window_2s | look_closely 이전 | Mujica: qwen3.5:9b | 1 | 웅덩이 사례를 윈도우 2초로 |
| 03_look_closely_pilot | `1dad02c` | Mujica: qwen3.5:9b, Sonnet | 1 | look_closely 도입 후 시험 |
| 04_look_closely | `1dad02c` | Mujica: Haiku, Sonnet | 5 | 본 실험 (애니·오타니 6개 사례) |
| 05_agent_pilot | `1dad02c` | 범용 에이전트: Sonnet | 1 | 공룡·웅덩이·홈런·인터뷰. 토큰 기록 전 |
| 06_big_buck_bunny | `1dad02c+dirty` | Mujica(조각 4개)와 범용 에이전트, Sonnet | 1 | BBB 3개 사례 |
| 08_blender_haiku_agent, 08_blender_haiku_mujica | `1dad02c+dirty` | 범용 에이전트(클립 직접 자르기), Mujica(조각 4개), Haiku | 1 | BBB·Spring·Sintel 9개 사례. Whisper 500 이 나면 20초 이하로 나눠 재시도 |
| 09_dialogue_haiku_agent_no_whisper, 09_dialogue_haiku_mujica | `1dad02c+dirty` | 범용 에이전트(받아쓰기 도구 없음), Mujica(조각 1개), Haiku | 1 | 애니·오타니 6개 사례 |
| 10_sintel_dialogue_haiku_agent_no_whisper, 10_sintel_dialogue_haiku_mujica | `1dad02c+dirty` | 범용 에이전트(받아쓰기 도구 없음), Mujica(조각 4개), Haiku | 1 | Sintel 대사 4개 사례. 에이전트 결과는 오염됨(아래) |
| 07_big_buck_bunny_haiku | `1dad02c+dirty` | Mujica(조각 4개)와 범용 에이전트, Haiku | 1 | BBB 3개 사례. 에이전트에게 받아쓰기 도구만 알려 주고 클립을 직접 자르게 함 |

### 04 결과 (Mujica, tIoU 평균 ± 표준편차, 5회)

| 사례 | Haiku 5.5 | Sonnet 5.5 |
|---|---|---|
| anime_ending_song | 0.66 ± 0.47 | **0.99 ± 0.00** |
| anime_dinosaur | **0.92 ± 0.03** | 0.75 ± 0.18 |
| anime_rain_puddle | 0.80 ± 0.23 | **0.87 ± 0.14** |
| ohtani_home_run | **0.32 ± 0.19** | 0.30 ± 0.22 |
| ohtani_interview_question | 0.75 ± 0.12 | **0.84 ± 0.00** |
| ohtani_first_run_answer | 0.68 ± 0.40 | 0.68 ± 0.39 |
| 사례당 평균 시간 | 59초 | 39초 |

### 05 결과 (범용 에이전트 Sonnet, 1회)

| 사례 | tIoU |
|---|---|
| anime_dinosaur | 0.93 |
| anime_rain_puddle | 1.00 |
| ohtani_home_run | 0.76 (두 홈런 모두 찾음) |
| ohtani_interview_question | 0.94 |

### 06 결과 (Big Buck Bunny, Sonnet, 1회)

| 사례 | Mujica tIoU / 정답률 | 범용 에이전트 tIoU / 정답률 | Mujica 시간·토큰(입력/출력) | 에이전트 시간·토큰·비용 |
|---|---|---|---|---|
| bbb_bow_making | 0.00 / 0.00 (못 찾음) | 0.35 / 0.00 (244–284로 넓게) | 67초, 73k / 10.6k | 39초, 24k / 1.6k, $0.04 |
| bbb_spike_trap | 0.00 / 0.00 (못 찾음) | **0.86 / 1.00** | 47초, 71k / 8.8k | 33초, 32k / 1.2k, $0.04 |
| bbb_flying_squirrel | 0.58 / 0.20 | **0.76 / 0.60** | 73초, 140k / 16.6k | 48초, 32k / 2.6k, $0.05 |

Mujica 실행 중 대사가 없는 구간 4곳(298–373, 447–485, 559–597초)에서 Whisper 서버가 500을 냈다. 30초 넘는 오디오에서 VAD 가 말소리를 하나도 못 찾으면 faster-whisper 가 빈 결과 대신 `RuntimeError: No clip timestamps found` 를 내는 서버 쪽 문제다. 원래 말소리가 없는 구간이라 잃은 대사는 없다.

### 07 결과 (Big Buck Bunny, Haiku, 1회)

범용 에이전트 프롬프트에서 ffmpeg·Read 안내를 빼고, 장면을 직접 잘라 `clips/` 에 저장하게 했다. 세 실행 모두 ffmpeg 를 스스로 찾아 클립을 만들었고, 클립 길이는 보고한 구간과 맞았다.

| 사례 | Mujica tIoU / 정답률 | 범용 에이전트 tIoU / 정답률 | Mujica 시간·토큰(입력/출력) | 에이전트 시간·토큰·비용·턴 |
|---|---|---|---|---|
| bbb_bow_making | 0.05 / **1.00** (256.5–266 맞춤, 447–597 덧붙임) | 0.25 / 0.00 (244–284.5 등 3개) | 148초, 304k / 40k | 169초, 203k / 10k, $0.01, 24턴 |
| bbb_spike_trap | 0.17 / 0.00 | **0.86 / 1.00** | 111초, 289k / 48k | 68초, 113k / 5.7k, $0.01, 18턴 |
| bbb_flying_squirrel | 0.43 / 0.20 | 0.20 / 0.60 (1–3초짜리 6개) | 120초, 266k / 39k | 111초, 271k / 14k, $0.01, 23턴 |

### 08 결과 (블렌더 영상 9개 사례, Haiku, 1회)

| 사례 | Mujica tIoU / 정답률 | 범용 에이전트 tIoU / 정답률 |
|---|---|---|
| bbb_bow_making | **0.60 / 1.00** | 0.37 / 0.00 |
| bbb_spike_trap | 0.66 / 1.00 | **0.86 / 1.00** |
| bbb_flying_squirrel | 0.18 / 0.20 | **0.77 / 1.00** |
| spring_flame | 0.07 / 0.00 | **0.62 / 1.00** |
| spring_overlook | 0.47 / 1.00 | **0.78 / 1.00** |
| spring_title | **0.82 / 1.00** | 0.50 / 1.00 |
| sintel_logo | 0.94 / 1.00 | **0.95 / 1.00** |
| sintel_shaman_question | 0.65 / 1.00 | **0.82 / 1.00** |
| sintel_dragon_grab | 0.00 / 0.00 | **0.43 / 0.00** |
| 평균 | 0.49 / 0.69 | **0.68 / 0.78** |
| 사례당 평균 시간 | 118초 | 102초 |
| 출력 토큰 | 2.4만–6.4만 | 0.4만–1.4만 |

Whisper 재시도를 20초까지 내린 뒤로는 받아쓰기 실패 경고가 한 번도 나오지 않았다.

### 09 결과 (애니·오타니 6개 사례, Haiku, 1회, 정답률과 시간)

| 사례 | Mujica | 범용 에이전트 (받아쓰기 없음) |
|---|---|---|
| anime_ending_song | **1.00**, 58초 | 0.00, 60초 (45–90초만 냄) |
| anime_dinosaur | 1.00, 61초 | 1.00, **30초** |
| anime_rain_puddle | 1.00, 76초 | 1.00, **64초** |
| ohtani_home_run | 0.00, **116초** | **1.00**, 192초 |
| ohtani_interview_question | 1.00, **31초** | 1.00, 437초 |
| ohtani_first_run_answer | 1.00, **63초** | 1.00, 469초 |
| 평균 | 0.83, **68초** | 0.83, 209초 |

받아쓰기 도구가 없어도 에이전트는 오타니 영상에 박혀 있는 영어 자막을 프레임에서 읽어 대사 사례를 풀었다. 대신 7분 넘게 걸렸다.
에이전트가 작업 폴더 밖의 `/tmp/frames` 에 프레임을 쓴 일이 있었다. 하네스는 파일 시스템을 막지 않는다.

### 10 결과 (Sintel 대사 4개 사례, Haiku, 1회, 정답률과 시간)

| 사례 | Mujica | 범용 에이전트 (받아쓰기 없음) |
|---|---|---|
| sintel_shaman_question | **1.00**, 74초 | 시간 초과 (600초) |
| sintel_fool_unprepared | 0.00, 53초 (110–120) | 0.00, 489초 (101–134) |
| sintel_good_night | 0.00, 176초 (222–248) | 0.00, 100초 (212–248) |
| sintel_dragon_lands | 0.00, 65초 (453–465, IoU 0.42) | 시간 초과 (600초) |

**에이전트 결과는 오염됐다.** 받아쓰기 도구를 주지 않았더니 에이전트가 `/tmp/venv` 에 faster-whisper 를 pip 로 설치하고
CPU 로 받아썼다(`/tmp/tx.py`, `/tmp/asr_out.txt` 등). `/tmp` 에는 다른 실행이 남긴 받아쓰기 파일도 있어 서로 읽었을 수 있다.
하네스가 네트워크와 작업 폴더 밖 파일 시스템을 막지 않기 때문이다. 격리(샌드박스) 후 다시 재야 한다.

## 관찰

- **look_closely 효과 (01 → 04)**: 01에서는 구간이 10초 윈도우 경계에 묶여 공룡·웅덩이가 0.54를 넘지 못했고, 홈런과 첫 실점 답변은 세 모델 모두 0이었다. 04에서는 경계가 초 단위로 맞춰지고 두 사례도 찾기 시작했다.
- **윈도우만 줄이는 것은 효과가 없었다 (02)**: 2초 윈도우에서는 비슷한 비·우산 장면까지 묶어 tIoU 0.12로 오히려 떨어졌다.
- **실행마다 편차가 크다**: Haiku는 엔딩곡 사례에서 5번 중 2번 아무것도 제출하지 않았다. 첫 실점 답변은 두 모델 모두 5번 중 1번 놓쳤다.
- **홈런이 가장 어렵다**: 장외 홈런(93–107)은 자주 찾지만 1회 선두타자 홈런(65–77)은 거의 놓친다. Sonnet은 123–135초를 홈런으로 자주 냈는데, 이 구간이 실제 세 번째 홈런인지 정답을 다시 확인해야 한다.
- **범용 에이전트가 더 빠르고 정확했다 (05, 06)**: 짧은 영상에서도, 10분짜리 BBB에서도 범용 에이전트가 tIoU가 높고 시간·토큰이 적었다. 1회 결과라 반복 실행으로 확인이 필요하다.
- **글자가 뜨는 장면은 둘 다 잘 찾는다 (08)**: 타이틀 로고는 두 방식 모두 0.82–0.95. 대사 사례(sintel_shaman_question)도 둘 다 찾았다.
- **spring_flame 정답 재검토 필요 (08)**: 두 방식 모두 5:04–5:29 근처(소녀가 불붙은 지팡이 머리를 손에 든 장면)를 같이 냈다. 요청을 넓게 읽으면 맞는 답일 수 있다.
- **sintel_dragon_grab 은 경계가 애매하다 (08)**: 에이전트는 320.8–330.7 로 동작 앞부분을 잡았다. Mujica 는 0–0.5초라는 엉뚱한 구간을 냈다.
- **Haiku 로 내리면 둘 다 흔들린다 (07)**: 범용 에이전트는 Sonnet 보다 시간·턴이 두 배쯤 늘었고, 활공 장면을 1–3초 조각으로 잘게 나눴다. Mujica Haiku 는 활 만들기를 정확히 찾았지만 마지막 조각(447.4–596.5초)을 통째로 같이 내서 tIoU 가 낮다.
