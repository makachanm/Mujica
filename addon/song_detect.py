"""노래·음악 구간 감지.

1초 단위 speech / singing / music / silence 타임라인을 만들고, 노래·음악이 이어진 구간(songs)과
받아쓰기에서 무음 처리할 구간(mute)을 돌려준다. Whisper가 가사를 받아적거나 BGM 위에서
"시청해주셔서 감사합니다" 같은 문장을 지어내는 걸 막는 용도다.

    from addon.song_detect import detect_songs
    info = detect_songs(video_path)            # {"songs": [...], "mute": [...], "regions": [...], "share": {...}}

PANNs Cnn14(AudioSet 527 클래스, 32kHz)를 쓴다. AudioSet에 Speech·Singing·Music이 따로 있어서
"말", "노래(가창)", "반주·BGM"을 나눠 볼 수 있다.

두 단계로 나눈다.
1) frame_probs: 무거운 신경망 단계. 1초마다 클래스 확률을 뽑아 audio_frames.npz에 캐시한다.
   GPU 없이 CPU에서도 빨리 돌도록, 합성곱 특징은 64초 블록마다 한 번만 계산하고
   그 위에서 2초 창을 1초씩 밀며 분류 헤드만 다시 돌린다 (창마다 모델을 다시 돌리지 않는다).
2) label_regions: 가벼운 규칙 단계. 임계값·스무딩·최소 길이로 구간을 만든다.
   설정값만 바꿔 다시 돌릴 수 있다.
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


SR = 32000
HOP_SEC = 1.0        # 출력 간격
WIN_SEC = 2.0        # 한 판정에 보는 길이
BLOCK_SEC = 64.0     # 합성곱 특징을 한 번에 계산하는 길이
CTX_SEC = 2.0        # 블록 경계 왜곡을 줄이려고 앞뒤로 더 읽는 길이
FEAT_SEC = 0.32      # Cnn14 마지막 특징 한 칸의 시간 (10ms hop × 32배 풀링)

# 가중치를 한 번만 받도록 여기에 저장해 둔다
MODEL_DIR = Path.home() / ".cache" / "mujica-addons" / "panns"
MODEL_FILE = "Cnn14_32k.pth"
# 공식 배포처(Zenodo)가 막힌 환경이 있어 같은 파일의 Hugging Face 사본을 쓴다 (크기 327,428,481바이트)
MODEL_URL = "https://huggingface.co/thelou1s/panns-inference/resolve/main/Cnn14_mAP%3D0.431.pth"
MODEL_SIZE = 327_428_481
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

# AudioSet 클래스 번호 (class_labels_indices.csv 기준)
CLASSES = {
    "speech": [0, 1, 2, 3, 4, 5],          # Speech, Male/Female/Child speech, Conversation, Narration
    "singing": [27, 28, 32, 33, 34, 35, 36],  # Singing, Choir, Male/Female/Child/Synthetic singing, Rapping
    "music": [137],                        # Music
}
LABELS = ("speech", "singing", "music", "silence")

# 판정 기준 기본값
DEFAULT_SETTINGS = {
    "silence_db": -50,          # 이보다 조용하면 silence
    "speech_th": 0.5,           # 말소리 확률이 이 이상이면 음악이 깔려 있어도 speech
    "singing_th": 0.3,          # 가창 확률 기준
    "singing_over_speech": 1.0, # 가창 확률이 말소리 확률 × 이 값 이상일 때만 singing
    "music_th": 0.5,            # 음악 확률 기준 (말소리가 약할 때)
    "smooth_sec": 5,            # 다수결 스무딩 창
    "song_min_sec": 30,         # 노래·음악이 이만큼 이어져야 노래 구간으로 확정
    "song_gap_sec": 5,          # 노래 사이 이 이하의 짧은 말·무음은 같은 노래로 본다
    "keep_speech_th": 0.15,     # 노래 구간 안이라도 말소리 확률이 이 이상이면 받아쓰기에 남긴다 (곡 위 코멘트)
    "keep_pad_sec": 3,          # 그 말소리 앞뒤로 남길 여유
    "mute_min_sec": 5,          # 이보다 짧은 노래 조각은 굳이 무음 처리하지 않는다
    # 노래 구간을 "노래"(보컬 있음)와 "음악·BGM"으로 나누는 기준. 반주가 큰 곡에서는 Cnn14 가창 확률이
    # 0.1~0.2 에 머물러 singing_th(0.3)로는 보컬을 못 잡는다. 가사가 받아써진 어쿠스틱 곡은 이 값 이상인
    # 프레임이 66%, 말 뒤 BGM·기타 전주는 0~3% 였다. 이 값 이상인 프레임이 30% 이상이면 "노래"로 본다.
    "song_vocal_th": 0.08,
}


def _ensure_model() -> Path:
    path = MODEL_DIR / MODEL_FILE
    if path.exists() and path.stat().st_size == MODEL_SIZE:
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"[song_detect] PANNs Cnn14 가중치 다운로드 (327MB) → {path}", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if tmp.stat().st_size != MODEL_SIZE:
        tmp.unlink()
        raise RuntimeError("PANNs 가중치 크기가 예상과 다릅니다. 다운로드를 다시 시도하세요.")
    tmp.replace(path)
    return path


def _ensure_labels_csv() -> None:
    """panns_inference는 import 시 ~/panns_data/class_labels_indices.csv가 없으면 wget을 호출한다.
    wget이 없는 환경도 있어서 미리 받아 둔다."""
    path = Path.home() / "panns_data" / "class_labels_indices.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LABELS_URL, path)


def _load_model():
    import torch

    _ensure_labels_csv()
    from panns_inference.models import Cnn14

    model = Cnn14(sample_rate=SR, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    ckpt = torch.load(_ensure_model(), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _features(model, audio: np.ndarray):
    """Cnn14 합성곱 부분만 통과시킨 시간축 특징 (T', 2048)."""
    import torch

    x = torch.from_numpy(audio)[None, :]
    x = model.spectrogram_extractor(x)
    x = model.logmel_extractor(x)
    x = x.transpose(1, 3)
    x = model.bn0(x)
    x = x.transpose(1, 3)
    for i, blk in enumerate(
        [model.conv_block1, model.conv_block2, model.conv_block3, model.conv_block4, model.conv_block5, model.conv_block6]
    ):
        x = blk(x, pool_size=(1, 1) if i == 5 else (2, 2), pool_type="avg")
    x = torch.mean(x, dim=3)  # 주파수 평균 → (1, 2048, T')
    return x[0].T  # (T', 2048)


def _head(model, windows):
    """창별 (max + mean) 풀링 → 분류 헤드. windows: (N, 2048) 두 개."""
    import torch
    import torch.nn.functional as F

    x1, x2 = windows
    x = F.relu_(model.fc1(x1 + x2))
    return torch.sigmoid(model.fc_audioset(x))


def _pcm_stream(video: Path, start: float = 0.0, duration: float | None = None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def frame_probs(
    video: Path, out_npz: Path, force: bool = False, start: float = 0.0, duration: float | None = None
) -> dict[str, np.ndarray]:
    """1초 간격 확률 타임라인. 키: t(초), speech, singing, music, rms_db. start/duration은 시험용."""
    if out_npz.exists() and not force:
        d = np.load(out_npz)
        return {k: d[k] for k in d.files}

    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = _load_model()
    proc = _pcm_stream(video, start, duration)
    assert proc.stdout is not None

    bytes_per_sec = SR * 4
    buf = np.zeros(0, dtype=np.float32)
    buf_t0 = 0.0  # buf[0]의 시각
    t0 = 0.0      # 이번 블록 시작 시각
    eof = False
    rows: list[np.ndarray] = []
    started = time.time()

    with torch.inference_mode():
        while True:
            need_until = t0 + BLOCK_SEC + CTX_SEC
            while not eof and buf_t0 + len(buf) / SR < need_until:
                chunk = proc.stdout.read(int(bytes_per_sec * 16))
                if not chunk:
                    eof = True
                    break
                buf = np.concatenate([buf, np.frombuffer(chunk, dtype=np.float32)])
            buf_end = buf_t0 + len(buf) / SR
            if t0 >= buf_end - 0.5:
                break

            a_start = max(buf_t0, t0 - CTX_SEC)
            audio = buf[int((a_start - buf_t0) * SR):]
            block_end = min(t0 + BLOCK_SEC, buf_end)

            feat = _features(model, np.ascontiguousarray(audio))            # (T', 2048)
            centers = a_start + (np.arange(feat.shape[0]) + 0.5) * FEAT_SEC
            secs = np.arange(t0, block_end, HOP_SEC)
            x1s, x2s = [], []
            for s in secs:
                c = s + HOP_SEC / 2
                idx = np.nonzero((centers >= c - WIN_SEC / 2) & (centers < c + WIN_SEC / 2))[0]
                if len(idx) == 0:
                    idx = np.array([min(int((c - a_start) / FEAT_SEC), feat.shape[0] - 1)])
                w = feat[idx[0]: idx[-1] + 1]
                x1s.append(w.max(dim=0).values)
                x2s.append(w.mean(dim=0))
            probs = _head(model, (torch.stack(x1s), torch.stack(x2s))).numpy()  # (N, 527)

            for j, s in enumerate(secs):
                seg = buf[int((s - buf_t0) * SR): int((s + HOP_SEC - buf_t0) * SR)]
                rms = float(np.sqrt(np.mean(seg**2))) if len(seg) else 0.0
                p = probs[j]
                rows.append(np.array([
                    s,
                    p[CLASSES["speech"]].max(),
                    p[CLASSES["singing"]].max(),
                    p[CLASSES["music"]].max(),
                    20 * np.log10(rms + 1e-9),
                ], dtype=np.float32))

            done = block_end
            if int(done) % 1800 < BLOCK_SEC:
                speed = done / max(1e-6, time.time() - started)
                print(f"[song_detect] {done / 3600:5.2f}시간 처리 · x{speed:.0f} 실시간", file=sys.stderr)

            t0 = block_end
            drop = int(max(0.0, (t0 - CTX_SEC) - buf_t0) * SR)
            buf, buf_t0 = buf[drop:], buf_t0 + drop / SR

    proc.wait()
    if not rows:
        raise RuntimeError(f"오디오를 읽지 못했습니다: {video}")
    arr = np.stack(rows)
    arr[:, 0] += start
    out = {"t": arr[:, 0], "speech": arr[:, 1], "singing": arr[:, 2], "music": arr[:, 3], "rms_db": arr[:, 4]}
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(out_npz)
    print(f"[song_detect] {len(arr)}초 분류 완료 ({time.time() - started:.0f}s)", file=sys.stderr)
    return out


# ---------- 규칙 단계 ----------

def _mode_filter(lab: np.ndarray, k: int) -> np.ndarray:
    """k초 창 다수결로 짧게 튀는 판정을 정리한다."""
    if k <= 1:
        return lab
    n, half = len(lab), k // 2
    onehot = np.zeros((n + 2 * half, len(LABELS)), dtype=np.int32)
    padded = np.concatenate([np.full(half, lab[0]), lab, np.full(half, lab[-1])])
    onehot[np.arange(len(padded)), padded] = 1
    csum = np.vstack([np.zeros((1, len(LABELS)), dtype=np.int32), np.cumsum(onehot, axis=0)])
    counts = csum[k:] - csum[:-k]
    out = counts.argmax(axis=1)[:n]
    # 동률이면 원래 값 유지
    keep = counts[np.arange(n), lab] == counts.max(axis=1)[:n]
    return np.where(keep, lab, out)


def _runs(values: np.ndarray, t: np.ndarray) -> list[tuple[int, float, float]]:
    """같은 값이 이어지는 구간들: (값, 시작, 끝)."""
    if len(values) == 0:
        return []
    cut = np.nonzero(np.diff(values))[0] + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(values)]
    return [(int(values[a]), float(t[a]), float(t[b - 1] + HOP_SEC)) for a, b in zip(starts, ends)]


def frame_labels(fp: dict[str, np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    """1초마다 speech/singing/music/silence 번호."""
    sp, si, mu, db = fp["speech"], fp["singing"], fp["music"], fp["rms_db"]
    lab = np.zeros(len(sp), dtype=np.int64)  # 기본 speech
    is_sil = db < cfg["silence_db"]
    # 노래(가창): 가창 확률이 충분하고 말소리보다 우세
    is_sing = (si >= cfg["singing_th"]) & (si >= sp * cfg["singing_over_speech"])
    # 반주·BGM만: 음악이 강하고 말소리가 약함
    is_music = (mu >= cfg["music_th"]) & (sp < cfg["speech_th"]) & ~is_sing
    lab[is_music] = LABELS.index("music")
    lab[is_sing] = LABELS.index("singing")
    lab[is_sil] = LABELS.index("silence")
    return _mode_filter(lab, int(cfg["smooth_sec"]))


def _intervals(mask: np.ndarray, t: np.ndarray, min_sec: float) -> list[dict[str, float]]:
    return [{"start": a, "end": b} for v, a, b in _runs(mask.astype(np.int64), t) if v and b - a >= min_sec]


def label_regions(fp: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, Any]:
    """구간 타임라인, 표시용 '노래 구간'(songs), STT에서 실제로 무음 처리할 구간(mute).

    songs: singing/music 판정이 song_min_sec 이상 이어진 곳. 사이에 낀 짧은 말·무음
    (song_gap_sec 이하)은 같은 노래로 본다. 그보다 짧은 노래·음악은 말 사이 효과음으로 보고 STT에 남긴다.
    mute: songs에서 말소리(speech ≥ keep_speech_th, 앞뒤 keep_pad_sec 여유)를 뺀 부분.
          곡 위에 말을 얹는 영상에서 그 말까지 지우지 않기 위해서다.
    """
    t = fp["t"]
    lab = frame_labels(fp, cfg)
    regions = [{"start": a, "end": b, "type": LABELS[v]} for v, a, b in _runs(lab, t)]

    songish = np.isin(lab, [LABELS.index("singing"), LABELS.index("music")]).astype(np.int64)
    songs: list[dict[str, Any]] = []
    for v, a, b in _runs(songish, t):
        if not v:
            continue
        if songs and a - songs[-1]["end"] <= cfg["song_gap_sec"]:
            songs[-1]["end"] = b
        else:
            songs.append({"start": a, "end": b})
    out_songs = []
    for s in songs:
        if s["end"] - s["start"] < cfg["song_min_sec"]:
            continue
        m = (t >= s["start"]) & (t < s["end"])
        sing_ratio = float(np.mean(fp["singing"][m] >= cfg["song_vocal_th"])) if m.any() else 0.0
        out_songs.append({**s, "type": "singing" if sing_ratio >= 0.3 else "music", "singing_ratio": round(sing_ratio, 2)})

    song_m = np.zeros(len(t), dtype=bool)
    for s in out_songs:
        song_m |= (t >= s["start"]) & (t < s["end"])
    pad = int(cfg.get("keep_pad_sec", 2))
    talk = fp["speech"] >= cfg.get("keep_speech_th", 0.3)
    if pad > 0:
        talk = np.convolve(talk.astype(np.int64), np.ones(2 * pad + 1, dtype=np.int64), "same") > 0
    mute = _intervals(song_m & ~talk, t, cfg.get("mute_min_sec", 5))

    total = float(t[-1] + HOP_SEC) if len(t) else 0.0
    share = {name: round(float(np.mean(lab == i)), 3) for i, name in enumerate(LABELS)} if len(lab) else {}
    return {"duration": total, "share": share, "regions": regions, "songs": out_songs, "mute": mute}


def detect_songs(video_path, cache_path=None, settings=None, force=False):
    """영상 전체를 분류해 label_regions 결과를 돌려준다.

    cache_path: 1초 확률 타임라인(.npz)을 저장할 곳. 같은 영상을 다시 돌릴 때 신경망 단계를 건너뛴다.
                None 이면 ~/.cache/mujica-addons/<파일 이름·크기>.npz
    settings:   DEFAULT_SETTINGS 중 바꿀 값만 넣는다.
    """
    video_path = Path(video_path)
    if cache_path is None:
        stat = video_path.stat()
        cache_path = Path.home() / ".cache" / "mujica-addons" / f"{video_path.stem}_{stat.st_size}.npz"
    fp = frame_probs(video_path, Path(cache_path), force=force)
    return label_regions(fp, {**DEFAULT_SETTINGS, **(settings or {})})
