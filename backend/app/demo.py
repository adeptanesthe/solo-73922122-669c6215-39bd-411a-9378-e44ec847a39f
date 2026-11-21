"""演示音频合成：生成一段确定性的双人“类语音”样本，便于网页演示与自测。

两个合成说话人具有不同基频与共振峰结构（MFCC 可区分），
中间留有静音段；同时给出与合成时间轴匹配的参考标注。
"""
from __future__ import annotations

import base64
import io

import numpy as np
from scipy.io import wavfile

from .audio import SAMPLE_RATE

# (说话人, 开始, 结束) 秒
SPEECH_PLAN = [
    ("A", 0.5, 3.2),
    ("B", 3.8, 6.4),
    ("A", 7.0, 9.0),
    ("B", 9.6, 11.5),
]

VOICES = {
    "A": dict(f0=110.0, formants=[(500.0, 150.0), (1500.0, 200.0), (2500.0, 300.0)], seed=11),
    "B": dict(f0=190.0, formants=[(700.0, 150.0), (1200.0, 200.0), (2800.0, 300.0)], seed=22),
}

DEMO_REFERENCES = {
    "A": [[0.6, 1.3], [1.5, 2.2], [2.4, 3.1]],
    "B": [[3.9, 4.6], [4.8, 5.5], [5.7, 6.3]],
}

DEMO_DURATION = 12.0


def synth_voice(duration: float, f0: float, formants, seed: int,
                syll_period: float = 0.26, syll_dur: float = 0.17) -> np.ndarray:
    """合成一段带音节包络的谐波“语音”，频谱包络由共振峰决定。"""
    rng = np.random.default_rng(seed)
    n = int(round(duration * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE

    f0t = f0 * (1.0 + 0.008 * np.sin(2.0 * np.pi * 4.5 * t))  # 轻微颤音
    phase = 2.0 * np.pi * np.cumsum(f0t) / SAMPLE_RATE

    y = np.zeros(n)
    for k in range(1, 40):
        fk = k * f0
        if fk > SAMPLE_RATE / 2.0:
            break
        env = sum(np.exp(-0.5 * ((fk - fc) / bw) ** 2) for fc, bw in formants)
        y += (env / k) * np.sin(k * phase + rng.uniform(0.0, 2.0 * np.pi))

    # 音节包络：模拟说话的断续感；音节间保留气息底噪，回合间仍是真静音
    env = np.zeros(n)
    s = 0.02
    while s < duration:
        i0 = int(s * SAMPLE_RATE)
        i1 = min(n, int((s + syll_dur) * SAMPLE_RATE))
        if i1 > i0:
            env[i0:i1] += np.sin(np.pi * np.arange(i1 - i0) / (i1 - i0)) ** 2
        s += syll_period
    env = 0.12 + 0.88 * np.clip(env, 0.0, 1.0)
    y *= env

    y += rng.normal(0.0, 0.002, n)  # 气息噪声
    return y


def build_demo() -> np.ndarray:
    """生成完整演示录音（float32, 16 kHz 单声道）。"""
    y = np.zeros(int(DEMO_DURATION * SAMPLE_RATE))
    for spk, start, end in SPEECH_PLAN:
        v = synth_voice(end - start, **VOICES[spk])
        i0 = int(start * SAMPLE_RATE)
        y[i0:i0 + v.size] += v
    y += np.random.default_rng(0).normal(0.0, 0.0008, y.size)  # 房间底噪
    y = 0.6 * y / np.max(np.abs(y))
    return y.astype(np.float32)


def to_wav_bytes(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16))
    return buf.getvalue()


_cache: dict = {}


def demo_payload() -> dict:
    """/api/demo 的响应内容（带缓存，确定性生成）。"""
    if "payload" not in _cache:
        y = build_demo()
        _cache["payload"] = {
            "audio_base64": base64.b64encode(to_wav_bytes(y)).decode("ascii"),
            "duration": DEMO_DURATION,
            "references": DEMO_REFERENCES,
        }
    return _cache["payload"]
