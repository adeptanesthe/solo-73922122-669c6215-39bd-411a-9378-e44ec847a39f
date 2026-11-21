"""场景合成工具：供自测构造“底噪 + 轻声”等难以用演示音频覆盖的录音。

所有函数都是确定性的：同样的参数得到逐采样相同的波形，便于回归测试。
"""
from __future__ import annotations

import numpy as np

from app.demo import VOICES, synth_voice
from app.features import SAMPLE_RATE

THIRD_VOICE = dict(f0=150.0, formants=[(900.0, 120.0), (1900.0, 150.0), (3100.0, 200.0)])


def voice(name: str, duration: float, seed_shift: int = 0) -> np.ndarray:
    """按演示音色库合成一段“类语音”（name ∈ {A, B}）。"""
    v = VOICES[name]
    return synth_voice(duration, v["f0"], v["formants"], v["seed"] + seed_shift)


def third_voice(duration: float, seed: int) -> np.ndarray:
    return synth_voice(duration, THIRD_VOICE["f0"], THIRD_VOICE["formants"], seed)


class Track:
    """在时间轴上逐段叠加语音与底噪。"""

    def __init__(self, duration: float, noise: float = 0.002, seed: int = 0):
        self.duration = duration
        self.y = np.zeros(int(round(duration * SAMPLE_RATE)))
        self.noise = noise
        self.seed = seed

    def add(self, wave: np.ndarray, start: float, gain: float = 1.0) -> "Track":
        i = int(round(start * SAMPLE_RATE))
        self.y[i:i + wave.size] += gain * wave[: max(0, self.y.size - i)]
        return self

    def build(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return (self.y + rng.normal(0.0, self.noise, self.y.size)).astype(np.float32)


def _place(track: Track, who: str, spans, gain: float, seed_shift: int = 0):
    for k, (start, duration) in enumerate(spans):
        track.add(voice(who, duration, seed_shift + k), start, gain)
    return track


def soft_reference_case(noise: float = 0.0015, soft_gain: float = 0.012) -> np.ndarray:
    """场景一：轻声说话人乙的**参考本身**就是很轻但清楚的语音。

    甲较响；乙的参考和参考以外的发言都明显轻于甲，只比底噪高一点。
    """
    t = Track(10.0, noise=noise, seed=5)
    _place(t, "A", [(0.4, 1.2), (1.7, 1.2)], gain=0.35)
    _place(t, "B", [(3.8, 0.8), (4.8, 0.8), (5.8, 0.8)], gain=soft_gain)
    _place(t, "B", [(7.0, 1.0), (8.2, 0.9)], gain=soft_gain, seed_shift=4)
    return t.build()


SOFT_REFERENCE_REFS = {
    "A": [[0.5, 1.0], [1.05, 1.5], [1.8, 2.6]],
    "B": [[3.9, 4.6], [4.9, 5.6], [5.9, 6.55]],
}
SOFT_REFERENCE_SPEECH = [("A", 0.5, 2.9), ("B", 3.9, 6.5), ("B", 7.0, 9.2)]
SOFT_REFERENCE_SILENCE = [(0.0, 0.35), (3.05, 3.7), (9.45, 10.0)]


def loud_reference_soft_elsewhere_case(noise: float = 0.002, soft_gain: float = 0.018) -> np.ndarray:
    """场景二：乙的参考选在较响处，乙在参考以外的轻声仍应归给乙。

    这正是“参考选在较响处，同一个人参考以外的轻声被标成静音”的复现。
    """
    t = Track(12.0, noise=noise, seed=7)
    _place(t, "A", [(0.4, 1.3), (2.0, 1.3), (10.0, 1.0)], gain=0.30)
    _place(t, "B", [(3.8, 0.9), (4.9, 0.9), (6.0, 0.9)], gain=0.12)
    _place(t, "B", [(7.4, 1.0), (8.7, 0.9)], gain=soft_gain, seed_shift=5)
    return t.build()


LOUD_REFERENCE_SOFT_REFS = {
    "A": [[0.5, 1.6], [2.1, 3.2], [10.1, 10.9]],
    "B": [[3.9, 4.7], [5.0, 5.8], [6.1, 6.85]],
}
LOUD_REFERENCE_SOFT_SPEECH = [("A", 0.5, 3.2), ("B", 3.9, 6.85), ("B", 7.5, 9.9)]
LOUD_REFERENCE_SOFT_SILENCE = [(0.0, 0.35), (3.4, 3.7), (7.05, 7.35), (11.0, 12.0)]


def third_speaker_and_mix_case(noise: float = 0.002) -> np.ndarray:
    """甲乙 + 第三说话人 + 甲乙等权混音，用于确认“归属不明 -> 待复核”。"""
    t = Track(12.0, noise=noise, seed=3)
    t.add(voice("A", 2.5), 0.5, 0.30)
    t.add(voice("B", 2.5, seed_shift=3), 3.5, 0.25)
    t.add(third_voice(2.5, seed=3), 6.5, 0.30)
    mixed = 0.5 * voice("A", 2.5, seed_shift=7) + 0.5 * voice("B", 2.5, seed_shift=9)
    t.add(mixed, 9.5, 0.30)
    return t.build()


THIRD_MIX_REFS = {
    "A": [[0.6, 1.2], [1.3, 2.0], [2.1, 2.8]],
    "B": [[3.6, 4.2], [4.3, 5.0], [5.1, 6.0]],
}
