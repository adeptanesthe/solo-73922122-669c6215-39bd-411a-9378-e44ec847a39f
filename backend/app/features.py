"""短时声学特征提取（仅使用 NumPy / SciPy）。

流程：预加重 -> 分帧(25ms 窗 / 10ms 帧移) -> 加窗 -> 功率谱
-> Mel 滤波器组 -> log -> DCT 得到 MFCC -> 拼接一阶差分与对数能量。
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dct

SAMPLE_RATE = 16000
FRAME_LEN = 400          # 25 ms @ 16 kHz
FRAME_HOP = 160          # 10 ms
N_FFT = 512
N_MELS = 26
N_MFCC = 13
FMIN = 20.0
FMAX = SAMPLE_RATE / 2.0


def frame_signal(y: np.ndarray, frame_len: int = FRAME_LEN, hop: int = FRAME_HOP) -> np.ndarray:
    """把一维信号切成 (n_frames, frame_len) 的重叠帧。"""
    y = np.asarray(y, dtype=np.float64)
    if y.size < frame_len:
        y = np.pad(y, (0, frame_len - y.size))
    n_frames = 1 + (y.size - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return y[idx]


def frame_times(n_frames: int, hop: int = FRAME_HOP, frame_len: int = FRAME_LEN,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """每帧中心对应的秒数。"""
    return (np.arange(n_frames) * hop + frame_len / 2.0) / sr


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """三角 Mel 滤波器组，形状 (n_mels, n_fft//2+1)。"""
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    mels = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), n_mels + 2)
    hz = _mel_to_hz(mels)
    fb = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        left, center, right = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - left) / max(center - left, 1e-9)
        down = (right - freqs) / max(right - center, 1e-9)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE):
    """提取短时特征。

    返回 (features, rms, times)：
    - features: (n_frames, 2*N_MFCC + 1)，MFCC + 一阶差分 + 对数能量
    - rms:      (n_frames,)，每帧短时 RMS（用于静音检测）
    - times:    (n_frames,)，每帧中心时间（秒）
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        y = np.zeros(FRAME_LEN)

    emphasized = np.append(y[0], y[1:] - 0.97 * y[:-1])
    frames = frame_signal(emphasized)
    raw_frames = frame_signal(y)

    rms = np.sqrt(np.mean(raw_frames ** 2, axis=1) + 1e-12)
    log_energy = np.log(np.mean(raw_frames ** 2, axis=1) + 1e-12)

    window = np.hamming(frames.shape[1])
    power = np.abs(np.fft.rfft(frames * window, N_FFT, axis=1)) ** 2 / N_FFT
    mel = power @ mel_filterbank(sr).T
    mel = np.maximum(mel, 1e-12)
    mfcc = dct(np.log(mel), type=2, axis=1, norm="ortho")[:, :N_MFCC]

    # 一阶差分（边缘用一阶差代替）
    delta = np.zeros_like(mfcc)
    n = mfcc.shape[0]
    if n >= 3:
        delta[1:-1] = (mfcc[2:] - mfcc[:-2]) / 2.0
        delta[0] = mfcc[1] - mfcc[0]
        delta[-1] = mfcc[-1] - mfcc[-2]
    elif n == 2:
        delta[0] = delta[1] = (mfcc[1] - mfcc[0]) / 2.0

    features = np.concatenate([mfcc, delta, log_energy[:, None]], axis=1)
    return features, rms, frame_times(n)
