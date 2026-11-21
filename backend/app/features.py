"""短时声学特征提取（仅使用 NumPy / SciPy）。

流程：预加重 -> 分帧(25ms 窗 / 10ms 帧移) -> 加窗 -> 功率谱
-> Mel 滤波器组，得到每帧 26 个频带的能量，作为后续
静音检测（按频带比较底噪）与说话人建模（倒谱特征）的共同输入。
"""
from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class Frames:
    """整条录音的分帧结果。"""

    mel: np.ndarray      # (n_frames, N_MELS) 各频带能量
    rms: np.ndarray      # (n_frames,) 短时 RMS，仅用于底噪初筛与展示
    times: np.ndarray    # (n_frames,) 帧中心时间（秒）


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


_FILTERBANK = mel_filterbank()


def extract_features(y: np.ndarray) -> Frames:
    """提取每帧 Mel 频带能量与 RMS。

    这里刻意不保留对数总能量：本项目的难点是“轻声与底噪响度接近”，
    总能量会把两者混为一谈；后续静音检测用逐频带信噪比，
    说话人建模用去掉能量项的倒谱（见 :func:`cepstral_features`）。
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        y = np.zeros(FRAME_LEN)

    emphasized = np.append(y[0], y[1:] - 0.97 * y[:-1])
    frames = frame_signal(emphasized)
    raw_frames = frame_signal(y)

    rms = np.sqrt(np.mean(raw_frames ** 2, axis=1) + 1e-12)

    window = np.hamming(frames.shape[1])
    power = np.abs(np.fft.rfft(frames * window, N_FFT, axis=1)) ** 2 / N_FFT
    mel = np.maximum(power @ _FILTERBANK.T, 1e-12)

    return Frames(mel=mel, rms=rms, times=frame_times(mel.shape[0]))


def _smooth_frames(x: np.ndarray, win: int) -> np.ndarray:
    """沿时间做滑动平均；两端用边缘值延拓，避免首尾帧被削弱。"""
    if win <= 1 or x.shape[0] == 0:
        return x
    win = min(win, x.shape[0])
    pad = win // 2
    padded = np.pad(x, ((pad, win - 1 - pad), (0, 0)), mode="edge")
    kernel = np.ones(win) / win
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid")
                     for j in range(x.shape[1])], axis=1)


def cepstral_features(mel: np.ndarray, noise_floor: np.ndarray,
                      alpha: float = 0.7, smooth_frames: int = 3) -> np.ndarray:
    """由 Mel 能量得到与响度基本无关的倒谱特征。

    1. 逐频带减去底噪（谱减），残差下限取底噪的 5%，避免出现零能量；
    2. DCT 后丢掉第 0 维（整体响度/增益），只保留 1～12 阶倒谱；
    3. 每帧减去倒谱均值（消去录音链路带来的整体谱倾斜）；
    4. 沿时间做 3 帧平滑，压低宽带噪声对单帧特征的扰动。

    第 1、2 步保证轻声与底噪接近时仍有可比较的谱形，
    第 3 步进一步消除“参考较响、别处较轻”带来的增益差异。
    """
    residual = np.maximum(mel - alpha * noise_floor, 0.05 * noise_floor)
    mfcc = dct(np.log(residual), type=2, axis=1, norm="ortho")[:, 1:N_MFCC]
    mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)
    return _smooth_frames(mfcc, smooth_frames)
