"""WAV 读取与输入校验。"""
from __future__ import annotations

import io

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 16000
MAX_DURATION = 30.0          # 秒
MIN_DURATION = 0.5           # 秒
MAX_FILE_BYTES = 8 * 1024 * 1024  # 30s/16k/16bit 约 960KB，留足余量


class AudioError(ValueError):
    """输入音频不满足约束（采样率 / 声道 / 时长 / 格式）。"""


def load_wav(data: bytes):
    """解析并校验 WAV 字节流，返回 (float32 单声道信号, 采样率, 时长秒)。

    约束：16 kHz、单声道、时长 <= 30 秒。任何不满足都抛 AudioError。
    """
    if not data:
        raise AudioError("未收到音频数据")
    if len(data) > MAX_FILE_BYTES:
        raise AudioError("文件过大（超过 8 MB），请上传不超过 30 秒的录音")
    try:
        sr, y = wavfile.read(io.BytesIO(data))
    except Exception:
        raise AudioError("无法解析音频文件，请上传标准 PCM WAV")

    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y[:, 0]
        else:
            raise AudioError(f"仅支持单声道 WAV，当前为 {y.shape[1]} 声道")
    if sr != SAMPLE_RATE:
        raise AudioError(f"采样率须为 16000 Hz，当前为 {sr} Hz")

    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    elif y.dtype == np.uint8:
        y = (y.astype(np.float32) - 128.0) / 128.0
    elif y.dtype in (np.float32, np.float64):
        y = y.astype(np.float32)
    else:
        raise AudioError(f"不支持的采样格式：{y.dtype}")

    duration = y.size / float(sr)
    if duration > MAX_DURATION + 1e-3:
        raise AudioError(f"录音时长须不超过 {MAX_DURATION:.0f} 秒，当前 {duration:.2f} 秒")
    if duration < MIN_DURATION:
        raise AudioError(f"录音太短（{duration:.2f} 秒），无法分析")
    if not np.all(np.isfinite(y)):
        raise AudioError("音频包含非法采样值")
    return y, sr, duration
