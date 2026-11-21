"""底噪估计、按频带信噪比的语音检测、帧级归属与合并。

为什么不能再用“响度阈值”判静音：双人录音里声音较轻的一方，
其轻声可能只比纯底噪响一点点；响度阈值要么把轻声当底噪，
要么把底噪当人声。这里改用**逐频带**与底噪谱比较：

1. 底噪谱取全片最安静若干帧的逐频带中位数（迭代几次，避免把弱语音算进底噪）；
2. 每帧取信噪比最高的若干频带，计算超出底噪的平均对数幅度，
   记作“语音证据”；它衡量的是“有新的谱结构出现”，而不是整体变响；
3. 用低分位噪声帧的证据分布定阈值，并用双阈值 + 滞回：
   低阈值抓轻声，高阈值确认强语音，中间只在邻近强语音时保留，
   这样轻声不被漏掉、纯底噪与停顿也不会被连成语音。

说话人归属使用去掉了整体能量/谱倾斜的倒谱特征（见 features.cepstral_features）；
对证据较弱的帧，按帧证据相对参考证据的比例放宽“典型度”门槛，
使同一人参考以外的轻声也能归给本人，而真正不像两人的声音仍标为“待复核”。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import Frames

LABEL_A = "A"
LABEL_B = "B"
LABEL_REVIEW = "review"
LABEL_SILENCE = "silence"

REVIEW_CONF_THRESHOLD = 0.65   # 置信度低于该值 -> 待复核
EVIDENCE_TOP_BANDS = 6         # 每帧参与计算证据的频带数（取信噪比最高的频带）
SMOOTH_FRAMES = 9              # 证据沿时间平滑的帧数（约 90 ms）
NOISE_QUANTILE = 0.2           # 初估底噪时取最安静的 20% 帧
NOISE_ITERATIONS = 4
VAD_BRIDGE_FRAMES = 15         # 低阈值片段间隔不超过该值时视为同一段
VAD_MIN_FRAMES = 8             # 无强语音支撑时允许的最短低阈值段
LOW_THRESHOLD_SPREAD = 3.0     # 低阈值 = 噪声证据中位数 + 3 倍稳健标准差…
LOW_THRESHOLD_MIN = 0.30       # …且至少高出该值
HIGH_THRESHOLD_SPREAD = 6.0
HIGH_THRESHOLD_MIN = 0.50
NOISE_CEIL_MARGIN = 0.15       # 阈值至少高于噪声帧证据上限这么多
LOW_TYPICALITY_GAMMA = 8.0     # 证据越弱，典型度门槛放宽越多
TYPICALITY_WIDTH = 0.25


@dataclass
class NoiseProfile:
    """整条录音的底噪与语音证据。"""

    floor: np.ndarray          # (N_MELS,) 逐频带底噪功率
    evidence: np.ndarray       # (n_frames,) 语音证据（对数谱超出底噪的程度）
    base: float                # 噪声帧证据中位数
    spread: float              # 噪声帧证据稳健标准差
    noise_hi: float            # 噪声帧证据的 99.5 分位
    speech_mask: np.ndarray    # (n_frames,) 是否判为语音（非静音）
    low_threshold: float
    high_threshold: float


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """沿时间做滑动平均；两端用边缘值延拓，避免首尾证据被削弱。"""
    if win <= 1 or x.size == 0:
        return x
    win = min(win, x.size)
    pad = win // 2
    padded = np.pad(x, (pad, win - 1 - pad), mode="edge")
    return np.convolve(padded, np.ones(win) / win, mode="valid")


def _band_evidence(snr: np.ndarray) -> np.ndarray:
    """每帧最显著的若干频带的对数信噪比（只取正值）均值。"""
    top = np.sort(snr, axis=1)[:, -EVIDENCE_TOP_BANDS:]
    return np.mean(np.maximum(top, 0.0), axis=1)


def _iterative_floor(mel: np.ndarray, rms: np.ndarray, n_noise: int):
    """从最安静的帧出发，迭代剔除非噪声帧后重估逐频带底噪。"""
    quiet = np.argsort(rms)[:n_noise]
    floor = np.median(mel[quiet], axis=0)
    for _ in range(NOISE_ITERATIONS):
        ev = _smooth(_band_evidence(np.log(mel) - np.log(floor)), SMOOTH_FRAMES)
        quiet = np.argsort(ev)[:n_noise]
        floor = np.median(mel[quiet], axis=0)
    return floor, quiet


def _thresholds(base: float, spread: float, noise_hi: float):
    low = max(base + max(LOW_THRESHOLD_SPREAD * spread, LOW_THRESHOLD_MIN),
              noise_hi + NOISE_CEIL_MARGIN * 0.5)
    high = max(base + max(HIGH_THRESHOLD_SPREAD * spread, HIGH_THRESHOLD_MIN),
               noise_hi + NOISE_CEIL_MARGIN)
    low = min(low, high - 1e-6)
    return float(low), float(high)


def _hysteresis_mask(ev: np.ndarray, low: float, high: float) -> np.ndarray:
    """双阈值滞回：低阈值抓到轻声，但只有成段或邻近强语音时才保留。"""
    candidate = ev > low
    mask = np.zeros(ev.size, dtype=bool)
    idx = np.where(candidate)[0]
    if idx.size == 0:
        return mask

    segments = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= VAD_BRIDGE_FRAMES + 1:
            prev = i
        else:
            segments.append((start, prev))
            start = prev = i
    segments.append((start, prev))

    for s, e in segments:
        length = e - s + 1
        if length >= VAD_MIN_FRAMES or bool((ev[s:e + 1] > high).any()):
            mask[s:e + 1] = True
    return mask


def estimate_noise(frames: Frames) -> NoiseProfile:
    """估计逐频带底噪并给出语音掩码。纯底噪 / 静音输入会得到全 False 的掩码。"""
    mel, rms = frames.mel, frames.rms
    n = mel.shape[0]
    n_noise = max(10, min(int(NOISE_QUANTILE * n), n))

    floor, quiet = _iterative_floor(mel, rms, n_noise)
    ev = _smooth(_band_evidence(np.log(mel) - np.log(floor)), SMOOTH_FRAMES)

    # 证据最低的若干帧视为噪声参考（迭代后可能变化，重取一次）
    quiet = np.argsort(ev)[:n_noise]
    base = float(np.median(ev[quiet]))
    spread = float(1.4826 * np.median(np.abs(ev[quiet] - base))) + 1e-3
    noise_hi = float(np.percentile(ev[quiet], 99.5))

    low, high = _thresholds(base, spread, noise_hi)
    speech = _hysteresis_mask(ev, low, high)

    # 极端兜底：底噪估计失效（证据几乎无起伏）而录音本身有明显能量起伏时，
    # 退回能量判据，避免把整段连续说话误判为静音。
    if speech.sum() < max(10, int(0.005 * n)):
        p10, p95 = np.percentile(rms, [10, 95])
        if p95 > 4.0 * max(p10, 1e-6) and p95 > 1e-3:
            energy_thr = max(1e-4, min(4.0 * float(p10), 0.3 * float(p95)))
            speech = rms > energy_thr
            low = high = float(energy_thr)

    return NoiseProfile(
        floor=floor, evidence=ev, base=base, spread=spread, noise_hi=noise_hi,
        speech_mask=speech, low_threshold=low, high_threshold=high,
    )


def refs_to_frame_indices(segments, times: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """参考时间段 -> 掩码内的帧下标集合。"""
    hits = [np.where((times >= s) & (times <= e) & mask)[0] for s, e in segments]
    if not hits:
        return np.empty(0, dtype=int)
    return np.unique(np.concatenate(hits)).astype(int)


def _fit_gaussian(x: np.ndarray, global_var: np.ndarray):
    # 协方差向全局方差收缩，参考帧较少时不至于过窄
    var = np.maximum(x.var(axis=0), 0.3 * global_var) + 1e-4
    return x.mean(axis=0), var


def _log_likelihood(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(var).sum() + ((x - mu) ** 2 / var).sum(axis=1))


def classify_frames(features: np.ndarray, ref_idx_a: np.ndarray, ref_idx_b: np.ndarray,
                    profile: NoiseProfile):
    """对所有帧给出标签与置信度（静音帧由调用方覆盖）。

    置信度 = 区分度 × 典型度：
    - 区分度：两人模型的对数似然差，按参考帧的似然差尺度归一；
    - 典型度：获胜模型似然相对参考典型水平的位置；对语音证据较弱的帧
      （轻声），按“证据相对参考的比例”放宽门槛，避免把同一人的轻声
      当成异常而标为待复核；证据中等、又确实不像两人的帧仍会被判为待复核。
    返回 (labels, conf)，labels ∈ {A, B, review}。
    """
    ref_all = np.concatenate([ref_idx_a, ref_idx_b])

    mean = features[ref_all].mean(axis=0)
    std = features[ref_all].std(axis=0) + 1e-8
    z = (features - mean) / std
    global_var = z[ref_all].var(axis=0) + 1e-6

    mu_a, var_a = _fit_gaussian(z[ref_idx_a], global_var)
    mu_b, var_b = _fit_gaussian(z[ref_idx_b], global_var)

    ll_a = _log_likelihood(z, mu_a, var_a)
    ll_b = _log_likelihood(z, mu_b, var_b)
    margin = ll_a - ll_b

    ref_margin = np.concatenate([np.abs(margin[ref_idx_a]), np.abs(margin[ref_idx_b])])
    tau = max(1.0, 0.25 * float(np.median(ref_margin)))
    conf_margin = 1.0 / (1.0 + np.exp(-np.abs(margin) / tau))

    own_ll = np.concatenate([ll_a[ref_idx_a], ll_b[ref_idx_b]])
    p5 = float(np.percentile(own_ll, 5))
    p90 = float(np.percentile(own_ll, 90))
    scale = max(1.0, p90 - p5)

    ref_ev = float(np.median(profile.evidence[ref_all]))
    rel = np.clip((profile.evidence - profile.base) / max(ref_ev - profile.base, 1e-6), 0.0, 1.0)
    typical_floor = p5 - LOW_TYPICALITY_GAMMA * scale * (1.0 - rel)
    typicality = 1.0 / (1.0 + np.exp(
        -(np.maximum(ll_a, ll_b) - typical_floor) / (TYPICALITY_WIDTH * scale)))

    conf = conf_margin * typicality

    labels = np.where(margin >= 0.0, LABEL_A, LABEL_B).astype(object)
    labels[conf < REVIEW_CONF_THRESHOLD] = LABEL_REVIEW
    return labels, conf


def merge_segments(labels, conf, times, duration: float):
    """合并相邻且标签相同的帧，输出带起止时间与平均置信度的分段。

    段边界取相邻帧中心的中点，使全部分段恰好铺满整条时间轴。
    """
    from .features import FRAME_LEN, SAMPLE_RATE

    frame_half = 0.5 * FRAME_LEN / SAMPLE_RATE
    segments = []
    n = len(labels)
    start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            t0 = max(0.0, float(times[start]) - frame_half) if start == 0 \
                else (float(times[start - 1]) + float(times[start])) / 2.0
            t1 = min(duration, float(times[i - 1]) + frame_half) if i == n \
                else (float(times[i - 1]) + float(times[i])) / 2.0
            segments.append({
                "label": str(labels[start]),
                "start": round(t0, 3),
                "end": round(t1, 3),
                "confidence": round(float(np.mean(conf[start:i])), 4),
                "frames": i - start,
            })
            start = i
    return segments
