"""帧级归属判断、置信度、待复核标记与相邻同标签帧合并。

模型：对每位说话人，汇总其全部参考片段内的非静音帧，
在标准化后的特征空间上拟合对角协方差高斯模型；
非静音帧按两个模型的对数似然比较归属，
似然差过小（与两人参考相似度接近）或对两个模型都不像的帧标为“待复核”。
"""
from __future__ import annotations

import numpy as np

from .features import FRAME_HOP, FRAME_LEN, SAMPLE_RATE

LABEL_A = "A"
LABEL_B = "B"
LABEL_REVIEW = "review"
LABEL_SILENCE = "silence"

REVIEW_CONF_THRESHOLD = 0.65   # 置信度低于该值 -> 待复核
OUTLIER_MARGIN = 3.0           # 对数似然低于参考典型值的偏移量（nats）


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def vad_mask(rms: np.ndarray):
    """自适应能量静音检测。返回 (非静音布尔掩码, 阈值)。"""
    p10 = float(np.percentile(rms, 10))
    p95 = float(np.percentile(rms, 95))
    # 噪声底的 4 倍，但不超过响亮段的 30%；绝对下限防止全静音误判
    thr = max(1e-4, min(4.0 * p10, 0.3 * p95))
    return rms > thr, thr


def _fit_gaussian(x: np.ndarray):
    mu = x.mean(axis=0)
    var = x.var(axis=0) + 1e-6
    return mu, var


def _log_likelihood(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(var).sum() + ((x - mu) ** 2 / var).sum(axis=1))


def classify_frames(features: np.ndarray, ref_idx_a: np.ndarray, ref_idx_b: np.ndarray):
    """对所有帧给出标签与置信度（静音帧由调用方另行覆盖）。

    置信度 = 两人间区分度 × 与参考语音的典型度，二者任一不足都会拉低置信度；
    置信度低于阈值（与两人参考相似度接近，或与两人都不像）的帧标为 review。
    返回 (labels, conf)，labels ∈ {A, B, review}。
    """
    ref_all = np.concatenate([ref_idx_a, ref_idx_b])

    # 用全部参考帧做标准化，保证两类模型在同一尺度上比较
    mean = features[ref_all].mean(axis=0)
    std = features[ref_all].std(axis=0) + 1e-8
    z = (features - mean) / std

    mu_a, var_a = _fit_gaussian(z[ref_idx_a])
    mu_b, var_b = _fit_gaussian(z[ref_idx_b])

    ll_a = _log_likelihood(z, mu_a, var_a)
    ll_b = _log_likelihood(z, mu_b, var_b)
    margin = ll_a - ll_b

    # 区分度：似然差经参考帧自适应尺度转换
    tau = max(1.0, 0.25 * float(np.median(np.abs(margin[ref_all]))))
    conf_margin = _sigmoid(np.abs(margin) / tau)

    # 典型度：获胜模型的对数似然相对参考典型水平的位置
    own_ll = np.concatenate([ll_a[ref_idx_a], ll_b[ref_idx_b]])
    ll_floor = float(np.percentile(own_ll, 5)) - OUTLIER_MARGIN
    typicality = _sigmoid(np.maximum(ll_a, ll_b) - ll_floor)

    conf = conf_margin * typicality

    labels = np.where(margin >= 0.0, LABEL_A, LABEL_B).astype(object)
    labels[conf < REVIEW_CONF_THRESHOLD] = LABEL_REVIEW
    return labels, conf


def merge_segments(labels, conf, times, duration: float):
    """合并相邻且标签相同的帧，输出带起止时间与平均置信度的分段。

    段边界取相邻帧中心的中点，使全部分段恰好铺满整条时间轴。
    """
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
