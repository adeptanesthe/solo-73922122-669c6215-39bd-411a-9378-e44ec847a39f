"""FastAPI 入口：接收录音与参考标注，返回帧级分段结果。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from .audio import AudioError, load_wav
from .classify import (
    LABEL_SILENCE,
    classify_frames,
    estimate_noise,
    merge_segments,
    refs_to_frame_indices,
)
from .demo import demo_payload
from .features import FRAME_HOP, SAMPLE_RATE, cepstral_features, extract_features

MIN_REFS = 3
MAX_REFS = 8
MIN_REF_LEN = 0.3        # 单段参考最短时长（秒）
MIN_REF_FRAMES = 10      # 每位说话人参考中至少需要的有效语音帧数
MIN_SPEECH_FRAMES = 30   # 整条录音至少需要的有效语音帧数

SPEAKER_NAMES = {"A": "甲", "B": "乙"}

app = FastAPI(title="双人录音复听分段", version="1.0.0")


class RefError(ValueError):
    """参考标注不合法（越界 / 重叠 / 数量不足等）。"""


def _err(status: int, msg: str):
    raise HTTPException(status_code=status, detail=msg)


def parse_references(raw: str, duration: float) -> dict:
    """解析并校验参考标注：每人 3~8 段、段内 start<end、不越界、互不重叠。"""
    try:
        refs = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        raise RefError("参考标注不是合法的 JSON")
    if not isinstance(refs, dict):
        raise RefError("参考标注格式错误：应为 {\"A\": [[s,e],...], \"B\": [[s,e],...]}")

    cleaned = {}
    for key in ("A", "B"):
        name = SPEAKER_NAMES[key]
        segs = refs.get(key)
        if not isinstance(segs, list):
            raise RefError(f"缺少说话人{name}的参考段列表")
        if not (MIN_REFS <= len(segs) <= MAX_REFS):
            raise RefError(f"说话人{name}的参考段数量须为 {MIN_REFS}～{MAX_REFS} 段，当前 {len(segs)} 段")
        out = []
        for i, seg in enumerate(segs):
            if not (isinstance(seg, (list, tuple)) and len(seg) == 2):
                raise RefError(f"说话人{name}的第 {i + 1} 段格式错误，应为 [开始, 结束]")
            try:
                s, e = float(seg[0]), float(seg[1])
            except (TypeError, ValueError):
                raise RefError(f"说话人{name}的第 {i + 1} 段包含非数值时间")
            if not (np.isfinite(s) and np.isfinite(e)):
                raise RefError(f"说话人{name}的第 {i + 1} 段包含非法时间")
            if s < 0.0 or e > duration + 1e-3:
                raise RefError(
                    f"说话人{name}的第 {i + 1} 段 [{s:.2f}, {e:.2f}] 超出录音范围 0～{duration:.2f} 秒")
            if e - s < MIN_REF_LEN:
                raise RefError(
                    f"说话人{name}的第 {i + 1} 段时长 {e - s:.2f} 秒，短于最小要求 {MIN_REF_LEN} 秒")
            out.append([s, e])
        cleaned[key] = sorted(out)

    # 所有参考段（含甲乙之间）互不重叠
    flat = sorted((s, e, SPEAKER_NAMES[k]) for k in ("A", "B") for s, e in cleaned[k])
    for (s0, e0, n0), (s1, e1, n1) in zip(flat, flat[1:]):
        if s1 < e0 - 1e-6:
            raise RefError(
                f"参考段重叠：{n0} [{s0:.2f}, {e0:.2f}] 与 {n1} [{s1:.2f}, {e1:.2f}]")
    return cleaned


def _ref_advice(key: str, frame_count: int, ref_max_evidence: float, low_threshold: float) -> str:
    name = SPEAKER_NAMES[key]
    if ref_max_evidence <= low_threshold:
        return (f"说话人{name}的参考语音不足：所选片段里几乎只有静音或底噪，"
                f"没有可辨认的发言（有效语音 {frame_count} 帧，至少需 {MIN_REF_FRAMES} 帧）。"
                f"请改选能听到该说话人清晰发言的片段")
    return (f"说话人{name}的参考语音不足：参考段里的说话声太轻，与底噪难以区分"
            f"（有效语音仅 {frame_count} 帧，至少需 {MIN_REF_FRAMES} 帧）。"
            f"请改选说话声更清楚、音量更高的片段，或请该说话人靠近话筒重录")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo")
def demo():
    """返回一段合成双人录音（base64 WAV）及配套参考标注，用于网页演示。"""
    return demo_payload()


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), references: str = Form(...)):
    data = await file.read()
    try:
        y, sr, duration = load_wav(data)
    except AudioError as exc:
        _err(400, str(exc))

    try:
        refs = parse_references(references, duration)
    except RefError as exc:
        _err(400, str(exc))

    frames = extract_features(y)
    profile = estimate_noise(frames)
    times = frames.times
    speech_mask = profile.speech_mask

    if int(speech_mask.sum()) < MIN_SPEECH_FRAMES:
        _err(400, "录音中有效语音过少（几乎全为静音或只有底噪），无法分析；"
                  "请确认为 16 kHz 单声道 WAV，且录音里确实有人在说话")

    idx_a = refs_to_frame_indices(refs["A"], times, speech_mask)
    idx_b = refs_to_frame_indices(refs["B"], times, speech_mask)
    for key, idx in (("A", idx_a), ("B", idx_b)):
        if idx.size < MIN_REF_FRAMES:
            ref_ev = [float(profile.evidence[i]) for i in idx]
            ref_max = max(ref_ev) if ref_ev else 0.0
            _err(400, _ref_advice(key, int(idx.size), ref_max, profile.low_threshold))

    z = cepstral_features(frames.mel, profile.floor)
    labels, conf = classify_frames(z, idx_a, idx_b, profile)
    labels[~speech_mask] = LABEL_SILENCE
    conf = np.where(speech_mask, conf, 1.0)

    segments = merge_segments(labels, conf, times, duration)
    stats = {k: 0 for k in ("A", "B", "review", "silence")}
    for seg in segments:
        stats[seg["label"]] += seg["frames"]

    return {
        "duration": round(duration, 3),
        "sample_rate": SAMPLE_RATE,
        "frame_count": int(len(labels)),
        "frame_hop_ms": round(1000.0 * FRAME_HOP / SAMPLE_RATE, 1),
        "vad_threshold": float(profile.low_threshold),
        "segments": segments,
        "stats": stats,
    }


def _static_dir() -> str:
    env = os.environ.get("STATIC_DIR")
    candidates = [env] if env else []
    here = Path(__file__).resolve()
    candidates += [str(here.parents[1] / "frontend"), str(here.parents[2] / "frontend")]
    for cand in candidates:
        if cand and Path(cand).is_dir():
            return cand
    raise RuntimeError("未找到前端静态目录，请设置 STATIC_DIR")


app.mount("/", StaticFiles(directory=_static_dir(), html=True), name="static")
