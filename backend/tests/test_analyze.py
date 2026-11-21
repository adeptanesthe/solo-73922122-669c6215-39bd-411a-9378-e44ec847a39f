"""自测：正常 / 静音 / 低置信输入，以及各类非法输入的拒绝。

运行：python -m pytest backend/tests -v
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from app.audio import SAMPLE_RATE
from app.demo import DEMO_REFERENCES, SPEECH_PLAN, build_demo, synth_voice, to_wav_bytes
from app.main import app

client = TestClient(app)


# ---------- 工具 ----------

def wav_bytes(y: np.ndarray, sr: int = SAMPLE_RATE, stereo: bool = False) -> bytes:
    data = np.clip(y, -1.0, 1.0)
    if stereo:
        data = np.stack([data, data], axis=1)
    buf = io.BytesIO()
    wavfile.write(buf, sr, (data * 32767.0).astype(np.int16))
    return buf.getvalue()


def analyze(y: np.ndarray, refs: dict, sr: int = SAMPLE_RATE, raw: bytes | None = None):
    files = {"file": ("test.wav", raw if raw is not None else wav_bytes(y, sr), "audio/wav")}
    data = {"references": json.dumps(refs)}
    return client.post("/api/analyze", files=files, data=data)


def label_coverage(segments, t0: float, t1: float) -> dict:
    """统计 [t0,t1] 区间内各标签覆盖的时长比例。"""
    cov: dict[str, float] = {}
    for seg in segments:
        ov = max(0.0, min(seg["end"], t1) - max(seg["start"], t0))
        cov[seg["label"]] = cov.get(seg["label"], 0.0) + ov
    total = t1 - t0
    return {k: v / total for k, v in cov.items()}


# ---------- 正常输入 ----------

def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_demo_endpoint():
    body = client.get("/api/demo").json()
    assert body["duration"] == pytest.approx(12.0)
    assert len(body["references"]["A"]) == 3
    assert len(base64.b64decode(body["audio_base64"])) > 10000


def test_analyze_normal_two_speakers():
    y = build_demo()
    resp = analyze(y, DEMO_REFERENCES)
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    segs = body["segments"]

    # 分段应覆盖整条时间轴且首尾相接（相邻同标签帧已合并）
    assert segs[0]["start"] == pytest.approx(0.0, abs=0.02)
    assert segs[-1]["end"] == pytest.approx(body["duration"], abs=0.02)
    for s0, s1 in zip(segs, segs[1:]):
        assert s1["start"] == pytest.approx(s0["end"], abs=1e-6)
        assert s0["label"] != s1["label"]  # 相邻段标签必不同

    # 每个说话时段内，非静音帧的归属应以正确标签为主、另一人标签极少
    for spk, t0, t1 in SPEECH_PLAN:
        cov = label_coverage(segs, t0 + 0.1, t1 - 0.1)
        other = "B" if spk == "A" else "A"
        voiced = cov.get(spk, 0.0) + cov.get(other, 0.0) + cov.get("review", 0.0)
        assert voiced > 0.3, f"{spk} 段语音未被检出: {cov}"
        assert cov.get(spk, 0.0) / voiced >= 0.85, f"{spk} 段归属正确率不足: {cov}"
        assert cov.get(other, 0.0) <= 0.05, f"{spk} 段误判为对方过多: {cov}"

    # 语音之间的空隙应判为静音
    cov_gap = label_coverage(segs, 3.4, 3.7)
    assert cov_gap.get("silence", 0.0) >= 0.8

    # 演示音频界限清晰，待复核帧应占少数
    total = sum(body["stats"].values())
    assert body["stats"]["review"] / total < 0.3


def test_analyze_with_uploaded_demo_wav():
    """走一遍和前端一致的流程：/api/demo 取音频 -> 上传 /api/analyze。"""
    demo = client.get("/api/demo").json()
    raw = base64.b64decode(demo["audio_base64"])
    files = {"file": ("demo.wav", raw, "audio/wav")}
    resp = client.post("/api/analyze", files=files,
                       data={"references": json.dumps(demo["references"])})
    assert resp.status_code == 200, resp.json()
    assert resp.json()["stats"]["A"] > 0
    assert resp.json()["stats"]["B"] > 0


# ---------- 静音输入 ----------

def test_all_silence_rejected():
    y = np.zeros(int(5 * SAMPLE_RATE), dtype=np.float32)
    refs = {"A": [[0.5, 1.1], [1.2, 1.8], [1.9, 2.5]],
            "B": [[2.6, 3.2], [3.3, 3.9], [4.0, 4.6]]}
    resp = analyze(y, refs)
    assert resp.status_code == 400
    assert "静音" in resp.json()["detail"] or "有效语音" in resp.json()["detail"]


def test_silence_tail_labeled():
    """语音之后的长静音尾巴应被标为静音段。"""
    y = build_demo()[: int(9.0 * SAMPLE_RATE)]
    tail = np.zeros(int(2.5 * SAMPLE_RATE), dtype=np.float32)
    y2 = np.concatenate([y, tail])
    refs = {"A": DEMO_REFERENCES["A"],
            "B": [[3.9, 4.6], [4.8, 5.5], [5.7, 6.3]]}
    resp = analyze(y2, refs)
    assert resp.status_code == 200, resp.json()
    cov = label_coverage(resp.json()["segments"], 9.6, 11.4)
    assert cov.get("silence", 0.0) >= 0.95


# ---------- 低置信输入 ----------

def test_unknown_speaker_marked_review():
    """第三说话人（与甲乙参考都不像）应产生待复核帧。"""
    total = 9.5
    y = np.zeros(int(total * SAMPLE_RATE), dtype=np.float32)
    va = synth_voice(2.5, 110.0, [(500.0, 150.0), (1500.0, 200.0), (2500.0, 300.0)], seed=1)
    vb = synth_voice(2.5, 190.0, [(700.0, 150.0), (1200.0, 200.0), (2800.0, 300.0)], seed=2)
    vc = synth_voice(2.5, 150.0, [(900.0, 120.0), (1900.0, 150.0), (3100.0, 200.0)], seed=3)
    y[int(0.5 * SAMPLE_RATE):int(0.5 * SAMPLE_RATE) + va.size] += va
    y[int(3.5 * SAMPLE_RATE):int(3.5 * SAMPLE_RATE) + vb.size] += vb
    y[int(6.5 * SAMPLE_RATE):int(6.5 * SAMPLE_RATE) + vc.size] += vc
    y = 0.6 * y / np.max(np.abs(y))

    refs = {"A": [[0.6, 1.3], [1.5, 2.2], [2.3, 2.9]],
            "B": [[3.6, 4.3], [4.5, 5.2], [5.3, 5.9]]}
    resp = analyze(y, refs)
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["stats"]["review"] > 0, "应出现待复核帧"
    cov = label_coverage(body["segments"], 6.6, 8.9)
    assert cov.get("review", 0.0) >= 0.2, f"第三说话人区域待复核覆盖不足: {cov}"


def test_mixed_voices_low_confidence():
    """甲乙声音等权叠加的区域与两人参考都接近，应出现待复核帧。"""
    total = 9.5
    y = np.zeros(int(total * SAMPLE_RATE), dtype=np.float32)
    va = synth_voice(2.5, 110.0, [(500.0, 150.0), (1500.0, 200.0), (2500.0, 300.0)], seed=4)
    vb = synth_voice(2.5, 190.0, [(700.0, 150.0), (1200.0, 200.0), (2800.0, 300.0)], seed=5)
    y[int(0.5 * SAMPLE_RATE):int(0.5 * SAMPLE_RATE) + va.size] += va
    y[int(3.5 * SAMPLE_RATE):int(3.5 * SAMPLE_RATE) + vb.size] += vb
    mix = 0.5 * va + 0.5 * vb
    y[int(6.5 * SAMPLE_RATE):int(6.5 * SAMPLE_RATE) + mix.size] += mix
    y = 0.6 * y / np.max(np.abs(y))

    refs = {"A": [[0.6, 1.3], [1.5, 2.2], [2.3, 2.9]],
            "B": [[3.6, 4.3], [4.5, 5.2], [5.3, 5.9]]}
    resp = analyze(y, refs)
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert body["stats"]["review"] > 0, "混音区域应出现待复核帧"


# ---------- 非法输入的拒绝 ----------

GOOD_REFS = {"A": [[0.6, 1.3], [1.5, 2.2], [2.4, 3.1]],
             "B": [[3.9, 4.6], [4.8, 5.5], [5.7, 6.3]]}


def test_reject_overlapping_refs():
    refs = {"A": [[0.6, 1.3], [1.5, 2.2], [2.4, 3.1]],
            "B": [[3.0, 4.0], [4.8, 5.5], [5.7, 6.3]]}  # 与 A 第三段重叠
    resp = analyze(build_demo(), refs)
    assert resp.status_code == 400
    assert "重叠" in resp.json()["detail"]


def test_reject_too_few_and_too_many_refs():
    y = build_demo()
    few = {"A": [[0.6, 1.3], [1.5, 2.2]], "B": GOOD_REFS["B"]}
    resp = analyze(y, few)
    assert resp.status_code == 400 and "3～8" in resp.json()["detail"]

    many = {"A": [[0.6 + 0.3 * i, 0.85 + 0.3 * i] for i in range(9)],
            "B": GOOD_REFS["B"]}
    resp = analyze(y, many)
    assert resp.status_code == 400 and "3～8" in resp.json()["detail"]


def test_reject_out_of_bounds_refs():
    refs = {"A": [[0.6, 1.3], [1.5, 2.2], [10.0, 13.0]],  # 超出 12s 时长
            "B": GOOD_REFS["B"]}
    resp = analyze(build_demo(), refs)
    assert resp.status_code == 400
    assert "超出" in resp.json()["detail"]


def test_reject_bad_audio():
    y = build_demo()
    assert analyze(y, GOOD_REFS, sr=8000).status_code == 400          # 采样率不符
    stereo_raw = wav_bytes(y, stereo=True)
    assert analyze(y, GOOD_REFS, raw=stereo_raw).status_code == 400   # 非单声道
    long_raw = wav_bytes(np.zeros(int(31 * SAMPLE_RATE), dtype=np.float32))
    assert analyze(y, GOOD_REFS, raw=long_raw).status_code == 400     # 超过 30 秒
    assert analyze(y, GOOD_REFS, raw=b"not a wav").status_code == 400  # 非 WAV


def test_reject_bad_references_json():
    files = {"file": ("t.wav", to_wav_bytes(build_demo()), "audio/wav")}
    resp = client.post("/api/analyze", files=files, data={"references": "not json"})
    assert resp.status_code == 400
