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
import scenarios as S

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


# ---------- 轻声场景 ----------

def syllable_truth(who: str, duration: float, seed_shift: int, start: float):
    """基于合成语音自身的音节包络给出“确实在发声”的时间点（相对录音起点）。"""
    wave = S.voice(who, duration, seed_shift)
    win, hop = 400, 160
    n = 1 + (wave.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(wave[idx] ** 2, axis=1))
    rel = rms / max(float(rms.max()), 1e-9)
    times = start + (np.arange(n) * hop + win / 2.0) / SAMPLE_RATE
    return times[rel > 0.30]


def label_at(segments, t: float) -> str | None:
    for seg in segments:
        if seg["start"] <= t < seg["end"]:
            return seg["label"]
    return None


def voice_stats(segments, truth: list[tuple[float, str]]) -> dict:
    counts = {"correct": 0, "silence": 0, "review": 0, "wrong": 0}
    for t, who in truth:
        got = label_at(segments, t)
        if got == who:
            counts["correct"] += 1
        elif got == "silence":
            counts["silence"] += 1
        elif got == "review":
            counts["review"] += 1
        else:
            counts["wrong"] += 1
    total = max(1, len(truth))
    return {k: v / total for k, v in counts.items()}


def test_soft_speaker_reference_builds_model():
    """场景一：轻声说话人的参考本身很轻，但参考仍须参与建模，分析能完成。"""
    y = S.soft_reference_case()
    resp = analyze(y, S.SOFT_REFERENCE_REFS)
    assert resp.status_code == 200, resp.json()
    segs = resp.json()["segments"]

    truth = []
    for who, duration, shift, start in [
        ("A", 1.2, 0, 0.4), ("A", 1.2, 0, 1.7),
        ("B", 0.8, 0, 3.8), ("B", 0.8, 1, 4.8), ("B", 0.8, 2, 5.8),
        ("B", 1.0, 4, 7.0), ("B", 0.9, 5, 8.2),
    ]:
        truth += [(t, who) for t in syllable_truth(who, duration, shift, start)]

    stats = voice_stats(segs, truth)
    assert stats["silence"] <= 0.05, f"轻声被误标为静音过多: {stats}"
    assert stats["wrong"] <= 0.05, f"轻声被误判给他人过多: {stats}"
    assert stats["correct"] >= 0.80, f"轻声归属正确率不足: {stats}"


def test_loud_reference_soft_speech_elsewhere_attributed():
    """场景二：乙的参考在较响处；参考以外乙的轻声仍应归给乙，而不是静音。"""
    y = S.loud_reference_soft_elsewhere_case()
    resp = analyze(y, S.LOUD_REFERENCE_SOFT_REFS)
    assert resp.status_code == 200, resp.json()
    segs = resp.json()["segments"]

    truth = []
    for who, duration, shift, start in [
        ("A", 1.3, 0, 0.4), ("A", 1.3, 0, 2.0), ("A", 1.0, 0, 10.0),
        ("B", 0.9, 0, 3.8), ("B", 0.9, 1, 4.9), ("B", 0.9, 2, 6.0),
        ("B", 1.0, 5, 7.4), ("B", 0.9, 6, 8.7),
    ]:
        truth += [(t, who) for t in syllable_truth(who, duration, shift, start)]

    stats = voice_stats(segs, truth)
    assert stats["silence"] <= 0.05, f"参考以外的轻声被标成静音: {stats}"
    assert stats["wrong"] <= 0.05, f"轻声被误判给他人: {stats}"
    assert stats["correct"] >= 0.80, f"参考以外的轻声归属正确率不足: {stats}"


def test_pauses_and_noise_are_silence_soft_speaker():
    """停顿与纯底噪必须标为静音，不能因为轻声判定放宽而被连成语音。"""
    for y, refs, silence_spans in [
        (S.soft_reference_case(), S.SOFT_REFERENCE_REFS, S.SOFT_REFERENCE_SILENCE),
        (S.loud_reference_soft_elsewhere_case(), S.LOUD_REFERENCE_SOFT_REFS,
         S.LOUD_REFERENCE_SOFT_SILENCE),
    ]:
        resp = analyze(y, refs)
        assert resp.status_code == 200, resp.json()
        segs = resp.json()["segments"]
        for t0, t1 in silence_spans:
            cov = label_coverage(segs, t0, t1)
            assert cov.get("silence", 0.0) >= 0.90, f"[{t0}, {t1}] 未判为静音: {cov}"


def test_noise_only_and_soft_reference_rejected(monkeypatch):
    """纯底噪录音拒绝分析；参考里只有底噪 / 语音过轻时给出具体原因。"""
    y = S.soft_reference_case()

    # 乙的参考全落在停顿的纯底噪区
    silent_refs = {"A": S.SOFT_REFERENCE_REFS["A"],
                   "B": [[3.3, 3.75], [9.3, 9.7], [6.6, 6.95]]}
    resp = analyze(y, silent_refs)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "乙" in detail and "参考语音不足" in detail
    assert "静音" in detail or "底噪" in detail

    # 乙的参考虽是发言，但音量与底噪几乎不可分
    too_soft = S.soft_reference_case(noise=0.003, soft_gain=0.0035)
    resp = analyze(too_soft, {"A": S.SOFT_REFERENCE_REFS["A"],
                              "B": S.SOFT_REFERENCE_REFS["B"]})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "乙" in detail and "参考语音不足" in detail
    assert "太轻" in detail or "底噪" in detail

    # 纯底噪录音没有可用参考
    noise_only = np.random.default_rng(0).normal(0, 0.004, int(10 * SAMPLE_RATE)).astype(np.float32)
    resp = analyze(noise_only, S.SOFT_REFERENCE_REFS)
    assert resp.status_code == 400
    assert "有效语音过少" in resp.json()["detail"]


def test_third_speaker_and_mix_still_review():
    """放宽轻声判定后，第三人与甲乙混音仍应标为待复核。"""
    resp = analyze(S.third_speaker_and_mix_case(), S.THIRD_MIX_REFS)
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    cov_third = label_coverage(body["segments"], 6.6, 8.9)
    cov_mix = label_coverage(body["segments"], 9.6, 11.9)
    assert cov_third.get("review", 0.0) >= 0.5, f"第三说话人未标待复核: {cov_third}"
    assert cov_mix.get("review", 0.0) >= 0.5, f"甲乙混音未标待复核: {cov_mix}"
    for spk, t0, t1 in [("A", 0.6, 2.9), ("B", 3.6, 6.2)]:
        cov = label_coverage(body["segments"], t0, t1)
        assert cov.get(spk, 0.0) >= 0.6, f"{spk} 的正常语音归属不足: {cov}"


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
