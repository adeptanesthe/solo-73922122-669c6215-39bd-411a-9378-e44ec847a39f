"use strict";

/* 全局状态 */
const state = {
  file: null,
  objectUrl: null,
  buffer: null,      // 解码后的 AudioBuffer（画波形用）
  duration: 0,
  refs: { A: [], B: [] },   // 每项 {start, end}
  sel: null,              // 当前选区 {start, end}
  result: null,
  stopAt: null,           // 片段复听的停止点
};

const LABELS = { A: "甲", B: "乙", review: "待复核", silence: "静音" };

const els = {};
["fileInput", "demoBtn", "fileInfo", "player", "wave", "selInfo",
 "playSelBtn", "clearSelBtn", "addABtn", "addBBtn",
 "countA", "countB", "listA", "listB",
 "analyzeBtn", "status", "legend", "timeline", "segTable",
].forEach((id) => { els[id] = document.getElementById(id); });

let audioCtx = null;

/* ---------- 工具 ---------- */
function fmt(t) { return t.toFixed(2) + "s"; }

function setStatus(msg, kind) {
  els.status.textContent = msg || "";
  els.status.className = kind || "muted";
}

function b64ToBlob(b64, type) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: type || "audio/wav" });
}

/* ---------- 载入音频 ---------- */
async function loadFile(file) {
  state.file = file;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  els.player.src = state.objectUrl;

  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await file.arrayBuffer();
    state.buffer = await audioCtx.decodeAudioData(buf.slice(0));
    state.duration = state.buffer.duration;
    els.fileInfo.textContent =
      `${file.name} · ${fmt(state.duration)} · ${state.buffer.sampleRate} Hz · ${state.buffer.numberOfChannels} 声道`;
  } catch (e) {
    state.buffer = null;
    state.duration = 0;
    els.fileInfo.textContent = file.name + "（浏览器无法解码，仍可尝试上传分析）";
  }
  state.sel = null;
  state.result = null;
  renderAll();
}

/* ---------- 波形绘制 ---------- */
function drawWave() {
  const c = els.wave;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight || 160;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  if (!state.buffer) {
    ctx.fillStyle = "#9ca3af";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("请先载入音频", w / 2, h / 2);
    return;
  }

  const dur = state.duration;

  // 参考段底色
  for (const who of ["A", "B"]) {
    ctx.fillStyle = who === "A" ? "rgba(59,130,246,0.18)" : "rgba(249,115,22,0.18)";
    for (const r of state.refs[who]) {
      const x0 = (r.start / dur) * w, x1 = (r.end / dur) * w;
      ctx.fillRect(x0, 0, x1 - x0, h);
    }
  }

  // 波形
  const data = state.buffer.getChannelData(0);
  ctx.strokeStyle = "#374151";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(data.length / w));
  for (let x = 0; x < w; x++) {
    const start = Math.floor((x / w) * data.length);
    let min = 1, max = -1;
    for (let i = start; i < start + step && i < data.length; i++) {
      const v = data[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x + 0.5, ((1 - max) / 2) * h);
    ctx.lineTo(x + 0.5, ((1 - min) / 2) * h + 0.5);
  }
  ctx.stroke();

  // 秒刻度
  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  for (let s = 0; s <= Math.floor(dur); s++) {
    const x = (s / dur) * w;
    ctx.fillRect(x, h - 10, 1, 6);
    ctx.fillText(s + "s", x + 2, h - 2);
  }

  // 当前选区
  if (state.sel) {
    const x0 = (state.sel.start / dur) * w, x1 = (state.sel.end / dur) * w;
    ctx.fillStyle = "rgba(17,24,39,0.18)";
    ctx.fillRect(x0, 0, x1 - x0, h);
    ctx.strokeStyle = "#111827";
    ctx.strokeRect(x0 + 0.5, 0.5, x1 - x0 - 1, h - 1);
  }

  // 播放头
  const t = els.player.currentTime;
  if (t > 0 && dur > 0) {
    const x = (t / dur) * w;
    ctx.strokeStyle = "#dc2626";
    ctx.beginPath();
    ctx.moveTo(x, 0); ctx.lineTo(x, h);
    ctx.stroke();
  }
}

/* ---------- 波形交互：拖动选区 / 单击跳转 ---------- */
let dragStart = null;

function waveTime(e) {
  const rect = els.wave.getBoundingClientRect();
  const x = Math.min(Math.max(e.clientX - rect.left, 0), rect.width);
  return (x / rect.width) * state.duration;
}

els.wave.addEventListener("pointerdown", (e) => {
  if (!state.buffer) return;
  dragStart = waveTime(e);
  els.wave.setPointerCapture(e.pointerId);
});

els.wave.addEventListener("pointermove", (e) => {
  if (dragStart === null) return;
  const t = waveTime(e);
  state.sel = { start: Math.min(dragStart, t), end: Math.max(dragStart, t) };
  renderSel();
  drawWave();
});

els.wave.addEventListener("pointerup", (e) => {
  if (dragStart === null) return;
  const t = waveTime(e);
  if (Math.abs(t - dragStart) < 0.03) {   // 视为单击：跳转播放
    els.player.currentTime = t;
    els.player.play();
    state.sel = null;
  } else {
    state.sel = { start: Math.min(dragStart, t), end: Math.max(dragStart, t) };
  }
  dragStart = null;
  renderSel();
  drawWave();
});

/* ---------- 选区与参考段 ---------- */
function renderSel() {
  const has = !!state.sel && state.sel.end - state.sel.start >= 0.05;
  els.selInfo.textContent = has
    ? `选区：${fmt(state.sel.start)} ~ ${fmt(state.sel.end)}（${fmt(state.sel.end - state.sel.start)}）`
    : "未选择片段";
  els.playSelBtn.disabled = !has;
  els.clearSelBtn.disabled = !has;
  els.addABtn.disabled = !has;
  els.addBBtn.disabled = !has;
}

function overlaps(a, b) { return a.start < b.end - 1e-6 && b.start < a.end - 1e-6; }

function addRef(who) {
  if (!state.sel) return;
  const seg = { start: +state.sel.start.toFixed(3), end: +state.sel.end.toFixed(3) };
  const all = state.refs.A.concat(state.refs.B);
  if (all.some((r) => overlaps(r, seg))) {
    setStatus("该选区与已有参考段重叠，请重新选择", "err");
    return;
  }
  state.refs[who].push(seg);
  state.refs[who].sort((x, y) => x.start - y.start);
  state.sel = null;
  setStatus("");
  renderAll();
}

function delRef(who, i) {
  state.refs[who].splice(i, 1);
  renderAll();
}

function playRange(start, end) {
  els.player.currentTime = Math.max(0, start + 1e-3);
  state.stopAt = end;
  els.player.play();
}

function renderRefs() {
  for (const who of ["A", "B"]) {
    const list = who === "A" ? els.listA : els.listB;
    list.innerHTML = "";
    state.refs[who].forEach((r, i) => {
      const li = document.createElement("li");
      const txt = document.createElement("span");
      txt.className = "grow";
      txt.textContent = `第${i + 1}段：${fmt(r.start)} ~ ${fmt(r.end)}`;
      const play = document.createElement("button");
      play.textContent = "试听";
      play.onclick = () => playRange(r.start, r.end);
      const del = document.createElement("button");
      del.textContent = "删除";
      del.onclick = () => delRef(who, i);
      li.append(txt, play, del);
      list.appendChild(li);
    });
    (who === "A" ? els.countA : els.countB).textContent = state.refs[who].length;
  }
}

/* ---------- 校验（提交前本地预检，服务端仍会复核） ---------- */
function validateRefs() {
  if (!state.file) return "请先载入录音文件";
  for (const who of ["A", "B"]) {
    const n = state.refs[who].length;
    if (n < 3 || n > 8) return `${LABELS[who]}的参考段须为 3～8 段，当前 ${n} 段`;
  }
  const all = state.refs.A.concat(state.refs.B).sort((a, b) => a.start - b.start);
  for (const r of all) {
    if (r.start < 0 || r.end > state.duration + 1e-3 || r.end - r.start < 0.3)
      return `参考段 [${fmt(r.start)}, ${fmt(r.end)}] 越界或过短`;
  }
  for (let i = 1; i < all.length; i++) {
    if (all[i].start < all[i - 1].end - 1e-6) return "存在相互重叠的参考段";
  }
  return null;
}

function refreshAnalyzeBtn() {
  els.analyzeBtn.disabled = !state.file ||
    state.refs.A.length < 3 || state.refs.B.length < 3;
}

/* ---------- 分析 ---------- */
async function analyze() {
  const err = validateRefs();
  if (err) { setStatus(err, "err"); return; }

  const fd = new FormData();
  fd.append("file", state.file);
  fd.append("references", JSON.stringify({
    A: state.refs.A.map((r) => [r.start, r.end]),
    B: state.refs.B.map((r) => [r.start, r.end]),
  }));

  els.analyzeBtn.disabled = true;
  setStatus("分析中……", "muted");
  try {
    const resp = await fetch("/api/analyze", { method: "POST", body: fd });
    const body = await resp.json();
    if (!resp.ok) {
      setStatus("分析被拒绝：" + (body.detail || resp.statusText), "err");
      return;
    }
    state.result = body;
    setStatus(
      `分析完成：甲 ${body.stats.A} 帧 / 乙 ${body.stats.B} 帧 / 待复核 ${body.stats.review} 帧 / 静音 ${body.stats.silence} 帧`,
      "ok");
    renderResult();
  } catch (e) {
    setStatus("请求失败：" + e.message, "err");
  } finally {
    refreshAnalyzeBtn();
  }
}

/* ---------- 结果展示 ---------- */
function renderResult() {
  const res = state.result;
  if (!res) {
    els.timeline.classList.add("hidden");
    els.segTable.classList.add("hidden");
    els.legend.classList.add("hidden");
    return;
  }
  els.legend.classList.remove("hidden");
  const tl = els.timeline;
  tl.classList.remove("hidden");
  tl.innerHTML = "";
  const dur = res.duration;

  for (const seg of res.segments) {
    const div = document.createElement("div");
    div.className = "seg " + seg.label;
    div.style.left = (seg.start / dur) * 100 + "%";
    div.style.width = Math.max(((seg.end - seg.start) / dur) * 100, 0.15) + "%";
    if (seg.label === "A" || seg.label === "B") {
      div.style.opacity = (0.45 + 0.55 * seg.confidence).toFixed(2);
    }
    const pct = Math.round(seg.confidence * 100) + "%";
    div.title = `${LABELS[seg.label]} · ${fmt(seg.start)}~${fmt(seg.end)} · 置信度 ${pct}`;
    if (seg.end - seg.start > 0.35 && seg.label !== "silence") {
      div.textContent = `${LABELS[seg.label]} ${pct}`;
    }
    div.onclick = () => playRange(seg.start, seg.end);
    tl.appendChild(div);
  }
  const ph = document.createElement("div");
  ph.className = "playhead";
  ph.id = "tlPlayhead";
  ph.style.left = "0%";
  tl.appendChild(ph);

  // 明细表
  const box = els.segTable;
  box.classList.remove("hidden");
  let html = "<table><thead><tr><th>#</th><th>标签</th><th>开始</th><th>结束</th><th>时长</th><th>置信度</th></tr></thead><tbody>";
  res.segments.forEach((seg, i) => {
    html += `<tr data-i="${i}"><td>${i + 1}</td><td>${LABELS[seg.label]}</td>` +
      `<td>${fmt(seg.start)}</td><td>${fmt(seg.end)}</td>` +
      `<td>${fmt(seg.end - seg.start)}</td><td>${Math.round(seg.confidence * 100)}%</td></tr>`;
  });
  html += "</tbody></table>";
  box.innerHTML = html;
  box.querySelectorAll("tr[data-i]").forEach((tr) => {
    tr.onclick = () => {
      const seg = state.result.segments[+tr.dataset.i];
      playRange(seg.start, seg.end);
    };
  });
}

/* ---------- 播放控制 ---------- */
els.player.addEventListener("timeupdate", () => {
  if (state.stopAt !== null && els.player.currentTime >= state.stopAt) {
    els.player.pause();
    state.stopAt = null;
  }
  drawWave();
  const res = state.result;
  const ph = document.getElementById("tlPlayhead");
  if (res && ph) ph.style.left = (els.player.currentTime / res.duration) * 100 + "%";
});

/* ---------- 演示 ---------- */
async function loadDemo() {
  setStatus("正在生成演示音频……", "muted");
  try {
    const resp = await fetch("/api/demo");
    const body = await resp.json();
    const blob = b64ToBlob(body.audio_base64);
    const file = new File([blob], "demo.wav", { type: "audio/wav" });
    await loadFile(file);
    state.refs.A = body.references.A.map(([s, e]) => ({ start: s, end: e }));
    state.refs.B = body.references.B.map(([s, e]) => ({ start: s, end: e }));
    setStatus("已载入演示音频并预填参考段，可直接点击「开始分析」", "ok");
    renderAll();
  } catch (e) {
    setStatus("演示载入失败：" + e.message, "err");
  }
}

/* ---------- 汇总渲染 ---------- */
function renderAll() {
  renderSel();
  renderRefs();
  refreshAnalyzeBtn();
  renderResult();
  drawWave();
}

/* ---------- 事件绑定 ---------- */
els.fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) loadFile(e.target.files[0]);
});
els.demoBtn.addEventListener("click", loadDemo);
els.playSelBtn.addEventListener("click", () => state.sel && playRange(state.sel.start, state.sel.end));
els.clearSelBtn.addEventListener("click", () => { state.sel = null; renderAll(); });
els.addABtn.addEventListener("click", () => addRef("A"));
els.addBBtn.addEventListener("click", () => addRef("B"));
els.analyzeBtn.addEventListener("click", analyze);
window.addEventListener("resize", drawWave);

renderAll();
