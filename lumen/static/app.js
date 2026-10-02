"use strict";

const S = {
  live: null,
  health: null,
  mounted: "",
  gpu: 0,
  hoverIndex: null,
  detailHover: null,
  points: [],
  detailPoints: [],
  mode: "gpu",
  runs: [],
  detail: null,
  ro: null,
  timer: null,
};

const METRIC_ORDER = ["step", "loss", "learning_rate", "lr", "grad_norm", "epoch", "tokens_per_sec", "ttft_ms", "prompt_tokens", "completion_tokens"];
const METRIC_LABEL = {
  step: "step",
  loss: "loss",
  learning_rate: "lr",
  lr: "lr",
  grad_norm: "grad",
  epoch: "epoch",
  tokens_per_sec: "tok/s",
  ttft_ms: "ttft",
  prompt_tokens: "prompt",
  completion_tokens: "completion",
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function fmtBytes(bytes) {
  if (bytes == null || Number.isNaN(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Number(bytes);
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  const digits = value >= 100 || index === 0 ? 0 : 1;
  return `${value.toFixed(digits)} ${units[index]}`;
}

function fmtBps(bytes) {
  if (bytes == null) return "—";
  return `${fmtBytes(bytes)}/s`;
}

function fmtDuration(seconds) {
  const sec = Math.max(0, Math.floor(seconds));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m`;
  if (m) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${s}s`;
}

function fmtUptime(seconds) {
  if (seconds == null) return "";
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d) return `Up ${d}d ${h}h`;
  if (h) return `Up ${h}h ${m}m`;
  return `Up ${m}m`;
}

function fmtClock(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function fmtDay(ts) {
  return new Date(ts * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function fmtPct(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return String(Math.round(Number(value)));
}

function fmtGB(bytes) {
  if (bytes == null) return "—";
  return (bytes / 1024 ** 3).toFixed(1);
}

function prettyModel(model) {
  if (!model) return "";
  const parts = String(model).split(/[\\/]/);
  return parts[parts.length - 1] || model;
}

function runTitle(run) {
  return prettyModel(run.model) || run.proc_name || "Process";
}

function runDuration(run, now) {
  const end = run.ended_at || now;
  return fmtDuration(end - run.started_at);
}

function fmtMetric(key, value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  if (key === "loss" || key === "grad_norm") return number.toFixed(3);
  if (key === "lr" || key === "learning_rate") return number.toExponential(1);
  if (key === "ttft_ms") return `${Math.round(number)} ms`;
  if (key === "epoch") return number.toFixed(2);
  if (key === "step" || key.includes("token")) return Math.round(number).toLocaleString();
  if (Number.isInteger(number)) return String(number);
  return number.toFixed(2);
}

function metricBits(data) {
  const bits = [];
  METRIC_ORDER.forEach((key) => {
    if (data[key] != null) bits.push(`${METRIC_LABEL[key] || key} ${fmtMetric(key, data[key])}`);
  });
  Object.keys(data).forEach((key) => {
    if (!METRIC_ORDER.includes(key) && bits.length < 6 && typeof data[key] === "number") {
      bits.push(`${key} ${fmtMetric(key, data[key])}`);
    }
  });
  return bits;
}

function metricSummary(data) {
  if (!data) return "";
  const bits = [];
  if (data.step != null) bits.push(`step ${Math.round(data.step)}`);
  if (data.loss != null) bits.push(`loss ${Number(data.loss).toFixed(3)}`);
  if (data.tokens_per_sec != null) bits.push(`${Math.round(data.tokens_per_sec).toLocaleString()} tok/s`);
  if (data.ttft_ms != null) bits.push(`ttft ${Math.round(data.ttft_ms)} ms`);
  return bits.join("  ·  ");
}

function parseRoute() {
  const parts = (location.hash || "#/").replace(/^#/, "").split("/").filter(Boolean);
  if (parts[0] === "runs" && parts[1]) return { page: "run", id: decodeURIComponent(parts[1]) };
  if (parts[0] === "runs") return { page: "runs" };
  return { page: "live" };
}

function segs(pct, tone) {
  const count = 28;
  const lit = Math.max(0, Math.min(count, Math.round(((Number(pct) || 0) / 100) * count)));
  let cells = "";
  for (let i = 0; i < count; i += 1) cells += `<i class="${i < lit ? "on" : ""}"></i>`;
  return `<div class="segs ${tone}">${cells}</div>`;
}

function meter(kicker, value, unit, pct, tone, foot) {
  return `<article class="meter">
    <p class="kicker">${esc(kicker)}</p>
    <div class="meter-val"><span class="num">${esc(value)}</span><span class="unit">${esc(unit)}</span></div>
    ${segs(pct, tone)}
    <p class="meter-foot">${esc(foot)}</p>
  </article>`;
}

function spark(values) {
  if (!values || values.length < 2) return "";
  const w = 148;
  const h = 32;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((value, index) => {
    const x = (index / (values.length - 1)) * w;
    const y = h - ((value - min) / span) * (h - 4) - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true"><polyline points="${pts}" /></svg>`;
}

function legendHTML(mode) {
  if (mode === "gpu") {
    return '<span><i class="swatch signal"></i>GPU</span><span><i class="swatch ice"></i>VRAM</span><span><i class="swatch heat"></i>Power</span>';
  }
  return '<span><i class="swatch ice"></i>CPU</span><span><i class="swatch signal"></i>Memory</span>';
}

function normalize(history, gpuIndex) {
  return (history || []).map((point) => {
    const gpu = (point.gpus || [])[gpuIndex] || null;
    const vram = gpu
      ? (gpu.vram != null ? gpu.vram : (gpu.mem_total ? (100 * gpu.mem_used) / gpu.mem_total : null))
      : null;
    return {
      ts: point.ts,
      cpu: point.cpu,
      ram: point.ram,
      gpu: gpu ? gpu.util : null,
      vram,
      power: gpu ? (gpu.power != null ? gpu.power : gpu.power_w) : null,
      temp: gpu ? gpu.temp : null,
      powerLimit: gpu ? (gpu.power_limit != null ? gpu.power_limit : gpu.power_limit_w) : null,
    };
  });
}

function tickClock() {
  const clock = document.getElementById("clock");
  if (clock) {
    clock.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
}

function setConn(mode) {
  const el = document.getElementById("conn");
  const label = document.getElementById("conn-label");
  if (!el || !label) return;
  el.classList.remove("live", "retry");
  if (mode === "live") {
    el.classList.add("live");
    label.textContent = "Live";
  } else if (mode === "retry") {
    el.classList.add("retry");
    label.textContent = "Reconnecting";
  } else {
    label.textContent = "Connecting";
  }
}

function paintShell() {
  const live = S.live;
  if (!live) return;
  document.body.classList.toggle("is-recording", !!(live.runs && live.runs.length));
  const pill = document.getElementById("demo-pill");
  if (pill) pill.classList.toggle("hidden", !live.demo);
  const host = live.host || {};
  const name = document.getElementById("host-name");
  const cpu = document.getElementById("host-cpu");
  const meta = document.getElementById("host-meta");
  const up = document.getElementById("host-up");
  if (name) name.textContent = host.hostname || "This machine";
  if (cpu) cpu.textContent = host.cpu || "";
  if (meta) meta.textContent = host.cores ? `${host.cores} cores · ${host.threads} threads` : "";
  if (up) up.textContent = fmtUptime(host.uptime_s);
  const badge = document.getElementById("run-count");
  if (badge) badge.textContent = live.run_count ? String(live.run_count) : "";
  const side = document.getElementById("side-meters");
  if (side) {
    const memory = live.memory || {};
    const disk = live.disk || {};
    const net = live.net || {};
    side.innerHTML = `
      ${bar("CPU", live.cpu && live.cpu.percent, "signal")}
      ${bar("Memory", memory.percent, "ice")}
      <div class="side-io">
        <span>Read ${esc(fmtBps(disk.read_bps))}</span>
        <span>Write ${esc(fmtBps(disk.write_bps))}</span>
        <span>Down ${esc(fmtBps(net.recv_bps))}</span>
        <span>Up ${esc(fmtBps(net.sent_bps))}</span>
      </div>`;
  }
}

function bar(label, pct, tone) {
  const width = Math.max(0, Math.min(100, Number(pct) || 0));
  return `<div class="sbar ${tone || ""}"><div class="sbar-top"><span>${esc(label)}</span><span>${Math.round(width)}%</span></div><div class="sbar-track"><div style="width:${width}%"></div></div></div>`;
}

function teardown() {
  if (S.ro) {
    S.ro.disconnect();
    S.ro = null;
  }
  if (S.timer) {
    clearInterval(S.timer);
    S.timer = null;
  }
}

function selectedGpu(live) {
  const gpus = live.gpus || [];
  if (!gpus.length) return null;
  if (S.gpu >= gpus.length) S.gpu = 0;
  return gpus[S.gpu];
}

function paintHero(live) {
  const gpus = live.gpus || [];
  const gpu = selectedGpu(live);
  const headline = document.getElementById("headline");
  const sub = document.getElementById("subline");
  const ollama = document.getElementById("ollama-line");
  const sw = document.getElementById("gpu-switch");
  if (sw) {
    if (gpus.length > 1) {
      sw.classList.remove("hidden");
      sw.innerHTML = gpus.map((item, index) => (
        `<button type="button" data-gpu="${index}" class="${index === S.gpu ? "on" : ""}">GPU ${index}</button>`
      )).join("");
    } else {
      sw.classList.add("hidden");
      sw.innerHTML = "";
    }
  }
  if (gpu && headline && sub) {
    headline.textContent = gpu.name || `GPU ${gpu.index}`;
    const bits = [`Device ${gpu.index}`];
    if (live.host && live.host.gpu_driver) bits.push(`driver ${live.host.gpu_driver}`);
    if (gpu.pcie) bits.push(gpu.pcie);
    if (gpu.clock_sm) bits.push(`${Math.round(gpu.clock_sm).toLocaleString()} MHz SM`);
    if (gpu.clock_mem) bits.push(`${Math.round(gpu.clock_mem).toLocaleString()} MHz mem`);
    const active = (live.runs || []).length;
    bits.push(active ? `recording ${active} ${active === 1 ? "run" : "runs"}` : "idle");
    sub.textContent = bits.join("   ·   ");
  } else if (headline && sub) {
    headline.textContent = (live.host && live.host.cpu) || "This machine";
    sub.textContent = live.gpu_note || "CPU and memory are recording.";
  }
  if (ollama) {
    if (live.ollama && live.ollama.online) {
      const names = (live.ollama.models || []).map((item) => item.name).filter(Boolean);
      ollama.textContent = names.length ? `Ollama loaded  ${names.join(", ")}` : "Ollama is up. No model is loaded.";
      ollama.classList.remove("hidden");
    } else {
      ollama.textContent = "";
      ollama.classList.add("hidden");
    }
  }
}

function gpuMeters(gpu) {
  const vramPct = gpu.mem_total ? (100 * (gpu.mem_used || 0)) / gpu.mem_total : 0;
  const powerPct = gpu.power_limit_w && gpu.power_w != null ? (100 * gpu.power_w) / gpu.power_limit_w : (gpu.power_w ? 70 : 0);
  const tempPct = gpu.temp == null ? 0 : Math.max(0, Math.min(100, ((gpu.temp - 30) / 65) * 100));
  let tempTone = "signal";
  if (gpu.temp >= 85) tempTone = "heat";
  else if (gpu.temp >= 75) tempTone = "warn";
  return [
    meter("GPU load", fmtPct(gpu.util), "%", gpu.util || 0, "signal", gpu.mem_util != null ? `memory controller ${Math.round(gpu.mem_util)}%` : "graphics"),
    meter("VRAM", fmtGB(gpu.mem_used), "GB", vramPct, "ice", `of ${fmtGB(gpu.mem_total)} GB`),
    meter("Power", gpu.power_w == null ? "—" : String(Math.round(gpu.power_w)), "W", powerPct, "heat", gpu.power_limit_w ? `limit ${Math.round(gpu.power_limit_w)} W` : "draw"),
    meter("Temp", gpu.temp == null ? "—" : String(Math.round(gpu.temp)), "°C", tempPct, tempTone, gpu.fan == null ? "die" : `fan ${Math.round(gpu.fan)}%`),
  ].join("");
}

function hostMeters(live) {
  const memory = live.memory || {};
  const disk = live.disk || {};
  const cpu = live.cpu || {};
  const readScale = Math.min(100, (disk.read_bps || 0) / 1e8 * 100);
  const writeScale = Math.min(100, (disk.write_bps || 0) / 1e8 * 100);
  return [
    meter("CPU", fmtPct(cpu.percent), "%", cpu.percent || 0, "signal", cpu.freq_mhz ? `${Math.round(cpu.freq_mhz).toLocaleString()} MHz` : "package"),
    meter("Memory", fmtGB(memory.used), "GB", memory.percent || 0, "ice", `of ${fmtGB(memory.total)} GB`),
    meter("Disk read", fmtBps(disk.read_bps), "", readScale, "ice", "bytes in"),
    meter("Disk write", fmtBps(disk.write_bps), "", writeScale, "heat", "bytes out"),
  ].join("");
}

function paintCpu(live) {
  const row = document.getElementById("cpu-row");
  if (!row) return;
  const cores = (live.cpu && live.cpu.per_core) || [];
  const bars = cores.map((value) => `<i style="height:${Math.max(4, Math.round(value))}%"></i>`).join("");
  const memory = live.memory || {};
  const disk = live.disk || {};
  const net = live.net || {};
  const freq = live.cpu && live.cpu.freq_mhz ? `${Math.round(live.cpu.freq_mhz).toLocaleString()} MHz` : "";
  row.innerHTML = `
    <div>
      <p class="kicker">CPU</p>
      <p class="cpu-num">${esc(fmtPct(live.cpu && live.cpu.percent))}<span>%</span></p>
      <p class="quiet">${esc(freq)}</p>
    </div>
    <div class="cores" aria-hidden="true">${bars}</div>
    <dl class="io">
      <div><dt>Memory</dt><dd>${esc(fmtBytes(memory.used))} <span>/ ${esc(fmtBytes(memory.total))}</span></dd></div>
      <div><dt>Read</dt><dd>${esc(fmtBps(disk.read_bps))}</dd></div>
      <div><dt>Write</dt><dd>${esc(fmtBps(disk.write_bps))}</dd></div>
      <div><dt>Net</dt><dd>${esc(fmtBps(net.recv_bps))} ↓ ${esc(fmtBps(net.sent_bps))} ↑</dd></div>
    </dl>`;
}

function paintRuns(live) {
  const el = document.getElementById("active-runs");
  if (!el) return;
  const runs = live.runs || [];
  if (!runs.length) {
    el.innerHTML = `<div class="empty"><p class="kicker">Waiting</p><h3>No model run is active.</h3><p>Start Ollama, LM Studio, llama.cpp, vLLM, or a training script. Lumen opens a session when it sees the process.</p></div>`;
    return;
  }
  const now = live.now || Date.now() / 1000;
  el.innerHTML = runs.map((run) => {
    const summary = metricSummary(run.last_metric);
    return `<article class="run">
      <div class="run-top"><span class="kind ${esc(run.kind)}">${esc(run.kind)}</span><span class="mono">${esc(runDuration(run, now))}</span></div>
      <h3 title="${esc(run.model || "")}">${esc(runTitle(run))}</h3>
      <p class="cmd" title="${esc(run.command || "")}">${esc(run.command || run.proc_name || "")}</p>
      ${summary ? `<p class="metric-now">${esc(summary)}</p>` : ""}
      ${spark(run.loss_tail || [])}
    </article>`;
  }).join("");
}

function paintProcs(live) {
  const el = document.getElementById("procs");
  if (!el) return;
  const gpu = selectedGpu(live);
  if (!gpu) {
    el.innerHTML = `<p class="quiet">GPU processes show up when an NVIDIA card is present.</p>`;
    return;
  }
  const procs = gpu.processes || [];
  if (!procs.length) {
    el.innerHTML = `<p class="quiet">This GPU is idle.</p>`;
    return;
  }
  el.innerHTML = `<ul class="procs">${procs.map((proc) => (
    `<li><span>${esc(proc.name)}</span><span class="quiet">pid ${esc(proc.pid)}</span><span class="mono">${esc(proc.mem ? fmtBytes(proc.mem) : "—")}</span></li>`
  )).join("")}</ul>`;
}

function paintLog(live) {
  const list = document.getElementById("metric-log");
  if (!list) return;
  const metrics = live.metrics || [];
  if (!metrics.length) {
    list.innerHTML = `<li class="quiet">Loss, learning rate, and tokens per second land here from LumenCallback or log_metrics().</li>`;
    return;
  }
  list.innerHTML = metrics.map((item) => {
    const bits = metricBits(item.data || {});
    return `<li><time>${esc(fmtClock(item.ts))}</time><span class="kind ${esc(item.type)}">${esc(item.type)}</span><span>${esc(bits.join("   ·   ") || item.model || "")}</span></li>`;
  }).join("");
}

function mountLive() {
  teardown();
  S.mounted = "live";
  S.hoverIndex = null;
  document.getElementById("main").innerHTML = `
    <section class="live">
      <div id="gpu-switch" class="gpu-switch hidden"></div>
      <header class="page-head">
        <h1 id="headline">—</h1>
        <p id="subline" class="subline"></p>
        <p id="ollama-line" class="ollama-line hidden"></p>
      </header>
      <div id="meters" class="meters"></div>
      <section class="card chart-card">
        <header class="card-head">
          <span class="kicker" id="chart-kicker">Last 3 minutes</span>
          <div class="legend" id="legend"></div>
        </header>
        <div class="chart-wrap">
          <canvas id="chart"></canvas>
          <div id="tip" class="tip hidden"></div>
        </div>
      </section>
      <section class="cpu-row card" id="cpu-row"></section>
      <section class="split">
        <div class="card">
          <header class="card-head"><span class="kicker">Active runs</span></header>
          <div id="active-runs"></div>
        </div>
        <div class="card">
          <header class="card-head"><span class="kicker">GPU processes</span></header>
          <div id="procs"></div>
        </div>
      </section>
      <section class="card" id="log-card">
        <header class="card-head"><span class="kicker">Signals</span></header>
        <ol id="metric-log" class="log"></ol>
      </section>
    </section>`;
  const canvas = document.getElementById("chart");
  canvas.addEventListener("mousemove", onChartMove);
  canvas.addEventListener("mouseleave", onChartLeave);
  S.ro = new ResizeObserver(() => drawLiveChart());
  S.ro.observe(canvas);
  updateLive();
}

function updateLive() {
  if (!S.live || !document.getElementById("headline")) return;
  const live = S.live;
  const gpu = selectedGpu(live);
  paintHero(live);
  const meters = document.getElementById("meters");
  if (meters) meters.innerHTML = gpu ? gpuMeters(gpu) : hostMeters(live);
  paintCpu(live);
  paintRuns(live);
  paintProcs(live);
  paintLog(live);
  drawLiveChart();
}

function drawLiveChart() {
  const canvas = document.getElementById("chart");
  if (!canvas || !S.live) return;
  const gpus = S.live.gpus || [];
  const index = gpus.length ? Math.min(S.gpu, gpus.length - 1) : 0;
  const points = normalize(S.live.history, index);
  S.points = points;
  const mode = points.some((point) => point.gpu != null) ? "gpu" : "cpu";
  S.mode = mode;
  const legend = document.getElementById("legend");
  if (legend) legend.innerHTML = legendHTML(mode);
  const kicker = document.getElementById("chart-kicker");
  if (kicker) kicker.textContent = mode === "gpu" ? "Last 3 minutes" : "Last 3 minutes · CPU";
  drawChart(canvas, points, mode, S.hoverIndex);
}

function onChartMove(event) {
  const canvas = event.currentTarget;
  const live = canvas.id === "chart";
  const points = live ? S.points : S.detailPoints;
  if (!points.length) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const index = indexAt(x, rect.width, points.length);
  if (live) {
    S.hoverIndex = index;
    drawLiveChart();
  } else {
    S.detailHover = index;
    drawDetailChart();
  }
  const tip = canvas.parentElement.querySelector(".tip");
  const point = points[index];
  const mode = live ? S.mode : (points.some((item) => item.gpu != null) ? "gpu" : "cpu");
  if (tip && point) {
    tip.innerHTML = tipHTML(point, mode);
    tip.classList.remove("hidden");
    tip.style.left = `${Math.min(Math.max(8, x + 14), rect.width - 150)}px`;
  }
}

function onChartLeave(event) {
  const live = event.currentTarget.id === "chart";
  if (live) {
    S.hoverIndex = null;
    drawLiveChart();
  } else {
    S.detailHover = null;
    drawDetailChart();
  }
  const tip = event.currentTarget.parentElement.querySelector(".tip");
  if (tip) tip.classList.add("hidden");
}

function tipHTML(point, mode) {
  const rows = [`<div>${esc(fmtClock(point.ts))}</div>`];
  if (mode === "gpu") {
    if (point.gpu != null) rows.push(`<div>GPU <b>${Math.round(point.gpu)}%</b></div>`);
    if (point.vram != null) rows.push(`<div>VRAM <b>${Math.round(point.vram)}%</b></div>`);
    if (point.power != null) rows.push(`<div>Power <b>${Math.round(point.power)} W</b></div>`);
    if (point.temp != null) rows.push(`<div>Temp <b>${Math.round(point.temp)}°</b></div>`);
  } else {
    if (point.cpu != null) rows.push(`<div>CPU <b>${Math.round(point.cpu)}%</b></div>`);
    if (point.ram != null) rows.push(`<div>Memory <b>${Math.round(point.ram)}%</b></div>`);
  }
  return rows.join("");
}

function indexAt(x, width, count) {
  const padL = 36;
  const padR = 12;
  const inner = Math.max(1, width - padL - padR);
  const t = (x - padL) / inner;
  return Math.max(0, Math.min(count - 1, Math.round(t * (count - 1))));
}

function drawChart(canvas, points, mode, hoverIndex) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = rect.width;
  const h = rect.height;
  if (w < 10 || h < 10) return;
  const nextW = Math.floor(w * dpr);
  const nextH = Math.floor(h * dpr);
  if (canvas.width !== nextW || canvas.height !== nextH) {
    canvas.width = nextW;
    canvas.height = nextH;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 36, r: 12, t: 14, b: 26 };
  ctx.font = "11px JetBrains Mono, Cascadia Mono, monospace";
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  if (!points.length) {
    ctx.fillStyle = "rgba(154,148,136,0.9)";
    ctx.textAlign = "center";
    ctx.fillText("Waiting for samples", w / 2, h / 2);
    return;
  }
  const innerH = h - pad.t - pad.b;
  const yOf = (value) => pad.t + innerH - (Math.max(0, Math.min(100, value)) / 100) * innerH;
  const xOf = (index) => {
    const innerW = w - pad.l - pad.r;
    if (points.length <= 1) return pad.l + innerW / 2;
    return pad.l + (index / (points.length - 1)) * innerW;
  };
  ctx.strokeStyle = "rgba(244,240,230,0.08)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(154,148,136,0.95)";
  ctx.textAlign = "left";
  [0, 50, 100].forEach((mark) => {
    const y = yOf(mark);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
    ctx.fillText(String(mark), 2, y + 4);
  });
  const limit = points.reduce((found, point) => point.powerLimit || found, null);
  const maxPower = Math.max(1, ...points.map((point) => point.power || 0));
  const powerScale = limit && limit > 0 ? limit : maxPower * 1.15;

  function segments(key, mapValue) {
    const groups = [];
    let current = [];
    points.forEach((point, index) => {
      let value = point[key];
      if (mapValue && value != null) value = mapValue(value);
      if (value == null || Number.isNaN(value)) {
        if (current.length) groups.push(current);
        current = [];
        return;
      }
      current.push({ x: xOf(index), y: yOf(value) });
    });
    if (current.length) groups.push(current);
    return groups;
  }
  function fill(groups, color) {
    const base = h - pad.b;
    ctx.beginPath();
    groups.forEach((group) => {
      if (!group.length) return;
      ctx.moveTo(group[0].x, base);
      group.forEach((pt) => ctx.lineTo(pt.x, pt.y));
      ctx.lineTo(group[group.length - 1].x, base);
      ctx.closePath();
    });
    ctx.fillStyle = color;
    ctx.fill();
  }
  function stroke(groups, color, width) {
    ctx.beginPath();
    groups.forEach((group) => {
      group.forEach((pt, index) => {
        if (index === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  if (mode === "gpu") {
    const gpuSeg = segments("gpu");
    fill(gpuSeg, "rgba(214,255,74,0.16)");
    stroke(segments("vram"), "#8eb7ff", 1.25);
    stroke(segments("power", (value) => (100 * value) / powerScale), "#ff5a36", 1.35);
    ctx.save();
    ctx.shadowColor = "rgba(214,255,74,0.55)";
    ctx.shadowBlur = 8;
    stroke(gpuSeg, "#d6ff4a", 1.7);
    ctx.restore();
  } else {
    const cpuSeg = segments("cpu");
    fill(cpuSeg, "rgba(142,183,255,0.16)");
    stroke(segments("ram"), "#d6ff4a", 1.35);
    ctx.save();
    ctx.shadowColor = "rgba(142,183,255,0.45)";
    ctx.shadowBlur = 8;
    stroke(cpuSeg, "#8eb7ff", 1.7);
    ctx.restore();
  }

  if (hoverIndex != null && points[hoverIndex]) {
    const x = xOf(hoverIndex);
    ctx.strokeStyle = "rgba(244,240,230,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, h - pad.b);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(154,148,136,0.95)";
  ctx.textAlign = "left";
  ctx.fillText(fmtTick(points[0].ts), pad.l, h - 8);
  ctx.textAlign = "right";
  ctx.fillText(fmtTick(points[points.length - 1].ts), w - pad.r, h - 8);
}

function fmtTick(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function mountRuns() {
  teardown();
  S.mounted = "runs";
  document.getElementById("main").innerHTML = `
    <section class="page">
      <header class="page-head">
        <h1>Runs</h1>
        <p class="subline">Each local training or inference process becomes a session.</p>
      </header>
      <div id="run-list" class="card runs"><p class="quiet pad">Loading sessions…</p></div>
    </section>`;
  loadRuns();
  S.timer = setInterval(loadRuns, 4000);
}

async function loadRuns() {
  try {
    const response = await fetch("/api/runs");
    const data = await response.json();
    S.runs = data.runs || [];
    if (S.mounted === "runs") fillRuns();
  } catch (err) {
    const el = document.getElementById("run-list");
    if (el && S.mounted === "runs") el.innerHTML = `<p class="quiet pad">The run list is unavailable.</p>`;
  }
}

function fillRuns() {
  const el = document.getElementById("run-list");
  if (!el) return;
  const runs = S.runs || [];
  if (!runs.length) {
    el.innerHTML = `<div class="empty"><p class="kicker">No sessions yet</p><h2>Lumen is watching for a model process.</h2><p>Start Ollama, LM Studio, llama.cpp, vLLM, or a training script on this machine.</p></div>`;
    return;
  }
  const now = (S.live && S.live.now) || Date.now() / 1000;
  const rows = runs.map((run) => `
    <a class="run-row" href="#/runs/${encodeURIComponent(run.id)}">
      <span class="kind ${esc(run.kind)}">${esc(run.kind)}</span>
      <span class="run-id"><strong title="${esc(run.model || run.command || "")}">${esc(runTitle(run))}</strong><em class="${run.status === "running" ? "live" : ""}">${run.status === "running" ? "Live" : esc(fmtDay(run.started_at))}</em></span>
      <span class="mono">${esc(runDuration(run, now))}</span>
      <span class="mono">${run.avg_gpu == null ? "—" : `${Math.round(run.avg_gpu)}%`}</span>
      <span class="mono">${run.peak_vram ? esc(fmtBytes(run.peak_vram)) : "—"}</span>
      <span class="mono">${run.peak_power == null ? "—" : `${Math.round(run.peak_power)} W`}</span>
    </a>`).join("");
  el.innerHTML = `<div class="run-head"><span>Kind</span><span>Run</span><span>Duration</span><span>Avg GPU</span><span>Peak VRAM</span><span>Peak power</span></div>${rows}`;
}

async function mountDetail(id) {
  teardown();
  S.mounted = "run";
  S.detailHover = null;
  const main = document.getElementById("main");
  main.innerHTML = `<section class="page"><p class="quiet pad">Loading run…</p></section>`;
  let data;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(id)}`);
    if (!response.ok) throw new Error("missing");
    data = await response.json();
  } catch (err) {
    if (S.mounted !== "run") return;
    main.innerHTML = `<section class="page"><a class="back" href="#/runs">All runs</a><div class="empty"><h2>That run is not in the local database.</h2></div></section>`;
    return;
  }
  const route = parseRoute();
  if (route.page !== "run" || route.id !== id) return;
  S.detail = data;
  paintDetail();
}

function paintDetail() {
  const data = S.detail;
  if (!data) return;
  const run = data.run;
  const now = (S.live && S.live.now) || Date.now() / 1000;
  const metrics = data.metrics || [];
  const older = run.proc_started_at && run.started_at - run.proc_started_at > 30
    ? `Process started ${fmtDuration(now - run.proc_started_at)} ago. This chart is the machine while the session was open.`
    : "Machine metrics while this session was open.";
  const keys = METRIC_ORDER.filter((key) => metrics.some((item) => item.data && item.data[key] != null));
  const shown = metrics.slice().reverse().slice(0, 18);
  const table = keys.length ? `
    <section class="card">
      <header class="card-head"><span class="kicker">Signals</span></header>
      <div class="table-wrap"><table class="metrics">
        <thead><tr><th>Time</th>${keys.map((key) => `<th>${esc(METRIC_LABEL[key] || key)}</th>`).join("")}</tr></thead>
        <tbody>
          ${shown.map((item) => `<tr><td>${esc(fmtClock(item.ts))}</td>${keys.map((key) => `<td>${item.data && item.data[key] != null ? esc(fmtMetric(key, item.data[key])) : "—"}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table></div>
    </section>` : `<section class="card"><p class="quiet">No step metrics for this run. Add LumenCallback to a Trainer, or call log_metrics(), to plot loss beside the GPU.</p></section>`;
  const lossCard = metrics.some((item) => item.data && typeof item.data.loss === "number")
    ? `<section class="card"><header class="card-head"><span class="kicker">Loss</span></header><div class="chart-wrap"><canvas id="loss-chart"></canvas></div></section>`
    : "";
  document.getElementById("main").innerHTML = `
    <section class="page">
      <a class="back" href="#/runs">All runs</a>
      <header class="page-head">
        <h1 title="${esc(run.model || "")}">${esc(runTitle(run))}</h1>
        <p class="subline">${esc(run.kind)} · ${esc(run.status)} · pid ${esc(run.pid)}${run.gpu_name ? ` · ${esc(run.gpu_name)}` : ""}</p>
      </header>
      <div class="stats">
        <article class="card stat"><span class="kicker">Duration</span><b>${esc(runDuration(run, now))}</b></article>
        <article class="card stat"><span class="kicker">Average GPU</span><b>${run.avg_gpu == null ? "—" : `${Math.round(run.avg_gpu)}%`}</b></article>
        <article class="card stat"><span class="kicker">Peak VRAM</span><b>${run.peak_vram ? esc(fmtBytes(run.peak_vram)) : "—"}</b></article>
        <article class="card stat"><span class="kicker">Peak power</span><b>${run.peak_power == null ? "—" : `${Math.round(run.peak_power)} W`}</b></article>
      </div>
      <p class="cmd-block">${esc(run.command || run.proc_name || "")}</p>
      <p class="quiet">${esc(older)}</p>
      <section class="card" style="margin-top:12px">
        <header class="card-head"><span class="kicker">Session</span><div class="legend" id="detail-legend"></div></header>
        <div class="chart-wrap">
          <canvas id="chart-detail"></canvas>
          <div class="tip hidden"></div>
        </div>
      </section>
      <div class="stack" style="margin-top:12px">${lossCard}${table}</div>
    </section>`;
  const canvas = document.getElementById("chart-detail");
  canvas.addEventListener("mousemove", onChartMove);
  canvas.addEventListener("mouseleave", onChartLeave);
  S.ro = new ResizeObserver(() => {
    drawDetailChart();
    drawDetailLoss();
  });
  S.ro.observe(canvas);
  requestAnimationFrame(() => {
    drawDetailChart();
    drawDetailLoss();
  });
}

function drawDetailChart() {
  const canvas = document.getElementById("chart-detail");
  if (!canvas || !S.detail) return;
  const points = normalize(S.detail.samples || [], 0);
  S.detailPoints = points;
  const mode = points.some((point) => point.gpu != null) ? "gpu" : "cpu";
  const legend = document.getElementById("detail-legend");
  if (legend) legend.innerHTML = legendHTML(mode);
  drawChart(canvas, points, mode, S.detailHover);
}

function drawDetailLoss() {
  const canvas = document.getElementById("loss-chart");
  if (!canvas || !S.detail) return;
  const points = (S.detail.metrics || [])
    .filter((item) => item.data && typeof item.data.loss === "number")
    .map((item) => ({ ts: item.ts, loss: item.data.loss }));
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = rect.width;
  const h = rect.height;
  if (w < 10 || h < 10 || points.length < 1) return;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pad = { l: 48, r: 12, t: 14, b: 26 };
  const min = Math.min(...points.map((point) => point.loss));
  const max = Math.max(...points.map((point) => point.loss));
  const span = (max - min) || 1;
  const yOf = (value) => pad.t + (1 - (value - min) / span) * (h - pad.t - pad.b);
  const xOf = (index) => {
    const inner = w - pad.l - pad.r;
    if (points.length <= 1) return pad.l + inner / 2;
    return pad.l + (index / (points.length - 1)) * inner;
  };
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = xOf(index);
    const y = yOf(point.loss);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#d6ff4a";
  ctx.lineWidth = 1.7;
  ctx.stroke();
  ctx.lineTo(xOf(points.length - 1), h - pad.b);
  ctx.lineTo(xOf(0), h - pad.b);
  ctx.closePath();
  ctx.fillStyle = "rgba(214,255,74,0.12)";
  ctx.fill();
  ctx.font = "11px JetBrains Mono, Cascadia Mono, monospace";
  ctx.fillStyle = "rgba(154,148,136,0.95)";
  ctx.textAlign = "left";
  ctx.fillText(max.toFixed(2), 2, pad.t + 8);
  ctx.fillText(min.toFixed(2), 2, h - pad.b);
}

function onMainClick(event) {
  const button = event.target.closest("[data-gpu]");
  if (!button) return;
  S.gpu = Number(button.dataset.gpu);
  updateLive();
}

function onLive(data) {
  S.live = data;
  setConn("live");
  paintShell();
  const route = parseRoute();
  if (route.page !== "live") return;
  if (S.mounted !== "live") mountLive();
  else updateLive();
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  let socket;
  try {
    socket = new WebSocket(`${proto}://${location.host}/api/stream`);
  } catch (err) {
    setTimeout(connect, 1500);
    return;
  }
  socket.onmessage = (event) => {
    try {
      onLive(JSON.parse(event.data));
    } catch (err) {
      /* ignore a bad frame */
    }
  };
  socket.onclose = () => setTimeout(connect, 1500);
}

async function poll() {
  try {
    const response = await fetch("/api/live");
    if (!response.ok) throw new Error("offline");
    onLive(await response.json());
  } catch (err) {
    setConn("retry");
  }
}

function render() {
  const route = parseRoute();
  document.getElementById("nav-live").classList.toggle("active", route.page === "live");
  document.getElementById("nav-runs").classList.toggle("active", route.page !== "live");
  if (route.page === "live") {
    if (S.mounted !== "live") mountLive();
    else updateLive();
  } else if (route.page === "runs") {
    mountRuns();
  } else {
    mountDetail(route.id);
  }
}

function boot() {
  tickClock();
  setInterval(tickClock, 1000);
  document.getElementById("main").addEventListener("click", onMainClick);
  window.addEventListener("hashchange", () => render());
  fetch("/api/health").then((response) => response.json()).then((health) => {
    S.health = health;
    const ver = document.getElementById("ver");
    if (ver) ver.textContent = `v${health.version}`;
    const foot = document.getElementById("foot");
    if (foot && health.db) foot.title = health.db;
  }).catch(() => {});
  poll();
  setInterval(poll, 1000);
  connect();
  render();
}

boot();
