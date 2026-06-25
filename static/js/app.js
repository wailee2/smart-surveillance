"use strict";
/* ═══════════════════════════════════════════════════════════════════════════
   Smart Surveillance System v3 — Control Room UI
   ═══════════════════════════════════════════════════════════════════════════ */

// ── Clock ───────────────────────────────────────────────────────────────────
function tickClock() {
  const now = new Date();
  document.getElementById("clock").textContent =
    [now.getHours(), now.getMinutes(), now.getSeconds()]
      .map(n => String(n).padStart(2, "0")).join(":");
}
tickClock();
setInterval(tickClock, 1000);

// ── Health check ────────────────────────────────────────────────────────────
const sysStatus  = document.getElementById("sys-status");
const sysDevice  = document.getElementById("sys-device");
const sysModels  = document.getElementById("sys-models");
const badge      = document.getElementById("analyzing-badge");

async function checkHealth() {
  try {
    const d = await fetch("/health").then(r => r.json());
    sysStatus.textContent = "ONLINE";
    sysStatus.className   = "sys-val ok";
    sysDevice.textContent = d.cpu_mode ? "CPU" : "GPU";
    sysModels.textContent = d.models_loaded ? "LOADED" : "LOADING…";
  } catch {
    sysStatus.textContent = "OFFLINE";
    sysStatus.className   = "sys-val err";
  }
}
checkHealth();
setInterval(checkHealth, 20000);

// ── VTL ─────────────────────────────────────────────────────────────────────
const vtlEnabled   = document.getElementById("vtl-enabled");
const vtlModeLabel = document.getElementById("vtl-mode-label");
const tlBulbs      = { red: document.getElementById("tl-red"), amber: document.getElementById("tl-amber"), green: document.getElementById("tl-green") };
const tlBtns       = { red: document.getElementById("vtl-red-btn"), amber: document.getElementById("vtl-amber-btn"), green: document.getElementById("vtl-green-btn") };
let currentSignal  = "green";

function applySignalUI(color) {
  currentSignal = color;
  // Bulbs
  Object.entries(tlBulbs).forEach(([c, b]) => b.classList.toggle("active", c === color));
  // Buttons
  Object.entries(tlBtns).forEach(([c, b]) => b.classList.toggle("active", c === color));
  // Mode label
  const labels = { red: "● RED – violation mode active", amber: "● AMBER – caution mode", green: "● GREEN – normal mode" };
  vtlModeLabel.textContent  = labels[color];
  vtlModeLabel.className    = `vtl-mode-label ${color}`;
}

async function setVtlColor(color) {
  applySignalUI(color);
  try {
    await fetch(`/api/vtl?color=${color}&override=${vtlEnabled.checked}`, { method: "POST" });
  } catch {}
}

Object.entries(tlBtns).forEach(([c, b]) => b.addEventListener("click", () => setVtlColor(c)));
vtlEnabled.addEventListener("change", () => setVtlColor(currentSignal));

// Sync VTL state on load
fetch("/api/vtl").then(r => r.json()).then(d => applySignalUI(d.color)).catch(() => {});

// ── File selection ───────────────────────────────────────────────────────────
const fileInput       = document.getElementById("file-input");
const browseBtn       = document.getElementById("browse-btn");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const footerFile      = document.getElementById("footer-file");

let selectedFile = null;

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) selectFile(fileInput.files[0]);
});

// Drag-and-drop onto the feed area
const feedWrap = document.getElementById("feed-wrap");
feedWrap.addEventListener("dragover", e => { e.preventDefault(); feedWrap.style.outline = "2px solid var(--accent)"; });
feedWrap.addEventListener("dragleave", () => feedWrap.style.outline = "");
feedWrap.addEventListener("drop", e => {
  e.preventDefault();
  feedWrap.style.outline = "";
  if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
});

function selectFile(file) {
  if (!file.type.startsWith("video/")) { alert("Please select a video file."); return; }
  selectedFile = file;
  footerFile.textContent = file.name;
  // Show first frame for stop-line drawing
  showFirstFrame(file);
  uploadPlaceholder.classList.add("hidden");
}

// ── First-frame preview for stop-line drawing ────────────────────────────────
const drawCanvas  = document.getElementById("draw-canvas");
const drawCtx     = drawCanvas.getContext("2d");
const lineStatusBox = document.getElementById("line-status-box");
const lineCoords  = document.getElementById("line-coords");
const stopLineYInput = document.getElementById("stop-line-y");
const drawLineBtn = document.getElementById("draw-line-btn");

let firstFrameImg = null;     // ImageBitmap of frame 0
let stopLineY_canvas = null;  // Y in canvas-native pixels
let isDragging = false;

function showFirstFrame(file) {
  const video = document.createElement("video");
  video.src = URL.createObjectURL(file);
  video.currentTime = 0.5;
  video.muted = true;
  video.addEventListener("seeked", async () => {
    // Draw to off-screen canvas to grab ImageBitmap
    const tmp = document.createElement("canvas");
    tmp.width  = video.videoWidth;
    tmp.height = video.videoHeight;
    tmp.getContext("2d").drawImage(video, 0, 0);
    firstFrameImg = await createImageBitmap(tmp);
    URL.revokeObjectURL(video.src);

    // Size the overlay canvas
    drawCanvas.width  = video.videoWidth;
    drawCanvas.height = video.videoHeight;
    drawCanvas.classList.remove("hidden");
    redrawCanvas();
  }, { once: true });
  video.load();
}

function redrawCanvas() {
  if (!firstFrameImg) return;
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.drawImage(firstFrameImg, 0, 0);
  if (stopLineY_canvas !== null) {
    drawCtx.strokeStyle = "#ff2d2d";
    drawCtx.lineWidth   = 3;
    drawCtx.beginPath();
    drawCtx.moveTo(0, stopLineY_canvas);
    drawCtx.lineTo(drawCanvas.width, stopLineY_canvas);
    drawCtx.stroke();
    drawCtx.fillStyle = "#ff2d2d";
    drawCtx.font      = "bold 16px 'Share Tech Mono'";
    drawCtx.fillText(`STOP LINE  Y=${stopLineY_canvas}`, 10, stopLineY_canvas - 8);
  }
}

// Click on canvas to set stop line
drawCanvas.addEventListener("mousedown", e => { isDragging = true; updateStopLine(e); });
drawCanvas.addEventListener("mousemove", e => { if (isDragging) updateStopLine(e); });
drawCanvas.addEventListener("mouseup",   () => { isDragging = false; });

function updateStopLine(e) {
  const rect  = drawCanvas.getBoundingClientRect();
  const scaleY = drawCanvas.height / rect.height;
  const y = Math.round((e.clientY - rect.top) * scaleY);
  stopLineY_canvas = y;
  stopLineYInput.value = y;
  lineStatusBox.classList.remove("hidden");
  lineCoords.textContent = `[0,${y}] → [${drawCanvas.width},${y}]`;
  redrawCanvas();
}

drawLineBtn.addEventListener("click", () => {
  if (!firstFrameImg) { alert("Select a video first."); return; }
  drawCanvas.classList.remove("hidden");
  redrawCanvas();
});

// Manual Y input sync
stopLineYInput.addEventListener("input", () => {
  const y = parseInt(stopLineYInput.value);
  if (!isNaN(y)) {
    stopLineY_canvas = y;
    lineStatusBox.classList.remove("hidden");
    lineCoords.textContent = `Y = ${y}px`;
    redrawCanvas();
  }
});

// ── Conf slider ──────────────────────────────────────────────────────────────
const confSlider = document.getElementById("conf-slider");
const confVal    = document.getElementById("conf-val");
confSlider.addEventListener("input", () => confVal.textContent = confSlider.value + "%");

// ── Start Analysis ───────────────────────────────────────────────────────────
const startBtn   = document.getElementById("start-btn");
const abortBtn   = document.getElementById("abort-btn");
const livePreview = document.getElementById("live-preview");
const videoResult = document.getElementById("output-video");
const videoResultWrap = document.getElementById("video-result-wrap");
const violationFlash  = document.getElementById("violation-flash");
const violationAlert  = document.getElementById("violation-alert");
const downloadStrip   = document.getElementById("download-strip");
const feedLabel       = document.getElementById("feed-label");
const feedMeta        = document.getElementById("feed-meta");
const footerJob       = document.getElementById("footer-job");
const footerTime      = document.getElementById("footer-time");

// Stats
const statFrames     = document.getElementById("stat-frames");
const statViolations = document.getElementById("stat-violations");
const statFpsVal     = document.getElementById("stat-fps-val");
const statProgress   = document.getElementById("stat-progress");
const progressFill   = document.getElementById("progress-fill");

let currentJobId  = null;
let pollInterval  = null;
let sseSource     = null;
let violationCount = 0;
let aborted       = false;

startBtn.addEventListener("click", async () => {
  if (!selectedFile) { alert("Select a video file first."); return; }

  const fd = new FormData();
  fd.append("file", selectedFile);
  const stopY = stopLineYInput.value;
  if (stopY) fd.append("stop_line_y", parseInt(stopY));
  fd.append("vtl_override", document.getElementById("vtl-override-proc").checked);
  fd.append("speed_limit",  parseFloat(document.getElementById("speed-limit").value) || 50);
  fd.append("conf_threshold", (parseInt(confSlider.value) || 40) / 100);

  // Reset UI
  violationCount = 0; aborted = false;
  statFrames.textContent = "0";
  statViolations.textContent = "0";
  statFpsVal.textContent = "—";
  statProgress.textContent = "0%";
  progressFill.style.width = "0%";
  document.getElementById("violation-list").innerHTML = "<div class='no-violations'>Processing…</div>";
  violationAlert.classList.add("hidden");
  downloadStrip.classList.add("hidden");
  videoResultWrap.classList.add("hidden");
  violationFlash.classList.add("hidden");
  livePreview.classList.add("hidden");
  drawCanvas.classList.add("hidden");
  badge.className = "analyzing-badge active";
  badge.innerHTML = "<span class='pulse-dot'></span> ANALYZING";
  startBtn.classList.add("hidden");
  abortBtn.classList.remove("hidden");

  try {
    const res = await fetch("/api/process", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    currentJobId = data.job_id;
    footerJob.textContent = `JOB: ${currentJobId}`;

    // Start MJPEG preview
    livePreview.src = `/api/job/${currentJobId}/preview?t=${Date.now()}`;
    livePreview.classList.remove("hidden");
    feedLabel.textContent = "LIVE FEED";
    feedMeta.textContent  = selectedFile.name;

    startSSE(currentJobId);
    startPolling(currentJobId);
  } catch (err) {
    alert("Failed to start: " + err.message);
    resetUI();
  }
});

abortBtn.addEventListener("click", async () => {
  aborted = true;
  clearInterval(pollInterval);
  if (sseSource) sseSource.close();
  livePreview.src = "";
  resetUI();
  badge.innerHTML = "<span class='pulse-dot'></span> ABORTED";
  badge.className = "analyzing-badge";
});

// ── SSE — live violations ────────────────────────────────────────────────────
function startSSE(jobId) {
  if (sseSource) sseSource.close();
  sseSource = new EventSource(`/api/job/${jobId}/events`);

  sseSource.onmessage = e => {
    const evt = JSON.parse(e.data);
    if (evt.type === "violation") {
      violationCount++;
      statViolations.textContent = violationCount;

      // Flash overlay
      violationFlash.classList.remove("hidden");
      violationAlert.classList.remove("hidden");
      setTimeout(() => {
        violationFlash.classList.add("hidden");
        violationAlert.classList.add("hidden");
      }, 2000);

      // Push placeholder card immediately (snapshot comes after OCR)
      pushViolationCard({
        track_id:      evt.track_id,
        vehicle_type:  evt.vehicle,
        plate_number:  "PROCESSING…",
        speed_kmh:     evt.speed,
        violation_type: evt.violation,
        timestamp_real: evt.timestamp,
        snapshot_path:  null,
      });

      footerTime.textContent = `LAST: ${evt.timestamp}s`;
    } else if (evt.type === "done") {
      sseSource.close();
    }
  };
  sseSource.onerror = () => sseSource.close();
}

// ── Polling ──────────────────────────────────────────────────────────────────
function startPolling(jobId) {
  clearInterval(pollInterval);
  pollInterval = setInterval(() => pollJob(jobId), 1500);
}

async function pollJob(jobId) {
  if (aborted) return;
  try {
    const d = await fetch(`/api/job/${jobId}`).then(r => r.json());
    const pct = d.progress || 0;

    statFrames.textContent   = (d.frame || 0).toLocaleString();
    statProgress.textContent = pct + "%";
    statFpsVal.textContent   = d.fps_proc > 0 ? d.fps_proc.toFixed(1) : "—";
    progressFill.style.width = pct + "%";
    if (d.total_frames) feedMeta.textContent = `${fmtNum(d.frame)} / ${fmtNum(d.total_frames)} frames`;

    if (d.status === "done") {
      clearInterval(pollInterval);
      onJobDone(jobId, d.result);
    } else if (d.status === "error") {
      clearInterval(pollInterval);
      alert("Processing error: " + (d.error || "Unknown error"));
      resetUI();
    }
  } catch {}
}

// ── Job done ─────────────────────────────────────────────────────────────────
function onJobDone(jobId, result) {
  badge.className = "analyzing-badge done";
  badge.innerHTML = "<span class='pulse-dot'></span> COMPLETE";

  // Hide live MJPEG, show final video player
  livePreview.src = "";
  livePreview.classList.add("hidden");
  drawCanvas.classList.add("hidden");
  videoResultWrap.classList.remove("hidden");
  const videoUrl = `/api/job/${jobId}/video?t=${Date.now()}`;
  videoResult.src = videoUrl;
  videoResult.load();

  // Download links
  document.getElementById("dl-video").href = videoUrl;
  document.getElementById("dl-csv").href   = `/api/job/${jobId}/csv`;
  downloadStrip.classList.remove("hidden");

  // Re-render violation cards with real OCR plates + snapshots
  const offenders = result.offenders || [];
  if (offenders.length) {
    document.getElementById("violation-list").innerHTML = "";
    offenders.forEach(o => pushViolationCard(o, true));
    statViolations.textContent = offenders.length;
  } else {
    document.getElementById("violation-list").innerHTML =
      "<div class='no-violations'>No violations detected</div>";
  }

  // Final stats update
  statFrames.textContent   = fmtNum(result.frames_processed || 0);
  statProgress.textContent = "100%";
  progressFill.style.width = "100%";
  feedMeta.textContent = `Done — ${fmtDur(result.processing_time_s || 0)}`;
  feedLabel.textContent = "ANALYSIS COMPLETE";

  startBtn.classList.remove("hidden");
  abortBtn.classList.add("hidden");
}

// ── Violation card builder ────────────────────────────────────────────────────
function pushViolationCard(o, replace = false) {
  const list = document.getElementById("violation-list");

  // Clear "no violations" placeholder
  if (list.querySelector(".no-violations")) list.innerHTML = "";

  // If replace mode: remove any existing card for this track
  if (replace) {
    const existing = list.querySelector(`[data-tid="${o.track_id}"]`);
    if (existing) existing.remove();
  }

  const types = (o.violation_type || "").split(",").filter(Boolean);
  const cls   = types.length > 1 ? "both" : (types[0] || "");
  const vioLabels = types.map(t =>
    `<span class="vcard-vio ${t.trim()}">${t.trim() === "speeding" ? "🚨 SPEEDING" : "🚦 RED LIGHT"}</span>`
  ).join(" ");

  const snapHtml = o.snapshot_path
    ? `<img class="vcard-snap" src="${o.snapshot_path}" alt="snap" loading="lazy"/>`
    : `<div class="vcard-snap-ph">🚗</div>`;

  const card = document.createElement("div");
  card.className = `vcard ${cls}`;
  card.dataset.tid = o.track_id;
  card.innerHTML = `
    ${snapHtml}
    <div class="vcard-body">
      <div class="vcard-plate">${o.plate_number || "—"}</div>
      <div class="vcard-row"><span>${o.vehicle_type || "—"}</span><span>#${o.track_id}</span></div>
      <div class="vcard-row"><span>${o.speed_kmh != null ? o.speed_kmh + " km/h" : "—"}</span><span>${o.timestamp_real != null ? o.timestamp_real + "s" : "—"}</span></div>
      <div>${vioLabels}</div>
    </div>`;

  // Prepend (newest on top)
  list.prepend(card);
}

// ── Reset UI ──────────────────────────────────────────────────────────────────
function resetUI() {
  badge.className = "analyzing-badge";
  badge.innerHTML = "<span class='pulse-dot'></span> STANDBY";
  startBtn.classList.remove("hidden");
  abortBtn.classList.add("hidden");
  livePreview.src = "";
  livePreview.classList.add("hidden");
}

// ── History drawer ────────────────────────────────────────────────────────────
const historyDrawer  = document.getElementById("history-drawer");
const drawerBackdrop = document.getElementById("drawer-backdrop");
const historyToggle  = document.getElementById("history-toggle-btn");
const historyClose   = document.getElementById("history-close-btn");

historyToggle.addEventListener("click", () => {
  historyDrawer.classList.remove("hidden");
  drawerBackdrop.classList.remove("hidden");
  loadHistory();
});

function closeHistory() {
  historyDrawer.classList.add("hidden");
  drawerBackdrop.classList.add("hidden");
}

historyClose.addEventListener("click", closeHistory);
drawerBackdrop.addEventListener("click", closeHistory);

async function loadHistory() {
  const list = document.getElementById("history-list");
  list.innerHTML = "<p style='padding:12px;font-family:var(--mono);font-size:10px;color:var(--text-dim)'>Loading…</p>";
  try {
    const sessions = await fetch("/api/sessions").then(r => r.json());
    if (!sessions.length) {
      list.innerHTML = "<p style='padding:12px;font-family:var(--mono);font-size:10px;color:var(--text-dim)'>No sessions yet.</p>";
      return;
    }
    list.innerHTML = sessions.map(s => `
      <div class="h-row" onclick="restoreSession('${s.job_id}')">
        <div class="h-dot ${s.status || 'queued'}"></div>
        <div class="h-info">
          <div class="h-fname">${s.filename || s.job_id}</div>
          <div class="h-meta">${s.created_at || ""} · ${s.duration_s != null ? fmtDur(s.duration_s) : ""}</div>
        </div>
        <div class="h-badges">
          ${s.speeding_count  ? `<span class="h-badge red">${s.speeding_count}🚨</span>` : ""}
          ${s.red_light_count ? `<span class="h-badge amber">${s.red_light_count}🚦</span>` : ""}
        </div>
      </div>`).join("");
  } catch {
    list.innerHTML = "<p style='padding:12px;font-family:var(--mono);font-size:10px;color:var(--red)'>Failed to load.</p>";
  }
}

async function restoreSession(jobId) {
  closeHistory();
  try {
    const d = await fetch(`/api/job/${jobId}`).then(r => r.json());
    if (d.status === "done" && d.result) {
      currentJobId = jobId;
      violationCount = d.result.offenders_count || 0;

      // Show the final video
      videoResultWrap.classList.remove("hidden");
      livePreview.classList.add("hidden");
      drawCanvas.classList.add("hidden");
      uploadPlaceholder.classList.add("hidden");
      const videoUrl = `/api/job/${jobId}/video?t=${Date.now()}`;
      videoResult.src = videoUrl;
      videoResult.load();

      document.getElementById("dl-video").href = videoUrl;
      document.getElementById("dl-csv").href   = `/api/job/${jobId}/csv`;
      downloadStrip.classList.remove("hidden");

      // Rebuild cards
      document.getElementById("violation-list").innerHTML = "";
      (d.result.offenders || []).forEach(o => pushViolationCard(o, false));
      statViolations.textContent = violationCount;
      statFrames.textContent     = fmtNum(d.result.frames_processed || 0);
      statProgress.textContent   = "100%";
      progressFill.style.width   = "100%";
      footerJob.textContent      = `JOB: ${jobId}`;
      feedLabel.textContent      = "RESTORED SESSION";
      badge.className = "analyzing-badge done";
      badge.innerHTML = "<span class='pulse-dot'></span> COMPLETE";
    } else {
      alert(`Session ${jobId}: status = ${d.status}`);
    }
  } catch {
    alert("Could not restore session " + jobId);
  }
}

// Expose for inline onclick
window.restoreSession = restoreSession;

// ── Utils ─────────────────────────────────────────────────────────────────────
function fmtDur(s) {
  if (!s) return "0s";
  if (s < 60) return s.toFixed(1) + "s";
  return `${Math.floor(s/60)}m ${Math.round(s%60)}s`;
}
function fmtNum(n) {
  return Number(n).toLocaleString();
}
