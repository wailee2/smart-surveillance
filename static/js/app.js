/* ── Smart Surveillance — Frontend App ──────────────────────────────────── */

"use strict";

// ── DOM refs ───────────────────────────────────────────────────────────────
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const browseBtn      = document.getElementById("browse-btn");
const fileInfo       = document.getElementById("file-info");
const fileName       = document.getElementById("file-name");
const fileSize       = document.getElementById("file-size");
const clearFileBtn   = document.getElementById("clear-file");
const processBtn     = document.getElementById("process-btn");
const speedLimitInp  = document.getElementById("speed-limit");
const confSlider     = document.getElementById("conf-threshold");
const confVal        = document.getElementById("conf-val");

const progressSection = document.getElementById("progress-section");
const progressFill    = document.getElementById("progress-fill");
const progressPct     = document.getElementById("progress-pct");
const progressBadge   = document.getElementById("progress-badge");
const statFrames      = document.getElementById("stat-frames");
const statFps         = document.getElementById("stat-fps");
const statJob         = document.getElementById("stat-job");

const resultsSection  = document.getElementById("results-section");
const offendersSection= document.getElementById("offenders-section");
const offendersBody   = document.getElementById("offenders-body");
const offenderCountBadge = document.getElementById("offender-count-badge");
const outputVideo     = document.getElementById("output-video");
const dlVideoBtn      = document.getElementById("dl-video");
const dlCsvBtn        = document.getElementById("dl-csv");
const newJobBtn       = document.getElementById("new-job-btn");

const statusDot       = document.getElementById("status-dot");
const statusText      = document.getElementById("status-text");
const deviceBadge     = document.getElementById("device-badge");

// ── State ──────────────────────────────────────────────────────────────────
let selectedFile = null;
let pollInterval = null;
let currentJobId = null;

// ── Health check ───────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch("/health");
    const data = await r.json();
    if (data.status === "ok") {
      statusDot.className = "status-dot ok";
      statusText.textContent = data.models_loaded ? "Ready" : "Models loading…";
      deviceBadge.textContent = data.cpu_mode ? "CPU Mode" : "GPU Mode";
      deviceBadge.style.color = data.cpu_mode ? "var(--warn)" : "var(--ok)";
    } else {
      statusDot.className = "status-dot error";
      statusText.textContent = "Error";
    }
  } catch {
    statusDot.className = "status-dot error";
    statusText.textContent = "Offline";
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

// ── Slider live value ──────────────────────────────────────────────────────
confSlider.addEventListener("input", () => {
  confVal.textContent = confSlider.value + "%";
});

// ── File selection ─────────────────────────────────────────────────────────
browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFile(fileInput.files[0]));

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
dropZone.addEventListener("click", e => {
  if (e.target !== browseBtn) fileInput.click();
});

function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("video/")) {
    alert("Please select a video file.");
    return;
  }
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  fileInfo.classList.remove("hidden");
  dropZone.classList.add("hidden");
  processBtn.disabled = false;
}

clearFileBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  fileInfo.classList.add("hidden");
  dropZone.classList.remove("hidden");
  processBtn.disabled = true;
});

// ── Process ────────────────────────────────────────────────────────────────
processBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append("file", selectedFile);

  // Patch config via URL params (simple approach for demo)
  const speed = parseInt(speedLimitInp.value) || 50;
  const conf  = (parseInt(confSlider.value) || 40) / 100;

  // Show progress
  showSection("progress");
  progressFill.style.width = "0%";
  progressPct.textContent = "0%";
  statJob.textContent = "Submitting…";
  progressBadge.textContent = "Running";
  processBtn.disabled = true;

  try {
    const r = await fetch(`/api/process?speed_limit=${speed}&conf=${conf}`, {
      method: "POST",
      body: formData,
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    currentJobId = data.job_id;
    statJob.textContent = currentJobId;
    startPolling(currentJobId);
  } catch (err) {
    alert("Upload failed: " + err.message);
    showSection("upload");
    processBtn.disabled = false;
  }
});

// ── Polling ────────────────────────────────────────────────────────────────
function startPolling(jobId) {
  clearInterval(pollInterval);
  pollInterval = setInterval(() => pollJob(jobId), 2000);
}

async function pollJob(jobId) {
  try {
    const r = await fetch(`/api/job/${jobId}`);
    if (!r.ok) return;
    const data = await r.json();

    const pct = data.progress || 0;
    progressFill.style.width = pct + "%";
    progressPct.textContent  = pct + "%";

    if (data.total_frames > 0) {
      statFrames.textContent = `${data.frame.toLocaleString()} / ${data.total_frames.toLocaleString()}`;
    }
    if (data.fps_proc > 0) statFps.textContent = data.fps_proc.toFixed(2) + " fps";

    if (data.status === "done") {
      clearInterval(pollInterval);
      renderResults(jobId, data.result);
    } else if (data.status === "error") {
      clearInterval(pollInterval);
      progressBadge.textContent = "Error";
      progressBadge.className = "badge badge-danger";
      alert("Processing error: " + (data.error || "Unknown error"));
      showSection("upload");
      processBtn.disabled = false;
    }
  } catch {/* network hiccup — keep polling */}
}

// ── Results ────────────────────────────────────────────────────────────────
function renderResults(jobId, result) {
  document.getElementById("res-frames").textContent    = (result.frames_processed || 0).toLocaleString();
  document.getElementById("res-duration").textContent  = formatDuration(result.duration_s || 0);
  document.getElementById("res-proc-time").textContent = formatDuration(result.processing_time_s || 0);
  document.getElementById("res-offenders").textContent = result.offenders_count || 0;
  document.getElementById("res-speeding").textContent  = result.speeding_count  || 0;
  document.getElementById("res-redlight").textContent  = result.red_light_count || 0;

  // Video
  const videoUrl = `/api/job/${jobId}/video`;
  outputVideo.src = videoUrl;
  dlVideoBtn.href = videoUrl;
  dlVideoBtn.download = `${jobId}_annotated.mp4`;

  // CSV
  dlCsvBtn.href = `/api/job/${jobId}/csv`;
  dlCsvBtn.download = `${jobId}_offenders.csv`;

  // Table
  const offenders = result.offenders || [];
  offenderCountBadge.textContent = offenders.length;
  renderOffendersTable(offenders);

  showSection("results");
}

function renderOffendersTable(offenders) {
  if (!offenders.length) {
    offendersBody.innerHTML = `<tr><td colspan="6" class="empty-msg">No violations detected.</td></tr>`;
    offendersSection.classList.add("hidden");
    return;
  }

  offendersSection.classList.remove("hidden");
  offendersBody.innerHTML = offenders.map(o => {
    const types = (o.violation_type || "").split(",").filter(Boolean);
    const chips  = types.map(t => `<span class="vio-chip vio-${t.trim()}">${t.trim().replace("_", " ")}</span>`).join(" ");
    const speed  = o.speed_kmh != null ? `<strong>${o.speed_kmh}</strong>` : "—";
    const ts     = o.timestamp_real != null ? `${o.timestamp_real}s` : "—";
    return `<tr>
      <td class="mono">#${o.track_id}</td>
      <td>${o.vehicle_type || "—"}</td>
      <td class="mono">${o.plate_number || "—"}</td>
      <td>${speed}</td>
      <td>${chips}</td>
      <td>${ts}</td>
    </tr>`;
  }).join("");
}

// ── New job ────────────────────────────────────────────────────────────────
newJobBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  currentJobId = null;
  fileInfo.classList.add("hidden");
  dropZone.classList.remove("hidden");
  processBtn.disabled = true;
  progressFill.style.width = "0%";
  outputVideo.src = "";
  showSection("upload");
});

// ── Section helper ─────────────────────────────────────────────────────────
function showSection(name) {
  const map = {
    upload:    ["upload-section"],
    progress:  ["upload-section", "progress-section"],
    results:   ["results-section"],
  };
  // Hide all
  ["upload-section", "progress-section", "results-section", "offenders-section"]
    .forEach(id => document.getElementById(id)?.classList.add("hidden"));

  (map[name] || []).forEach(id => document.getElementById(id)?.classList.remove("hidden"));

  if (name === "results") {
    // offenders section shown separately by renderResults
    document.getElementById("results-section").classList.remove("hidden");
  }

  // Update nav
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.section === name);
  });
}

// ── Nav links ──────────────────────────────────────────────────────────────
document.querySelectorAll(".nav-item").forEach(link => {
  link.addEventListener("click", e => {
    const section = link.dataset.section;
    if (section === "results" && !currentJobId) { e.preventDefault(); return; }
    if (section === "offenders") {
      document.getElementById("offenders-section")?.classList.toggle("hidden");
      e.preventDefault();
      return;
    }
  });
});

// ── Utils ──────────────────────────────────────────────────────────────────
function formatBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024 ** 2) return (n / 1024).toFixed(1) + " KB";
  if (n < 1024 ** 3) return (n / 1024 ** 2).toFixed(1) + " MB";
  return (n / 1024 ** 3).toFixed(2) + " GB";
}

function formatDuration(s) {
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  return `${m}m ${sec}s`;
}
