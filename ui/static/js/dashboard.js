// Clean White & Gold Dashboard Logic - Dual Side-by-Side Comparison
document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("videoFileInput");
  const selectedFileInfo = document.getElementById("selectedFileInfo");
  const runBtn = document.getElementById("runBtn");
  const btnText = document.getElementById("btnText");

  // Dual Video Players
  const rawVideoPlayer = document.getElementById("rawVideoPlayer");
  const rawEmptyState = document.getElementById("rawEmptyState");
  const rawVideoInfo = document.getElementById("rawVideoInfo");

  const detectedVideoPlayer = document.getElementById("detectedVideoPlayer");
  const detectedEmptyState = document.getElementById("detectedEmptyState");
  const detectedVideoInfo = document.getElementById("detectedVideoInfo");

  // Synchronized Controls
  const syncControls = document.getElementById("syncControls");
  const playBothBtn = document.getElementById("playBothBtn");
  const pauseBothBtn = document.getElementById("pauseBothBtn");
  const syncTimeBtn = document.getElementById("syncTimeBtn");

  // Metrics and Logs
  const progressBar = document.getElementById("progressBar");
  const progressContainer = document.getElementById("progressContainer");
  const incidentList = document.getElementById("incidentList");
  const totalIncidentsEl = document.getElementById("totalIncidents");
  const activeEntitiesEl = document.getElementById("activeEntities");
  const statusStateEl = document.getElementById("statusState");

  let currentVideoPath = null;
  let isAnalyzing = false;
  let pollInterval = null;
  let isSyncing = false;

  // 1. File Upload Interaction
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      handleFileUpload(fileInput.files[0]);
    }
  });

  async function handleFileUpload(file) {
    if (!file) return;

    selectedFileInfo.textContent = `Uploading ${file.name}...`;
    runBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Upload failed");
      }

      const data = await res.json();
      currentVideoPath = data.path;

      // Update Left Video (Raw Footage)
      selectedFileInfo.textContent = `📁 ${data.name}`;
      rawVideoInfo.textContent = data.name;
      rawVideoPlayer.src = `/api/video_stream?file=${encodeURIComponent(data.path)}`;
      rawEmptyState.style.display = "none";
      rawVideoPlayer.style.display = "block";
      rawVideoPlayer.load();

      // Reset Right Video (AI Detected) to empty state until analysis runs
      detectedVideoPlayer.pause();
      detectedVideoPlayer.src = "";
      detectedVideoPlayer.style.display = "none";
      detectedEmptyState.style.display = "flex";
      detectedVideoInfo.textContent = "Awaiting Detection";
      syncControls.style.display = "none";

      runBtn.disabled = false;
      statusStateEl.textContent = "Video Ready";
    } catch (err) {
      alert("Error uploading video: " + err.message);
      selectedFileInfo.textContent = "Upload failed. Try again.";
    }
  }

  // 2. Synchronized Playback Handlers
  function syncSeek(source, target) {
    if (isSyncing || !target.duration) return;
    isSyncing = true;
    if (Math.abs(target.currentTime - source.currentTime) > 0.15) {
      target.currentTime = source.currentTime;
    }
    setTimeout(() => { isSyncing = false; }, 80);
  }

  rawVideoPlayer.addEventListener("seeking", () => {
    if (detectedVideoPlayer.src) syncSeek(rawVideoPlayer, detectedVideoPlayer);
  });

  detectedVideoPlayer.addEventListener("seeking", () => {
    if (rawVideoPlayer.src) syncSeek(detectedVideoPlayer, rawVideoPlayer);
  });

  rawVideoPlayer.addEventListener("play", () => {
    if (detectedVideoPlayer.src && detectedVideoPlayer.paused) {
      detectedVideoPlayer.play().catch(() => {});
    }
  });

  rawVideoPlayer.addEventListener("pause", () => {
    if (detectedVideoPlayer.src && !detectedVideoPlayer.paused) {
      detectedVideoPlayer.pause();
    }
  });

  detectedVideoPlayer.addEventListener("play", () => {
    if (rawVideoPlayer.src && rawVideoPlayer.paused) {
      rawVideoPlayer.play().catch(() => {});
    }
  });

  detectedVideoPlayer.addEventListener("pause", () => {
    if (rawVideoPlayer.src && !rawVideoPlayer.paused) {
      rawVideoPlayer.pause();
    }
  });

  // Sync Toolbar Buttons
  playBothBtn.addEventListener("click", () => {
    rawVideoPlayer.play().catch(() => {});
    detectedVideoPlayer.play().catch(() => {});
  });

  pauseBothBtn.addEventListener("click", () => {
    rawVideoPlayer.pause();
    detectedVideoPlayer.pause();
  });

  syncTimeBtn.addEventListener("click", () => {
    detectedVideoPlayer.currentTime = rawVideoPlayer.currentTime;
  });

  // 3. Fetch and Render Incident Audit Log
  async function loadIncidents() {
    try {
      const res = await fetch("/api/events");
      const data = await res.json();
      renderIncidents(data);
    } catch (err) {
      console.error("Failed to fetch events:", err);
    }
  }

  function renderIncidents(data) {
    const events = data.events || [];
    totalIncidentsEl.textContent = events.length;

    const entityCount = data.total_tracked_entities !== undefined && data.total_tracked_entities > 0
      ? data.total_tracked_entities
      : new Set(events.map((e) => e.entity_id)).size;
    activeEntitiesEl.textContent = entityCount;

    if (events.length === 0) {
      incidentList.innerHTML = `
        <div style="color: var(--text-muted); text-align: center; padding: 3rem 1rem;">
          No incidents detected yet. Run detection to analyze.
        </div>`;
      return;
    }

    incidentList.innerHTML = "";
    events.slice().reverse().forEach((evt) => {
      const card = document.createElement("div");
      card.className = "incident-card";
      card.innerHTML = `
        <div class="incident-top">
          <span class="incident-entity">${escapeHtml(evt.entity_id)}</span>
          <span class="incident-badge">${escapeHtml(evt.action_type)}</span>
        </div>
        <div class="incident-time">
          ⏱ <strong>${escapeHtml(evt.start_time)}</strong> → <strong>${escapeHtml(evt.end_time)}</strong> (${evt.duration_seconds}s)
        </div>
        <div class="incident-reason">
          ${escapeHtml(evt.reason)}
        </div>
      `;
      incidentList.appendChild(card);
    });
  }

  // 4. Trigger Analysis on Uploaded Video
  runBtn.addEventListener("click", async () => {
    if (!currentVideoPath || isAnalyzing) return;

    isAnalyzing = true;
    runBtn.disabled = true;
    btnText.textContent = "Analyzing on GPU...";
    progressContainer.style.display = "block";
    progressBar.style.width = "0%";
    statusStateEl.textContent = "CUDA Inference Active";

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_path: currentVideoPath }),
      });
      const data = await res.json();

      if (!res.ok) {
        alert(data.error || "Failed to start analysis");
        resetRunState();
        return;
      }

      // Start polling
      pollInterval = setInterval(checkStatus, 400);
    } catch (err) {
      alert("Error starting analysis: " + err.message);
      resetRunState();
    }
  });

  async function checkStatus() {
    try {
      const res = await fetch("/api/status");
      const status = await res.json();

      if (status.running) {
        const pct = Math.round(status.progress * 100);
        progressBar.style.width = `${pct}%`;
        loadIncidents();
      } else {
        clearInterval(pollInterval);
        progressBar.style.width = "100%";
        setTimeout(() => {
          progressContainer.style.display = "none";
        }, 1200);

        // Load Right Video (Annotated Output with cache-busting timestamp)
        if (status.output_video) {
          detectedVideoInfo.textContent = "AI Annotated (Active)";
          detectedVideoPlayer.src = `/api/video_stream?file=${encodeURIComponent(status.output_video)}&t=${Date.now()}`;
          detectedEmptyState.style.display = "none";
          detectedVideoPlayer.style.display = "block";
          detectedVideoPlayer.load();

          // Reveal Synchronized Controls
          syncControls.style.display = "flex";

          // Align timestamps & play both side-by-side
          setTimeout(() => {
            rawVideoPlayer.currentTime = 0;
            detectedVideoPlayer.currentTime = 0;
            rawVideoPlayer.play().catch(() => {});
            detectedVideoPlayer.play().catch(() => {});
          }, 400);
        }

        loadIncidents();
        resetRunState();
        statusStateEl.textContent = "Analysis Complete";
      }
    } catch (err) {
      console.error("Status check failed:", err);
    }
  }

  // 5. Auto-load existing uploaded and annotated videos on refresh
  async function checkExistingVideos() {
    try {
      const res = await fetch("/api/videos");
      const data = await res.json();
      const inputVid = data.videos.find((v) => v.name.includes("[Input]"));
      const annotated = data.videos.find((v) => v.name.includes("[Annotated]"));

      if (inputVid) {
        currentVideoPath = inputVid.path;
        const cleanName = inputVid.name.replace("[Input] ", "");
        rawVideoInfo.textContent = cleanName;
        selectedFileInfo.textContent = `📁 ${cleanName}`;
        rawVideoPlayer.src = `/api/video_stream?file=${encodeURIComponent(inputVid.path)}`;
        rawEmptyState.style.display = "none";
        rawVideoPlayer.style.display = "block";
        rawVideoPlayer.load();
        runBtn.disabled = false;
        statusStateEl.textContent = "Video Ready";
      }

      if (annotated) {
        detectedVideoInfo.textContent = "AI Annotated (Active)";
        detectedVideoPlayer.src = `/api/video_stream?file=${encodeURIComponent(annotated.path)}&t=${Date.now()}`;
        detectedEmptyState.style.display = "none";
        detectedVideoPlayer.style.display = "block";
        detectedVideoPlayer.load();
        syncControls.style.display = "flex";
      }
    } catch (e) {
      console.log("No previous videos to restore", e);
    }
  }

  function resetRunState() {
    isAnalyzing = false;
    runBtn.disabled = false;
    btnText.textContent = "Detect Behaviour";
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/[&<>"']/g, (m) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[m]));
  }

  // Initial load
  loadIncidents();
  checkExistingVideos();
});
