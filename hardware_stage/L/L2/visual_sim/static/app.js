const boardCanvas = document.getElementById("boardCanvas");
const cameraCanvas = document.getElementById("cameraCanvas");
const statsEl = document.getElementById("stats");
const logBox = document.getElementById("logBox");

const boardCtx = boardCanvas.getContext("2d");
const camCtx = cameraCanvas.getContext("2d");

let sim = null;
const camStateHistory = [];

const pieceMap = {
  0: ".",
  1: "wP", 2: "wN", 3: "wB", 4: "wQ", 5: "wK",
  6: "bP", 7: "bN", 8: "bB", 9: "bQ", 10: "bK"
};

async function post(path) {
  await fetch(path, { method: "POST" });
}

async function postJson(path, body) {
  await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
}

async function refreshState() {
  try {
    const res = await fetch("/api/state");
    sim = await res.json();
    render();
  } catch (err) {
    console.error(err);
  }
}

function worldToPx(x, y, size, margin) {
  const min = -180;
  const max = 180;
  const inner = size - margin * 2;
  const px = margin + ((x - min) / (max - min)) * inner;
  const py = size - (margin + ((y - min) / (max - min)) * inner);
  return [px, py];
}

function drawBoard() {
  const w = boardCanvas.width;
  const h = boardCanvas.height;
  boardCtx.clearRect(0, 0, w, h);

  const margin = 70;
  const size = Math.min(w, h);
  const boardSize = size - margin * 2;
  const cell = boardSize / 6;

  boardCtx.fillStyle = "#07101a";
  boardCtx.fillRect(0, 0, w, h);

  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      const x = margin + c * cell;
      const y = margin + (5 - r) * cell;
      boardCtx.fillStyle = (r + c) % 2 === 0 ? "#2a4358" : "#1f3142";
      boardCtx.fillRect(x, y, cell, cell);
    }
  }

  boardCtx.strokeStyle = "#5a86ad";
  boardCtx.lineWidth = 2;
  boardCtx.strokeRect(margin, margin, boardSize, boardSize);

  boardCtx.fillStyle = "#9dc2df";
  boardCtx.font = "16px Segoe UI";
  for (let i = 0; i < 6; i++) {
    const file = String.fromCharCode(65 + i);
    const rank = String(i + 1);
    boardCtx.fillText(file, margin + i * cell + cell * 0.46, margin + boardSize + 24);
    boardCtx.fillText(rank, margin - 28, margin + (5 - i) * cell + cell * 0.56);
  }

  if (!sim) {
    return;
  }

  boardCtx.textAlign = "center";
  boardCtx.textBaseline = "middle";
  boardCtx.font = "bold 24px Segoe UI";

  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      const p = sim.board[r][c];
      if (!p) continue;
      const x = margin + c * cell + cell / 2;
      const y = margin + (5 - r) * cell + cell / 2;

      boardCtx.fillStyle = p <= 5 ? "#f5e7ce" : "#0f171f";
      boardCtx.strokeStyle = p <= 5 ? "#dcbf8f" : "#6b879e";
      boardCtx.lineWidth = 2;
      boardCtx.beginPath();
      boardCtx.arc(x, y, cell * 0.33, 0, Math.PI * 2);
      boardCtx.fill();
      boardCtx.stroke();

      boardCtx.fillStyle = p <= 5 ? "#273c53" : "#d8e9f8";
      boardCtx.fillText(pieceMap[p], x, y);
    }
  }

  const [ax, ay] = worldToPx(sim.arm.x, sim.arm.y, size, margin);
  const azNorm = Math.max(0, Math.min(1, sim.arm.z / 200));
  const color = sim.arm.holding ? "#ffb84d" : "#66d9ff";

  boardCtx.strokeStyle = "rgba(120,180,220,0.45)";
  boardCtx.lineWidth = 3;
  boardCtx.beginPath();
  boardCtx.moveTo(ax, margin - 26);
  boardCtx.lineTo(ax, ay);
  boardCtx.stroke();

  boardCtx.fillStyle = color;
  boardCtx.beginPath();
  boardCtx.arc(ax, ay, 12 + (1 - azNorm) * 8, 0, Math.PI * 2);
  boardCtx.fill();

  boardCtx.fillStyle = "#b9d4eb";
  boardCtx.font = "14px Segoe UI";
  boardCtx.fillText(`z=${sim.arm.z.toFixed(1)}`, ax + 38, ay);
}

function drawCamera() {
  const w = cameraCanvas.width;
  const h = cameraCanvas.height;
  camCtx.clearRect(0, 0, w, h);

  const cfg = sim && sim.config ? sim.config : {};
  const lagMs = cfg.camera_lag_ms || 0;
  const noise = cfg.camera_noise || 0;

  if (sim) {
    camStateHistory.push({ ts: performance.now(), board: JSON.parse(JSON.stringify(sim.board)) });
    while (camStateHistory.length > 200) camStateHistory.shift();
  }

  let boardView = sim ? sim.board : null;
  if (lagMs > 0 && camStateHistory.length) {
    const targetTs = performance.now() - lagMs;
    for (let i = camStateHistory.length - 1; i >= 0; i--) {
      if (camStateHistory[i].ts <= targetTs) {
        boardView = camStateHistory[i].board;
        break;
      }
    }
  }

  camCtx.fillStyle = "#0b131c";
  camCtx.fillRect(0, 0, w, h);

  camCtx.save();
  camCtx.translate(w * 0.5, h * 0.53);
  camCtx.rotate((-7 * Math.PI) / 180);
  camCtx.scale(1.04, 0.86);

  const size = 280;
  const cell = size / 6;
  const ox = -size / 2;
  const oy = -size / 2;

  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < 6; c++) {
      const x = ox + c * cell;
      const y = oy + r * cell;
      camCtx.fillStyle = (r + c) % 2 === 0 ? "#3e5665" : "#2d414f";
      camCtx.fillRect(x, y, cell, cell);
    }
  }

  if (boardView) {
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        const p = boardView[r][c];
        if (!p) continue;
        const x = ox + c * cell + cell / 2;
        const y = oy + (5 - r) * cell + cell / 2;

        camCtx.fillStyle = p <= 5 ? "#f4e5c9" : "#101722";
        camCtx.beginPath();
        camCtx.arc(x, y, cell * 0.28, 0, Math.PI * 2);
        camCtx.fill();

        camCtx.strokeStyle = "#89c3ff";
        camCtx.lineWidth = 1.5;
        camCtx.strokeRect(x - 8, y - 8, 16, 16);
      }
    }
  }

  camCtx.restore();

  for (let y = 0; y < h; y += 4) {
    camCtx.fillStyle = "rgba(12, 20, 30, 0.08)";
    camCtx.fillRect(0, y, w, 1);
  }

  if (noise > 0) {
    const dots = Math.floor(500 * noise);
    for (let i = 0; i < dots; i++) {
      const x = Math.random() * w;
      const y = Math.random() * h;
      const a = 0.03 + Math.random() * 0.2 * noise;
      camCtx.fillStyle = `rgba(190,220,245,${a.toFixed(3)})`;
      camCtx.fillRect(x, y, 1, 1);
    }
  }

  camCtx.fillStyle = "#9bc2df";
  camCtx.font = "12px Consolas";
  const mode = sim ? (sim.running ? "RUN" : "PAUSE") : "-";
  camCtx.fillText(`CAM MODE: ${mode}`, 12, 20);
}

function renderStats() {
  if (!sim) return;
  statsEl.textContent = `Turn: ${sim.turn_white ? "White" : "Black"} | Move #: ${sim.move_count} | Last: ${sim.last_move || "-"} | Phase: ${sim.current_phase}`;

  const lines = sim.log.map(line => `<div class="log-line">${line}</div>`).join("");
  logBox.innerHTML = lines;
}

function bindFaultControlsFromState() {
  if (!sim || !sim.config) return;
  const c = sim.config;
  document.getElementById("simSpeed").value = c.sim_speed;
  document.getElementById("apiDelay").value = c.api_delay_ms;
  document.getElementById("planDelay").value = c.plan_delay_ms;
  document.getElementById("dropProb").value = c.command_drop_prob;
  document.getElementById("invalidProb").value = c.invalid_move_prob;
  document.getElementById("stallProb").value = c.arm_stall_prob;
  document.getElementById("camLag").value = c.camera_lag_ms;
  document.getElementById("camNoise").value = c.camera_noise;
  updateFaultLabels();
}

function updateFaultLabels() {
  document.getElementById("simSpeedVal").textContent = `${Number(document.getElementById("simSpeed").value).toFixed(1)}x`;
  document.getElementById("apiDelayVal").textContent = document.getElementById("apiDelay").value;
  document.getElementById("planDelayVal").textContent = document.getElementById("planDelay").value;
  document.getElementById("dropProbVal").textContent = Number(document.getElementById("dropProb").value).toFixed(2);
  document.getElementById("invalidProbVal").textContent = Number(document.getElementById("invalidProb").value).toFixed(2);
  document.getElementById("stallProbVal").textContent = Number(document.getElementById("stallProb").value).toFixed(2);
  document.getElementById("camLagVal").textContent = document.getElementById("camLag").value;
  document.getElementById("camNoiseVal").textContent = Number(document.getElementById("camNoise").value).toFixed(2);
}

async function applyFaults() {
  const body = {
    sim_speed: Number(document.getElementById("simSpeed").value),
    api_delay_ms: Number(document.getElementById("apiDelay").value),
    plan_delay_ms: Number(document.getElementById("planDelay").value),
    command_drop_prob: Number(document.getElementById("dropProb").value),
    invalid_move_prob: Number(document.getElementById("invalidProb").value),
    arm_stall_prob: Number(document.getElementById("stallProb").value),
    camera_lag_ms: Number(document.getElementById("camLag").value),
    camera_noise: Number(document.getElementById("camNoise").value)
  };
  await postJson("/api/config", body);
}

function render() {
  drawBoard();
  drawCamera();
  renderStats();
}

function wireControls() {
  document.getElementById("startBtn").addEventListener("click", () => post("/api/start"));
  document.getElementById("pauseBtn").addEventListener("click", () => post("/api/pause"));
  document.getElementById("stepBtn").addEventListener("click", () => post("/api/step"));
  document.getElementById("resetBtn").addEventListener("click", () => post("/api/reset"));
  document.getElementById("applyFaultsBtn").addEventListener("click", applyFaults);

  ["simSpeed", "apiDelay", "planDelay", "dropProb", "invalidProb", "stallProb", "camLag", "camNoise"]
    .forEach(id => document.getElementById(id).addEventListener("input", updateFaultLabels));
}

wireControls();
setInterval(refreshState, 90);
refreshState();

setTimeout(bindFaultControlsFromState, 350);
