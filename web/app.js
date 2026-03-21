const uploadForm = document.getElementById("uploadForm");
const chatForm = document.getElementById("chatForm");
const fileInput = document.getElementById("fileInput");
const speakerInput = document.getElementById("speakerInput");
const modelInput = document.getElementById("modelInput");
const maxLengthInput = document.getElementById("maxLengthInput");
const gradAccumInput = document.getElementById("gradAccumInput");
const epochsInput = document.getElementById("epochsInput");
const conversationIdInput = document.getElementById("conversationId");
const jobIdInput = document.getElementById("jobId");
const statusBox = document.getElementById("statusBox");
const refreshBtn = document.getElementById("refreshBtn");
const pollBtn = document.getElementById("pollBtn");
const stopPollBtn = document.getElementById("stopPollBtn");
const chatBox = document.getElementById("chatBox");
const chatInput = document.getElementById("chatInput");

let pollTimer = null;
const history = [];

function setStatus(obj) {
  statusBox.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function addMsg(role, text) {
  const row = document.createElement("div");
  row.className = "msg";
  row.innerHTML = `<span class="role">${role}:</span><span>${text}</span>`;
  chatBox.appendChild(row);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function refreshStatus() {
  const jobId = jobIdInput.value.trim();
  if (!jobId) {
    setStatus("No job id yet.");
    return;
  }
  const res = await fetch(`/train-status/${encodeURIComponent(jobId)}`);
  const data = await res.json();
  setStatus(data);
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Pick a .jsonl file.");
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  fd.append("target_speaker", speakerInput.value.trim());
  fd.append("model_id", modelInput.value.trim());
  fd.append("max_length", maxLengthInput.value.trim());
  fd.append("grad_accum", gradAccumInput.value.trim());
  fd.append("epochs", epochsInput.value.trim());

  setStatus("Uploading...");
  const res = await fetch("/upload", { method: "POST", body: fd });
  const data = await res.json();
  conversationIdInput.value = data.conversation_id || "";
  jobIdInput.value = data.job_id || "";
  setStatus(data);
});

refreshBtn.addEventListener("click", async () => {
  await refreshStatus();
});

pollBtn.addEventListener("click", () => {
  if (pollTimer) return;
  refreshStatus();
  pollTimer = setInterval(refreshStatus, 10000);
  setStatus("Auto poll started (10s).");
});

stopPollBtn.addEventListener("click", () => {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  setStatus("Auto poll stopped.");
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const conversationId = conversationIdInput.value.trim();
  const text = chatInput.value.trim();
  if (!conversationId) {
    setStatus("Upload first to get a conversation id.");
    return;
  }
  if (!text) return;

  addMsg("you", text);
  history.push({ role: "user", content: text });
  chatInput.value = "";

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      message: text,
      history,
    }),
  });
  const data = await res.json();
  addMsg(`ghost (${data.mode || "unknown"})`, data.reply || "(no reply)");
  history.push({ role: "assistant", content: data.reply || "" });
});
