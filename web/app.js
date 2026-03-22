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
const refreshConversationsBtn = document.getElementById("refreshConversationsBtn");
const conversationList = document.getElementById("conversationList");
const chatTitle = document.getElementById("chatTitle");
const chatBox = document.getElementById("chatBox");
const chatInput = document.getElementById("chatInput");
const retrievalOnlyInput = document.getElementById("retrievalOnlyInput");

let pollTimer = null;
let currentConversationId = null;
let conversations = [];
const STORAGE_KEY = "ghost_thread_state_v1";
let threadState = loadThreadState();

function loadThreadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return {};
    return parsed;
  } catch {
    return {};
  }
}

function saveThreadState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(threadState));
}

function getCurrentHistory() {
  if (!currentConversationId) return [];
  if (!Array.isArray(threadState[currentConversationId])) {
    threadState[currentConversationId] = [];
  }
  return threadState[currentConversationId];
}

function setStatus(obj) {
  statusBox.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function addMsg(role, text, clear = false) {
  if (clear) chatBox.innerHTML = "";
  const row = document.createElement("div");
  row.className = "msg";
  row.innerHTML = `<span class="role">${role}:</span><span>${text}</span>`;
  chatBox.appendChild(row);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function renderChat() {
  chatBox.innerHTML = "";
  if (!currentConversationId) {
    chatTitle.textContent = "No chat selected";
    return;
  }
  const convo = conversations.find((c) => c.conversation_id === currentConversationId);
  const name = convo ? convo.target_speaker : currentConversationId.slice(0, 8);
  chatTitle.textContent = `Chat with ${name}`;
  const history = getCurrentHistory();
  for (const item of history) {
    const role = item.role === "assistant" ? "ghost" : "you";
    addMsg(role, item.content);
  }
}

function selectConversation(conversationId) {
  currentConversationId = conversationId;
  conversationIdInput.value = conversationId || "";
  const convo = conversations.find((c) => c.conversation_id === conversationId);
  jobIdInput.value = convo?.latest_job_id || "";
  renderConversationList();
  renderChat();
}

function renderConversationList() {
  conversationList.innerHTML = "";
  if (!conversations.length) {
    conversationList.innerHTML = `<div class="conversationMeta" style="padding:10px;">No chats yet. Upload a file to create one.</div>`;
    return;
  }

  for (const convo of conversations) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `conversationItem${convo.conversation_id === currentConversationId ? " active" : ""}`;
    btn.innerHTML = `
      <div class="conversationName">${convo.target_speaker}</div>
      <div class="conversationMeta">${convo.model_id}</div>
      <div class="conversationMeta">Job: ${convo.latest_job_status || "n/a"}</div>
    `;
    btn.addEventListener("click", () => selectConversation(convo.conversation_id));
    conversationList.appendChild(btn);
  }
}

async function loadConversations(preferConversationId = null) {
  const res = await fetch("/conversations");
  const data = await res.json();
  conversations = Array.isArray(data) ? data : [];
  renderConversationList();

  if (preferConversationId) {
    selectConversation(preferConversationId);
    return;
  }
  if (currentConversationId && conversations.some((c) => c.conversation_id === currentConversationId)) {
    selectConversation(currentConversationId);
    return;
  }
  if (conversations.length) {
    selectConversation(conversations[0].conversation_id);
  } else {
    currentConversationId = null;
    conversationIdInput.value = "";
    jobIdInput.value = "";
    renderChat();
  }
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
  if (data.conversation_id && !threadState[data.conversation_id]) {
    threadState[data.conversation_id] = [];
    saveThreadState();
  }
  await loadConversations(data.conversation_id || null);
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

refreshConversationsBtn.addEventListener("click", async () => {
  await loadConversations(currentConversationId);
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const conversationId = (currentConversationId || conversationIdInput.value || "").trim();
  const text = chatInput.value.trim();
  if (!conversationId) {
    setStatus("Select a chat first.");
    return;
  }
  if (!text) return;

  if (!threadState[conversationId]) {
    threadState[conversationId] = [];
  }

  addMsg("you", text);
  threadState[conversationId].push({ role: "user", content: text });
  saveThreadState();
  chatInput.value = "";

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      message: text,
      history: threadState[conversationId],
      retrieval_only: !!retrievalOnlyInput?.checked,
      retrieval_k: 4,
    }),
  });
  const data = await res.json();
  addMsg(`ghost (${data.mode || "unknown"})`, data.reply || "(no reply)");
  threadState[conversationId].push({ role: "assistant", content: data.reply || "" });
  saveThreadState();
});

loadConversations().catch((err) => {
  setStatus(`Could not load conversations: ${String(err)}`);
});
