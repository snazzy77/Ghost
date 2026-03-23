# Ghost

Local chat-mimic app with:
- Instant chat after upload (retrieval + profile)
- Optional background LoRA training (Redis + RQ worker)
- Multi-chat browser UI (thread list like WhatsApp)

## Quick Start (Windows)

### 1) Setup venv

```powershell
cd C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Check interpreter:

```powershell
python -c "import sys; print(sys.executable)"
```

It should end with `Ghost\.venv\Scripts\python.exe`.

### 2) Start services (3 terminals)

Terminal A (Redis):

```powershell
docker start ghost-redis
```

If it does not exist yet:

```powershell
docker run --name ghost-redis -p 6379:6379 redis:7
```

Terminal B (API):

```powershell
cd C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```

Terminal C (Worker):

```powershell
cd C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost
.\.venv\Scripts\Activate.ps1
& "C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost\.venv\Scripts\python.exe" .\run_worker.py
```

### 3) Open UI

Open:

`http://127.0.0.1:8000/`

---

## Browser Workflow

1. Upload `.jsonl` chat file.
2. Enter `target_speaker` exactly as it appears in `speaker`.
3. Click `Upload + Start Training`.
4. Use left sidebar to switch chat threads.
5. Use `Refresh Status` / `Auto Poll` for training status.

Each upload creates a new `conversation_id` thread.

## Modes

### Instant vs Tuned

- `instant`: chat works immediately using base model + retrieval/profile.
- `tuned`: LoRA adapter finished and is active for that conversation.

When `/train-status/{job_id}` returns `completed`, that conversation should use tuned mode.

### Retrieval Modes (Chat)

- Default (`Retrieval-Only` unchecked):
  - Retrieves top similar examples
  - Generates a new response with those examples in prompt
- Retrieval-only (`Retrieval-Only` checked):
  - Returns closest real historical reply directly

---

## API Commands (Optional)

Use `curl.exe` in PowerShell (not `curl` alias).

### Upload

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload" `
  -F "file=@data/message_1_converted_50_dedup.jsonl" `
  -F "target_speaker=user20063110" `
  -F "model_id=Qwen/Qwen2.5-1.5B-Instruct" `
  -F "max_length=384" `
  -F "grad_accum=64" `
  -F "epochs=0.5"
```

### Status

```powershell
curl.exe "http://127.0.0.1:8000/train-status/YOUR_JOB_ID"
```

Auto-poll:

```powershell
while ($true) { curl.exe "http://127.0.0.1:8000/train-status/YOUR_JOB_ID"; Start-Sleep 10 }
```

### Chat (default retrieval + generation)

```powershell
curl.exe -X POST "http://127.0.0.1:8000/chat" `
  -H "Content-Type: application/json" `
  -d "{\"conversation_id\":\"YOUR_CONVERSATION_ID\",\"message\":\"hey what are you up to?\",\"history\":[],\"retrieval_only\":false,\"retrieval_k\":4}"
```

### Chat (retrieval-only)

```powershell
curl.exe -X POST "http://127.0.0.1:8000/chat" `
  -H "Content-Type: application/json" `
  -d "{\"conversation_id\":\"YOUR_CONVERSATION_ID\",\"message\":\"hey what are you up to?\",\"history\":[],\"retrieval_only\":true,\"retrieval_k\":4}"
```

### List conversations

```powershell
curl.exe "http://127.0.0.1:8000/conversations"
```

---

## Troubleshooting

### Docker error: daemon not running

If you see `failed to connect to the docker API...dockerDesktopLinuxEngine`:

1. Start Docker Desktop.
2. Wait until it is fully running.
3. Run `docker version`.
4. Then run `docker start ghost-redis`.

### `No module named redis`

```powershell
python -m pip install redis "rq>=1.16.0,<2"
python -m pip install -r requirements.txt
```

### Worker says `can't open file ... run_worker.py`

You are in wrong folder. Must be inside `...\Ghost`.

```powershell
cd C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost
& "C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost\.venv\Scripts\python.exe" .\run_worker.py
```

### `/upload` fails: `No messages found for friend 'X'`

`target_speaker` does not match dataset speaker exactly.

```powershell
Get-Content data\message_1_converted_50_dedup.jsonl | ForEach-Object { ($_ | ConvertFrom-Json).speaker } | Sort-Object -Unique
```

### Job status is `failed`

Check the log path from `/train-status`:

```powershell
Get-Content "C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost\app_state\logs\YOUR_JOB_ID.log" -Tail 120
```

### GPU check

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## Stop Everything

API terminal: `Ctrl + C`  
Worker terminal: `Ctrl + C`

Stop Redis:

```powershell
docker stop ghost-redis
```

Optional remove container:

```powershell
docker rm ghost-redis
```

---

## Input File Format

JSONL, one message per line:

```json
{"speaker":"you","text":"are you free?","timestamp":"2025-01-02T18:11:00Z"}
{"speaker":"alex","text":"yeah around 7 works","timestamp":"2025-01-02T18:11:19Z"}
```

Required keys: `speaker`, `text`.

## Scripts

- `prepare_data.py`
- `train_lora_qwen3.py`
- `chat_llm.py`
