# Ghost API (Instant + Background LoRA)

Ghost supports:
- Instant chat right after upload (base model + retrieval/profile)
- Background LoRA training
- Automatic switch to tuned adapter when ready

## 1) Setup (Always Use a venv)

Always run Ghost inside `Ghost\.venv` (never global Python).

### Windows PowerShell

```powershell
cd C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Every new terminal

```powershell
cd C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

Expected output ends with:
`...\Ghost\.venv\Scripts\python.exe`

### Optional NVIDIA CUDA torch install

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 2) Start Everything (4 terminals)

### Terminal 1: Redis

First time:
```powershell
docker run --name ghost-redis -p 6379:6379 redis:7
```

Later runs:
```powershell
docker start ghost-redis
```

If conflict/stuck:
```powershell
docker rm -f ghost-redis
docker run --name ghost-redis -p 6379:6379 redis:7
```

### Terminal 2: API

```powershell
cd C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```

### Terminal 3: Worker

```powershell
cd C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost
.\.venv\Scripts\Activate.ps1
& "C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost\.venv\Scripts\python.exe" run_worker.py
```

### Terminal 4: Commands / checks

Use this for `curl.exe`, status checks, and pip installs.

## 3) Browser UI Flow

Open:
`http://127.0.0.1:8000/`

In the UI:
1. Choose file (`.jsonl`)
2. Enter `target_speaker` exactly as it appears in `speaker`
3. Click `Upload + Start Training`
4. Use `Refresh Status` / `Auto Poll`
5. Chat immediately in instant mode; it auto-switches to tuned mode when training completes

## 3.1) What "Instant Mode" Means

- `instant` = upload is processed and chat works immediately, but background LoRA training is still running.
- `tuned` = training finished and the adapter is active for that conversation.

You can verify with:

```powershell
curl.exe "http://127.0.0.1:8000/train-status/YOUR_JOB_ID"
```

When `status` becomes `completed`, chat for that `conversation_id` uses the tuned adapter.

## 4) Terminal API Flow (optional)

### Upload

Use `curl.exe` in PowerShell (not `curl` alias):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/upload" `
  -F "file=@data/message_1_converted_50_dedup.jsonl" `
  -F "target_speaker=user20063110" `
  -F "model_id=Qwen/Qwen2.5-1.5B-Instruct" `
  -F "max_length=384" `
  -F "grad_accum=64" `
  -F "epochs=0.5"
```

### Check status

```powershell
curl.exe "http://127.0.0.1:8000/train-status/YOUR_JOB_ID"
```

Auto-poll:
```powershell
while ($true) { curl.exe "http://127.0.0.1:8000/train-status/YOUR_JOB_ID"; Start-Sleep 10 }
```

### Chat

```powershell
curl.exe -X POST "http://127.0.0.1:8000/chat" `
  -H "Content-Type: application/json" `
  -d "{\"conversation_id\":\"YOUR_CONVERSATION_ID\",\"message\":\"hey what are you up to?\",\"history\":[]}"
```

## 5) Troubleshooting

### `No module named redis` / missing packages

```powershell
python -m pip install redis "rq>=1.16.0,<2"
python -m pip install -r requirements.txt
```

### Wrong Python interpreter keeps getting used

Run with absolute Ghost interpreter:

```powershell
& "C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --port 8000
& "C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost\.venv\Scripts\python.exe" run_worker.py
```

### `/upload` fails: `No messages found for friend 'X'`

`target_speaker` is wrong. List speakers first:

```powershell
Get-Content data\message_1_converted_50_dedup.jsonl | ForEach-Object { ($_ | ConvertFrom-Json).speaker } | Sort-Object -Unique
```

Then upload using one exact speaker name.

### Job says `failed`

Check log path from `/train-status`:

```powershell
Get-Content "C:\Users\girid\OneDrive\Documents\Gokul_Projects\Gokul_Workspace\Ghost\app_state\logs\YOUR_JOB_ID.log" -Tail 120
```

### GPU not used (`no accelerator is found`)

Check:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If `False`, install CUDA torch in Ghost venv (see setup section).

### Job status is `queued` for too long

- Make sure Terminal 3 worker is running and shows:
  `*** Listening on ghost-train...`
- Re-upload once to create a fresh job if an old queued job is stale.

### Why quality can feel weak

- 50 lines is very small for style tuning.
- Use more chat lines when possible (`message_1_converted.jsonl` instead of only 50 lines).
- Increase epochs moderately (for small data, try `1.5` to `3.0`).

### What epochs mean

- `1.0` epoch = one full pass through dataset
- `2.0` epochs = two full passes
- `0.5` epoch = half pass

More epochs can improve style fit, but too many on tiny data can overfit.

## 6) Stop Commands (Clean Shutdown)

### Stop worker (`run_worker.py`)

In worker terminal:
`Ctrl + C`

### Stop API (`uvicorn`)

In API terminal:
`Ctrl + C`

### Stop Redis container

```powershell
docker stop ghost-redis
```

Optional remove container:

```powershell
docker rm ghost-redis
```

## Upload File Format

JSONL, one message per line:

```json
{"speaker":"you","text":"are you free?","timestamp":"2025-01-02T18:11:00Z"}
{"speaker":"alex","text":"yeah around 7 works","timestamp":"2025-01-02T18:11:19Z"}
```

Required keys: `speaker`, `text`.

## CLI scripts (still available)

- `prepare_data.py`
- `train_lora_qwen3.py`
- `chat_llm.py`
