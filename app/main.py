import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from redis import Redis
from rq import Queue

from prepare_data import build_pairs, build_profile, build_style_system_prompt, load_jsonl

from .config import APP_STATE_DIR, BASE_DIR, DEFAULT_MODEL, REDIS_URL, UPLOADS_DIR
from .db import (
    create_training_job,
    get_conversation,
    get_training_job,
    init_db,
    list_conversations,
    update_training_job,
    upsert_conversation,
)
from .llm_runtime import runtime
from .schemas import ChatRequest, ChatResponse, ConversationSummary, TrainStatusResponse, UploadResponse
from .tasks import run_training_job


app = FastAPI(title="Ghost API", version="0.1.0")
WEB_DIR = BASE_DIR / "web"


def try_get_queue() -> Queue | None:
    try:
        redis_conn = Redis.from_url(REDIS_URL)
        redis_conn.ping()
        return Queue("ghost-train", connection=redis_conn)
    except Exception:
        return None


@app.on_event("startup")
def startup() -> None:
    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found")
    return FileResponse(page)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/conversations", response_model=list[ConversationSummary])
def conversations() -> list[ConversationSummary]:
    rows = list_conversations()
    return [
        ConversationSummary(
            conversation_id=row["id"],
            target_speaker=row["target_speaker"],
            model_id=row["model_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            latest_job_id=row["latest_job_id"],
            latest_job_status=row["latest_job_status"],
        )
        for row in rows
    ]


@app.post("/upload", response_model=UploadResponse)
async def upload_chat(
    file: UploadFile = File(...),
    target_speaker: str = Form(...),
    model_id: str = Form(DEFAULT_MODEL),
    enqueue_training: bool = Form(True),
    max_length: int = Form(512),
    batch_size: int = Form(1),
    grad_accum: int = Form(64),
    epochs: float = Form(0.5),
) -> UploadResponse:
    if not target_speaker.strip():
        raise HTTPException(status_code=400, detail="target_speaker is required")

    conversation_id = uuid4().hex
    convo_dir = UPLOADS_DIR / conversation_id
    convo_dir.mkdir(parents=True, exist_ok=True)

    input_path = convo_dir / "input.jsonl"
    profile_path = convo_dir / "profile.json"
    pairs_path = convo_dir / "pairs.json"
    sft_path = convo_dir / "sft_train.jsonl"

    raw = await file.read()
    input_path.write_bytes(raw)

    messages = load_jsonl(input_path)
    profile = build_profile(messages, target_speaker)
    pairs = build_pairs(messages, target_speaker)
    if not pairs:
        raise HTTPException(status_code=400, detail="No user->target_speaker pairs found in uploaded file.")

    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    pairs_path.write_text(
        json.dumps([{"user_text": u, "friend_reply": a} for u, a in pairs], indent=2),
        encoding="utf-8",
    )
    system_prompt = build_style_system_prompt(profile)
    with sft_path.open("w", encoding="utf-8") as f:
        for user_text, friend_reply in pairs:
            row = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": friend_reply},
                ]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    upsert_conversation(
        conversation_id=conversation_id,
        target_speaker=target_speaker,
        model_id=model_id,
        input_path=input_path,
        profile_path=profile_path,
        pairs_path=pairs_path,
        sft_path=sft_path,
    )

    job_id = None
    queue = try_get_queue()
    if enqueue_training and queue is not None:
        job_id = uuid4().hex
        rq_job = queue.enqueue(
            run_training_job,
            job_id,
            conversation_id,
            model_id,
            str(sft_path),
            max_length,
            batch_size,
            grad_accum,
            epochs,
            job_timeout="24h",
            result_ttl=7 * 24 * 3600,
        )
        create_training_job(
            job_id=job_id,
            conversation_id=conversation_id,
            status="queued",
            queue_job_id=rq_job.id,
        )
    elif enqueue_training:
        job_id = uuid4().hex
        create_training_job(
            job_id=job_id,
            conversation_id=conversation_id,
            status="failed",
            error="Redis queue unavailable. Start Redis and RQ worker to enable training.",
        )

    return UploadResponse(
        conversation_id=conversation_id,
        job_id=job_id,
        ready_mode="instant",
        message="Upload processed. Chat is ready in instant mode.",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    convo = get_conversation(req.conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="conversation_id not found")

    adapter_path = convo["active_adapter_path"]
    mode = "tuned" if adapter_path and Path(adapter_path).exists() else "instant"
    reply = runtime.generate_reply(
        model_id=convo["model_id"],
        profile_path=convo["profile_path"],
        pairs_path=convo["pairs_path"],
        message=req.message,
        history=req.history,
        adapter_path=adapter_path if mode == "tuned" else None,
        retrieval_only=req.retrieval_only,
        retrieval_k=req.retrieval_k,
    )
    return ChatResponse(conversation_id=req.conversation_id, mode=mode, reply=reply)


@app.get("/train-status/{job_id}", response_model=TrainStatusResponse)
def train_status(job_id: str) -> TrainStatusResponse:
    row = get_training_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job_id not found")

    queue = try_get_queue()
    if queue is not None and row["queue_job_id"] and row["status"] in {"queued", "running"}:
        rq_job = queue.fetch_job(row["queue_job_id"])
        if rq_job is not None:
            if rq_job.is_queued and row["status"] != "queued":
                update_training_job(job_id, status="queued")
            elif rq_job.is_started and row["status"] != "running":
                update_training_job(job_id, status="running")
            elif rq_job.is_failed and row["status"] != "failed":
                update_training_job(job_id, status="failed", error=str(rq_job.exc_info or "Worker failed"))

            row = get_training_job(job_id)

    assert row is not None
    return TrainStatusResponse(
        job_id=row["id"],
        status=row["status"],
        conversation_id=row["conversation_id"],
        adapter_path=row["adapter_path"],
        log_path=row["log_path"],
        error=row["error"],
    )
