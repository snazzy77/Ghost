import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from .config import ADAPTERS_DIR, BASE_DIR, LOGS_DIR
from .db import set_active_adapter, update_training_job


def run_training_job(
    job_id: str,
    conversation_id: str,
    model_id: str,
    sft_path: str,
    max_length: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
) -> dict:
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    adapter_path = ADAPTERS_DIR / f"{conversation_id}-{uuid4().hex[:8]}"
    log_path = LOGS_DIR / f"{job_id}.log"
    update_training_job(job_id, status="running", adapter_path=adapter_path, log_path=log_path)

    train_script = BASE_DIR / "train_lora_qwen3.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--model",
        model_id,
        "--data",
        sft_path,
        "--output",
        str(adapter_path),
        "--max-length",
        str(max_length),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--epochs",
        str(epochs),
    ]

    env = os.environ.copy()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )

    if proc.returncode != 0:
        update_training_job(job_id, status="failed", error=f"Training failed. See log: {log_path}")
        return {"ok": False, "job_id": job_id, "log_path": str(log_path)}

    set_active_adapter(conversation_id, adapter_path)
    update_training_job(job_id, status="completed", adapter_path=adapter_path, log_path=log_path)
    return {"ok": True, "job_id": job_id, "adapter_path": str(adapter_path), "log_path": str(log_path)}

