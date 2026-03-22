import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import DB_PATH


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                target_speaker TEXT NOT NULL,
                model_id TEXT NOT NULL,
                input_path TEXT NOT NULL,
                profile_path TEXT NOT NULL,
                pairs_path TEXT NOT NULL,
                sft_path TEXT NOT NULL,
                active_adapter_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                queue_job_id TEXT,
                status TEXT NOT NULL,
                adapter_path TEXT,
                log_path TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )


def upsert_conversation(
    conversation_id: str,
    target_speaker: str,
    model_id: str,
    input_path: Path,
    profile_path: Path,
    pairs_path: Path,
    sft_path: Path,
    active_adapter_path: Path | None = None,
) -> None:
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO conversations (
                id, target_speaker, model_id, input_path, profile_path, pairs_path, sft_path,
                active_adapter_path, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                target_speaker=excluded.target_speaker,
                model_id=excluded.model_id,
                input_path=excluded.input_path,
                profile_path=excluded.profile_path,
                pairs_path=excluded.pairs_path,
                sft_path=excluded.sft_path,
                active_adapter_path=COALESCE(excluded.active_adapter_path, conversations.active_adapter_path),
                updated_at=excluded.updated_at
            """,
            (
                conversation_id,
                target_speaker,
                model_id,
                str(input_path),
                str(profile_path),
                str(pairs_path),
                str(sft_path),
                str(active_adapter_path) if active_adapter_path else None,
                ts,
                ts,
            ),
        )


def get_conversation(conversation_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()


def set_active_adapter(conversation_id: str, adapter_path: Path) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET active_adapter_path=?, updated_at=? WHERE id=?",
            (str(adapter_path), now_iso(), conversation_id),
        )


def create_training_job(
    job_id: str,
    conversation_id: str,
    status: str,
    queue_job_id: str | None = None,
    adapter_path: Path | None = None,
    log_path: Path | None = None,
    error: str | None = None,
) -> None:
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO training_jobs (
                id, conversation_id, queue_job_id, status, adapter_path, log_path, error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                conversation_id,
                queue_job_id,
                status,
                str(adapter_path) if adapter_path else None,
                str(log_path) if log_path else None,
                error,
                ts,
                ts,
            ),
        )


def update_training_job(
    job_id: str,
    status: str | None = None,
    adapter_path: Path | None = None,
    log_path: Path | None = None,
    error: str | None = None,
    queue_job_id: str | None = None,
) -> None:
    fields = []
    values = []
    if status is not None:
        fields.append("status=?")
        values.append(status)
    if adapter_path is not None:
        fields.append("adapter_path=?")
        values.append(str(adapter_path))
    if log_path is not None:
        fields.append("log_path=?")
        values.append(str(log_path))
    if error is not None:
        fields.append("error=?")
        values.append(error)
    if queue_job_id is not None:
        fields.append("queue_job_id=?")
        values.append(queue_job_id)

    fields.append("updated_at=?")
    values.append(now_iso())
    values.append(job_id)

    with get_conn() as conn:
        conn.execute(f"UPDATE training_jobs SET {', '.join(fields)} WHERE id=?", values)


def get_training_job(job_id: str) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,)).fetchone()


def list_conversations() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                c.*,
                tj.id AS latest_job_id,
                tj.status AS latest_job_status
            FROM conversations c
            LEFT JOIN training_jobs tj
              ON tj.id = (
                  SELECT t2.id
                  FROM training_jobs t2
                  WHERE t2.conversation_id = c.id
                  ORDER BY t2.created_at DESC
                  LIMIT 1
              )
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
