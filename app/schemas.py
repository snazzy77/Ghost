from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    conversation_id: str
    job_id: str | None
    ready_mode: str
    message: str


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    history: list[dict] = Field(default_factory=list)
    retrieval_only: bool = False
    retrieval_k: int = 4


class ChatResponse(BaseModel):
    conversation_id: str
    mode: str
    reply: str


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str
    conversation_id: str
    adapter_path: str | None = None
    log_path: str | None = None
    error: str | None = None


class ConversationSummary(BaseModel):
    conversation_id: str
    target_speaker: str
    model_id: str
    created_at: str
    updated_at: str
    latest_job_id: str | None = None
    latest_job_status: str | None = None
