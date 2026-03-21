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

