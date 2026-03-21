from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
APP_STATE_DIR = BASE_DIR / "app_state"
UPLOADS_DIR = APP_STATE_DIR / "uploads"
ADAPTERS_DIR = APP_STATE_DIR / "adapters"
LOGS_DIR = APP_STATE_DIR / "logs"
DB_PATH = APP_STATE_DIR / "ghost.db"

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REDIS_URL = "redis://localhost:6379/0"

