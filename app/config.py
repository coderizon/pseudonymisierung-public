import os
from pathlib import Path

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:26b")
DATA_DIR = os.getenv("DATA_DIR", "/data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
DB_PATH = os.path.join(DATA_DIR, "mappings.db")
ENCRYPTION_KEY_PATH = os.path.join(DATA_DIR, ".encryption_key")
CATEGORIES_PATH = os.getenv(
    "CATEGORIES_PATH",
    str(Path(__file__).parent / "categories.yaml"),
)
TAXONOMIES_DIR = os.getenv(
    "TAXONOMIES_DIR",
    str(Path(__file__).resolve().parent.parent / "config" / "taxonomies"),
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_model(model: str) -> None:
    """Override the active model at runtime."""
    global OLLAMA_MODEL
    OLLAMA_MODEL = model
