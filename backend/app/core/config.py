import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
PREVIEWS_DIR = DATA_DIR / "previews"
DEMO_DIR = DATA_DIR / "demo"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    app_name: str = "SatQuery AI"
    version: str = "1.0.0"
    tagline: str = "Vision-Language Assistant for Satellite Imagery"
    sih_problem_id: str = "SIH26167"
    organization: str = "ISRO"
    theme: str = "Space Technology"
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ]
    database_url: str = f"sqlite:///{DATA_DIR / 'satquery.db'}"
    demo_mode_enabled: bool = True
    active_pipeline: str = "Auto-Routed"
    max_upload_size_mb: int = 100
    allowed_extensions: list[str] = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

settings = Settings()
