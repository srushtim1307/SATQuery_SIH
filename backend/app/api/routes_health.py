from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(prefix="/api", tags=["health"])

@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.version,
        "tagline": settings.tagline,
        "sih_problem_id": settings.sih_problem_id,
        "organization": settings.organization,
        "theme": settings.theme,
        "demo_mode": settings.demo_mode_enabled,
        "pipeline": settings.active_pipeline,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "modalities_supported": [
            "Optical (Sentinel-2, Landsat, Cartosat)",
            "SAR (Sentinel-1, RISAT C-Band)",
            "Cross-Modal Pairs (Optical + SAR)",
            "Bi-Temporal Pairs (T1 Baseline + T2 Current)"
        ],
        "specialist_capabilities": [
            "Remote-Sensing VQA",
            "Text-Guided Visual Grounding",
            "Bi-Temporal Change Detection",
            "Optical + SAR Multi-Sensor Fusion",
            "Scene Captioning & Land-Cover Description"
        ]
    }
