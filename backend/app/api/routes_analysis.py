"""
FastAPI Routes for Analysis & Agent Orchestration.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import AnalysisRecord, ImageAsset
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.services.agent import SatQueryAgent
from app.models.registry import registry

router = APIRouter(prefix="/api", tags=["analysis"])

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_imagery(
    request: AnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Submits a natural-language query and staged satellite imagery to the SatQuery AI Agent.
    Routes to the appropriate specialist model and returns a structured analysis result
    with observable execution trace.
    """
    result = SatQueryAgent.run(request=request, db=db)
    return result

@router.get("/history", response_model=List[Dict[str, Any]])
@router.get("/analyses", response_model=List[Dict[str, Any]])
async def list_analysis_history(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Lists persisted satellite analysis records from the database, ordered chronologically.
    Includes input image thumbnails and metadata for history restoration.
    """
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).limit(limit).all()
    history_list = []
    for r in records:
        data = r.to_dict()
        asset_ids = data.get("input_asset_ids", [])
        if asset_ids:
            assets = db.query(ImageAsset).filter(ImageAsset.id.in_(asset_ids)).all()
            data["input_assets"] = [
                {
                    "id": a.id,
                    "filename": a.filename,
                    "preview_url": a.preview_url,
                    "modality": a.modality,
                    "resolution": a.resolution
                } for a in assets
            ]
        else:
            data["input_assets"] = []
        history_list.append(data)
    return history_list

@router.get("/analysis/{analysis_id}", response_model=Dict[str, Any])
async def get_analysis_record(
    analysis_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieves a previously executed analysis record and its execution trace by ID.
    """
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")
    return record.to_dict()

@router.get("/models", response_model=List[Dict[str, Any]])
async def list_registered_models():
    """
    Lists all specialist models currently registered in the model registry.
    """
    return registry.list()

@router.get("/benchmarks", response_model=List[Dict[str, Any]])
async def list_benchmarks():
    """
    Returns the evaluation status and measurements for standard remote sensing vision-language benchmarks.
    """
    from app.evaluation.benchmarks import get_benchmark_status
    return get_benchmark_status()

