from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import ImageAsset
from app.schemas.requests import ValidateRequest, ValidateResponse
from app.services.validator import validate_analysis_configuration

router = APIRouter(prefix="/api", tags=["validation"])

@router.post("/validate", response_model=ValidateResponse)
def validate_inputs(
    payload: ValidateRequest,
    db: Session = Depends(get_db)
):
    """
    Validates uploaded images against intended analysis requirements before inference.
    Checks single vs pair, optical vs SAR, and bi-temporal separation rules.
    """
    if not payload.image_ids:
        raise HTTPException(
            status_code=400,
            detail="Validation requires at least one image ID."
        )

    # Fetch image asset records from SQLite
    assets = db.query(ImageAsset).filter(ImageAsset.id.in_(payload.image_ids)).all()
    
    # Maintain user-submitted ordering
    asset_map = {a.id: a for a in assets}
    ordered_assets = [asset_map[aid] for aid in payload.image_ids if aid in asset_map]

    if len(ordered_assets) != len(payload.image_ids):
        missing = set(payload.image_ids) - set(asset_map.keys())
        raise HTTPException(
            status_code=404,
            detail=f"One or more image assets not found: {list(missing)}"
        )

    # Execute validation rules
    result = validate_analysis_configuration(ordered_assets, payload.mode)
    return result
