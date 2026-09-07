from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ValidateRequest(BaseModel):
    image_ids: List[str]
    mode: str = "Auto Detect (Recommended)"

class ValidateResponse(BaseModel):
    valid: bool
    mode: str
    count: int
    images: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]

class ImageAssetResponse(BaseModel):
    id: str
    filename: str
    format: str
    modality: str
    width: Optional[int]
    height: Optional[int]
    bands: int
    crs: Optional[str]
    resolution: Optional[str]
    acquisition_date: Optional[str]
    preview_url: Optional[str]
    status: str
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}
