import os
import uuid
import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.config import settings, UPLOADS_DIR, PROCESSED_DIR
from app.db.database import get_db, Base, engine
from app.db.models import ImageAsset
from app.services.ingestion import sanitize_filename, inspect_and_extract_metadata, generate_web_preview

# Ensure database tables exist
Base.metadata.create_all(bind=engine)

router = APIRouter(prefix="/api", tags=["uploads"])

@router.post("/uploads", response_model=List[dict])
async def upload_satellite_images(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    modality_hint: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Accepts one or more satellite images (GeoTIFF, TIFF, PNG, JPEG).
    Validates file formats, sanitizes filenames, extracts metadata without loading raw rasters into memory,
    generates optimized web previews, and persists records in SQLite.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    uploaded_assets = []

    for file in files:
        original_filename = file.filename or "unnamed_raster.tif"
        ext = Path(original_filename).suffix.lower()

        # 1. Validate file extension
        if ext not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Supported formats: GeoTIFF (.tif, .tiff), PNG (.png), JPEG (.jpg, .jpeg)."
            )

        # 2. Sanitize and generate unique storage filename
        asset_id = uuid.uuid4().hex
        clean_name = sanitize_filename(original_filename)
        storage_filename = f"{asset_id}_{clean_name}"
        storage_path = UPLOADS_DIR / storage_filename

        # 3. Stream content to disk while enforcing max file size limit
        bytes_read = 0
        max_bytes = settings.max_upload_size_mb * 1024 * 1024

        try:
            with open(storage_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    bytes_read += len(chunk)
                    if bytes_read > max_bytes:
                        # Clean up partial file
                        buffer.close()
                        if storage_path.exists():
                            storage_path.unlink()
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{original_filename}' exceeds maximum allowed size of {settings.max_upload_size_mb}MB."
                        )
                    buffer.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            if storage_path.exists():
                storage_path.unlink()
            raise HTTPException(status_code=500, detail=f"Failed to save upload '{original_filename}': {str(e)}")

        # 4. Inspect file header and extract metadata
        try:
            meta = inspect_and_extract_metadata(storage_path, clean_name)
        except Exception as e:
            if storage_path.exists():
                storage_path.unlink()
            raise HTTPException(
                status_code=400,
                detail=f"Image file '{original_filename}' appears corrupt or unreadable: {str(e)}"
            )

        # Allow user modality hint if provided and safe
        if modality_hint and modality_hint.lower() in ["optical", "sar", "multispectral"]:
            meta["modality"] = modality_hint.lower()

        # 5. Generate browser-friendly web preview
        preview_filename = f"{asset_id}_preview.png"
        preview_path = PROCESSED_DIR / preview_filename
        try:
            preview_url = generate_web_preview(storage_path, preview_path)
        except Exception as e:
            meta["warnings"].append(f"Preview generation warning: {str(e)}")
            preview_url = None

        # 6. Save ImageAsset in SQLite Database
        asset_record = ImageAsset(
            id=asset_id,
            session_id=session_id,
            filename=clean_name,
            file_path=str(storage_path),
            preview_path=str(preview_path),
            preview_url=preview_url,
            modality=meta["modality"],
            format=meta["format"],
            width=meta["width"],
            height=meta["height"],
            bands=meta["bands"],
            acquisition_date=meta["acquisition_date"],
            crs=meta["crs"],
            resolution=meta["resolution"],
            metadata_json=json.dumps(meta),
            status="ready"
        )
        db.add(asset_record)
        db.commit()
        db.refresh(asset_record)

        # 7. Add to output bundle
        res = asset_record.to_dict()
        res["warnings"] = meta["warnings"]
        uploaded_assets.append(res)

    return uploaded_assets

@router.delete("/uploads/{asset_id}")
def delete_uploaded_image(asset_id: str, db: Session = Depends(get_db)):
    """Safely delete an uploaded image and its generated preview."""
    asset = db.query(ImageAsset).filter(ImageAsset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found.")

    # Safely remove files
    try:
        fpath = Path(asset.file_path)
        if fpath.exists() and fpath.is_file():
            fpath.unlink()
        ppath = Path(asset.preview_path) if asset.preview_path else None
        if ppath and ppath.exists() and ppath.is_file():
            ppath.unlink()
    except Exception:
        pass

    db.delete(asset)
    db.commit()
    return {"message": "Asset deleted successfully", "id": asset_id}
