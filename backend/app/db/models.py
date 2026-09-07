import json
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from app.db.database import Base

class ImageAsset(Base):
    __tablename__ = "image_assets"

    id = Column(String(64), primary_key=True, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    preview_path = Column(String(512), nullable=True)
    preview_url = Column(String(512), nullable=True)
    modality = Column(String(32), default="unknown")  # optical, sar, multispectral, unknown
    format = Column(String(32), nullable=False)        # GeoTIFF, TIFF, PNG, JPEG
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    bands = Column(Integer, default=1)
    acquisition_date = Column(String(64), nullable=True)
    crs = Column(String(128), nullable=True)
    resolution = Column(String(64), nullable=True)
    metadata_json = Column(Text, nullable=True)
    status = Column(String(32), default="ready")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        meta = {}
        if self.metadata_json:
            try:
                meta = json.loads(self.metadata_json)
            except Exception:
                meta = {}
        return {
            "id": self.id,
            "session_id": self.session_id,
            "filename": self.filename,
            "format": self.format,
            "modality": self.modality,
            "width": self.width,
            "height": self.height,
            "bands": self.bands,
            "crs": self.crs,
            "resolution": self.resolution,
            "acquisition_date": self.acquisition_date,
            "preview_url": self.preview_url,
            "status": self.status,
            "metadata": meta,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id = Column(String(64), primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    mode = Column(String(64), default="auto")
    status = Column(String(32), default="active")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(String(64), primary_key=True, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    query = Column(Text, nullable=False)
    input_asset_ids = Column(Text, nullable=False, default="[]")  # JSON list
    detected_task = Column(String(64), nullable=False)
    selected_model_id = Column(String(64), nullable=True)
    model_version = Column(String(32), nullable=True)
    execution_trace_json = Column(Text, nullable=False, default="[]")  # JSON list
    result_json = Column(Text, nullable=True)  # JSON dict
    backend_type = Column(String(32), default="demo")
    status = Column(String(32), default="completed")  # completed, incompatible, unsupported, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        try:
            asset_ids = json.loads(self.input_asset_ids) if self.input_asset_ids else []
        except Exception:
            asset_ids = []

        try:
            trace = json.loads(self.execution_trace_json) if self.execution_trace_json else []
        except Exception:
            trace = []

        try:
            result = json.loads(self.result_json) if self.result_json else {}
        except Exception:
            result = {}

        return {
            "analysis_id": self.id,
            "session_id": self.session_id,
            "query": self.query,
            "input_asset_ids": asset_ids,
            "detected_task": self.detected_task,
            "selected_model": {
                "id": self.selected_model_id,
                "version": self.model_version
            } if self.selected_model_id else None,
            "execution_trace": trace,
            "result": result,
            "backend": self.backend_type,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

