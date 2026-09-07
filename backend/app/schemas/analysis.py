"""
Pydantic schemas for the Agentic Task Classification, Model Registry & Analysis Pipeline.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class TraceStep(BaseModel):
    name: str
    status: str = "pending"  # completed, in_progress, pending, failed
    detail: Optional[str] = None

class ModelDescriptor(BaseModel):
    id: str
    name: str
    version: str
    task: str
    backend: str = "demo"

class RoutingDecision(BaseModel):
    task: str
    specialist: str
    confidence: float
    reason: str
    required_inputs: List[str]
    compatible: bool

class AnalysisRequest(BaseModel):
    query: str = Field(default="", description="Natural-language question or analysis command.")
    image_ids: List[str] = Field(..., description="List of uploaded image asset IDs.")
    mode: str = Field(default="auto", description="Execution mode ('auto', 'vqa', 'grounding', 'change_detection', 'optical_sar', 'captioning').")
    session_id: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    query: str
    task: str
    selected_model: Optional[ModelDescriptor] = None
    routing: Optional[RoutingDecision] = None
    execution_trace: List[TraceStep]
    result: Dict[str, Any] = {}
    status: str = "completed"  # completed, incompatible, unsupported, failed
    error_message: Optional[str] = None
    backend: str = "demo"
