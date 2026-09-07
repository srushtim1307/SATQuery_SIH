"""
SatQuery Agent Orchestrator.

Implements the end-to-end vision-language pipeline:
Natural-language query + image context -> Validation -> Task Classification -> Model Registry -> Adapter Execution -> Audit Persistence.
"""

import uuid
import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.core.constants import TRACE_STEPS
from app.db.models import ImageAsset, AnalysisRecord
from app.services.validator import validate_analysis_configuration
from app.services.classifier import TaskClassifier
from app.models.registry import registry
from app.schemas.analysis import AnalysisRequest, AnalysisResponse, TraceStep, ModelDescriptor, RoutingDecision

class SatQueryAgent:
    """
    Core orchestrator agent for SatQuery AI.
    Converts natural language queries and satellite image assets into observable specialist analyses.
    """

    @classmethod
    def run(cls, request: AnalysisRequest, db: Session) -> Dict[str, Any]:
        analysis_id = f"analysis-{uuid.uuid4().hex[:12]}"
        query = request.query.strip()
        mode = request.mode or "auto"
        session_id = request.session_id

        # Initialize observable trace steps
        trace_steps = [TraceStep(name=step_name, status="pending") for step_name in TRACE_STEPS]

        def update_step(index: int, status: str, detail: Optional[str] = None):
            trace_steps[index].status = status
            if detail:
                trace_steps[index].detail = detail

        # ----------------------------------------------------
        # STEP 1: Understanding your question
        # ----------------------------------------------------
        update_step(0, "in_progress")
        if not query:
            update_step(0, "failed", "Empty natural-language query provided.")
            return cls._build_response(
                analysis_id=analysis_id,
                query=query,
                task="none",
                trace_steps=trace_steps,
                status="incompatible",
                error_message="Query cannot be empty. Please ask a question about your satellite imagery.",
                db=db,
                request=request
            )
        update_step(0, "completed", f"Parsed query: \"{query}\"")

        # ----------------------------------------------------
        # Retrieve Image Assets from Database
        # ----------------------------------------------------
        assets: List[ImageAsset] = []
        if request.image_ids:
            # Preserve ordering matching request.image_ids
            asset_map = {a.id: a for a in db.query(ImageAsset).filter(ImageAsset.id.in_(request.image_ids)).all()}
            assets = [asset_map[aid] for aid in request.image_ids if aid in asset_map]

            # Enforce requirement 4: If an asset ID is invalid, return a clear error
            missing_ids = [aid for aid in request.image_ids if aid not in asset_map]
            if missing_ids:
                update_step(1, "failed", f"Requested image assets not found in database: {missing_ids}")
                return cls._build_response(
                    analysis_id=analysis_id,
                    query=query,
                    task="none",
                    trace_steps=trace_steps,
                    status="incompatible",
                    error_message="Uploaded image could not be found. Please upload the image again.",
                    db=db,
                    request=request
                )

        # ----------------------------------------------------
        # STEP 2: Checking image compatibility
        # ----------------------------------------------------
        update_step(1, "in_progress")
        if not assets:
            update_step(1, "failed", "No valid satellite images found for the provided asset IDs.")
            return cls._build_response(
                analysis_id=analysis_id,
                query=query,
                task="none",
                trace_steps=trace_steps,
                status="incompatible",
                error_message="No satellite imagery provided. Please upload at least one image before asking a question.",
                db=db,
                request=request
            )

        # Enforce requirement 6: Image Identity Logging
        print(f"\n[SatQueryAgent] ================= ANALYSIS INPUT =================")
        print(f"[SatQueryAgent] query = '{query}'")
        print(f"[SatQueryAgent] asset_ids = {[a.id for a in assets]}")
        print(f"[SatQueryAgent] filenames = {[a.filename for a in assets]}")
        print(f"[SatQueryAgent] modalities = {[a.modality for a in assets]}")
        print(f"[SatQueryAgent] file_paths = {[a.file_path for a in assets]}")

        # ----------------------------------------------------
        # STEP 3: Identifying analysis task
        # ----------------------------------------------------
        update_step(2, "in_progress")
        routing = TaskClassifier.classify(query=query, assets=assets, mode=mode)
        detected_task = routing["task"]

        # Check if query is unsupported (out-of-scope)
        if routing.get("is_unsupported"):
            update_step(1, "completed", f"{len(assets)} image(s) verified.")
            update_step(2, "failed", routing["reason"])
            return cls._build_response(
                analysis_id=analysis_id,
                query=query,
                task="unsupported",
                trace_steps=trace_steps,
                status="unsupported",
                error_message=routing["reason"],
                routing=routing,
                db=db,
                request=request
            )

        # Check input compatibility
        if not routing.get("compatible", True):
            update_step(1, "failed", routing["reason"])
            update_step(2, "completed", f"Task identified: {detected_task}")
            return cls._build_response(
                analysis_id=analysis_id,
                query=query,
                task=detected_task,
                trace_steps=trace_steps,
                status="incompatible",
                error_message=routing["reason"],
                routing=routing,
                db=db,
                request=request
            )

        update_step(1, "completed", f"{len(assets)} image(s) verified for {detected_task.upper()}.")
        update_step(2, "completed", f"Task routed to {routing['specialist']}: {routing['reason']}")

        # ----------------------------------------------------
        # STEP 4: Selecting specialist model
        # ----------------------------------------------------
        update_step(3, "in_progress")
        modalities = [a.modality for a in assets]
        input_type = "bi_temporal_pair" if len(assets) == 2 and detected_task == "change_detection" else (
            "optical_sar_pair" if len(assets) == 2 and detected_task == "optical_sar" else "single_image"
        )
        model_entry = registry.find_capable_model(task=detected_task, modalities=modalities, input_type=input_type)

        if not model_entry:
            update_step(3, "failed", f"No active model registered for task '{detected_task}'.")
            return cls._build_response(
                analysis_id=analysis_id,
                query=query,
                task=detected_task,
                trace_steps=trace_steps,
                status="failed",
                error_message=f"Model registry could not find a capable specialist model for '{detected_task}'.",
                routing=routing,
                db=db,
                request=request
            )

        model_desc = ModelDescriptor(
            id=model_entry.id,
            name=model_entry.name,
            version=model_entry.version,
            task=model_entry.task,
            backend=model_entry.backend_type
        )
        update_step(3, "completed", f"Selected {model_entry.name} ({model_entry.id} v{model_entry.version})")

        print(f"[SatQueryAgent] SELECTED MODEL = {model_entry.id} ({model_entry.name})")
        print(f"[SatQueryAgent] BACKEND = {model_entry.backend_type}")
        print(f"[SatQueryAgent] ====================================================\n")

        # ----------------------------------------------------
        # STEP 5: Analyzing imagery (Adapter Execution)
        # ----------------------------------------------------
        update_step(4, "in_progress")
        try:
            adapter = model_entry.adapter
            result_payload = adapter.analyze(query=query, assets=assets, context={"routing": routing})
            update_step(4, "completed", "Specialist model execution finished.")
        except Exception as e:
            update_step(4, "failed", str(e))
            return cls._build_response(
                analysis_id=analysis_id,
                query=query,
                task=detected_task,
                selected_model=model_desc,
                trace_steps=trace_steps,
                status="failed",
                error_message=f"Model execution error: {str(e)}",
                routing=routing,
                db=db,
                request=request
            )

        # ----------------------------------------------------
        # STEP 6: Preparing visual evidence
        # ----------------------------------------------------
        update_step(5, "in_progress")
        evidence_count = len(result_payload.get("evidence", []))
        update_step(5, "completed", f"Generated {evidence_count} visual evidence item(s).")

        # ----------------------------------------------------
        # Persist and return response
        # ----------------------------------------------------
        return cls._build_response(
            analysis_id=analysis_id,
            query=query,
            task=detected_task,
            selected_model=model_desc,
            trace_steps=trace_steps,
            status="completed",
            routing=routing,
            result=result_payload,
            db=db,
            request=request
        )

    @classmethod
    def _build_response(
        cls,
        analysis_id: str,
        query: str,
        task: str,
        trace_steps: List[TraceStep],
        status: str,
        db: Session,
        request: AnalysisRequest,
        error_message: Optional[str] = None,
        selected_model: Optional[ModelDescriptor] = None,
        routing: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        trace_dicts = [step.model_dump() for step in trace_steps]
        res_dict = result or {}

        # Persist AnalysisRecord to SQLite database
        try:
            record = AnalysisRecord(
                id=analysis_id,
                session_id=request.session_id,
                query=query,
                input_asset_ids=json.dumps(request.image_ids),
                detected_task=task,
                selected_model_id=selected_model.id if selected_model else None,
                model_version=selected_model.version if selected_model else None,
                execution_trace_json=json.dumps(trace_dicts),
                result_json=json.dumps(res_dict),
                backend_type=selected_model.backend if selected_model else "demo",
                status=status,
                error_message=error_message
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            # Non-blocking persistence warning
            print(f"[WARN] Failed to persist AnalysisRecord: {e}")

        routing_obj = None
        if routing:
            routing_obj = {
                "task": routing.get("task", task),
                "specialist": routing.get("specialist", "none"),
                "confidence": routing.get("confidence", 0.0),
                "reason": routing.get("reason", ""),
                "required_inputs": routing.get("required_inputs", []),
                "compatible": routing.get("compatible", False)
            }

        return {
            "analysis_id": analysis_id,
            "query": query,
            "task": task,
            "selected_model": selected_model.model_dump() if selected_model else None,
            "routing": routing_obj,
            "execution_trace": trace_dicts,
            "result": res_dict,
            "status": status,
            "error_message": error_message,
            "backend": selected_model.backend if selected_model else "demo"
        }
