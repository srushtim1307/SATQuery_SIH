"""
Text-Guided Visual Grounding Model Adapter (OWL-ViT Specialist).

Integrates the Google OWL-ViT open-vocabulary detector for zero-shot satellite feature
localization, bounding coordinate extraction, and annotated overlay rendering.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.adapters.base import BaseModelAdapter
from app.db.models import ImageAsset
from app.demo.mock_engines import DemoExecutionEngine
from app.models.manager import model_manager
from app.models.grounding_model import load_owlvit_model, run_owlvit_grounding

logger = logging.getLogger(__name__)

class GroundingAdapter(BaseModelAdapter):
    def __init__(
        self, 
        model_id: str = "rs-grounding-focal", 
        name: str = "OWL-ViT Open-Vocabulary Grounding Specialist", 
        version: str = "1.0.0",
        backend: str = "real"
    ):
        super().__init__(model_id=model_id, name=name, version=version, task="grounding", backend=backend)

        # Register factory with ModelManager if real backend
        if self.backend == "real":
            model_manager.register_factory(
                model_id=self.model_id,
                factory=lambda: load_owlvit_model(device=model_manager.device),
                backend_type="real",
                metadata={"architecture": "OWL-ViT (ViT-B/32)", "task": "open-vocabulary-grounding"}
            )

    def validate_input(self, assets: List[ImageAsset]) -> Tuple[bool, List[str]]:
        errors = []
        if len(assets) != 1:
            errors.append(f"Grounding specialist requires exactly 1 satellite image. Received {len(assets)}.")
        return len(errors) == 0, errors

    def _resolve_asset_path(self, asset: ImageAsset) -> Optional[str]:
        """Prefers preprocessed RGB preview image; falls back to original raster file path."""
        if asset.preview_path and os.path.exists(asset.preview_path):
            return asset.preview_path
        if asset.file_path and os.path.exists(asset.file_path):
            return asset.file_path
        return None

    def analyze(
        self, 
        query: str, 
        assets: List[ImageAsset], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        valid, errors = self.validate_input(assets)
        if not valid:
            raise ValueError(f"Input validation failed for Grounding: {', '.join(errors)}")

        asset = assets[0]

        # 1. Fallback to Demo Engine if explicitly configured
        if self.backend == "demo":
            return DemoExecutionEngine.execute_grounding(query, assets)

        # 2. Check if physical image exists on disk (fallback to demo for synthetic DB records)
        image_path = self._resolve_asset_path(asset)
        if not image_path:
            logger.info(f"Physical image file not on disk for asset '{asset.id}'. Returning demo execution.")
            return DemoExecutionEngine.execute_grounding(query, assets)

        # 3. Real AI Inference via OWL-ViT
        try:
            model_tuple = model_manager.get_model(self.model_id)
            if model_tuple is None:
                logger.warning(f"Could not load '{self.model_id}'. Falling back to demo mode.")
                res = DemoExecutionEngine.execute_grounding(query, assets)
                res["metadata"]["fallback_reason"] = "Model weights could not be loaded."
                return res

            grounding_output = run_owlvit_grounding(
                model_tuple=model_tuple,
                image_path=image_path,
                query=query,
                device=model_manager.device,
                max_boxes=6
            )

            boxes = grounding_output["bounding_boxes"]
            primary_target = grounding_output["primary_target"]
            overall_confidence = grounding_output["overall_confidence"]
            annotated_url = grounding_output["annotated_image_url"]

            if len(boxes) > 0:
                answer = (
                    f"Visual grounding with OWL-ViT located {len(boxes)} spatial instance(s) of '{primary_target}' "
                    f"within '{asset.filename}'. Grounded bounding coordinates and confidence metrics are mapped below."
                )
            else:
                answer = (
                    f"Visual grounding with OWL-ViT did not identify distinct spatial instances matching '{query}' "
                    f"in '{asset.filename}' above detection threshold."
                )

            evidence_item = {
                "type": "bounding_coordinates",
                "target_feature": primary_target,
                "imageUrl": asset.preview_url or f"/static/uploads/{asset.filename}",
                "annotatedImageUrl": annotated_url,
                "scaleLabel": f"GSD: {asset.resolution or '10m'}",
                "satelliteLabel": f"{asset.modality.upper() if asset.modality else 'Optical'} · {asset.filename}",
                "boundingBoxes": boxes,
                "bounding_boxes": boxes,
                "is_demo": False
            }

            return {
                "task": "grounding",
                "backend": "real",
                "confidence": overall_confidence,
                "answer": answer,
                "evidence": [evidence_item],
                "metadata": {
                    "model_id": self.model_id,
                    "model_name": grounding_output["model_name"],
                    "detected_objects_count": len(boxes),
                    "primary_target": primary_target,
                    "candidate_queries": grounding_output["candidate_queries"],
                    "annotated_image_url": annotated_url,
                    "confidence_source": "OWL-ViT class logit distribution sigmoid",
                    "is_demo": False
                }
            }

        except Exception as e:
            logger.error(f"Error during visual grounding inference: {e}", exc_info=True)
            raise RuntimeError(f"Visual grounding execution failed: {e}")
