"""
Remote-Sensing Visual Question Answering (VQA) Model Adapter.

Integrates the SmolVLM-256M multimodal vision-language model for open-domain natural-language
reasoning over satellite rasters with real token probability metrics.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.adapters.base import BaseModelAdapter
from app.db.models import ImageAsset
from app.demo.mock_engines import DemoExecutionEngine
from app.models.manager import model_manager
from app.models.vqa_model import load_vqa_model, run_vqa_inference

logger = logging.getLogger(__name__)

class VQAAdapter(BaseModelAdapter):
    def __init__(
        self, 
        model_id: str = "rs-vqa-base", 
        name: str = "SmolVLM Remote-Sensing VQA Specialist", 
        version: str = "1.0.0",
        backend: str = "real"
    ):
        super().__init__(model_id=model_id, name=name, version=version, task="vqa", backend=backend)

        # Register factory with ModelManager if real backend
        if self.backend == "real":
            model_manager.register_factory(
                model_id=self.model_id,
                factory=lambda: load_vqa_model(device=model_manager.device),
                backend_type="real",
                metadata={"architecture": "SmolVLM-256M (Idefics3)", "task": "remote-sensing-vqa"}
            )

    def validate_input(self, assets: List[ImageAsset]) -> Tuple[bool, List[str]]:
        errors = []
        if len(assets) != 1:
            errors.append(f"VQA specialist requires exactly 1 satellite image. Received {len(assets)}.")
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
            raise ValueError(f"Input validation failed for VQA: {', '.join(errors)}")

        asset = assets[0]

        # 1. Fallback to Demo Engine if explicitly configured
        if self.backend == "demo":
            return DemoExecutionEngine.execute_vqa(query, assets)

        # 2. Check if physical raster exists on disk
        image_path = self._resolve_asset_path(asset)
        if not image_path:
            logger.info(f"Physical image file not on disk for asset '{asset.id}'. Returning demo execution.")
            return DemoExecutionEngine.execute_vqa(query, assets)

        # 3. Real AI Inference via SmolVLM
        try:
            model_tuple = model_manager.get_model(self.model_id)
            if model_tuple is None:
                logger.warning(f"Could not load '{self.model_id}'. Falling back to demo mode.")
                res = DemoExecutionEngine.execute_vqa(query, assets)
                res["metadata"]["fallback_reason"] = "Model weights could not be loaded."
                return res

            vqa_output = run_vqa_inference(
                model_tuple=model_tuple,
                image_path=image_path,
                query=query,
                device=model_manager.device,
                max_new_tokens=45
            )

            answer = vqa_output["answer"]
            confidence = vqa_output["confidence"]

            evidence_item = {
                "type": "visual_qa",
                "question": query,
                "answer": answer,
                "imageUrl": asset.preview_url or f"/static/uploads/{asset.filename}",
                "scaleLabel": f"GSD: {asset.resolution or '10m'}",
                "satelliteLabel": f"{asset.modality.upper() if asset.modality else 'Optical'} · {asset.filename}",
                "confidence": confidence,
                "is_demo": False
            }

            return {
                "task": "vqa",
                "backend": "real",
                "confidence": confidence,
                "answer": answer,
                "evidence": [evidence_item],
                "metadata": {
                    "model_id": self.model_id,
                    "model_name": vqa_output["model_name"],
                    "confidence_source": "mean token softmax probability distribution",
                    "tokens_generated": vqa_output["tokens_generated"],
                    "is_demo": False
                }
            }

        except Exception as e:
            logger.error(f"Error during visual question answering inference: {e}", exc_info=True)
            raise RuntimeError(f"VQA execution failed: {e}")
