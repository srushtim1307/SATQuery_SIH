"""
Bi-Temporal Change Detection Model Adapter (TinyCD Specialist).

Integrates the TinyCD deep learning architecture for dense change mask estimation,
pixel-level alteration metrics, and visual overlay synthesis.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.adapters.base import BaseModelAdapter
from app.db.models import ImageAsset
from app.demo.mock_engines import DemoExecutionEngine
from app.models.manager import model_manager
from app.models.change_detection_model import load_tinycd_model, run_tinycd_inference

logger = logging.getLogger(__name__)

class ChangeDetectionAdapter(BaseModelAdapter):
    def __init__(
        self, 
        model_id: str = "rs-change-diff", 
        name: str = "TinyCD Bi-Temporal Change Specialist", 
        version: str = "1.0.0",
        backend: str = "real"
    ):
        super().__init__(model_id=model_id, name=name, version=version, task="change_detection", backend=backend)
        
        # Register factory with ModelManager if real backend
        if self.backend == "real":
            model_manager.register_factory(
                model_id=self.model_id,
                factory=lambda: load_tinycd_model(device=model_manager.device),
                backend_type="real",
                metadata={"architecture": "TinyCD (EfficientNet-B4 + MAMB)", "input_resolution": [256, 256]}
            )

    def validate_input(self, assets: List[ImageAsset]) -> Tuple[bool, List[str]]:
        errors = []
        if len(assets) != 2:
            errors.append(f"Change detection specialist requires exactly 2 images (before and after). Received {len(assets)}.")
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
            raise ValueError(f"Input validation failed for Change Detection: {', '.join(errors)}")

        t1_asset = assets[0]
        t2_asset = assets[1]

        # 1. Fallback to Demo Engine if explicitly configured
        if self.backend == "demo":
            return DemoExecutionEngine.execute_change_detection(query, assets)

        # 2. Check if physical files exist on disk. If not (e.g. simulated fixture), fallback to demo
        t1_path = self._resolve_asset_path(t1_asset)
        t2_path = self._resolve_asset_path(t2_asset)

        if not t1_path or not t2_path:
            logger.info("Physical raster files not present on disk for assets. Providing demo execution fallback.")
            return DemoExecutionEngine.execute_change_detection(query, assets)

        # 3. Real AI Inference via TinyCD
        try:
            model = model_manager.get_model(self.model_id)
            if model is None:
                logger.warning(f"Could not load '{self.model_id}'. Falling back to demo mode.")
                res = DemoExecutionEngine.execute_change_detection(query, assets)
                res["metadata"]["fallback_reason"] = "Model weights could not be loaded."
                return res

            # Execute real neural network inference
            cd_output = run_tinycd_inference(
                model=model,
                t1_path=t1_path,
                t2_path=t2_path,
                device=model_manager.device,
                threshold=0.35
            )

            change_pct = cd_output["change_percentage"]
            changed_pixels = cd_output["changed_pixels"]
            total_pixels = cd_output["total_pixels"]
            confidence = cd_output["confidence"]
            boxes = cd_output["bounding_boxes"]

            t1_name = t1_asset.filename
            t2_name = t2_asset.filename

            # Formulate structured natural-language response based on real outputs
            if change_pct > 0.05:
                answer = (
                    f"Bi-temporal neural change analysis with TinyCD identified {change_pct}% surface alteration "
                    f"({changed_pixels:,} of {total_pixels:,} pixels) across {len(boxes)} distinct clusters between "
                    f"'{t1_name}' (T1) and '{t2_name}' (T2). Significant variations are highlighted in the overlay mask."
                )
            else:
                answer = (
                    f"Bi-temporal neural change analysis with TinyCD indicates negligible spatial variation ({change_pct}% altered) "
                    f"between '{t1_name}' (T1) and '{t2_name}' (T2). No significant structural or land-cover changes detected."
                )

            t1_preview = t1_asset.preview_url or f"/static/uploads/{t1_asset.filename}"
            t2_preview = t2_asset.preview_url or f"/static/uploads/{t2_asset.filename}"
            overlay_url = cd_output["overlay_url"]

            evidence_item = {
                "type": "bi_temporal_differential",
                "t1_filename": t1_name,
                "t2_filename": t2_name,
                "primary_change_class": f"Surface Variation ({change_pct}% altered)",
                "change_percentage": change_pct,
                "changed_pixels": changed_pixels,
                "total_pixels": total_pixels,
                "mask_url": cd_output["mask_url"],
                "overlay_url": overlay_url,
                "t2_preview_url": t2_preview,
                "bounding_boxes": boxes,
                "is_demo": False,
                # Fields matching frontend ChangeComparisonEvidence schema:
                "beforeImageUrl": t1_preview,
                "afterImageUrl": overlay_url,
                "beforeLabel": f"T1 Baseline · {t1_name}",
                "beforeDate": t1_asset.acquisition_date or "Baseline Acquisition",
                "beforeGsd": f"GSD: {t1_asset.resolution or '10m'} · EPSG:{t1_asset.crs or '4326'}",
                "afterLabel": f"T2 Current · {t2_name}",
                "afterDate": t2_asset.acquisition_date or "Recent Acquisition",
                "afterOverlayTag": f"TinyCD Mask: {change_pct}% Alteration ({len(boxes)} Regions)"
            }

            return {
                "task": "change_detection",
                "backend": "real",
                "confidence": confidence,
                "answer": answer,
                "evidence": [evidence_item],
                "metadata": {
                    "model_id": self.model_id,
                    "model_name": cd_output["model_name"],
                    "architecture": "Siamese Sliced EfficientNet-B4 + Mixing Attention",
                    "device": model_manager.device,
                    "pair_count": len(assets),
                    "change_percentage": change_pct,
                    "changed_pixels": changed_pixels,
                    "total_pixels": total_pixels,
                    "detected_regions_count": len(boxes),
                    "confidence_source": "mean probability across detected change mask pixels",
                    "is_demo": False
                }
            }

        except Exception as e:
            logger.error(f"Error during real change detection inference: {e}", exc_info=True)
            raise RuntimeError(f"Change detection execution failed: {e}")
