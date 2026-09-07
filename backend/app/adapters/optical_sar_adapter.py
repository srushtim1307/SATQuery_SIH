"""
Optical + SAR Cross-Modal Fusion Model Adapter (Phase 5E).

Integrates the CrossModalOpticalSARNet neural network for dual-stream
optical spectral and SAR microwave backscatter co-analysis.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.adapters.base import BaseModelAdapter
from app.db.models import ImageAsset
from app.demo.mock_engines import DemoExecutionEngine
from app.models.manager import model_manager
from app.models.optical_sar_model import load_optical_sar_model, run_optical_sar_fusion

logger = logging.getLogger(__name__)

class OpticalSARAdapter(BaseModelAdapter):
    def __init__(
        self, 
        model_id: str = "rs-optical-sar-fusion", 
        name: str = "Optical + SAR Cross-Modal Fusion Specialist", 
        version: str = "1.0.0",
        backend: str = "real"
    ):
        super().__init__(model_id=model_id, name=name, version=version, task="optical_sar", backend=backend)

        # Register factory with ModelManager if real backend
        if self.backend == "real":
            model_manager.register_factory(
                model_id=self.model_id,
                factory=lambda: load_optical_sar_model(device=model_manager.device),
                backend_type="real",
                metadata={"architecture": "CrossModalOpticalSARNet (Dual-Stream + Cross-Attention)", "resolution": [256, 256]}
            )

    def validate_input(self, assets: List[ImageAsset]) -> Tuple[bool, List[str]]:
        errors = []
        if len(assets) != 2:
            errors.append(f"Optical + SAR fusion requires exactly 2 images (1 optical/multispectral, 1 SAR). Received {len(assets)}.")
            return False, errors

        modalities = [a.modality.lower() if a.modality else "unknown" for a in assets]
        has_optical = any(m in ["optical", "multispectral"] for m in modalities)
        has_sar = any(m == "sar" for m in modalities)

        if not has_optical:
            errors.append("Missing optical/multispectral imagery in input pair.")
        if not has_sar:
            errors.append("Missing SAR radar imagery in input pair.")

        return len(errors) == 0, errors

    def _resolve_asset_path(self, asset: ImageAsset) -> Optional[str]:
        """Prefers preprocessed preview image; falls back to original raster file path."""
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
            raise ValueError(f"Input validation failed for Optical + SAR: {', '.join(errors)}")

        opt_asset = next((a for a in assets if a.modality and a.modality.lower() in ["optical", "multispectral"]), assets[0])
        sar_asset = next((a for a in assets if a.modality and a.modality.lower() == "sar"), assets[1])

        # 1. Fallback to Demo Engine if explicitly configured
        if self.backend == "demo":
            return DemoExecutionEngine.execute_optical_sar(query, assets)

        # 2. Check physical raster paths on disk
        opt_path = self._resolve_asset_path(opt_asset)
        sar_path = self._resolve_asset_path(sar_asset)

        if not opt_path or not sar_path:
            logger.info("Physical raster files not present on disk for assets. Providing demo execution fallback.")
            return DemoExecutionEngine.execute_optical_sar(query, assets)

        # 3. Real Neural Inference via CrossModalOpticalSARNet
        try:
            model = model_manager.get_model(self.model_id)
            if model is None:
                logger.warning(f"Could not load '{self.model_id}'. Falling back to demo mode.")
                res = DemoExecutionEngine.execute_optical_sar(query, assets)
                res["metadata"]["fallback_reason"] = "Model weights could not be loaded."
                return res

            fusion_res = run_optical_sar_fusion(
                model=model,
                optical_path=opt_path,
                sar_path=sar_path,
                device=model_manager.device,
                query=query
            )

            opt_preview = opt_asset.preview_url or f"/static/uploads/{opt_asset.filename}"
            sar_preview = sar_asset.preview_url or f"/static/uploads/{sar_asset.filename}"

            evidence_item = {
                "type": "sensor_cross_correlation",
                "opticalImageUrl": opt_preview,
                "sarImageUrl": sar_preview,
                "fusedImageUrl": fusion_res["fused_url"],
                "synergyMapUrl": fusion_res["synergy_url"],
                "opticalLabel": f"Multispectral Optical · {opt_asset.filename}",
                "sarLabel": f"SAR Microwave Backscatter · {sar_asset.filename}",
                "fusionSummary": fusion_res["fusion_summary"],
                "metrics": fusion_res["metrics"],
                "is_demo": False
            }

            return {
                "task": "optical_sar",
                "backend": "real",
                "confidence": fusion_res["confidence"],
                "answer": fusion_res["answer"],
                "evidence": [evidence_item],
                "metadata": {
                    "model_id": self.model_id,
                    "architecture": "CrossModalOpticalSARNet (Dual-Stream + Bidirectional Attention)",
                    "device": model_manager.device,
                    "optical_source": opt_asset.filename,
                    "sar_source": sar_asset.filename,
                    "sar_mean_db": fusion_res["metrics"]["sar_backscatter_mean_db"],
                    "edge_correlation": fusion_res["metrics"]["cross_modal_edge_correlation"],
                    "confidence_source": "Sobel structural edge correlation and spatial cross-attention coherence",
                    "is_demo": False
                }
            }

        except Exception as e:
            logger.error(f"Error during real optical-sar fusion inference: {e}", exc_info=True)
            raise RuntimeError(f"Optical + SAR fusion execution failed: {e}")
