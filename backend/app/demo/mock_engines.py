"""
Demo / Simulated Execution Engines for SatQuery AI.

IMPORTANT:
- All executions here are explicitly marked with backend: "demo".
- No fabricated measurements or model confidences are presented as real model outputs.
- Model confidence is explicitly set to None (or clearly labeled simulated_confidence).
"""

from typing import List, Dict, Any, Optional
from app.db.models import ImageAsset

class DemoExecutionEngine:
    @staticmethod
    def execute_vqa(query: str, assets: List[ImageAsset]) -> Dict[str, Any]:
        asset = assets[0] if assets else None
        res_info = asset.resolution if (asset and asset.resolution) else "Standard Resolution"
        format_info = asset.format if asset else "Raster"
        dims = f"{asset.width}x{asset.height}" if (asset and asset.width) else "1024x1024"
        
        q_lower = query.lower()
        if "building" in q_lower or "structure" in q_lower:
            answer = (
                f"[Demo Execution] High-density structural clusters and linear geometric boundaries are observed in the {format_info} imagery ({dims}). "
                f"Built-up footprints align with local transportation corridors."
            )
        elif "water" in q_lower or "river" in q_lower or "lake" in q_lower:
            answer = (
                f"[Demo Execution] Distinct low-reflectance water bodies and drainage channels are identifiable in the spectral bands. "
                f"Surface boundaries indicate natural riparian wetlands."
            )
        elif "vegetation" in q_lower or "forest" in q_lower or "crop" in q_lower:
            answer = (
                f"[Demo Execution] Moderate to dense vegetative canopy is indicated across the non-urban sectors of the scene."
            )
        else:
            answer = (
                f"[Demo Execution] Visual inspection of the {format_info} raster ({dims}, {res_info}) indicates diverse surface reflectance consistent with mixed land-cover patterns."
            )

        return {
            "task": "vqa",
            "backend": "demo",
            "confidence": None,  # Real confidence is None for demo execution
            "answer": answer,
            "evidence": [
                {
                    "type": "spectral_band_analysis",
                    "description": f"Analyzed {asset.bands if asset else 1} channel(s) across {dims} grid",
                    "sensor_context": asset.modality if asset else "optical",
                    "is_demo": True
                }
            ],
            "metadata": {
                "source_raster": asset.filename if asset else "unknown",
                "format": format_info,
                "resolution": res_info,
                "note": "Demo simulated output for pipeline demonstration."
            }
        }

    @staticmethod
    def execute_grounding(query: str, assets: List[ImageAsset]) -> Dict[str, Any]:
        asset = assets[0] if assets else None
        img_url = asset.preview_url if (asset and asset.preview_url) else "/demo/sentinel2_estuary_delta.png"
        
        # Determine target label from query
        q_lower = query.lower()
        if "water" in q_lower:
            label = "Water Body / Estuary Channel"
            boxes = [
                {"id": "box-demo-1", "label": label, "top": 35, "left": 48, "width": 30, "height": 28, "is_demo": True},
                {"id": "box-demo-2", "label": "Tributary Inflow", "top": 18, "left": 62, "width": 18, "height": 20, "is_demo": True}
            ]
        elif "road" in q_lower or "highway" in q_lower:
            label = "Primary Transit Corridor"
            boxes = [
                {"id": "box-demo-1", "label": label, "top": 20, "left": 10, "width": 75, "height": 15, "is_demo": True}
            ]
        elif "building" in q_lower or "urban" in q_lower:
            label = "Built-Up Settlement Zone"
            boxes = [
                {"id": "box-demo-1", "label": label, "top": 42, "left": 25, "width": 32, "height": 34, "is_demo": True}
            ]
        else:
            label = "Identified Feature of Interest"
            boxes = [
                {"id": "box-demo-1", "label": label, "top": 30, "left": 40, "width": 35, "height": 30, "is_demo": True}
            ]

        return {
            "task": "grounding",
            "backend": "demo",
            "confidence": None,
            "answer": f"[Demo Execution] Successfully grounded referenced target '{label}' within the imagery.",
            "evidence": [
                {
                    "type": "bounding_coordinates",
                    "image_url": img_url,
                    "target_feature": label,
                    "bounding_boxes": boxes,
                    "is_demo": True
                }
            ],
            "metadata": {
                "detected_objects_count": len(boxes),
                "source_raster": asset.filename if asset else "unknown",
                "note": "Demo simulated bounding coordinates."
            }
        }

    @staticmethod
    def execute_change_detection(query: str, assets: List[ImageAsset]) -> Dict[str, Any]:
        t1_name = assets[0].filename if len(assets) > 0 else "baseline_raster.tif"
        t2_name = assets[1].filename if len(assets) > 1 else "current_raster.tif"
        
        return {
            "task": "change_detection",
            "backend": "demo",
            "confidence": None,
            "answer": (
                f"[Demo Execution] Bi-temporal analysis between '{t1_name}' (T1) and '{t2_name}' (T2) reveals notable spatial dynamics. "
                "Observable expansion is concentrated along eastern perimeter corridors, alongside surface moisture redistribution."
            ),
            "evidence": [
                {
                    "type": "bi_temporal_differential",
                    "t1_filename": t1_name,
                    "t2_filename": t2_name,
                    "primary_change_class": "Urban Infill & Vegetation Transition",
                    "is_demo": True
                }
            ],
            "metadata": {
                "pair_count": len(assets),
                "alignment": "Co-registered (simulated)",
                "note": "Demo simulated change detection output."
            }
        }

    @staticmethod
    def execute_optical_sar(query: str, assets: List[ImageAsset]) -> Dict[str, Any]:
        opt_asset = next((a for a in assets if a.modality == "optical"), assets[0] if assets else None)
        sar_asset = next((a for a in assets if a.modality == "sar"), assets[1] if len(assets) > 1 else None)
        
        return {
            "task": "optical_sar",
            "backend": "demo",
            "confidence": None,
            "answer": (
                f"[Demo Execution] Cross-modal fusion between Optical multispectral reflectance ('{opt_asset.filename if opt_asset else 'optical'}') "
                f"and SAR microwave backscatter ('{sar_asset.filename if sar_asset else 'sar'}') completed. "
                "Optical channels delineate spectral land-cover classes while SAR backscatter penetrates cloud cover and resolves dielectric roughness."
            ),
            "evidence": [
                {
                    "type": "sensor_cross_correlation",
                    "optical_source": opt_asset.filename if opt_asset else "optical",
                    "sar_source": sar_asset.filename if sar_asset else "sar",
                    "fusion_technique": "Multispectral & Radar Cross-Attention (Demo)",
                    "is_demo": True
                }
            ],
            "metadata": {
                "optical_bands": opt_asset.bands if opt_asset else 3,
                "sar_polarization": "VV/VH (Inferred)",
                "note": "Demo simulated cross-sensor fusion output."
            }
        }

    @staticmethod
    def execute_captioning(query: str, assets: List[ImageAsset]) -> Dict[str, Any]:
        asset = assets[0] if assets else None
        return {
            "task": "captioning",
            "backend": "demo",
            "confidence": None,
            "answer": (
                f"[Demo Execution] Comprehensive scene caption: Satellite acquisition of '{asset.filename if asset else 'raster'}' "
                "illustrates a heterogeneous coastal estuary environment characterized by meandering tidal waterways, "
                "agricultural parcels in the hinterland, and scattered settlement clusters along transport corridors."
            ),
            "evidence": [
                {
                    "type": "land_cover_summary",
                    "classes": ["Water Bodies (34%)", "Agricultural Land (42%)", "Built-Up Settlements (18%)", "Bare Soil (6%)"],
                    "is_demo": True
                }
            ],
            "metadata": {
                "source_raster": asset.filename if asset else "unknown",
                "format": asset.format if asset else "TIFF",
                "note": "Demo simulated scene description."
            }
        }
