from typing import List, Dict, Any, Optional
from app.db.models import ImageAsset

def validate_analysis_configuration(
    assets: List[ImageAsset], 
    intended_mode: str
) -> Dict[str, Any]:
    """
    Validates uploaded imagery against task configuration rules before model execution.
    Produces human-readable errors and non-blocking warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []
    count = len(assets)
    mode_normalized = intended_mode.lower().replace(" ", "_").replace("+", "_")

    if count == 0:
        errors.append("No satellite images were provided. Please upload or stage at least one image.")
        return {
            "valid": False,
            "mode": intended_mode,
            "count": count,
            "images": [],
            "warnings": warnings,
            "errors": errors
        }

    # 1. Single-Image Modes (VQA, Grounding, Captioning, Single)
    if mode_normalized in ["single", "vqa", "grounding", "captioning"]:
        if count != 1:
            errors.append(f"{intended_mode.upper()} analysis requires exactly one satellite image. You provided {count}.")
        else:
            asset = assets[0]
            if asset.modality == "unknown":
                warnings.append("Image modality is unknown. Analysis will proceed using default multispectral processing.")

    # 2. Bi-Temporal Change Detection
    elif mode_normalized in ["change_detection", "bi_temporal", "change"]:
        if count != 2:
            errors.append("Two images are required for change analysis. Please upload a before and after image.")
        else:
            # Check if acquisition dates or filenames suggest temporal separation
            dates = [a.acquisition_date for a in assets if a.acquisition_date]
            if len(dates) < 2:
                warnings.append("Acquisition timestamp metadata could not be fully verified from TIFF tags. Proceeding with spatial diff.")
            
            # Check dimensions compatibility
            if assets[0].width and assets[1].width:
                if assets[0].width != assets[1].width or assets[0].height != assets[1].height:
                    warnings.append(
                        f"Image dimensions differ ({assets[0].width}x{assets[0].height} vs {assets[1].width}x{assets[1].height}). "
                        "Automated spatial resampling will be required during analysis."
                    )

    # 3. Cross-Modal Optical + SAR Fusion
    elif mode_normalized in ["optical_sar", "optical_sar_fusion", "fusion"]:
        if count != 2:
            errors.append("Optical + SAR analysis requires exactly two images: one optical image and one SAR image.")
        else:
            modalities = [a.modality for a in assets]
            has_optical = "optical" in modalities
            has_sar = "sar" in modalities

            if not has_optical and not has_sar:
                errors.append("Optical + SAR analysis requires one optical image and one SAR image. Neither could be detected.")
            elif not has_optical:
                errors.append("Missing optical image in upload pair. Please upload an optical image alongside the SAR image.")
            elif not has_sar:
                errors.append("Missing SAR image in upload pair. Please upload a SAR image alongside the optical image.")
            else:
                warnings.append("Dual-sensor pair accepted. Automated co-registration will align spectral and radar grids.")

    # 4. Auto Detect mode
    elif mode_normalized in ["auto", "auto_detect", "auto_detect_(recommended)"]:
        if count == 1:
            # Single image auto-routed to VQA/Grounding
            pass
        elif count == 2:
            modalities = [a.modality for a in assets]
            if "optical" in modalities and "sar" in modalities:
                warnings.append("Auto-detected Optical + SAR cross-modal pair.")
            else:
                warnings.append("Auto-detected Bi-temporal pair for change analysis.")
        else:
            errors.append(f"Auto-detect currently supports single images, bi-temporal pairs, or optical+SAR pairs. Received {count} images.")

    return {
        "valid": len(errors) == 0,
        "mode": intended_mode,
        "count": count,
        "images": [a.to_dict() for a in assets],
        "warnings": warnings,
        "errors": errors
    }
