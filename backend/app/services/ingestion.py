import os
import re
import json
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from PIL import Image, TiffTags
import tifffile
from app.core.config import settings, UPLOADS_DIR, PROCESSED_DIR

# Common GeoTIFF Tag IDs
TAG_MODEL_PIXEL_SCALE = 33550
TAG_MODEL_TIEPOINT = 33922
TAG_GEO_KEY_DIRECTORY = 34735
TAG_GEO_DOUBLE_PARAMS = 34736
TAG_GEO_ASCII_PARAMS = 34737

def sanitize_filename(filename: str) -> str:
    """Sanitize original filename to prevent path traversal and shell injection."""
    name = Path(filename).name
    clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', name)
    clean = clean.lstrip('.')
    return clean or "unnamed_raster.tif"

def detect_modality_safely(filename: str, bands: int, tiff_tags: Dict[int, Any]) -> str:
    """
    Safely determine modality based on metadata, band count, and filename hints.
    Never fabricates certainty.
    """
    fn_lower = filename.lower()
    
    # 1. Check strong SAR hints
    if any(k in fn_lower for k in ["sar", "sentinel-1", "sentinel1", "s1", "risat", "c-band", "grd", "slc"]):
        return "sar"

    # 2. Check strong Optical hints
    if any(k in fn_lower for k in ["optical", "sentinel-2", "sentinel2", "s2", "landsat", "cartosat", "truecolor", "rgb", "ndvi", "estuary"]):
        return "optical"

    # 3. Check bands & color space
    if bands >= 3:
        return "optical"
    elif bands == 1 and any(k in fn_lower for k in ["radar", "backscatter", "amp", "intensity"]):
        return "sar"

    return "unknown"

def inspect_and_extract_metadata(file_path: Path, filename: str) -> Dict[str, Any]:
    """
    Inspect image / GeoTIFF headers without loading the entire raster into memory.
    Extracts dimensions, band count, format, GeoTIFF CRS/resolution if present.
    """
    warnings: List[str] = []
    ext = file_path.suffix.lower()
    format_type = "TIFF" if ext in [".tif", ".tiff"] else ("PNG" if ext == ".png" else "JPEG")
    
    width: Optional[int] = None
    height: Optional[int] = None
    bands: int = 1
    crs: Optional[str] = None
    resolution: Optional[str] = None
    acquisition_date: Optional[str] = None
    is_geotiff = False

    try:
        if ext in [".tif", ".tiff"]:
            with tifffile.TiffFile(file_path) as tf:
                page = tf.pages[0]
                width = int(page.imagewidth)
                height = int(page.imagelength)
                bands = int(getattr(page, 'samplesperpixel', 1))
                if bands == 1 and len(page.shape) == 3:
                    bands = int(page.shape[2])

                # Check for GeoTIFF specific tags
                if TAG_GEO_KEY_DIRECTORY in page.tags or TAG_MODEL_PIXEL_SCALE in page.tags or TAG_MODEL_TIEPOINT in page.tags:
                    is_geotiff = True
                    format_type = "GeoTIFF"

                    if TAG_MODEL_PIXEL_SCALE in page.tags:
                        scales = page.tags[TAG_MODEL_PIXEL_SCALE].value
                        if len(scales) >= 2:
                            resolution = f"{round(scales[0], 2)}m Ground Resolution"

                    if TAG_GEO_ASCII_PARAMS in page.tags:
                        ascii_val = str(page.tags[TAG_GEO_ASCII_PARAMS].value).strip("|\x00 ")
                        if ascii_val:
                            crs = ascii_val.split("|")[0]
                    
                    if not crs and TAG_GEO_KEY_DIRECTORY in page.tags:
                        crs = "WGS 84 / UTM (GeoKey Projected)"

                # Extract acquisition date if present in tags
                if 306 in page.tags:  # DateTime tag
                    acquisition_date = str(page.tags[306].value)
                
                # Date hint from filename (e.g. 2023, 2024, May, Oct)
                if not acquisition_date:
                    date_match = re.search(r'(20\d\d[-_]?\d\d?[-_]?\d\d?|may\d{4}|oct\d{4})', filename.lower())
                    if date_match:
                        acquisition_date = date_match.group(0)

        else:
            # Standard PNG / JPEG
            with Image.open(file_path) as img:
                width, height = img.size
                bands = len(img.getbands())
                format_type = img.format or format_type

    except Exception as e:
        warnings.append(f"Header inspection encountered error: {str(e)}")
        # Fallback to Pillow
        with Image.open(file_path) as img:
            width, height = img.size
            bands = len(img.getbands())

    # Determine modality
    modality = detect_modality_safely(filename, bands, {})
    if modality == "unknown":
        warnings.append("Modality could not be definitively identified from metadata alone.")

    return {
        "format": format_type,
        "is_geotiff": is_geotiff,
        "width": width,
        "height": height,
        "bands": bands,
        "crs": crs,
        "resolution": resolution,
        "acquisition_date": acquisition_date,
        "modality": modality,
        "warnings": warnings
    }

def generate_web_preview(file_path: Path, output_preview_path: Path, max_dim: int = 1200) -> str:
    """
    Generate an optimized, visually grounded PNG web preview.
    Safely normalizes uint16/multispectral/grayscale rasters without distortion.
    """
    output_preview_path.parent.mkdir(parents=True, exist_ok=True)
    ext = file_path.suffix.lower()

    if ext in [".tif", ".tiff"]:
        try:
            arr = tifffile.imread(file_path)
            
            # Dimensionality handling
            if arr.ndim == 3:
                if arr.shape[0] in [1, 3, 4] and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                    arr = np.transpose(arr, (1, 2, 0))

                if arr.shape[2] >= 3:
                    rgb = arr[:, :, :3].astype(np.float32)
                else:
                    rgb = arr[:, :, 0].astype(np.float32)
            else:
                rgb = arr.astype(np.float32)

            # Robust 2% - 98% percentile normalization
            if rgb.ndim == 3:
                norm_bands = []
                for c in range(3):
                    b = rgb[:, :, c]
                    p2, p98 = np.percentile(b, (2, 98))
                    if p98 > p2:
                        b_norm = np.clip((b - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        b_norm = np.clip(b, 0, 255).astype(np.uint8)
                    norm_bands.append(b_norm)
                preview_img = Image.fromarray(np.stack(norm_bands, axis=-1))
            else:
                p2, p98 = np.percentile(rgb, (2, 98))
                if p98 > p2:
                    gray_norm = np.clip((rgb - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
                else:
                    gray_norm = np.clip(rgb, 0, 255).astype(np.uint8)
                preview_img = Image.fromarray(gray_norm).convert("RGB")

        except Exception:
            with Image.open(file_path) as pimg:
                preview_img = pimg.convert("RGB")
    else:
        with Image.open(file_path) as pimg:
            preview_img = pimg.convert("RGB")

    # Resize preserving aspect ratio if larger than max_dim
    w, h = preview_img.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        preview_img = preview_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Save preview as PNG
    preview_img.save(output_preview_path, format="PNG", optimize=True)
    return f"/static/processed/{output_preview_path.name}"
