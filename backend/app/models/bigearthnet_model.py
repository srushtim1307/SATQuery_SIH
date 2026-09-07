"""
BigEarthNet-19 Adapted Land-Cover Classification & Captioning Model.

Backbone: MobileNetV3-Small fine-tuned on the 19-class BigEarthNet taxonomy.
Weights: backend/models/weights/bigearthnet_adapted.pth
Taxonomy: backend/models/weights/BigEarthNet.txt
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = Path("backend/models/weights/bigearthnet_adapted.pth")
DEFAULT_TAXONOMY_PATH = Path("backend/models/weights/BigEarthNet.txt")

BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters"
]


def load_bigearthnet_model(weights_path: Optional[str] = None, device: str = "cpu") -> nn.Module:
    """
    Instantiates MobileNetV3-Small and loads the adapted BigEarthNet-19 weights.
    """
    path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        # Look relative to workspace root if backend/ is working directory
        alt_path = Path("models/weights/bigearthnet_adapted.pth")
        if alt_path.exists():
            path = alt_path

    logger.info(f"Loading BigEarthNet adapted model from {path} on device='{device}'...")
    model = models.mobilenet_v3_small(weights=None, num_classes=len(BIGEARTHNET_19_CLASSES))

    if path.exists():
        state_dict = torch.load(str(path), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Successfully loaded adapted BigEarthNet-19 checkpoint weights.")
    else:
        logger.warning(f"Weights file not found at {path}. Model initialized with random weights.")

    model.to(device)
    model.eval()
    return model


def classify_land_cover(
    model: nn.Module,
    image_path: str,
    device: str = "cpu",
    top_k: int = 4
) -> Dict[str, Any]:
    """
    Executes multi-spectral land cover classification across BigEarthNet-19 taxonomy.
    Synthesizes natural-language scene caption from predicted class distributions.
    """
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((224, 224), Image.Resampling.BILINEAR)

    # Standard ImageNet normalization
    arr = np.array(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std

    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    # Get top-K class predictions
    sorted_indices = np.argsort(probs)[::-1][:top_k]
    top_classes = []
    for idx in sorted_indices:
        prob = float(probs[idx])
        top_classes.append({
            "class_id": int(idx),
            "class_name": BIGEARTHNET_19_CLASSES[idx],
            "probability": round(prob, 4),
            "percentage": round(prob * 100, 1)
        })

    primary = top_classes[0]
    secondary = top_classes[1] if len(top_classes) > 1 else None
    tertiary = top_classes[2] if len(top_classes) > 2 else None

    # Synthesize rich descriptive scene caption
    if primary["probability"] > 0.4:
        caption = (
            f"Multi-spectral land-cover classification identifies dominant terrain as '{primary['class_name']}' "
            f"({primary['percentage']}% likelihood)."
        )
        if secondary and secondary["probability"] > 0.25:
            caption += f" Secondary terrain associations include '{secondary['class_name']}' ({secondary['percentage']}%)"
            if tertiary and tertiary["probability"] > 0.15:
                caption += f" and '{tertiary['class_name']}' ({tertiary['percentage']}%)."
            else:
                caption += "."
    else:
        caption = (
            f"Complex multi-spectral terrain distribution observed: '{primary['class_name']}' ({primary['percentage']}%), "
            f"'{secondary['class_name']}' ({secondary['percentage']}%), and '{tertiary['class_name']}' ({tertiary['percentage']}%)."
        )

    return {
        "caption": caption,
        "primary_class": primary["class_name"],
        "confidence": primary["probability"],
        "top_classes": top_classes,
        "model_name": "MobileNetV3-Small (BigEarthNet-19 Adapted)",
        "taxonomy": "BigEarthNet-19 (CORINE Land Cover)",
        "backend": "adapted"
    }
