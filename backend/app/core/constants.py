"""
Central constants and normalized definitions for SatQuery AI.
"""

from typing import Dict, Any

# 1. Normalized Internal Task IDs
TASK_VQA = "vqa"
TASK_GROUNDING = "grounding"
TASK_CHANGE_DETECTION = "change_detection"
TASK_OPTICAL_SAR = "optical_sar"
TASK_CAPTIONING = "captioning"

SUPPORTED_TASKS = [
    TASK_VQA,
    TASK_GROUNDING,
    TASK_CHANGE_DETECTION,
    TASK_OPTICAL_SAR,
    TASK_CAPTIONING
]

# 2. Specialist Capability Metadata
SPECIALIST_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    TASK_VQA: {
        "id": "remote_sensing_vqa",
        "task": TASK_VQA,
        "name": "Remote-Sensing VQA Specialist",
        "description": "Answers natural-language visual questions regarding spectral, structural, and semantic attributes of satellite imagery.",
        "supported_input_config": ["single_image"],
        "min_images": 1,
        "max_images": 1,
        "supported_modalities": ["optical", "sar", "multispectral", "unknown"],
        "default_model_id": "rs-vqa-base"
    },
    TASK_GROUNDING: {
        "id": "remote_sensing_grounding",
        "task": TASK_GROUNDING,
        "name": "Text-Guided Visual Grounding Specialist",
        "description": "Localizes and bounds visual objects, natural formations, and infrastructure described in text queries.",
        "supported_input_config": ["single_image"],
        "min_images": 1,
        "max_images": 1,
        "supported_modalities": ["optical", "multispectral", "sar", "unknown"],
        "default_model_id": "rs-grounding-focal"
    },
    TASK_CHANGE_DETECTION: {
        "id": "bi_temporal_change_detection",
        "task": TASK_CHANGE_DETECTION,
        "name": "Bi-Temporal Change Detection Specialist",
        "description": "Identifies structural, surface, and land-cover changes across before/after temporal image pairs.",
        "supported_input_config": ["bi_temporal_pair"],
        "min_images": 2,
        "max_images": 2,
        "supported_modalities": ["optical", "multispectral"],
        "default_model_id": "rs-change-diff"
    },
    TASK_OPTICAL_SAR: {
        "id": "optical_sar_cross_modal_fusion",
        "task": TASK_OPTICAL_SAR,
        "name": "Optical + SAR Cross-Modal Fusion Specialist",
        "description": "Fuses multi-spectral optical imagery with synthetic aperture radar (SAR) backscatter for all-weather feature identification.",
        "supported_input_config": ["optical_sar_pair"],
        "min_images": 2,
        "max_images": 2,
        "supported_modalities": ["optical", "sar"],
        "default_model_id": "rs-optical-sar-fusion"
    },
    TASK_CAPTIONING: {
        "id": "scene_captioning_landcover",
        "task": TASK_CAPTIONING,
        "name": "Scene Captioning & Land-Cover Specialist",
        "description": "Produces structured natural-language descriptions of overall scene context, terrain types, and land use distribution.",
        "supported_input_config": ["single_image"],
        "min_images": 1,
        "max_images": 1,
        "supported_modalities": ["optical", "multispectral", "sar", "unknown"],
        "default_model_id": "rs-captioning-landcover"
    }
}

# 3. Observable Execution Trace Steps
TRACE_STEPS = [
    "Understanding your question",
    "Checking image compatibility",
    "Identifying analysis task",
    "Selecting specialist model",
    "Analyzing imagery",
    "Preparing visual evidence"
]
