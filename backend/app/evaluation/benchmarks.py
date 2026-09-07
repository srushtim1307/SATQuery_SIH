"""
SatQuery AI — Remote Sensing Benchmark & Evaluation Framework (Phase 6).

Maintains evaluation metadata, measurement hooks, and verification reporting for
standard remote sensing vision-language benchmarks:
- BigEarthNet-19 (CORINE Land Cover Multi-Spectral Scene Classification)
- VRSBench (Visual Remote Sensing Grounding & Open-Vocabulary Localization)
- RSVQA (Remote Sensing Visual Question Answering)
- CDVQA (Change Detection Visual Question Answering & Bi-temporal Difference)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

BENCHMARK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "BigEarthNet-19": {
        "name": "BigEarthNet-19 (CORINE Taxonomy)",
        "task": "captioning",
        "description": "Multi-spectral Sentinel-2 land cover classification and scene breakdown across 19 standard CORINE classes.",
        "target_model": "MobileNetV3-Small (Adapted)",
        "evaluation_status": "evaluated",
        "dataset_source": "BigEarthNet / DLR Remote Sensing (19 CORINE classes)",
        "metrics": {
            "validation_accuracy": 0.9289,
            "final_epoch_loss": 0.1604,
            "training_epochs": 5,
            "measured_date": "2026-09-07",
            "source_log": "backend/models/weights/training_log.json"
        },
        "notes": "Adapted model trained on remote-sensing spectral distributions matching CORINE-19 taxonomy."
    },
    "VRSBench": {
        "name": "VRSBench (Visual Grounding)",
        "task": "grounding",
        "description": "Text-guided bounding-box localization benchmark for remote-sensing optical satellite imagery.",
        "target_model": "OWL-ViT (google/owlvit-base-patch32)",
        "evaluation_status": "not_executed",
        "dataset_source": "VRSBench Official Dataset",
        "metrics": None,
        "notes": "Evaluation not yet executed. Grounding specialist operates using zero-shot open-vocabulary vision-text embeddings."
    },
    "RSVQA": {
        "name": "RSVQA (Remote Sensing VQA)",
        "task": "vqa",
        "description": "Visual question answering benchmark on high-resolution and low-resolution satellite imagery.",
        "target_model": "SmolVLM-256M-Instruct (Idefics3)",
        "evaluation_status": "not_executed",
        "dataset_source": "RSVQA-HR / RSVQA-LR",
        "metrics": None,
        "notes": "Evaluation not yet executed. Specialist executes autoregressive multi-modal generation with token confidence extraction."
    },
    "CDVQA": {
        "name": "CDVQA / Bi-Temporal Change Detection",
        "task": "change_detection",
        "description": "Bi-temporal satellite change verification and conversational change reasoning.",
        "target_model": "TinyCD (EfficientNet-B4 + MAMB)",
        "evaluation_status": "not_executed",
        "dataset_source": "LEVIR-CD / WHU-CD / CDVQA",
        "metrics": None,
        "notes": "Evaluation not yet executed. Inference utilizes pretrained weights from HZDR-FWGEL/UCD-MNCD256-TinyCD."
    }
}


def get_benchmark_status() -> List[Dict[str, Any]]:
    """
    Returns the current status and genuine metrics for all remote sensing benchmarks.
    Strictly preserves evaluation integrity without fabricated numbers.
    """
    benchmarks = []
    for key, val in BENCHMARK_REGISTRY.items():
        entry = dict(val)
        entry["id"] = key
        benchmarks.append(entry)
    return benchmarks


def get_single_benchmark(benchmark_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves metadata and status for a specific benchmark."""
    return BENCHMARK_REGISTRY.get(benchmark_id)
