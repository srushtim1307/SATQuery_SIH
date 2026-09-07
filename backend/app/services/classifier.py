"""
Deterministic Task Classifier & Context-Aware Router for SatQuery AI.

Translates natural-language queries and uploaded image context into normalized specialist tasks.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from app.core.constants import (
    TASK_VQA,
    TASK_GROUNDING,
    TASK_CHANGE_DETECTION,
    TASK_OPTICAL_SAR,
    TASK_CAPTIONING,
    SPECIALIST_DEFINITIONS
)
from app.db.models import ImageAsset

class TaskClassifier:
    """
    Deterministic intent and context classifier for multimodal satellite remote sensing.
    """

    # 1. Out-of-scope intent triggers (non-remote-sensing requests)
    UNSUPPORTED_PATTERNS = [
        r"\b(weather\s+forecast|predict\s+(tomorrow|next\s+week)|forecast\s+weather)\b",
        r"\b(write\s+(a\s+)?(poem|story|code|essay|song|joke))\b",
        r"\b(who\s+won|capital\s+of|president\s+of|stock\s+price|translate\s+to)\b",
        r"\b(python|javascript|sql|c\+\+|html)\s+(script|code|program)\b",
        r"\b(recipe|cook|restaurant|flight\s+ticket)\b"
    ]

    # 2. Change detection triggers
    CHANGE_PATTERNS = [
        r"\b(what\s+changed|how\s+has\s+.*changed|changes?\s+between|difference\s+between)\b",
        r"\b(before\s+and\s+after|temporal\s+change|expansion|loss|decrease|increase|growth)\b",
        r"\b(has\s+the\s+.*increased|has\s+the\s+.*decreased|deforestation|urbanization)\b",
        r"\b(compare\s+(the\s+)?(two|both|temporal|before|after)\s+images?)\b"
    ]

    # 3. Optical + SAR cross-modal triggers
    OPTICAL_SAR_PATTERNS = [
        r"\b(optical\s+and\s+sar|sar\s+and\s+optical|radar\s+and\s+optical)\b",
        r"\b(compare\s+optical\s+and\s+sar|fuse\s+radar|cross-modal|multi-sensor)\b",
        r"\b(use\s+both\s+images\s+to\s+identify|what\s+does\s+sar\s+reveal)\b",
        r"\b(complementary\s+sensor|backscatter\s+and\s+reflectance)\b"
    ]

    # 4. Visual grounding triggers (locative verbs, spatial directives)
    GROUNDING_PATTERNS = [
        r"\b(where\s+(is|are)|locate|highlight|pinpoint|find|bound|bbox|detect\s+(the|all)?)\b",
        r"\b(show\s+me\s+where|identify\s+the\s+position|draw\s+(a\s+)?box)\b",
        r"\b(point\s+out|mark\s+the|spatial\s+coordinates)\b"
    ]

    # 5. Scene captioning & land-cover triggers
    CAPTIONING_PATTERNS = [
        r"\b(describe\s+(this|the)\s+(scene|image|imagery|landscape|area|terrain))\b",
        r"\b(describe\s+(the\s+)?land[- ]*cover|summarize\s+land[- ]*use)\b",
        r"\b(scene\s+(caption|description)|overview\s+of\s+this|generate\s+caption)\b",
        r"\b(land[- ]*cover\s+(classes|breakdown|classification|description|mapping|type|distribution))\b",
        r"\b(what\s+kind\s+of\s+terrain|general\s+description|scene\s+overview)\b"
    ]

    # 6. Ambiguous / generic prompts
    AMBIGUOUS_PATTERNS = [
        r"^(analyze|check|inspect|look\s+at)\s+(this(\s+(image|imagery|raster))?|the\s+(image|imagery|raster)|image|imagery|raster)$",
        r"^(analyze|inspect|check\s+this|look\s+at\s+this|satquery\s+look)$"
    ]


    @classmethod
    def classify(
        cls, 
        query: str, 
        assets: List[ImageAsset], 
        mode: str = "auto"
    ) -> Dict[str, Any]:
        """
        Classifies intent and checks input context compatibility.
        Returns structured routing decision.
        """
        clean_q = query.strip()
        q_lower = clean_q.lower()
        img_count = len(assets)
        modalities = [a.modality for a in assets]
        has_optical = "optical" in modalities
        has_sar = "sar" in modalities

        # Step A: Check for out-of-scope / unsupported requests
        for pattern in cls.UNSUPPORTED_PATTERNS:
            if re.search(pattern, q_lower):
                return {
                    "task": "unsupported",
                    "specialist": "none",
                    "confidence": 0.99,
                    "reason": "This request is outside SatQuery AI's supported analysis tasks.",
                    "required_inputs": [],
                    "compatible": False,
                    "is_unsupported": True
                }

        # Step B: Manual mode override
        norm_mode = mode.lower().replace(" ", "_").replace("+", "_")
        if norm_mode not in ["auto", "auto_detect", "auto_detect_(recommended)"]:
            return cls._resolve_manual_mode(norm_mode, query, assets)

        # Step C: Ambiguous query handling
        for pattern in cls.AMBIGUOUS_PATTERNS:
            if re.search(pattern, q_lower.strip(" .!?")):
                return {
                    "task": TASK_CAPTIONING,
                    "specialist": SPECIALIST_DEFINITIONS[TASK_CAPTIONING]["id"],
                    "confidence": 0.60,
                    "reason": "General analysis requested. Defaulting to Scene Captioning & Land-Cover Description.",
                    "required_inputs": ["single_image"],
                    "compatible": img_count == 1,
                    "is_unsupported": False
                }

        # Step D: Intent classification from query
        detected_task = None
        confidence = 0.85
        reason = ""

        # 1. Check Change Detection intent
        if any(re.search(p, q_lower) for p in cls.CHANGE_PATTERNS):
            detected_task = TASK_CHANGE_DETECTION
            confidence = 0.94
            reason = "The query asks to compare differences across temporal states."

        # 2. Check Optical + SAR cross-modal intent
        elif any(re.search(p, q_lower) for p in cls.OPTICAL_SAR_PATTERNS):
            detected_task = TASK_OPTICAL_SAR
            confidence = 0.95
            reason = "The query explicitly references multi-sensor or Optical + SAR cross-modal observations."

        # 3. Check Grounding intent
        elif any(re.search(p, q_lower) for p in cls.GROUNDING_PATTERNS):
            detected_task = TASK_GROUNDING
            confidence = 0.92
            reason = "The query asks to locate, bound, or highlight specific visual regions."

        # 4. Check Scene Captioning intent
        elif any(re.search(p, q_lower) for p in cls.CAPTIONING_PATTERNS):
            detected_task = TASK_CAPTIONING
            confidence = 0.91
            reason = "The query requests a descriptive overview of scene composition and land cover."

        # 5. Fallback: Questions starting with Is/Are/What/How/Does -> VQA
        else:
            detected_task = TASK_VQA
            confidence = 0.82
            reason = "The query asks a visual question regarding features or attributes."

        # Step E: Context-Aware Input Compatibility Coupling
        return cls._couple_intent_with_context(detected_task, confidence, reason, assets)

    @classmethod
    def _couple_intent_with_context(
        cls, 
        task: str, 
        confidence: float, 
        reason: str, 
        assets: List[ImageAsset]
    ) -> Dict[str, Any]:
        """
        Couples classified intent with available imagery to determine execution compatibility.
        """
        img_count = len(assets)
        modalities = [a.modality for a in assets]
        has_optical = "optical" in modalities
        has_sar = "sar" in modalities

        # Context Check for Change Detection
        if task == TASK_CHANGE_DETECTION:
            if img_count != 2:
                return {
                    "task": TASK_CHANGE_DETECTION,
                    "specialist": SPECIALIST_DEFINITIONS[TASK_CHANGE_DETECTION]["id"],
                    "confidence": confidence,
                    "reason": "Two images are required for change analysis. Please upload a before and after image.",
                    "required_inputs": ["bi_temporal_pair"],
                    "compatible": False,
                    "is_unsupported": False
                }
            return {
                "task": TASK_CHANGE_DETECTION,
                "specialist": SPECIALIST_DEFINITIONS[TASK_CHANGE_DETECTION]["id"],
                "confidence": confidence,
                "reason": reason,
                "required_inputs": ["bi_temporal_pair"],
                "compatible": True,
                "is_unsupported": False
            }

        # Context Check for Optical + SAR
        if task == TASK_OPTICAL_SAR:
            if img_count != 2 or not (has_optical and has_sar):
                err = "Optical + SAR analysis requires exactly two images: one optical image and one SAR image."
                if img_count == 2 and not has_sar:
                    err = "Missing SAR image in upload pair. Please upload a SAR image alongside the optical image."
                return {
                    "task": TASK_OPTICAL_SAR,
                    "specialist": SPECIALIST_DEFINITIONS[TASK_OPTICAL_SAR]["id"],
                    "confidence": confidence,
                    "reason": err,
                    "required_inputs": ["optical_sar_pair"],
                    "compatible": False,
                    "is_unsupported": False
                }
            return {
                "task": TASK_OPTICAL_SAR,
                "specialist": SPECIALIST_DEFINITIONS[TASK_OPTICAL_SAR]["id"],
                "confidence": confidence,
                "reason": reason,
                "required_inputs": ["optical_sar_pair"],
                "compatible": True,
                "is_unsupported": False
            }

        # If user uploaded 2 images (Optical + SAR) but query was general/grounding:
        if img_count == 2 and has_optical and has_sar:
            # Auto-route to Optical + SAR cross-modal specialist
            return {
                "task": TASK_OPTICAL_SAR,
                "specialist": SPECIALIST_DEFINITIONS[TASK_OPTICAL_SAR]["id"],
                "confidence": 0.88,
                "reason": f"Input consists of an Optical + SAR pair. Routing '{task}' query through cross-modal specialist.",
                "required_inputs": ["optical_sar_pair"],
                "compatible": True,
                "is_unsupported": False
            }

        # If user uploaded 2 optical images but query was general:
        if img_count == 2 and task in [TASK_VQA, TASK_GROUNDING, TASK_CAPTIONING]:
            # Auto-route to Change Detection
            return {
                "task": TASK_CHANGE_DETECTION,
                "specialist": SPECIALIST_DEFINITIONS[TASK_CHANGE_DETECTION]["id"],
                "confidence": 0.80,
                "reason": "Two images provided in auto mode. Routing to Bi-Temporal Change Detection specialist.",
                "required_inputs": ["bi_temporal_pair"],
                "compatible": True,
                "is_unsupported": False
            }

        # Single-image tasks (VQA, Grounding, Captioning)
        if img_count != 1:
            return {
                "task": task,
                "specialist": SPECIALIST_DEFINITIONS[task]["id"],
                "confidence": confidence,
                "reason": f"{SPECIALIST_DEFINITIONS[task]['name']} requires exactly one satellite image. You provided {img_count}.",
                "required_inputs": ["single_image"],
                "compatible": False,
                "is_unsupported": False
            }

        return {
            "task": task,
            "specialist": SPECIALIST_DEFINITIONS[task]["id"],
            "confidence": confidence,
            "reason": reason,
            "required_inputs": ["single_image"],
            "compatible": True,
            "is_unsupported": False
        }

    @classmethod
    def _resolve_manual_mode(cls, mode: str, query: str, assets: List[ImageAsset]) -> Dict[str, Any]:
        """
        Validates manual specialist selection against input constraints.
        Manual mode does NOT bypass input compatibility checks.
        """
        img_count = len(assets)
        modalities = [a.modality for a in assets]

        task_map = {
            "vqa": TASK_VQA,
            "grounding": TASK_GROUNDING,
            "change_detection": TASK_CHANGE_DETECTION,
            "change": TASK_CHANGE_DETECTION,
            "optical_sar": TASK_OPTICAL_SAR,
            "optical_sar_cross_modal_fusion": TASK_OPTICAL_SAR,
            "captioning": TASK_CAPTIONING
        }

        task = task_map.get(mode, TASK_VQA)
        spec = SPECIALIST_DEFINITIONS.get(task, SPECIALIST_DEFINITIONS[TASK_VQA])

        if task == TASK_CHANGE_DETECTION:
            compatible = (img_count == 2)
            reason = (
                "Manual mode selected: Bi-Temporal Change Detection." if compatible
                else "Two images are required for change analysis. Please upload a before and after image."
            )
            return {
                "task": task,
                "specialist": spec["id"],
                "confidence": 1.0,  # Explicit manual override
                "reason": reason,
                "required_inputs": ["bi_temporal_pair"],
                "compatible": compatible,
                "is_unsupported": False
            }

        elif task == TASK_OPTICAL_SAR:
            compatible = (img_count == 2 and "optical" in modalities and "sar" in modalities)
            reason = (
                "Manual mode selected: Optical + SAR Fusion." if compatible
                else "Optical + SAR analysis requires exactly two images: one optical image and one SAR image."
            )
            return {
                "task": task,
                "specialist": spec["id"],
                "confidence": 1.0,
                "reason": reason,
                "required_inputs": ["optical_sar_pair"],
                "compatible": compatible,
                "is_unsupported": False
            }

        else:  # Single image tasks
            compatible = (img_count == 1)
            reason = (
                f"Manual mode selected: {spec['name']}." if compatible
                else f"{spec['name']} requires exactly one satellite image. You provided {img_count}."
            )
            return {
                "task": task,
                "specialist": spec["id"],
                "confidence": 1.0,
                "reason": reason,
                "required_inputs": ["single_image"],
                "compatible": compatible,
                "is_unsupported": False
            }
