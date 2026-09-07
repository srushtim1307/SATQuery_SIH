"""
Central Model Registry for SatQuery AI.

Maintains available specialist model adapters, their modalities, input constraints,
and runtime capabilities.
"""

from typing import Dict, List, Optional, Any
from app.adapters.base import BaseModelAdapter
from app.adapters.vqa_adapter import VQAAdapter
from app.adapters.grounding_adapter import GroundingAdapter
from app.adapters.change_adapter import ChangeDetectionAdapter
from app.adapters.optical_sar_adapter import OpticalSARAdapter
from app.adapters.captioning_adapter import CaptioningAdapter

class ModelEntry:
    def __init__(
        self,
        model_id: str,
        name: str,
        version: str,
        task: str,
        supported_modalities: List[str],
        supported_input_types: List[str],
        adapter: BaseModelAdapter,
        backend_type: str = "demo",
        status: str = "active"
    ):
        self.id = model_id
        self.name = name
        self.version = version
        self.task = task
        self.supported_modalities = supported_modalities
        self.supported_input_types = supported_input_types
        self.adapter = adapter
        self.backend_type = backend_type
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "supported_modalities": self.supported_modalities,
            "supported_input_types": self.supported_input_types,
            "backend_type": self.backend_type,
            "status": self.status
        }

class ModelRegistry:
    _instance = None

    def __init__(self):
        self._models: Dict[str, ModelEntry] = {}
        self._register_default_models()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_default_models(self):
        # 1. VQA Specialist (SmolVLM)
        vqa_adapter = VQAAdapter(backend="real")
        self.register(ModelEntry(
            model_id=vqa_adapter.model_id,
            name=vqa_adapter.name,
            version=vqa_adapter.version,
            task="vqa",
            supported_modalities=["optical", "sar", "multispectral", "unknown"],
            supported_input_types=["single_image"],
            adapter=vqa_adapter,
            backend_type=vqa_adapter.backend
        ))

        # 2. Grounding Specialist (OWL-ViT)
        grounding_adapter = GroundingAdapter(backend="real")
        self.register(ModelEntry(
            model_id=grounding_adapter.model_id,
            name=grounding_adapter.name,
            version=grounding_adapter.version,
            task="grounding",
            supported_modalities=["optical", "multispectral", "sar", "unknown"],
            supported_input_types=["single_image"],
            adapter=grounding_adapter,
            backend_type=grounding_adapter.backend
        ))

        # 3. Change Detection Specialist (TinyCD)
        change_adapter = ChangeDetectionAdapter(backend="real")
        self.register(ModelEntry(
            model_id=change_adapter.model_id,
            name=change_adapter.name,
            version=change_adapter.version,
            task="change_detection",
            supported_modalities=["optical", "multispectral", "sar", "unknown"],
            supported_input_types=["bi_temporal_pair"],
            adapter=change_adapter,
            backend_type=change_adapter.backend
        ))

        # 4. Optical + SAR Cross-Modal Specialist
        optical_sar_adapter = OpticalSARAdapter(backend="real")
        self.register(ModelEntry(
            model_id=optical_sar_adapter.model_id,
            name=optical_sar_adapter.name,
            version=optical_sar_adapter.version,
            task="optical_sar",
            supported_modalities=["optical", "sar", "multispectral"],
            supported_input_types=["optical_sar_pair"],
            adapter=optical_sar_adapter,
            backend_type=optical_sar_adapter.backend
        ))

        # 5. Scene Captioning Specialist (BigEarthNet-19 Adapted)
        captioning_adapter = CaptioningAdapter(backend="adapted")
        self.register(ModelEntry(
            model_id=captioning_adapter.model_id,
            name=captioning_adapter.name,
            version=captioning_adapter.version,
            task="captioning",
            supported_modalities=["optical", "multispectral", "sar", "unknown"],
            supported_input_types=["single_image"],
            adapter=captioning_adapter,
            backend_type=captioning_adapter.backend
        ))

    def register(self, entry: ModelEntry) -> None:
        self._models[entry.id] = entry

    def unregister(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.pop(model_id, None)

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.get(model_id)

    def list(self) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in self._models.values()]

    def find_capable_model(
        self,
        task: str,
        modalities: Optional[List[str]] = None,
        input_type: Optional[str] = None
    ) -> Optional[ModelEntry]:
        """
        Queries registry for an active model matching the target task and input constraints.
        """
        task_normalized = task.lower()

        for entry in self._models.values():
            if entry.status != "active":
                continue
            if entry.task != task_normalized:
                continue

            # Check input type if requested
            if input_type and input_type not in entry.supported_input_types:
                continue

            # Check modalities if requested
            if modalities:
                # If all requested modalities are compatible with entry
                if not all(m in entry.supported_modalities for m in modalities if m != "unknown"):
                    continue

            return entry

        # Fallback to any model matching task
        for entry in self._models.values():
            if entry.status == "active" and entry.task == task_normalized:
                return entry

        return None

# Global registry accessor
registry = ModelRegistry.get_instance()
