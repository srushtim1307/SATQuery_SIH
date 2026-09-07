"""
Base Model Adapter definition for SatQuery AI specialist vision-language models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from app.db.models import ImageAsset

class BaseModelAdapter(ABC):
    """
    Standard interface that all specialist model adapters must implement.
    Allows seamlessly swapping between demo engines and real deep learning models
    (HuggingFace, PyTorch, ONNX, etc.) without affecting the orchestrator or API.
    """

    def __init__(self, model_id: str, name: str, version: str, task: str, backend: str = "demo"):
        self.model_id = model_id
        self.name = name
        self.version = version
        self.task = task
        self.backend = backend

    @abstractmethod
    def validate_input(self, assets: List[ImageAsset]) -> Tuple[bool, List[str]]:
        """
        Validates that the provided images meet the adapter's structural and modality constraints.
        Returns (is_valid, list_of_error_strings).
        """
        pass

    @abstractmethod
    def analyze(
        self, 
        query: str, 
        assets: List[ImageAsset], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes the vision-language analysis and returns a normalized result dictionary.
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Returns capability descriptor for registry lookup.
        """
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "backend": self.backend
        }
