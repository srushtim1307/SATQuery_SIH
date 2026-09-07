"""
SatQuery AI — Real Model Manager & Device Infrastructure (Phase 5A).

Manages model lifecycle, lazy loading, memory-safe single-active model policy,
CPU/GPU device auto-detection, and graceful fallback handling.
"""

import os
import gc
import logging
from typing import Dict, Any, Optional, Callable, List
from collections import OrderedDict

import torch

logger = logging.getLogger("satquery.model_manager")

class ModelManager:
    """
    Central manager for deep learning models in SatQuery AI.
    Enforces lazy loading, device abstraction (CUDA / CPU fallback),
    and dynamic LRU caching to prevent memory exhaustion.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self, max_active_models: int = 1):
        self.max_active_models = max_active_models
        self.device = self._detect_device()
        self._loaded_models: OrderedDict[str, Any] = OrderedDict()
        self._model_factories: Dict[str, Callable[[], Any]] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}

        logger.info(f"[ModelManager] Initialized on device='{self.device}'. Max active models={self.max_active_models}")

    @classmethod
    def get_instance(cls, max_active_models: int = 1) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls(max_active_models=max_active_models)
        return cls._instance

    def _detect_device(self) -> str:
        """
        Detects optimal execution device with explicit environment override support.
        """
        env_device = os.environ.get("SATQUERY_DEVICE", "").strip().lower()
        if env_device in ["cuda", "gpu"]:
            if torch.cuda.is_available():
                return "cuda:0"
            logger.warning("[ModelManager] SATQUERY_DEVICE='cuda' requested but CUDA is not available. Falling back to CPU.")
            return "cpu"
        elif env_device == "cpu":
            return "cpu"

        # Auto-detect
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                logger.info(f"[ModelManager] Detected GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")
                return "cuda:0"
            except Exception as e:
                logger.warning(f"[ModelManager] CUDA detected but initialization failed: {e}. Falling back to CPU.")
                return "cpu"

        logger.info("[ModelManager] Operating in CPU mode.")
        return "cpu"

    def register_factory(
        self, 
        model_id: str, 
        factory: Callable[[], Any],
        backend_type: str = "real",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registers a factory function for lazy model instantiation.
        The model is NOT loaded into memory until explicitly requested.
        """
        self._model_factories[model_id] = factory
        self._model_metadata[model_id] = {
            "backend_type": backend_type,
            "metadata": metadata or {},
            "status": "registered"
        }
        logger.debug(f"[ModelManager] Registered factory for model '{model_id}' ({backend_type})")

    def get_model(self, model_id: str) -> Optional[Any]:
        """
        Retrieves an active model instance, loading it lazily if necessary.
        Enforces LRU memory limits by offloading older models before loading new ones.
        Returns None if loading fails gracefully.
        """
        # If already in memory, mark as most recently used and return
        if model_id in self._loaded_models:
            self._loaded_models.move_to_end(model_id)
            return self._loaded_models[model_id]

        # Check if factory exists
        if model_id not in self._model_factories:
            logger.warning(f"[ModelManager] No factory registered for model '{model_id}'")
            return None

        # Check memory limit and evict LRU model if needed
        while len(self._loaded_models) >= self.max_active_models:
            oldest_id, _ = self._loaded_models.popitem(last=False)
            self._evict_model(oldest_id)

        # Lazy load via factory
        try:
            logger.info(f"[ModelManager] Lazily instantiating model '{model_id}' on device '{self.device}'...")
            factory = self._model_factories[model_id]
            model_instance = factory()
            
            # Place in loaded cache
            self._loaded_models[model_id] = model_instance
            if model_id in self._model_metadata:
                self._model_metadata[model_id]["status"] = "loaded"

            logger.info(f"[ModelManager] Model '{model_id}' successfully loaded and ready.")
            return model_instance
        except Exception as e:
            logger.error(f"[ModelManager] Graceful failure loading model '{model_id}': {e}", exc_info=True)
            if model_id in self._model_metadata:
                self._model_metadata[model_id]["status"] = "failed"
                self._model_metadata[model_id]["error"] = str(e)
            return None

    def _evict_model(self, model_id: str) -> None:
        """
        Safely offloads a model from memory and reclaims VRAM / RAM.
        """
        logger.info(f"[ModelManager] Offloading model '{model_id}' to preserve memory limit...")
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
            
        if model_id in self._model_metadata:
            self._model_metadata[model_id]["status"] = "registered"

        # Explicit garbage collection and CUDA cache flush
        gc.collect()
        if "cuda" in self.device and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def unload_all(self) -> None:
        """
        Clears all loaded models from memory and triggers cleanup.
        """
        loaded_keys = list(self._loaded_models.keys())
        for model_id in loaded_keys:
            self._evict_model(model_id)
        self._loaded_models.clear()
        gc.collect()
        if "cuda" in self.device and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[ModelManager] All models unloaded from memory.")

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded_models

    def get_backend_type(self, model_id: str) -> str:
        """
        Returns explicit backend classification: 'real', 'adapted', or 'demo'.
        """
        if model_id in self._model_metadata:
            return self._model_metadata[model_id].get("backend_type", "demo")
        return "demo"

    def get_system_status(self) -> Dict[str, Any]:
        """
        Returns current device, VRAM, and loaded model status for diagnostic observability.
        """
        cuda_avail = torch.cuda.is_available()
        gpu_name = None
        vram_info = None

        if cuda_avail and "cuda" in self.device:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                tot = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
                vram_info = {
                    "total_mb": round(tot, 1),
                    "allocated_mb": round(alloc, 1),
                    "free_mb": round(tot - alloc, 1)
                }
            except Exception:
                pass

        return {
            "device": self.device,
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
            "vram": vram_info,
            "loaded_models": list(self._loaded_models.keys()),
            "registered_models": list(self._model_factories.keys()),
            "max_active_models": self.max_active_models
        }

# Global singleton instance
model_manager = ModelManager.get_instance()
