"""
Unit tests for Phase 5A: Real Model Infrastructure & ModelManager.
"""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.manager import ModelManager

class DummyNeuralNet(nn.Module):
    def __init__(self, name: str = "dummy"):
        super().__init__()
        self.name = name
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

def test_model_manager_initialization():
    """Verify ModelManager initializes, detects device, and exposes telemetry."""
    manager = ModelManager(max_active_models=1)
    status = manager.get_system_status()

    assert "device" in status
    assert status["device"] in ["cpu", "cuda:0"]
    assert "cuda_available" in status
    assert status["max_active_models"] == 1
    assert isinstance(status["loaded_models"], list)
    assert isinstance(status["registered_models"], list)

def test_lazy_loading_and_registration():
    """Verify models are registered lazily and not loaded until invoked."""
    manager = ModelManager(max_active_models=2)
    instantiated = False

    def create_dummy():
        nonlocal instantiated
        instantiated = True
        return DummyNeuralNet("lazy_test")

    manager.register_factory("dummy_lazy", create_dummy, backend_type="real")

    # Upon registration, factory must NOT have run yet
    assert instantiated is False
    assert manager.is_loaded("dummy_lazy") is False

    # Calling get_model triggers factory execution
    model = manager.get_model("dummy_lazy")
    assert instantiated is True
    assert model is not None
    assert manager.is_loaded("dummy_lazy") is True
    assert model.name == "lazy_test"

def test_single_active_model_lru_eviction():
    """Verify strict LRU policy evicts oldest model to keep memory bounded."""
    manager = ModelManager(max_active_models=1)

    manager.register_factory("model_alpha", lambda: DummyNeuralNet("alpha"), backend_type="real")
    manager.register_factory("model_beta", lambda: DummyNeuralNet("beta"), backend_type="real")

    # Load Model Alpha
    alpha = manager.get_model("model_alpha")
    assert alpha is not None
    assert manager.is_loaded("model_alpha") is True
    assert manager.is_loaded("model_beta") is False

    # Load Model Beta -> Alpha must be evicted
    beta = manager.get_model("model_beta")
    assert beta is not None
    assert manager.is_loaded("model_beta") is True
    assert manager.is_loaded("model_alpha") is False
    assert len(manager.get_system_status()["loaded_models"]) <= 1

def test_graceful_model_load_failure():
    """Verify loading errors are captured cleanly without crashing the server."""
    manager = ModelManager(max_active_models=1)

    def failing_factory():
        raise FileNotFoundError("Simulated missing model weights checkpoint file")

    manager.register_factory("corrupted_model", failing_factory, backend_type="real")

    # Attempting to load must not throw an unhandled exception
    model = manager.get_model("corrupted_model")
    assert model is None
    assert manager.is_loaded("corrupted_model") is False

def test_backend_type_tagging():
    """Verify explicit distinction between real, adapted, and demo backends."""
    manager = ModelManager(max_active_models=2)

    manager.register_factory("real_model", lambda: DummyNeuralNet(), backend_type="real")
    manager.register_factory("adapted_model", lambda: DummyNeuralNet(), backend_type="adapted")

    assert manager.get_backend_type("real_model") == "real"
    assert manager.get_backend_type("adapted_model") == "adapted"
    assert manager.get_backend_type("unknown_nonexistent") == "demo"

def test_unload_all():
    """Verify unload_all clears all models and reclaims memory."""
    manager = ModelManager(max_active_models=2)
    manager.register_factory("m1", lambda: DummyNeuralNet("m1"))
    manager.register_factory("m2", lambda: DummyNeuralNet("m2"))

    manager.get_model("m1")
    manager.get_model("m2")
    assert len(manager.get_system_status()["loaded_models"]) == 2

    manager.unload_all()
    assert len(manager.get_system_status()["loaded_models"]) == 0
    assert manager.is_loaded("m1") is False
    assert manager.is_loaded("m2") is False
