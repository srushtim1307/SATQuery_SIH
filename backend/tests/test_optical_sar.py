"""
Comprehensive Test Suite for Phase 5E: Optical + SAR Cross-Modal Fusion Specialist.

Verifies:
1. Neural network architecture (CrossModalOpticalSARNet) forward pass and output dimensions.
2. Calibrated weights checkpoint and training log persistence.
3. Radar backscatter (sigma0 dB), optical GLI, and Sobel gradient edge correlation.
4. OpticalSARAdapter input validation and real execution.
5. End-to-end API integration (/api/analyze) with dual-sensor optical + SAR assets.
"""

import os
import json
import uuid
import sys
from pathlib import Path
import pytest
from PIL import Image
import numpy as np
import torch
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.db.database import SessionLocal, Base, engine
from app.db.models import ImageAsset
from app.models.optical_sar_model import (
    CrossModalOpticalSARNet,
    load_optical_sar_model,
    run_optical_sar_fusion,
    calculate_sar_backscatter_db,
    calculate_optical_vegetation_index,
    calculate_structural_edge_correlation,
    WEIGHTS_PATH
)
from app.adapters.optical_sar_adapter import OpticalSARAdapter
from app.models.manager import model_manager

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_test_environment():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    test_dir = Path("data/test_fixtures")
    test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create synthetic Optical raster (vegetation + estuary)
    opt_img = Image.new("RGB", (256, 256), (34, 139, 34))
    opt_path = test_dir / "test_fixture_optical.png"
    opt_img.save(opt_path)

    opt_asset_id = f"test-asset-opt-{uuid.uuid4().hex[:6]}"
    opt_asset = ImageAsset(
        id=opt_asset_id,
        filename="test_fixture_optical.png",
        file_path=str(opt_path),
        preview_path=str(opt_path),
        preview_url="/static/previews/test_opt.png",
        modality="optical",
        format="PNG",
        width=256,
        height=256,
        resolution="10.0m GSD",
        crs="WGS 84 / UTM 43N",
        acquisition_date="2024-07-01",
        status="ready"
    )
    db.add(opt_asset)

    # 2. Create synthetic SAR raster (C-band microwave radar backscatter)
    sar_img = Image.new("RGB", (256, 256), (90, 90, 90))
    sar_path = test_dir / "test_fixture_sar.png"
    sar_img.save(sar_path)

    sar_asset_id = f"test-asset-sar-{uuid.uuid4().hex[:6]}"
    sar_asset = ImageAsset(
        id=sar_asset_id,
        filename="test_fixture_sar.png",
        file_path=str(sar_path),
        preview_path=str(sar_path),
        preview_url="/static/previews/test_sar.png",
        modality="sar",
        format="PNG",
        width=256,
        height=256,
        resolution="10.0m GSD",
        crs="WGS 84 / UTM 43N",
        acquisition_date="2024-07-01",
        status="ready"
    )
    db.add(sar_asset)
    db.commit()

    yield {
        "opt_asset_id": opt_asset_id,
        "sar_asset_id": sar_asset_id,
        "opt_path": str(opt_path),
        "sar_path": str(sar_path),
        "db": db
    }

    # Cleanup DB records
    db.query(ImageAsset).filter(ImageAsset.id.in_([opt_asset_id, sar_asset_id])).delete(synchronize_session=False)
    db.commit()
    db.close()


def test_optical_sar_architecture():
    """Verifies CrossModalOpticalSARNet forward pass and tensor dimensions."""
    model = CrossModalOpticalSARNet(feature_dim=64)
    model.eval()

    b, c, h, w = 2, 3, 128, 128
    x_opt = torch.rand(b, c, h, w)
    x_sar = torch.rand(b, c, h, w)

    with torch.no_grad():
        outputs = model(x_opt, x_sar)

    assert "fused_composite" in outputs
    assert "synergy_map" in outputs
    assert "coherence" in outputs
    assert outputs["fused_composite"].shape == (b, 3, h, w)
    assert outputs["synergy_map"].shape == (b, 1, h, w)
    assert outputs["coherence"].ndim == 0 or outputs["coherence"].numel() == 1


def test_optical_sar_weights_and_log():
    """Verifies that calibrated weights and training log exist and load cleanly."""
    assert WEIGHTS_PATH.exists(), f"Weights file missing at {WEIGHTS_PATH}"
    assert WEIGHTS_PATH.stat().st_size > 500_000, "Weights file unexpectedly small"

    log_path = WEIGHTS_PATH.parent / "optical_sar_training_log.json"
    assert log_path.exists(), f"Training log missing at {log_path}"

    with open(log_path, "r") as f:
        log_data = json.load(f)

    assert log_data["model"] == "CrossModalOpticalSARNet"
    assert len(log_data["history"]) >= 5
    assert log_data["history"][-1]["reconstruction_loss"] < 0.05

    # Test loading
    model = load_optical_sar_model(device="cpu")
    assert isinstance(model, CrossModalOpticalSARNet)


def test_physical_metrics_calculation(setup_test_environment):
    """Verifies calculation of radar backscatter dB, optical GLI, and Sobel edge correlation."""
    env = setup_test_environment
    opt_np = np.array(Image.open(env["opt_path"]))
    sar_gray = np.array(Image.open(env["sar_path"]).convert("L"))
    opt_gray = np.array(Image.open(env["opt_path"]).convert("L"))

    # SAR backscatter
    sar_metrics = calculate_sar_backscatter_db(sar_gray)
    assert "mean_db" in sar_metrics
    assert sar_metrics["min_db"] <= sar_metrics["mean_db"] <= sar_metrics["max_db"]
    assert 0.0 <= sar_metrics["specular_water_fraction"] <= 1.0

    # Optical GLI
    opt_metrics = calculate_optical_vegetation_index(opt_np)
    assert "mean_gli" in opt_metrics
    assert -1.0 <= opt_metrics["mean_gli"] <= 1.0

    # Structural Edge Correlation
    edge_corr = calculate_structural_edge_correlation(opt_gray, sar_gray)
    assert -1.0 <= edge_corr <= 1.0


def test_optical_sar_adapter_validation(setup_test_environment):
    """Verifies OpticalSARAdapter input validation rules."""
    env = setup_test_environment
    db = env["db"]
    adapter = OpticalSARAdapter(backend="real")

    opt_asset = db.query(ImageAsset).filter_by(id=env["opt_asset_id"]).first()
    sar_asset = db.query(ImageAsset).filter_by(id=env["sar_asset_id"]).first()

    # 1. Single image rejection
    valid, errors = adapter.validate_input([opt_asset])
    assert not valid
    assert "requires exactly 2 images" in errors[0]

    # 2. Missing SAR rejection (2 optical images)
    valid, errors = adapter.validate_input([opt_asset, opt_asset])
    assert not valid
    assert any("Missing SAR" in e for e in errors)

    # 3. Valid optical + SAR pair
    valid, errors = adapter.validate_input([opt_asset, sar_asset])
    assert valid
    assert len(errors) == 0


def test_optical_sar_adapter_real_execution(setup_test_environment):
    """Verifies OpticalSARAdapter execution returning real non-fabricated confidence and artifacts."""
    env = setup_test_environment
    db = env["db"]
    adapter = OpticalSARAdapter(backend="real")

    opt_asset = db.query(ImageAsset).filter_by(id=env["opt_asset_id"]).first()
    sar_asset = db.query(ImageAsset).filter_by(id=env["sar_asset_id"]).first()

    result = adapter.analyze("Compare optical and SAR radar characteristics", [opt_asset, sar_asset])

    assert result["task"] == "optical_sar"
    assert result["backend"] == "real"
    assert result["confidence"] is not None
    assert 0.05 <= result["confidence"] <= 0.99
    assert len(result["evidence"]) == 1

    ev = result["evidence"][0]
    assert ev["type"] == "sensor_cross_correlation"
    assert "fusedImageUrl" in ev
    assert "synergyMapUrl" in ev
    assert "sarLabel" in ev
    assert "opticalLabel" in ev
    assert ev["is_demo"] is False


def test_api_analyze_optical_sar_end_to_end(setup_test_environment):
    """Verifies POST /api/analyze routing and execution for Optical + SAR fusion."""
    env = setup_test_environment

    payload = {
        "query": "Perform optical and SAR cross-modal fusion to analyze ground features and radar backscatter",
        "image_ids": [env["opt_asset_id"], env["sar_asset_id"]],
        "mode": "auto"
    }

    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200, f"Analysis failed: {response.text}"

    data = response.json()
    assert data["status"] == "completed"
    assert data["task"] == "optical_sar"
    assert data["backend"] == "real"
    assert data["selected_model"]["id"] == "rs-optical-sar-fusion"
    assert data["selected_model"]["backend"] == "real"
    assert data["result"]["confidence"] is not None

    evidence = data["result"]["evidence"][0]
    assert evidence["type"] == "sensor_cross_correlation"
    assert "/static/processed/" in evidence["fusedImageUrl"]
    assert "/static/processed/" in evidence["synergyMapUrl"]
    assert "fusionSummary" in evidence
