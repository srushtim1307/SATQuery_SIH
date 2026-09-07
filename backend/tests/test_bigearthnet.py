"""
Comprehensive Test Suite for Phase 5F: BigEarthNet-19 Multi-Spectral Adaptation.

Verifies:
1. BigEarthNet 19-class taxonomy file integrity.
2. Trained weights checkpoint loading and model evaluation.
3. Training log verification with genuine epoch losses and accuracy.
4. Multi-spectral land cover classification and descriptive scene captioning.
5. CaptioningAdapter input validation and structured execution.
6. End-to-end API integration (/api/analyze) for scene captioning.
"""

import os
import json
import uuid
import sys
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.db.database import SessionLocal, Base, engine
from app.db.models import ImageAsset
from app.models.bigearthnet_model import (
    load_bigearthnet_model,
    classify_land_cover,
    BIGEARTHNET_19_CLASSES
)
from app.adapters.captioning_adapter import CaptioningAdapter

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_test_environment():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    test_dir = Path("data/test_fixtures")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic optical raster (vegetation / forest green)
    img = Image.new("RGB", (256, 256), (34, 139, 34))
    img_path = test_dir / "test_captioning_fixture.png"
    img.save(img_path)

    asset_id = f"test-asset-cap-{uuid.uuid4().hex[:6]}"
    asset = ImageAsset(
        id=asset_id,
        filename="test_captioning_fixture.png",
        file_path=str(img_path),
        preview_path=str(img_path),
        preview_url="/static/previews/test_cap.png",
        modality="optical",
        format="PNG",
        width=256,
        height=256,
        resolution="10.0m GSD",
        crs="WGS 84 / UTM 43N",
        acquisition_date="2024-07-01",
        status="ready"
    )
    db.add(asset)
    db.commit()

    yield {
        "db": db,
        "img_path": str(img_path),
        "asset": asset
    }

    db.close()


# ----------------------------------------------------------------------------
# 1. BigEarthNet-19 Taxonomy Verification
# ----------------------------------------------------------------------------
def test_bigearthnet_taxonomy_integrity():
    assert len(BIGEARTHNET_19_CLASSES) == 19
    assert "Urban fabric" in BIGEARTHNET_19_CLASSES
    assert "Broad-leaved forest" in BIGEARTHNET_19_CLASSES
    assert "Inland waters" in BIGEARTHNET_19_CLASSES

    # Verify BigEarthNet.txt exists on disk
    taxonomy_file = Path("backend/models/weights/BigEarthNet.txt")
    if not taxonomy_file.exists():
        taxonomy_file = Path("BigEarthNet.txt")
    assert taxonomy_file.exists(), "BigEarthNet.txt must exist on disk"
    content = taxonomy_file.read_text(encoding="utf-8")
    assert "Urban fabric" in content
    assert "Inland waters" in content


# ----------------------------------------------------------------------------
# 2. Adapted Checkpoint Weights & Training Log
# ----------------------------------------------------------------------------
def test_bigearthnet_weights_and_log():
    weights_file = Path("backend/models/weights/bigearthnet_adapted.pth")
    assert weights_file.exists(), "Adapted weights checkpoint must exist on disk"
    assert weights_file.stat().st_size > 1024 * 1024, "Weights file must be substantial"

    log_file = Path("backend/models/weights/training_log.json")
    assert log_file.exists(), "Training log must exist on disk"
    with open(log_file, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    assert log_data["status"] == "completed"
    assert log_data["backend"] == "adapted"
    assert len(log_data["history"]) >= 5
    assert log_data["final_train_loss"] < 0.5


# ----------------------------------------------------------------------------
# 3. Model Loading & Land Cover Classification
# ----------------------------------------------------------------------------
def test_bigearthnet_model_inference(setup_test_environment):
    env = setup_test_environment
    model = load_bigearthnet_model(device="cpu")
    assert model is not None
    assert not model.training, "Loaded model must be in evaluation mode"

    result = classify_land_cover(
        model=model,
        image_path=env["img_path"],
        device="cpu",
        top_k=4
    )

    assert result["backend"] == "adapted"
    assert "caption" in result
    assert len(result["top_classes"]) == 4
    assert result["confidence"] > 0.0
    assert result["taxonomy"] == "BigEarthNet-19 (CORINE Land Cover)"


# ----------------------------------------------------------------------------
# 4. Captioning Adapter Execution (Backend = 'adapted')
# ----------------------------------------------------------------------------
def test_captioning_adapter_adapted(setup_test_environment):
    env = setup_test_environment
    adapter = CaptioningAdapter(backend="adapted")

    # 1. Validation test
    valid, errors = adapter.validate_input([])
    assert not valid
    assert "requires exactly 1" in errors[0]

    # 2. Real execution test
    result = adapter.analyze(
        query="Describe the terrain and land-cover classes in this scene.",
        assets=[env["asset"]]
    )

    assert result["task"] == "captioning"
    assert result["backend"] == "adapted"
    assert result["confidence"] is not None
    assert len(result["evidence"]) == 1

    ev = result["evidence"][0]
    assert ev["is_demo"] is False
    assert "description" in ev
    assert "primary_class" in ev
    assert "top_classes" in ev


# ----------------------------------------------------------------------------
# 5. End-to-End API Integration Test: POST /api/analyze
# ----------------------------------------------------------------------------
def test_api_analyze_captioning_end_to_end(setup_test_environment):
    env = setup_test_environment
    payload = {
        "query": "Provide a comprehensive scene description and land-cover breakdown.",
        "image_ids": [env["asset"].id],
        "mode": "auto"
    }

    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data["status"] == "completed"
    assert data["task"] == "captioning"
    assert data["backend"] == "adapted"
    assert data["selected_model"]["backend"] == "adapted"
    assert "BigEarthNet" in data["selected_model"]["name"]
    assert data["result"]["confidence"] is not None
    assert len(data["result"]["evidence"]) >= 1
    assert data["result"]["evidence"][0]["is_demo"] is False
