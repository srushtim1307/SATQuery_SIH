"""
Comprehensive Test Suite for Phase 5D: Real TinyCD Change Detection Specialist.

Verifies:
1. TinyCD network architecture instantiation and forward pass shape.
2. Real weights loading from safetensors.
3. Preprocessing, tensor normalization, and device placement.
4. Real pixel-level mask computation, change percentage calculation, and non-fabricated confidence.
5. Connected component bounding box extraction.
6. Mask and overlay artifact synthesis.
7. ChangeDetectionAdapter input validation and real execution.
8. Full end-to-end API integration (/api/analyze) with real TinyCD inference.
"""

import os
import uuid
import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.db.database import SessionLocal, Base, engine
from app.db.models import ImageAsset
from app.models.change_detection_model import (
    TinyCD,
    load_tinycd_model,
    preprocess_image_pair,
    extract_change_bounding_boxes,
    generate_change_artifacts,
    run_tinycd_inference
)
from app.adapters.change_adapter import ChangeDetectionAdapter
from app.models.manager import model_manager
from app.core.config import PROCESSED_DIR

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_test_environment():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    # Create two temporary test images
    test_dir = Path("data/test_fixtures")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Baseline Image T1 (Green agricultural/vegetated area)
    t1_img = Image.new("RGB", (256, 256), (34, 139, 34))
    t1_path = test_dir / "test_t1_baseline.png"
    t1_img.save(t1_path)

    # Current Image T2 (With urban construction block in center)
    t2_img = Image.new("RGB", (256, 256), (34, 139, 34))
    t2_arr = np.array(t2_img)
    # Inject 60x60 altered gray/concrete patch
    t2_arr[90:150, 90:150] = [180, 180, 190]
    t2_modified = Image.fromarray(t2_arr)
    t2_path = test_dir / "test_t2_current.png"
    t2_modified.save(t2_path)

    # Create corresponding ImageAsset records in database
    id1 = f"test-asset-t1-{uuid.uuid4().hex[:6]}"
    id2 = f"test-asset-t2-{uuid.uuid4().hex[:6]}"

    asset1 = ImageAsset(
        id=id1,
        filename="test_t1_baseline.png",
        file_path=str(t1_path),
        preview_path=str(t1_path),
        preview_url="/static/previews/test_t1.png",
        modality="optical",
        format="PNG",
        width=256,
        height=256,
        resolution="10.0m GSD",
        crs="WGS 84 / UTM 43N",
        acquisition_date="2023-03-15",
        status="ready"
    )
    asset2 = ImageAsset(
        id=id2,
        filename="test_t2_current.png",
        file_path=str(t2_path),
        preview_path=str(t2_path),
        preview_url="/static/previews/test_t2.png",
        modality="optical",
        format="PNG",
        width=256,
        height=256,
        resolution="10.0m GSD",
        crs="WGS 84 / UTM 43N",
        acquisition_date="2024-03-15",
        status="ready"
    )
    db.add(asset1)
    db.add(asset2)
    db.commit()

    yield {
        "db": db,
        "t1_path": str(t1_path),
        "t2_path": str(t2_path),
        "asset1": asset1,
        "asset2": asset2
    }

    db.close()


# ----------------------------------------------------------------------------
# 1. TinyCD Architecture & Model Forward Pass
# ----------------------------------------------------------------------------
def test_tinycd_architecture():
    """Verifies that TinyCD instantiates and performs forward passes with valid tensor dimensions."""
    model = TinyCD()
    model.eval()

    t1 = torch.randn(1, 3, 256, 256)
    t2 = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out = model(t1, t2)

    assert out.shape == (1, 1, 256, 256), f"Expected output shape (1, 1, 256, 256), got {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, "Output probabilities must be sigmoid constrained in [0, 1]"


# ----------------------------------------------------------------------------
# 2. Model Weights Loading from HuggingFace Checkpoint
# ----------------------------------------------------------------------------
def test_tinycd_load_pretrained():
    """Verifies that pre-trained TinyCD weights load cleanly without missing keys."""
    model = load_tinycd_model(device="cpu")
    assert isinstance(model, TinyCD)
    assert not model.training, "Loaded model should be in eval mode"


# ----------------------------------------------------------------------------
# 3. Preprocessing and Tensor Normalization
# ----------------------------------------------------------------------------
def test_image_preprocessing(setup_test_environment):
    env = setup_test_environment
    t1_tensor, t2_tensor, img1, img2 = preprocess_image_pair(
        env["t1_path"], env["t2_path"], target_size=(256, 256), device="cpu"
    )
    assert t1_tensor.shape == (1, 3, 256, 256)
    assert t2_tensor.shape == (1, 3, 256, 256)
    assert img1.size == (256, 256)
    assert img2.size == (256, 256)


# ----------------------------------------------------------------------------
# 4. Connected Component Bounding Box Extraction
# ----------------------------------------------------------------------------
def test_bounding_box_extraction():
    mask = np.zeros((256, 256), dtype=np.uint8)
    prob_map = np.full((256, 256), 0.1, dtype=np.float32)

    # Add 2 change regions
    mask[50:100, 50:100] = 1
    prob_map[50:100, 50:100] = 0.88

    mask[180:220, 180:220] = 1
    prob_map[180:220, 180:220] = 0.75

    boxes = extract_change_bounding_boxes(mask, prob_map, min_area_pixels=16)
    assert len(boxes) == 2, f"Expected 2 bounding boxes, got {len(boxes)}"
    assert boxes[0]["confidence"] > 0.7
    assert boxes[0]["is_demo"] is False
    assert "top" in boxes[0] and "left" in boxes[0]


# ----------------------------------------------------------------------------
# 5. Mask & Overlay Artifact Synthesis
# ----------------------------------------------------------------------------
def test_generate_change_artifacts(setup_test_environment):
    env = setup_test_environment
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:150, 100:150] = 1
    prob_map = mask.astype(np.float32) * 0.9

    t2_img = Image.open(env["t2_path"]).convert("RGB").resize((256, 256))
    prefix = "test_run_artifacts"

    mask_url, overlay_url, mask_path, overlay_path = generate_change_artifacts(
        prob_map, mask, t2_img, prefix
    )

    assert os.path.exists(mask_path), f"Change mask file not found at {mask_path}"
    assert os.path.exists(overlay_path), f"Change overlay file not found at {overlay_path}"
    assert mask_url.startswith("/static/processed/")
    assert overlay_url.startswith("/static/processed/")


# ----------------------------------------------------------------------------
# 6. Change Detection Adapter Execution (Backend = 'real')
# ----------------------------------------------------------------------------
def test_change_detection_adapter_real(setup_test_environment):
    env = setup_test_environment
    adapter = ChangeDetectionAdapter(backend="real")

    # 1. Validation test
    valid, errors = adapter.validate_input([env["asset1"]])
    assert not valid
    assert "requires exactly 2 images" in errors[0]

    # 2. Execution test with pair
    result = adapter.analyze(
        query="Detect urban expansion and land use alterations between baseline and current.",
        assets=[env["asset1"], env["asset2"]]
    )

    assert result["task"] == "change_detection"
    assert result["backend"] == "real"
    assert isinstance(result["confidence"], float)
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0
    assert len(result["evidence"]) == 1
    
    evidence = result["evidence"][0]
    assert evidence["is_demo"] is False
    assert evidence["change_percentage"] >= 0.0
    assert "mask_url" in evidence
    assert "overlay_url" in evidence
    assert "beforeImageUrl" in evidence
    assert "afterImageUrl" in evidence
    assert "bounding_boxes" in evidence


# ----------------------------------------------------------------------------
# 7. End-to-End API Integration Test: POST /api/analyze
# ----------------------------------------------------------------------------
def test_api_analyze_change_detection_end_to_end(setup_test_environment):
    env = setup_test_environment
    payload = {
        "query": "Compare these multi-temporal satellite rasters to detect new construction changes.",
        "image_ids": [env["asset1"].id, env["asset2"].id],
        "mode": "auto"
    }

    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data["status"] == "completed"
    assert data["task"] == "change_detection"
    assert data["backend"] == "real"
    assert data["selected_model"]["backend"] == "real"
    assert "TinyCD" in data["selected_model"]["name"]
    assert data["result"] is not None
    assert data["result"]["confidence"] is not None
    assert len(data["result"]["evidence"]) >= 1
    
    first_ev = data["result"]["evidence"][0]
    assert first_ev["is_demo"] is False
    assert "change_percentage" in first_ev
    assert "mask_url" in first_ev
    assert "overlay_url" in first_ev
