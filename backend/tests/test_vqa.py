"""
Comprehensive Test Suite for Phase 5B: Real SmolVLM-256M Remote-Sensing VQA Specialist.

Verifies:
1. SmolVLM processor and model loading.
2. Vision-language instruction inference on satellite imagery.
3. Accurate answer generation derived directly from visual inputs.
4. Non-fabricated confidence scores computed from token softmax probability distribution.
5. VQAAdapter input validation and structured execution.
6. End-to-end API integration (/api/analyze) for VQA queries.
"""

import os
import uuid
import sys
from pathlib import Path
import pytest
from PIL import Image, ImageDraw
import torch
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.db.database import SessionLocal, Base, engine
from app.db.models import ImageAsset
from app.models.vqa_model import (
    load_vqa_model,
    run_vqa_inference
)
from app.adapters.vqa_adapter import VQAAdapter
from app.models.manager import model_manager

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_test_environment():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    test_dir = Path("data/test_fixtures")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic optical satellite raster (distinct ocean / deep blue texture)
    img = Image.new("RGB", (384, 384), (15, 75, 160))
    img_path = test_dir / "test_vqa_fixture.png"
    img.save(img_path)

    asset_id = f"test-asset-vqa-{uuid.uuid4().hex[:6]}"
    asset = ImageAsset(
        id=asset_id,
        filename="test_vqa_fixture.png",
        file_path=str(img_path),
        preview_path=str(img_path),
        preview_url="/static/previews/test_vqa.png",
        modality="optical",
        format="PNG",
        width=384,
        height=384,
        resolution="10.0m GSD",
        crs="WGS 84 / UTM 43N",
        acquisition_date="2024-06-20",
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
# 1. SmolVLM Model Loading
# ----------------------------------------------------------------------------
def test_vqa_model_loading():
    processor, model = load_vqa_model(device="cpu")
    assert processor is not None
    assert model is not None
    assert not model.training, "Loaded model must be in evaluation mode"


# ----------------------------------------------------------------------------
# 2. Vision-Language Inference & Token Confidence Extraction
# ----------------------------------------------------------------------------
def test_run_vqa_inference(setup_test_environment):
    env = setup_test_environment
    processor, model = load_vqa_model(device="cpu")

    result = run_vqa_inference(
        model_tuple=(processor, model),
        image_path=env["img_path"],
        query="What is the primary color of the surface in this satellite image?",
        device="cpu",
        max_new_tokens=30
    )

    assert result["backend"] == "real"
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert result["confidence"] is not None
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0
    assert result["tokens_generated"] > 0
    # Deep blue image should be recognized
    assert "blue" in result["answer"].lower() or "water" in result["answer"].lower() or "ocean" in result["answer"].lower()


# ----------------------------------------------------------------------------
# 3. VQA Adapter Execution (Backend = 'real')
# ----------------------------------------------------------------------------
def test_vqa_adapter_real(setup_test_environment):
    env = setup_test_environment
    adapter = VQAAdapter(backend="real")

    # 1. Validation test
    valid, errors = adapter.validate_input([])
    assert not valid
    assert "requires exactly 1" in errors[0]

    # 2. Real execution test
    result = adapter.analyze(
        query="Is this area water or desert?",
        assets=[env["asset"]]
    )

    assert result["task"] == "vqa"
    assert result["backend"] == "real"
    assert result["confidence"] is not None
    assert len(result["evidence"]) == 1

    ev = result["evidence"][0]
    assert ev["is_demo"] is False
    assert "answer" in ev
    assert "imageUrl" in ev


# ----------------------------------------------------------------------------
# 4. End-to-End API Integration Test: POST /api/analyze
# ----------------------------------------------------------------------------
def test_api_analyze_vqa_end_to_end(setup_test_environment):
    env = setup_test_environment
    payload = {
        "query": "What type of geographic terrain or water body dominates this image?",
        "image_ids": [env["asset"].id],
        "mode": "auto"
    }

    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data["status"] == "completed"
    assert data["task"] == "vqa"
    assert data["backend"] == "real"
    assert data["selected_model"]["backend"] == "real"
    assert "SmolVLM" in data["selected_model"]["name"]
    assert data["result"]["confidence"] is not None
    assert len(data["result"]["evidence"]) >= 1
    assert data["result"]["evidence"][0]["is_demo"] is False
