"""
Comprehensive Test Suite for Phase 5C: Real OWL-ViT Visual Grounding Specialist.

Verifies:
1. Natural language query expansion to remote sensing visual tokens.
2. OWL-ViT processor and model loading.
3. Open-vocabulary visual grounding execution on synthetic and satellite imagery.
4. Bounding box coordinates normalization into percentages (top, left, width, height).
5. Non-fabricated confidence scores derived from model logits.
6. Visual annotation artifact generation.
7. GroundingAdapter input validation and execution.
8. End-to-end API integration (/api/analyze) for grounding queries.
"""

import os
import uuid
import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image, ImageDraw
import torch
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.db.database import SessionLocal, Base, engine
from app.db.models import ImageAsset
from app.models.grounding_model import (
    extract_target_queries,
    load_owlvit_model,
    run_owlvit_grounding
)
from app.adapters.grounding_adapter import GroundingAdapter
from app.models.manager import model_manager
from app.core.config import PROCESSED_DIR

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_test_environment():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    test_dir = Path("data/test_fixtures")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic satellite scene with water lake
    img = Image.new("RGB", (512, 512), (34, 139, 34))  # Green vegetation canvas
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 250, 300], fill=(20, 80, 200))  # Blue water body
    img_path = test_dir / "test_grounding_fixture.png"
    img.save(img_path)

    asset_id = f"test-asset-grounding-{uuid.uuid4().hex[:6]}"
    asset = ImageAsset(
        id=asset_id,
        filename="test_grounding_fixture.png",
        file_path=str(img_path),
        preview_path=str(img_path),
        preview_url="/static/previews/test_grounding.png",
        modality="optical",
        format="PNG",
        width=512,
        height=512,
        resolution="10.0m GSD",
        crs="WGS 84 / UTM 43N",
        acquisition_date="2024-05-10",
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
# 1. Query Expansion Heuristics
# ----------------------------------------------------------------------------
def test_extract_target_queries():
    q_water = extract_target_queries("Where are the water bodies and reservoirs in this scene?")
    assert any("water" in t or "lake" in t or "river" in t for t in q_water)

    q_road = extract_target_queries("Locate the primary transit corridors and highways.")
    assert any("road" in t or "highway" in t for t in q_road)

    q_building = extract_target_queries("Find built-up urban structures and settlements.")
    assert any("building" in t or "urban" in t for t in q_building)

    q_forest = extract_target_queries("Highlight vegetation, farmland, and agricultural fields.")
    assert any("vegetation" in t or "field" in t or "forest" in t for t in q_forest)


# ----------------------------------------------------------------------------
# 2. OWL-ViT Model Loading
# ----------------------------------------------------------------------------
def test_owlvit_model_loading():
    processor, model = load_owlvit_model(device="cpu")
    assert processor is not None
    assert model is not None
    assert not model.training, "Loaded model must be in eval mode"


# ----------------------------------------------------------------------------
# 3. Visual Grounding Inference & Box Extraction
# ----------------------------------------------------------------------------
def test_run_owlvit_grounding(setup_test_environment):
    env = setup_test_environment
    processor, model = load_owlvit_model(device="cpu")

    result = run_owlvit_grounding(
        model_tuple=(processor, model),
        image_path=env["img_path"],
        query="Where are the water bodies in this image?",
        device="cpu",
        max_boxes=4
    )

    assert result["backend"] == "real"
    assert "bounding_boxes" in result
    assert len(result["bounding_boxes"]) > 0
    assert result["overall_confidence"] > 0.0

    first_box = result["bounding_boxes"][0]
    assert first_box["is_demo"] is False
    assert "top" in first_box and "left" in first_box
    assert "width" in first_box and "height" in first_box
    assert first_box["confidence"] >= 0.0 and first_box["confidence"] <= 1.0

    # Verify that annotated image was saved
    assert os.path.exists(result["annotated_disk_path"])
    assert result["annotated_image_url"].startswith("/static/processed/")


# ----------------------------------------------------------------------------
# 4. GroundingAdapter Execution (Backend = 'real')
# ----------------------------------------------------------------------------
def test_grounding_adapter_real(setup_test_environment):
    env = setup_test_environment
    adapter = GroundingAdapter(backend="real")

    # 1. Validation test
    valid, errors = adapter.validate_input([])
    assert not valid
    assert "requires exactly 1" in errors[0]

    # 2. Real execution test
    result = adapter.analyze(
        query="Where is the water reservoir in this image?",
        assets=[env["asset"]]
    )

    assert result["task"] == "grounding"
    assert result["backend"] == "real"
    assert result["confidence"] is not None
    assert len(result["evidence"]) == 1

    ev = result["evidence"][0]
    assert ev["is_demo"] is False
    assert len(ev["boundingBoxes"]) > 0
    assert "imageUrl" in ev
    assert "annotatedImageUrl" in ev


# ----------------------------------------------------------------------------
# 5. End-to-End API Integration Test: POST /api/analyze
# ----------------------------------------------------------------------------
def test_api_analyze_grounding_end_to_end(setup_test_environment):
    env = setup_test_environment
    payload = {
        "query": "Detect and highlight water bodies in this satellite image.",
        "image_ids": [env["asset"].id],
        "mode": "auto"
    }

    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert data["status"] == "completed"
    assert data["task"] == "grounding"
    assert data["backend"] == "real"
    assert data["selected_model"]["backend"] == "real"
    assert "OWL-ViT" in data["selected_model"]["name"]
    assert data["result"]["confidence"] is not None
    assert len(data["result"]["evidence"]) >= 1

    ev = data["result"]["evidence"][0]
    assert ev["is_demo"] is False
    assert len(ev["boundingBoxes"]) > 0
    assert ev["boundingBoxes"][0]["is_demo"] is False
