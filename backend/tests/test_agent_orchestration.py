"""
Comprehensive test suite for Phase 4: Agentic Task Classification, Model Registry,
Adapter Architecture, and Orchestration.
"""

import io
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app

from app.db.database import SessionLocal, Base, engine
from app.db.models import ImageAsset, AnalysisRecord
from app.models.registry import registry
from app.services.classifier import TaskClassifier
from app.services.agent import SatQueryAgent
from app.schemas.analysis import AnalysisRequest
from app.core.constants import (
    TASK_VQA,
    TASK_GROUNDING,
    TASK_CHANGE_DETECTION,
    TASK_OPTICAL_SAR,
    TASK_CAPTIONING,
    TRACE_STEPS
)

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    yield db
    db.close()

def create_test_image_asset(db, asset_id: str, name: str, modality: str = "optical") -> ImageAsset:
    asset = db.query(ImageAsset).filter(ImageAsset.id == asset_id).first()
    if not asset:
        asset = ImageAsset(
            id=asset_id,
            filename=name,
            file_path=f"data/uploads/{asset_id}_{name}",
            preview_url=f"/static/processed/{asset_id}_preview.png",
            modality=modality,
            format="GeoTIFF" if name.endswith(".tif") else "PNG",
            width=512,
            height=512,
            bands=4 if modality == "optical" else 1,
            resolution="10.0m Ground Resolution",
            crs="WGS 84 / UTM zone 43N",
            status="ready"
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)
    return asset

# ----------------------------------------------------
# 1. Model Registry Tests
# ----------------------------------------------------
def test_model_registry_operations():
    """Verify listing, getting, and querying capable models from the registry."""
    models = registry.list()
    assert len(models) >= 5
    task_ids = {m["task"] for m in models}
    assert {TASK_VQA, TASK_GROUNDING, TASK_CHANGE_DETECTION, TASK_OPTICAL_SAR, TASK_CAPTIONING}.issubset(task_ids)

    # Test get
    vqa_model = registry.get("rs-vqa-base")
    assert vqa_model is not None
    assert vqa_model.task == "vqa"
    assert vqa_model.backend_type in ["demo", "real"]

    # Test find_capable_model
    grounding = registry.find_capable_model("grounding")
    assert grounding is not None
    assert grounding.task == "grounding"

    change_model = registry.find_capable_model("change_detection", input_type="bi_temporal_pair")
    assert change_model is not None
    assert change_model.task == "change_detection"

# ----------------------------------------------------
# 2. Adapter Execution Tests
# ----------------------------------------------------
def test_adapter_demo_execution(setup_database):
    """Verify adapters return structured outputs with backend='demo' and confidence=None."""
    db = setup_database
    img1 = create_test_image_asset(db, "test-adapter-img-1", "sentinel2_estuary.tif", "optical")
    img2 = create_test_image_asset(db, "test-adapter-img-2", "sentinel1_radar.tif", "sar")

    # VQA
    vqa_model = registry.get("rs-vqa-base")
    res_vqa = vqa_model.adapter.analyze("Are there buildings in this area?", [img1])
    assert res_vqa["backend"] == "demo"
    assert res_vqa["confidence"] is None  # Must not invent real confidence
    assert "[Demo Execution]" in res_vqa["answer"]

    # Grounding
    grounding_model = registry.get("rs-grounding-focal")
    res_ground = grounding_model.adapter.analyze("Where are the water bodies?", [img1])
    assert res_ground["backend"] == "demo"
    assert len(res_ground["evidence"][0]["bounding_boxes"]) > 0
    assert res_ground["evidence"][0]["bounding_boxes"][0]["is_demo"] is True

    # Change Detection
    change_model = registry.get("rs-change-diff")
    res_change = change_model.adapter.analyze("What changed between these two dates?", [img1, img1])
    assert res_change["backend"] == "demo"
    assert "Bi-temporal analysis" in res_change["answer"]

    # Optical + SAR Fusion
    fusion_model = registry.get("rs-optical-sar-fusion")
    res_fusion = fusion_model.adapter.analyze("Compare optical and SAR data.", [img1, img2])
    assert res_fusion["backend"] == "demo"
    assert "Cross-modal fusion" in res_fusion["answer"]

    # Captioning
    caption_model = registry.get("rs-captioning-landcover")
    res_cap = caption_model.adapter.analyze("Describe this scene.", [img1])
    assert res_cap["backend"] == "demo"
    assert "Comprehensive scene caption" in res_cap["answer"]

# ----------------------------------------------------
# 3. Task Classification & Routing Tests
# ----------------------------------------------------
def test_task_classification_routing(setup_database):
    """Verify natural-language query routing across all 5 specialist capabilities."""
    db = setup_database
    opt = create_test_image_asset(db, "test-cls-opt", "sentinel2.tif", "optical")
    sar = create_test_image_asset(db, "test-cls-sar", "sentinel1.tif", "sar")

    # 1. VQA
    r_vqa = TaskClassifier.classify("Are there any industrial buildings here?", [opt])
    assert r_vqa["task"] == TASK_VQA
    assert r_vqa["compatible"] is True

    # 2. Grounding
    r_grd = TaskClassifier.classify("Where are the water bodies?", [opt])
    assert r_grd["task"] == TASK_GROUNDING
    assert r_grd["compatible"] is True

    r_grd2 = TaskClassifier.classify("Highlight the roads and bridges.", [opt])
    assert r_grd2["task"] == TASK_GROUNDING
    assert r_grd2["compatible"] is True

    # 3. Change Detection (2 images)
    r_chg = TaskClassifier.classify("What changed between these two images?", [opt, opt])
    assert r_chg["task"] == TASK_CHANGE_DETECTION
    assert r_chg["compatible"] is True

    # 4. Optical + SAR (1 optical + 1 SAR)
    r_sar = TaskClassifier.classify("Compare the optical and SAR images.", [opt, sar])
    assert r_sar["task"] == TASK_OPTICAL_SAR
    assert r_sar["compatible"] is True

    # 5. Scene Captioning
    r_cap = TaskClassifier.classify("Describe this scene and its land cover.", [opt])
    assert r_cap["task"] == TASK_CAPTIONING
    assert r_cap["compatible"] is True

# ----------------------------------------------------
# 4. Input Context Rejection Tests
# ----------------------------------------------------
def test_context_aware_rejections(setup_database):
    """Verify routing rejects requests when imagery does not match task constraints."""
    db = setup_database
    opt = create_test_image_asset(db, "test-rej-opt", "sentinel2.tif", "optical")

    # Change detection requested with 1 image -> Incompatible
    r_chg_fail = TaskClassifier.classify("What changed between these two images?", [opt])
    assert r_chg_fail["task"] == TASK_CHANGE_DETECTION
    assert r_chg_fail["compatible"] is False
    assert "Two images are required" in r_chg_fail["reason"]

    # Optical + SAR requested with only 1 optical image -> Incompatible
    r_sar_fail = TaskClassifier.classify("Compare optical and SAR data.", [opt])
    assert r_sar_fail["task"] == TASK_OPTICAL_SAR
    assert r_sar_fail["compatible"] is False

    # Optical + SAR requested with 2 optical images -> Incompatible (missing SAR)
    r_sar_fail2 = TaskClassifier.classify("Compare optical and SAR images.", [opt, opt])
    assert r_sar_fail2["task"] == TASK_OPTICAL_SAR
    assert r_sar_fail2["compatible"] is False
    assert "Missing SAR" in r_sar_fail2["reason"]

# ----------------------------------------------------
# 5. Unsupported & Ambiguous Query Tests
# ----------------------------------------------------
def test_unsupported_and_ambiguous_queries(setup_database):
    """Verify out-of-scope queries are rejected and ambiguous queries default safely."""
    db = setup_database
    opt = create_test_image_asset(db, "test-unsupp-opt", "sentinel2.tif", "optical")

    # Unsupported query: weather forecast
    r_unsupp1 = TaskClassifier.classify("Predict tomorrow's weather from this satellite image.", [opt])
    assert r_unsupp1["task"] == "unsupported"
    assert r_unsupp1["compatible"] is False
    assert "outside SatQuery AI's supported analysis tasks" in r_unsupp1["reason"]

    # Unsupported query: poetry / code
    r_unsupp2 = TaskClassifier.classify("Write a python script to invert a matrix.", [opt])
    assert r_unsupp2["task"] == "unsupported"
    assert r_unsupp2["compatible"] is False

    # Ambiguous query: "analyze this image" -> Scene Captioning fallback
    r_ambig = TaskClassifier.classify("analyze this image", [opt])
    assert r_ambig["task"] == TASK_CAPTIONING
    assert "Defaulting to Scene Captioning" in r_ambig["reason"]

# ----------------------------------------------------
# 6. Manual Mode Validation Tests
# ----------------------------------------------------
def test_manual_mode_enforcement(setup_database):
    """Verify manual mode enforces input constraints without bypassing validation."""
    db = setup_database
    opt = create_test_image_asset(db, "test-man-opt", "sentinel2.tif", "optical")

    # User manually selects Change Detection, but provides 1 image -> Incompatible
    r_man_fail = TaskClassifier.classify("Where are the roads?", [opt], mode="change_detection")
    assert r_man_fail["task"] == TASK_CHANGE_DETECTION
    assert r_man_fail["compatible"] is False
    assert "Two images are required" in r_man_fail["reason"]

    # User manually selects Change Detection with 2 images -> Compatible
    r_man_ok = TaskClassifier.classify("Any question", [opt, opt], mode="change_detection")
    assert r_man_ok["task"] == TASK_CHANGE_DETECTION
    assert r_man_ok["compatible"] is True

# ----------------------------------------------------
# 7. End-to-End SatQuery Agent Orchestration & Execution Trace Tests
# ----------------------------------------------------
def test_agent_full_orchestration_and_trace(setup_database):
    """Verify agent executes 6 observable steps, selects model, and persists record."""
    db = setup_database
    opt = create_test_image_asset(db, "test-agent-opt-1", "sentinel2_scene.tif", "optical")

    request = AnalysisRequest(
        query="Where are the water bodies?",
        image_ids=[opt.id],
        mode="auto"
    )

    response = SatQueryAgent.run(request, db)
    assert response["status"] == "completed"
    assert response["task"] == TASK_GROUNDING
    assert response["selected_model"]["id"] == "rs-grounding-focal"
    assert response["selected_model"]["backend"] in ["demo", "real"]
    assert response["backend"] in ["demo", "real"]

    # Verify observable trace steps
    trace = response["execution_trace"]
    assert len(trace) == len(TRACE_STEPS)
    for i, step_name in enumerate(TRACE_STEPS):
        assert trace[i]["name"] == step_name
        assert trace[i]["status"] == "completed"

    # Verify SQLite database persistence
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == response["analysis_id"]).first()
    assert record is not None
    assert record.query == "Where are the water bodies?"
    assert record.detected_task == TASK_GROUNDING
    assert record.status == "completed"
    assert record.backend_type in ["demo", "real"]

# ----------------------------------------------------
# 8. API Endpoint POST /api/analyze & GET /api/analysis/{id}
# ----------------------------------------------------
def test_api_analyze_and_get_endpoints(setup_database):
    """Verify POST /api/analyze and GET /api/analysis/{id} HTTP contracts."""
    db = setup_database
    opt = create_test_image_asset(db, "test-api-opt-1", "sentinel2_coastal.tif", "optical")

    # Valid request
    res = client.post(
        "/api/analyze",
        json={
            "query": "Describe this scene and summarize the terrain.",
            "image_ids": [opt.id],
            "mode": "auto"
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["task"] == TASK_CAPTIONING
    assert data["selected_model"]["id"] == "rs-captioning-landcover"
    assert data["backend"] in ["demo", "adapted", "real"]
    analysis_id = data["analysis_id"]

    # Verify retrieval via GET /api/analysis/{analysis_id}
    get_res = client.get(f"/api/analysis/{analysis_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["analysis_id"] == analysis_id
    assert get_data["detected_task"] == TASK_CAPTIONING
    assert get_data["status"] == "completed"

    # Incompatible request: Change detection with 1 image
    res_incomp = client.post(
        "/api/analyze",
        json={
            "query": "What changed between these two images?",
            "image_ids": [opt.id],
            "mode": "auto"
        }
    )
    assert res_incomp.status_code == 200  # Returns structured result, not raw 500
    data_incomp = res_incomp.json()
    assert data_incomp["status"] == "incompatible"
    assert "Two images are required" in data_incomp["error_message"]

    # Unsupported request: Weather forecast
    res_unsupp = client.post(
        "/api/analyze",
        json={
            "query": "Predict tomorrow's weather from this satellite image.",
            "image_ids": [opt.id],
            "mode": "auto"
        }
    )
    assert res_unsupp.status_code == 200
    data_unsupp = res_unsupp.json()
    assert data_unsupp["status"] == "unsupported"
    assert "outside SatQuery AI's supported analysis tasks" in data_unsupp["error_message"]
