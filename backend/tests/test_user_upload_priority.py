"""
Integration test suite verifying user upload priority and identity preservation.

Ensures that when a user uploads their own satellite image(s) and asks a question:
1. The backend ingestion stores the real image asset and returns its unique ID.
2. POST /api/analyze uses the user's uploaded asset ID, NOT demo images.
3. The specialist models (OWL-ViT, SmolVLM, TinyCD, BigEarthNet, CrossModal) receive the uploaded image file path.
4. The visual evidence returned in the response references the uploaded asset's preview/filename.
5. Multiple sequential uploads isolate their inputs properly without sticky demo assets.
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

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    yield db
    db.close()

def create_synthetic_image_bytes(color=(30, 144, 255), size=(256, 256)) -> io.BytesIO:
    """Creates an in-memory PNG image bytes buffer."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def test_user_single_upload_and_vqa_analysis(setup_db):
    """
    Test flow:
    1. User uploads custom_water_scene.png.
    2. Receives asset ID.
    3. User asks 'Where are the water bodies in this image?'.
    4. Verify the analysis routes the uploaded asset, not demo data.
    """
    img_bytes = create_synthetic_image_bytes(color=(0, 100, 200))
    files = [("files", ("custom_water_scene.png", img_bytes, "image/png"))]

    upload_res = client.post("/api/uploads", files=files)
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    uploaded_assets = upload_res.json()
    assert len(uploaded_assets) == 1
    asset = uploaded_assets[0]
    asset_id = asset["id"]
    assert asset["filename"] == "custom_water_scene.png"

    # Analyze with user query
    analyze_payload = {
        "query": "Where are the water bodies in this image?",
        "image_ids": [asset_id],
        "mode": "auto",
        "session_id": f"test-sess-{asset_id}"
    }

    analyze_res = client.post("/api/analyze", json=analyze_payload)
    assert analyze_res.status_code == 200, f"Analyze failed: {analyze_res.text}"
    res_data = analyze_res.json()

    assert res_data["status"] == "completed"
    assert res_data["task"] == "grounding"

    # Verify visual evidence points to the user's uploaded asset preview
    evidence_list = res_data["result"].get("evidence", [])
    assert len(evidence_list) >= 1
    evidence = evidence_list[0]
    assert asset["preview_url"] in evidence["imageUrl"]
    assert "sentinel2_estuary_delta.png" not in evidence["imageUrl"]

    # Verify execution trace shows image verification
    trace = res_data["execution_trace"]
    assert any("image(s) verified" in t.get("detail", "") for t in trace)

def test_user_sequential_uploads_are_isolated(setup_db):
    """
    Verify that uploading Image A and then Image B routes Image B when asked about Image B.
    No sticky state or fallback to Image A or demo images.
    """
    # Upload Image A
    buf_a = create_synthetic_image_bytes(color=(10, 200, 50))
    res_a = client.post("/api/uploads", files=[("files", ("image_alpha_forest.png", buf_a, "image/png"))])
    asset_a = res_a.json()[0]

    # Upload Image B
    buf_b = create_synthetic_image_bytes(color=(220, 50, 10))
    res_b = client.post("/api/uploads", files=[("files", ("image_beta_desert.png", buf_b, "image/png"))])
    asset_b = res_b.json()[0]

    assert asset_a["id"] != asset_b["id"]

    # Query specifically with Image B
    analyze_payload = {
        "query": "Describe the land-cover and surface features in this scene.",
        "image_ids": [asset_b["id"]],
        "mode": "auto"
    }
    analyze_res = client.post("/api/analyze", json=analyze_payload)
    assert analyze_res.status_code == 200
    res_data = analyze_res.json()

    assert res_data["status"] == "completed"
    evidence = res_data["result"]["evidence"][0]
    # Evidence must point to asset B, not asset A or demo
    assert asset_b["preview_url"] in evidence["imageUrl"]
    assert asset_a["preview_url"] not in evidence["imageUrl"]
    assert "sentinel2_estuary_delta" not in evidence["imageUrl"]

def test_user_bi_temporal_upload_change_detection(setup_db):
    """
    Verify uploading 2 images for change analysis routes both uploaded images to TinyCD.
    """
    buf_t1 = create_synthetic_image_bytes(color=(30, 30, 30))
    buf_t2 = create_synthetic_image_bytes(color=(200, 200, 200))

    res_t1 = client.post("/api/uploads", files=[("files", ("user_pre_event.png", buf_t1, "image/png"))])
    res_t2 = client.post("/api/uploads", files=[("files", ("user_post_event.png", buf_t2, "image/png"))])

    asset_t1 = res_t1.json()[0]
    asset_t2 = res_t2.json()[0]

    analyze_payload = {
        "query": "What changed between these two images?",
        "image_ids": [asset_t1["id"], asset_t2["id"]],
        "mode": "auto"
    }

    analyze_res = client.post("/api/analyze", json=analyze_payload)
    assert analyze_res.status_code == 200
    res_data = analyze_res.json()

    assert res_data["task"] == "change_detection"
    assert res_data["status"] == "completed"
    assert res_data["selected_model"]["id"] == "rs-change-diff"

    evidence = res_data["result"]["evidence"][0]
    assert asset_t1["preview_url"] in evidence["beforeImageUrl"]
    assert asset_t2["preview_url"] in evidence["t2_preview_url"]
    assert "/static/processed/change_overlay" in evidence["overlay_url"]
    assert evidence["t1_filename"] == "user_pre_event.png"
    assert evidence["t2_filename"] == "user_post_event.png"
