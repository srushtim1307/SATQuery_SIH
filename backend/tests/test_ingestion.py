import io
import sys
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
from fastapi.testclient import TestClient

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.services.ingestion import detect_modality_safely, sanitize_filename

client = TestClient(app)

def create_sample_png_bytes(width=100, height=80, color=(100, 150, 200)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def create_sample_tiff_bytes(width=80, height=80, bands=4) -> bytes:
    data = np.random.randint(0, 10000, size=(height, width, bands), dtype=np.uint16)
    buf = io.BytesIO()
    tifffile.imwrite(
        buf,
        data,
        planarconfig='contig',
        metadata={'axes': 'YXS'},
        description="Sentinel-2 Multispectral Sample"
    )
    buf.seek(0)
    return buf.getvalue()

def test_sanitize_filename():
    assert sanitize_filename("../../../secret.tif") == "secret.tif"
    assert sanitize_filename("safe_raster-10m.tif") == "safe_raster-10m.tif"
    assert sanitize_filename(".hidden.tif") == "hidden.tif"

def test_modality_detection_heuristics():
    assert detect_modality_safely("sentinel2_estuary_delta.tif", 4, {}) == "optical"
    assert detect_modality_safely("gulf_coastal_sar.tif", 1, {}) == "sar"
    assert detect_modality_safely("random_scene.tif", 3, {}) == "optical"
    assert detect_modality_safely("unlabeled_band.tif", 1, {}) == "unknown"

def test_upload_valid_png():
    png_bytes = create_sample_png_bytes(120, 90)
    response = client.post(
        "/api/uploads",
        files=[("files", ("test_image.png", png_bytes, "image/png"))]
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    asset = data[0]
    assert asset["format"] == "PNG"
    assert asset["width"] == 120
    assert asset["height"] == 90
    assert asset["bands"] == 3
    assert asset["status"] == "ready"
    assert asset["preview_url"] is not None

def test_upload_valid_geotiff():
    tiff_bytes = create_sample_tiff_bytes(80, 80, bands=4)
    response = client.post(
        "/api/uploads",
        files=[("files", ("sentinel2_estuary_test.tif", tiff_bytes, "image/tiff"))]
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    asset = data[0]
    assert asset["format"] in ["TIFF", "GeoTIFF"]
    assert asset["width"] == 80
    assert asset["height"] == 80
    assert asset["bands"] == 4
    assert asset["modality"] == "optical"
    assert asset["preview_url"] is not None

def test_upload_unsupported_extension():
    response = client.post(
        "/api/uploads",
        files=[("files", ("script.py", b"print('malicious')", "text/x-python"))]
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_upload_corrupted_image():
    corrupted_bytes = b"NOT_A_VALID_IMAGE_CONTENT_AT_ALL_12345"
    response = client.post(
        "/api/uploads",
        files=[("files", ("corrupt_raster.png", corrupted_bytes, "image/png"))]
    )
    assert response.status_code == 400
    assert "corrupt or unreadable" in response.json()["detail"]

def test_validation_api_single_and_bitemporal():
    # 1. Upload 2 images
    img1 = create_sample_png_bytes(100, 100)
    img2 = create_sample_png_bytes(100, 100)
    
    upload_res = client.post(
        "/api/uploads",
        files=[
            ("files", ("sentinel2_may2023.png", img1, "image/png")),
            ("files", ("sentinel2_oct2024.png", img2, "image/png"))
        ]
    )
    assert upload_res.status_code == 200
    assets = upload_res.json()
    id1 = assets[0]["id"]
    id2 = assets[1]["id"]

    # Test single-image validation on 1 image -> PASS
    val_single = client.post("/api/validate", json={"image_ids": [id1], "mode": "single"})
    assert val_single.status_code == 200
    assert val_single.json()["valid"] is True

    # Test single-image validation on 2 images -> FAIL
    val_single_fail = client.post("/api/validate", json={"image_ids": [id1, id2], "mode": "single"})
    assert val_single_fail.status_code == 200
    assert val_single_fail.json()["valid"] is False
    assert len(val_single_fail.json()["errors"]) > 0

    # Test bi-temporal change detection on 2 images -> PASS
    val_bitemporal = client.post("/api/validate", json={"image_ids": [id1, id2], "mode": "change_detection"})
    assert val_bitemporal.status_code == 200
    assert val_bitemporal.json()["valid"] is True

    # Test bi-temporal change detection on 1 image -> FAIL
    val_bitemporal_fail = client.post("/api/validate", json={"image_ids": [id1], "mode": "change_detection"})
    assert val_bitemporal_fail.status_code == 200
    assert val_bitemporal_fail.json()["valid"] is False
    assert "Two images are required" in val_bitemporal_fail.json()["errors"][0]

def test_validation_api_optical_sar():
    # Upload one optical and one SAR
    opt_bytes = create_sample_png_bytes()
    sar_bytes = create_sample_png_bytes()

    upload_res = client.post(
        "/api/uploads",
        files=[
            ("files", ("sentinel2_optical.png", opt_bytes, "image/png")),
            ("files", ("sentinel1_sar.png", sar_bytes, "image/png"))
        ]
    )
    assert upload_res.status_code == 200
    assets = upload_res.json()
    opt_id = assets[0]["id"]
    sar_id = assets[1]["id"]

    # Validate optical + SAR pair -> PASS
    val_fusion = client.post("/api/validate", json={"image_ids": [opt_id, sar_id], "mode": "optical_sar"})
    assert val_fusion.status_code == 200
    assert val_fusion.json()["valid"] is True

    # Validate 2 optical images for optical+SAR -> FAIL
    val_fusion_fail = client.post("/api/validate", json={"image_ids": [opt_id, opt_id], "mode": "optical_sar"})
    assert val_fusion_fail.status_code == 200
    assert val_fusion_fail.json()["valid"] is False
    assert any("Missing SAR" in err for err in val_fusion_fail.json()["errors"])

def test_upload_file_size_exceeded():
    from app.core.config import settings
    orig_limit = settings.max_upload_size_mb
    try:
        # Set limit to 1MB
        settings.max_upload_size_mb = 1
        # Create a payload of 1.5MB
        oversized_bytes = b"0" * (1024 * 1024 + 512 * 1024)
        response = client.post(
            "/api/uploads",
            files=[("files", ("oversized_image.png", oversized_bytes, "image/png"))]
        )
        assert response.status_code == 413
        assert "exceeds maximum allowed size" in response.json()["detail"]
    finally:
        settings.max_upload_size_mb = orig_limit

