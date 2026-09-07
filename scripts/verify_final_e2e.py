"""
SatQuery AI — Final Phase 6 Master End-to-End Acceptance Suite.

Validates:
1. Health, Model Registry & Benchmark Endpoints (/api/health, /api/models, /api/benchmarks)
2. Ingestion & Security (PNG, GeoTIFF, invalid extension rejection, corrupted image rejection)
3. Graceful Error Handling & Rejection Rules (empty query, out-of-scope query, modality mismatch)
4. JURY DEMO SCENARIO 1 — Visual Grounding ("Where are the water bodies in this image?")
5. JURY DEMO SCENARIO 2 — Bi-Temporal Change Detection ("What changed between these two images?")
6. JURY DEMO SCENARIO 3 — Optical + SAR Fusion ("Identify built-up and water-covered regions using both images.")
7. Remote-Sensing VQA Specialist (SmolVLM-256M)
8. Land-Cover Scene Captioning (BigEarthNet-19)
9. Real vs Demo Backend Integrity & Zero-Fabrication Confidence Guarantees
10. Database Persistence & History Restoration (/api/history and /api/analysis/{id})
11. ModelManager Single-Active Model Memory Eviction Policy
"""

import sys
import io
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
import tifffile
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.models.manager import model_manager

client = TestClient(app)

def log_test(num: int, name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] #{num:02d} {name} — {detail}")
    return passed

def upload_png_raster(filename: str, modality: str, color=(34, 139, 34), draw_water=False) -> str:
    buf = io.BytesIO()
    img = Image.new("RGB", (256, 256), color=color)
    if draw_water:
        # Draw distinct blue water body in center
        arr = np.array(img)
        arr[80:180, 60:200] = [15, 65, 175] # Blue water
        img = Image.fromarray(arr)
    img.save(buf, format="PNG")
    buf.seek(0)
    files = {"files": (filename, buf, "image/png")}
    data = {"modality_hint": modality}
    res = client.post("/api/uploads", data=data, files=files)
    assert res.status_code == 200, f"Upload failed: {res.text}"
    return res.json()[0]["id"]

def upload_tiff_raster(filename: str, modality: str) -> str:
    buf = io.BytesIO()
    arr = np.random.randint(20, 220, size=(128, 128), dtype=np.uint8)
    tifffile.imwrite(buf, arr)
    buf.seek(0)
    files = {"files": (filename, buf, "image/tiff")}
    data = {"modality_hint": modality}
    res = client.post("/api/uploads", data=data, files=files)
    assert res.status_code == 200, f"TIFF upload failed: {res.text}"
    return res.json()[0]["id"]

def run_master_e2e_verification():
    print("=================================================================")
    print("SATQUERY AI — FINAL PHASE 6 MASTER E2E ACCEPTANCE SUITE")
    print("=================================================================")

    results = {}

    # 1. API Health & Status
    print("\n--- [1] API HEALTH & ROUTE DISCOVERY ---")
    res_health = client.get("/api/health")
    passed_health = (res_health.status_code == 200 and res_health.json().get("status") == "healthy")
    results[1] = log_test(1, "API Health Endpoint", passed_health, f"Status: {res_health.json().get('status')}")

    # 2. Model Registry & Benchmark Discovery
    print("\n--- [2] MODEL REGISTRY & BENCHMARK AUDIT ---")
    res_models = client.get("/api/models")
    res_benchmarks = client.get("/api/benchmarks")
    models = res_models.json()
    benchmarks = res_benchmarks.json()
    
    passed_models = len(models) >= 5
    passed_benchmarks = len(benchmarks) >= 4 and any(b["id"] == "BigEarthNet-19" and b["evaluation_status"] == "evaluated" for b in benchmarks)
    results[2] = log_test(2, "Model Registry & Benchmarks", passed_models and passed_benchmarks, 
                          f"{len(models)} models registered, {len(benchmarks)} benchmarks tracked")

    # 3. Ingestion & Security Validations
    print("\n--- [3] INGESTION & SECURITY VALIDATION ---")
    # A. Valid PNG upload
    opt_water_id = upload_png_raster("jury_opt_coastal_water.png", "optical", color=(34, 139, 34), draw_water=True)
    opt_t1_id = upload_png_raster("jury_opt_baseline_2023.png", "optical", color=(45, 145, 45))
    opt_t2_id = upload_png_raster("jury_opt_current_2024.png", "optical", color=(65, 165, 65))
    sar_radar_id = upload_png_raster("jury_sar_sentinel1.png", "sar", color=(90, 90, 90))
    tiff_id = upload_tiff_raster("jury_geotiff_raster.tif", "optical")

    # B. Invalid file extension rejection (HTTP 400)
    res_bad_ext = client.post("/api/uploads", files={"files": ("malicious.exe", b"binary", "application/octet-stream")})
    passed_bad_ext = (res_bad_ext.status_code == 400)

    # C. Corrupted image rejection (HTTP 400)
    res_corrupt = client.post("/api/uploads", files={"files": ("corrupt.png", b"not a valid png file header", "image/png")})
    passed_corrupt = (res_corrupt.status_code == 400)

    results[3] = log_test(3, "Upload & Security Constraints", passed_bad_ext and passed_corrupt,
                          "PNG/TIFF accepted; .exe rejected (400); corrupt binary rejected (400)")

    # 4. Graceful Error Handling & Rejection Rules
    print("\n--- [4] GRACEFUL ERROR HANDLING & REJECTION RULES ---")
    # A. Empty query
    res_empty = client.post("/api/analyze", json={"query": "", "image_ids": [opt_water_id]})
    passed_empty = (res_empty.status_code == 200 and res_empty.json()["status"] == "incompatible")

    # B. Unsupported / Out-of-scope query
    res_unsupported = client.post("/api/analyze", json={
        "query": "Write a python script to calculate fibonacci numbers",
        "image_ids": [opt_water_id]
    })
    passed_unsupported = (res_unsupported.status_code == 200 and res_unsupported.json()["status"] == "unsupported")

    # C. 1 image provided for change query (incompatible input)
    res_cd_1img = client.post("/api/analyze", json={
        "query": "What changed between these images?",
        "image_ids": [opt_water_id]
    })
    passed_cd_1img = (res_cd_1img.status_code == 200 and res_cd_1img.json()["status"] == "incompatible")

    # D. Missing SAR in optical+sar query
    res_missing_sar = client.post("/api/analyze", json={
        "query": "Compare optical and SAR radar characteristics",
        "image_ids": [opt_t1_id, opt_t2_id]
    })
    passed_missing_sar = (res_missing_sar.status_code == 200 and res_missing_sar.json()["status"] == "incompatible")

    results[4] = log_test(4, "Graceful Input Rejection Rules", 
                          passed_empty and passed_unsupported and passed_cd_1img and passed_missing_sar,
                          "All invalid inputs rejected gracefully with human-friendly messages and zero HTTP 500s")

    # 5. JURY DEMO SCENARIO 1 — Visual Grounding
    print("\n--- [5] JURY DEMO SCENARIO 1: VISUAL GROUNDING ---")
    t0 = time.time()
    res_demo1 = client.post("/api/analyze", json={
        "query": "Where are the water bodies in this image?",
        "image_ids": [opt_water_id],
        "mode": "auto"
    })
    dt_demo1 = time.time() - t0
    d1 = res_demo1.json()
    passed_demo1 = (
        res_demo1.status_code == 200 and
        d1["status"] == "completed" and
        d1["task"] == "grounding" and
        d1["backend"] == "real" and
        d1["result"]["confidence"] is not None and
        len(d1["result"]["evidence"][0]["boundingBoxes"]) > 0
    )
    results[5] = log_test(5, "Jury Demo 1: Grounding", passed_demo1,
                          f"Time: {dt_demo1:.2f}s | Confidence: {d1['result']['confidence']} | "
                          f"Boxes: {len(d1['result']['evidence'][0]['boundingBoxes'])} | Trace: {len(d1['execution_trace'])} steps")

    # 6. JURY DEMO SCENARIO 2 — Bi-Temporal Change Detection
    print("\n--- [6] JURY DEMO SCENARIO 2: CHANGE DETECTION ---")
    t0 = time.time()
    res_demo2 = client.post("/api/analyze", json={
        "query": "What changed between these two images?",
        "image_ids": [opt_t1_id, opt_t2_id],
        "mode": "auto"
    })
    dt_demo2 = time.time() - t0
    d2 = res_demo2.json()
    passed_demo2 = (
        res_demo2.status_code == 200 and
        d2["status"] == "completed" and
        d2["task"] == "change_detection" and
        d2["backend"] == "real" and
        d2["result"]["confidence"] is not None and
        "overlay_url" in d2["result"]["evidence"][0]
    )
    results[6] = log_test(6, "Jury Demo 2: Change Detection", passed_demo2,
                          f"Time: {dt_demo2:.2f}s | Confidence: {d2['result']['confidence']} | "
                          f"Change: {d2['result']['evidence'][0].get('change_percentage')}% | Overlay: {d2['result']['evidence'][0].get('overlay_url')}")

    # 7. JURY DEMO SCENARIO 3 — Optical + SAR Cross-Modal Fusion
    print("\n--- [7] JURY DEMO SCENARIO 3: OPTICAL + SAR FUSION ---")
    t0 = time.time()
    res_demo3 = client.post("/api/analyze", json={
        "query": "Identify built-up and water-covered regions using both images.",
        "image_ids": [opt_water_id, sar_radar_id],
        "mode": "auto"
    })
    dt_demo3 = time.time() - t0
    d3 = res_demo3.json()
    passed_demo3 = (
        res_demo3.status_code == 200 and
        d3["status"] == "completed" and
        d3["task"] == "optical_sar" and
        d3["backend"] == "real" and
        d3["result"]["confidence"] is not None and
        "fusedImageUrl" in d3["result"]["evidence"][0] and
        "synergyMapUrl" in d3["result"]["evidence"][0]
    )
    results[7] = log_test(7, "Jury Demo 3: Optical + SAR Fusion", passed_demo3,
                          f"Time: {dt_demo3:.2f}s | Confidence: {d3['result']['confidence']} | "
                          f"SAR dB: {d3['result']['evidence'][0]['metrics']['sar_backscatter_mean_db']} dB | "
                          f"Edge Corr: {d3['result']['evidence'][0]['metrics']['cross_modal_edge_correlation']}")

    # 8. Remote-Sensing VQA Specialist (SmolVLM-256M)
    print("\n--- [8] REMOTE-SENSING VQA SPECIALIST ---")
    t0 = time.time()
    res_vqa = client.post("/api/analyze", json={
        "query": "Is this image mostly water or green vegetation?",
        "image_ids": [opt_water_id],
        "mode": "auto"
    })
    dt_vqa = time.time() - t0
    d_vqa = res_vqa.json()
    vqa_ans = d_vqa.get("result", {}).get("answer", "").strip()
    passed_vqa = (
        res_vqa.status_code == 200 and
        d_vqa["status"] == "completed" and
        d_vqa["task"] == "vqa" and
        d_vqa["backend"] == "real" and
        d_vqa["result"]["confidence"] is not None and
        len(vqa_ans) > 0
    )
    results[8] = log_test(8, "Remote-Sensing VQA", passed_vqa,
                          f"Time: {dt_vqa:.2f}s | Confidence: {d_vqa['result']['confidence']} | Answer: '{vqa_ans}'")

    # 9. Scene Captioning Specialist (BigEarthNet-19)
    print("\n--- [9] SCENE CAPTIONING SPECIALIST ---")
    t0 = time.time()
    res_cap = client.post("/api/analyze", json={
        "query": "Describe the land cover classification and scene breakdown",
        "image_ids": [opt_water_id],
        "mode": "auto"
    })
    dt_cap = time.time() - t0
    d_cap = res_cap.json()
    ev_cap = d_cap.get("result", {}).get("evidence", [{}])[0]
    top_c = ev_cap.get("top_classes", ev_cap.get("classes", []))
    passed_cap = (
        res_cap.status_code == 200 and
        d_cap["status"] == "completed" and
        d_cap["task"] == "captioning" and
        d_cap["backend"] == "adapted" and
        d_cap["result"]["confidence"] is not None and
        len(top_c) > 0
    )
    results[9] = log_test(9, "Land Cover Captioning", passed_cap,
                          f"Time: {dt_cap:.2f}s | Confidence: {d_cap['result']['confidence']} | Top Class: {top_c[0].get('class_name') if isinstance(top_c[0], dict) else top_c[0]}")

    # 10. Database Persistence & History Restoration
    print("\n--- [10] DATABASE PERSISTENCE & HISTORY RESTORATION ---")
    res_hist = client.get("/api/history")
    hist = res_hist.json()
    demo3_id = d3["analysis_id"]
    res_single = client.get(f"/api/analysis/{demo3_id}")
    passed_hist = (
        res_hist.status_code == 200 and
        len(hist) > 0 and
        res_single.status_code == 200 and
        res_single.json()["analysis_id"] == demo3_id and
        len(res_single.json()["execution_trace"]) >= 5
    )
    results[10] = log_test(10, "History Persistence & Restoration", passed_hist,
                           f"History records in DB: {len(hist)} | Loaded record {demo3_id} with {len(res_single.json().get('execution_trace', []))} trace steps")

    # 11. ModelManager Single-Active Model Memory Eviction
    print("\n--- [11] MODEL MANAGER LRU MEMORY SAFETY ---")
    sys_status = model_manager.get_system_status()
    loaded_models = sys_status["loaded_models"]
    max_active = sys_status["max_active_models"]
    passed_lru = len(loaded_models) <= max_active
    results[11] = log_test(11, "Single-Active LRU Model Limit", passed_lru,
                           f"Active model in memory: {loaded_models} (Device: {sys_status['device']}, Limit: {max_active})")

    # Final Summary
    print("\n=================================================================")
    all_passed = all(results.values())
    total_passed = sum(1 for v in results.values() if v)
    print(f"MASTER ACCEPTANCE RESULT: {total_passed}/{len(results)} SUITES PASSED")
    print("=================================================================")
    assert all_passed, "One or more master acceptance test suites failed."
    return all_passed

if __name__ == "__main__":
    success = run_master_e2e_verification()
    sys.exit(0 if success else 1)
