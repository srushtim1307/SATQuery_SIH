"""
SatQuery AI — Phase 5 Full End-to-End System Verification Suite.

Validates the full agentic AI stack with real deep learning specialist models:
1. Model Registry Discovery & Verification (All 5 models active with real/adapted backends)
2. Task 1: Change Detection via TinyCD (Siamese Sliced EfficientNet-B4 + MAMB)
3. Task 2: Open-Vocabulary Visual Grounding via OWL-ViT
4. Task 3: Remote-Sensing VQA via SmolVLM-256M Instruct
5. Task 4: Land-Cover Scene Captioning via BigEarthNet-19 MobileNetV3-Small
6. Task 5: Cross-Modal Optical + SAR Fusion via CrossModalOpticalSARNet
7. Single-Active Model Memory Management & LRU Eviction Policy
8. Database Persistence & Audit Retrieval (/api/analysis/{id})
"""

import sys
import io
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.models.manager import model_manager

client = TestClient(app)

def log_step(num: int, name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] #{num:02d} {name} — {detail}")
    return passed

def create_and_upload_asset(filename: str, modality: str, color=(34, 139, 34)) -> str:
    buf = io.BytesIO()
    img = Image.new("RGB", (256, 256), color=color)
    img.save(buf, format="PNG")
    buf.seek(0)

    files = {"files": (filename, buf, "image/png")}
    data = {"modality_hint": modality}
    res = client.post("/api/uploads", data=data, files=files)
    assert res.status_code == 200, f"Upload failed: {res.text}"
    return res.json()[0]["id"]

def run_phase5_e2e_verification():
    print("=================================================================")
    print("SATQUERY AI — PHASE 5 FULL SYSTEM VERIFICATION SUITE")
    print("=================================================================")

    results = {}

    # Step 1: Model Registry Verification
    print("\n--- [1] MODEL REGISTRY DISCOVERY & BACKEND AUDIT ---")
    res = client.get("/api/models")
    assert res.status_code == 200
    models = res.json()
    model_map = {m["task"]: m for m in models}
    
    passed_reg = (
        len(models) >= 5 and
        model_map["change_detection"]["backend_type"] == "real" and
        model_map["grounding"]["backend_type"] == "real" and
        model_map["vqa"]["backend_type"] == "real" and
        model_map["captioning"]["backend_type"] == "adapted" and
        model_map["optical_sar"]["backend_type"] == "real"
    )
    results[1] = log_step(
        1, "Model Registry Real Backends", passed_reg,
        f"{len(models)} models registered. CD: {model_map['change_detection']['backend_type']}, "
        f"Grounding: {model_map['grounding']['backend_type']}, VQA: {model_map['vqa']['backend_type']}, "
        f"Captioning: {model_map['captioning']['backend_type']}, Optical-SAR: {model_map['optical_sar']['backend_type']}"
    )

    # Ingest assets for multi-modal analysis
    print("\n--- INGESTING TEST ASSETS ---")
    opt1_id = create_and_upload_asset("e2e_opt_may2023.png", "optical", color=(40, 140, 40))
    opt2_id = create_and_upload_asset("e2e_opt_oct2024.png", "optical", color=(60, 160, 60))
    sar_id = create_and_upload_asset("e2e_sar_radar.png", "sar", color=(95, 95, 95))
    print(f"Uploaded: opt1={opt1_id}, opt2={opt2_id}, sar={sar_id}")

    # Step 2: Change Detection Specialist (TinyCD)
    print("\n--- [2] CHANGE DETECTION SPECIALIST (TinyCD) ---")
    payload_cd = {
        "query": "Identify surface changes and alterations between May 2023 and October 2024",
        "image_ids": [opt1_id, opt2_id],
        "mode": "auto"
    }
    t0 = time.time()
    res_cd = client.post("/api/analyze", json=payload_cd)
    dt_cd = time.time() - t0
    data_cd = res_cd.json()
    passed_cd = (
        res_cd.status_code == 200 and
        data_cd["status"] == "completed" and
        data_cd["task"] == "change_detection" and
        data_cd["backend"] == "real" and
        data_cd["result"]["confidence"] is not None and
        "overlay_url" in data_cd["result"]["evidence"][0]
    )
    results[2] = log_step(
        2, "Change Detection (TinyCD)", passed_cd,
        f"Time: {dt_cd:.2f}s | Confidence: {data_cd['result']['confidence']} | "
        f"Change%: {data_cd['result']['evidence'][0].get('change_percentage')}% | "
        f"Overlay: {data_cd['result']['evidence'][0].get('overlay_url')}"
    )

    # Step 3: Visual Grounding Specialist (OWL-ViT)
    print("\n--- [3] VISUAL GROUNDING SPECIALIST (OWL-ViT) ---")
    payload_gr = {
        "query": "Locate and highlight green vegetation and forest canopies",
        "image_ids": [opt1_id],
        "mode": "auto"
    }
    t0 = time.time()
    res_gr = client.post("/api/analyze", json=payload_gr)
    dt_gr = time.time() - t0
    data_gr = res_gr.json()
    passed_gr = (
        res_gr.status_code == 200 and
        data_gr["status"] == "completed" and
        data_gr["task"] == "grounding" and
        data_gr["backend"] == "real" and
        data_gr["result"]["confidence"] is not None and
        "boundingBoxes" in data_gr["result"]["evidence"][0]
    )
    results[3] = log_step(
        3, "Visual Grounding (OWL-ViT)", passed_gr,
        f"Time: {dt_gr:.2f}s | Confidence: {data_gr['result']['confidence']} | "
        f"Boxes: {len(data_gr['result']['evidence'][0]['boundingBoxes'])} grounded"
    )

    # Step 4: Remote-Sensing VQA Specialist (SmolVLM-256M)
    print("\n--- [4] REMOTE-SENSING VQA SPECIALIST (SmolVLM-256M) ---")
    payload_vqa = {
        "query": "Is this image mostly water or green vegetation?",
        "image_ids": [opt1_id],
        "mode": "auto"
    }
    t0 = time.time()
    res_vqa = client.post("/api/analyze", json=payload_vqa)
    dt_vqa = time.time() - t0
    data_vqa = res_vqa.json()
    vqa_ans = data_vqa.get("result", {}).get("answer", "").strip()
    passed_vqa = (
        res_vqa.status_code == 200 and
        data_vqa["status"] == "completed" and
        data_vqa["task"] == "vqa" and
        data_vqa["backend"] == "real" and
        data_vqa["result"]["confidence"] is not None and
        len(vqa_ans) > 0
    )
    results[4] = log_step(
        4, "Remote-Sensing VQA (SmolVLM)", passed_vqa,
        f"Time: {dt_vqa:.2f}s | Confidence: {data_vqa['result']['confidence']} | "
        f"Answer: '{vqa_ans}'"
    )

    # Step 5: Scene Captioning Specialist (BigEarthNet-19 Adapted)
    print("\n--- [5] SCENE CAPTIONING SPECIALIST (BigEarthNet-19) ---")
    payload_cap = {
        "query": "Describe the land cover classification and scene breakdown",
        "image_ids": [opt1_id],
        "mode": "auto"
    }
    t0 = time.time()
    res_cap = client.post("/api/analyze", json=payload_cap)
    dt_cap = time.time() - t0
    data_cap = res_cap.json()
    ev_cap = data_cap.get("result", {}).get("evidence", [{}])[0]
    classes = ev_cap.get("top_classes", ev_cap.get("classes", []))
    passed_cap = (
        res_cap.status_code == 200 and
        data_cap["status"] == "completed" and
        data_cap["task"] == "captioning" and
        data_cap["backend"] == "adapted" and
        data_cap["result"]["confidence"] is not None and
        len(classes) > 0
    )
    results[5] = log_step(
        5, "Scene Captioning (BigEarthNet-19)", passed_cap,
        f"Time: {dt_cap:.2f}s | Confidence: {data_cap['result']['confidence']} | "
        f"Top Classes: {[c['class_name'] if isinstance(c, dict) else c for c in classes[:2]]}"
    )

    # Step 6: Optical + SAR Fusion Specialist (CrossModalOpticalSARNet)
    print("\n--- [6] OPTICAL + SAR FUSION SPECIALIST ---")
    payload_fusion = {
        "query": "Perform cross-modal fusion comparing optical reflectance and SAR radar backscatter",
        "image_ids": [opt1_id, sar_id],
        "mode": "auto"
    }
    t0 = time.time()
    res_fus = client.post("/api/analyze", json=payload_fusion)
    dt_fus = time.time() - t0
    data_fus = res_fus.json()
    passed_fus = (
        res_fus.status_code == 200 and
        data_fus["status"] == "completed" and
        data_fus["task"] == "optical_sar" and
        data_fus["backend"] == "real" and
        data_fus["result"]["confidence"] is not None and
        "fusedImageUrl" in data_fus["result"]["evidence"][0] and
        "synergyMapUrl" in data_fus["result"]["evidence"][0]
    )
    results[6] = log_step(
        6, "Optical + SAR Fusion", passed_fus,
        f"Time: {dt_fus:.2f}s | Confidence: {data_fus['result']['confidence']} | "
        f"SAR dB: {data_fus['result']['evidence'][0]['metrics']['sar_backscatter_mean_db']} dB | "
        f"Fused URL: {data_fus['result']['evidence'][0]['fusedImageUrl']}"
    )

    # Step 7: Single-Active LRU Model Eviction & Memory Management
    print("\n--- [7] MODEL MANAGER LRU EVICTION AUDIT ---")
    sys_status = model_manager.get_system_status()
    loaded_models = sys_status["loaded_models"]
    max_active = sys_status["max_active_models"]
    passed_lru = len(loaded_models) <= max_active
    results[7] = log_step(
        7, "Single-Active LRU Model Policy", passed_lru,
        f"Loaded models in memory: {loaded_models} (Limit: {max_active}, Device: {sys_status['device']})"
    )

    # Step 8: Database Audit Persistence & Retrieval
    print("\n--- [8] AUDIT PERSISTENCE & TRACE RETRIEVAL ---")
    analysis_id = data_fus["analysis_id"]
    res_get = client.get(f"/api/analysis/{analysis_id}")
    passed_audit = (
        res_get.status_code == 200 and
        res_get.json()["analysis_id"] == analysis_id and
        len(res_get.json()["execution_trace"]) >= 5
    )
    results[8] = log_step(
        8, "Audit Persistence Retrieval", passed_audit,
        f"Record ID: {analysis_id} | Trace Steps: {len(res_get.json().get('execution_trace', []))}"
    )

    print("\n=================================================================")
    all_passed = all(results.values())
    total_passed = sum(1 for v in results.values() if v)
    print(f"PHASE 5 VERIFICATION RESULT: {total_passed}/{len(results)} PASSED")
    print("=================================================================")
    assert all_passed, "One or more Phase 5 verification tests failed."
    return all_passed

if __name__ == "__main__":
    success = run_phase5_e2e_verification()
    sys.exit(0 if success else 1)
