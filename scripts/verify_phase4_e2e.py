"""
End-to-End Verification Suite for Phase 4:
Agentic Task Classification, Routing, Model Registry & Adapter Architecture.
"""

import io
import json
import urllib.request
import urllib.error
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5173"

def log_test(num: int, name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] #{num:02d} {name} - {detail}")
    return passed

def post_json(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as res:
        return res.status, json.loads(res.read().decode())

def get_json(url: str):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as res:
        return res.status, json.loads(res.read().decode())

def create_multipart(fields, files):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())
    for key, filename, content, content_type in files:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode())
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return boundary, bytes(body)

def upload_sample_image(filename: str, modality: str) -> str:
    buf = io.BytesIO()
    if filename.endswith(".png"):
        Image.new("RGB", (256, 256), color=(60, 120, 180)).save(buf, format="PNG")
        ct = "image/png"
    else:
        arr = np.random.randint(0, 255, size=(128, 128), dtype=np.uint8)
        tifffile.imwrite(buf, arr)
        ct = "image/tiff"

    boundary, body = create_multipart({"modality_hint": modality}, [("files", filename, buf.getvalue(), ct)])
    req = urllib.request.Request(
        f"{BASE_URL}/api/uploads",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read().decode())
        return data[0]["id"]

def run_phase4_verification():
    results = {}
    print("==================================================")
    print("SATQUERY AI - PHASE 4 FULL E2E ACCEPTANCE SUITE")
    print("==================================================")

    # Ingest test assets for E2E testing
    opt1_id = upload_sample_image("e2e_optical_scene1.png", "optical")
    opt2_id = upload_sample_image("e2e_optical_scene2.png", "optical")
    sar_id = upload_sample_image("e2e_sar_scene.tif", "sar")

    # 1. Model Registry Listing
    try:
        st, models = get_json(f"{BASE_URL}/api/models")
        passed_1 = (st == 200 and len(models) >= 5)
        tasks = {m["task"] for m in models}
        results[1] = log_test(1, "Model Registry Discovery", passed_1, f"{len(models)} specialist models discovered: {tasks}")
    except Exception as e:
        results[1] = log_test(1, "Model Registry Discovery", False, str(e))

    # 2. VQA Query Routing
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "Are there buildings in this image?",
            "image_ids": [opt1_id],
            "mode": "auto"
        })
        passed_2 = (st == 200 and res["task"] == "vqa" and res["status"] == "completed" and res["selected_model"]["id"] == "rs-vqa-base")
        results[2] = log_test(2, "VQA Task Routing", passed_2, f"Task={res['task']}, Model={res['selected_model']['id']}")
    except Exception as e:
        results[2] = log_test(2, "VQA Task Routing", False, str(e))

    # 3. Grounding Query Routing
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "Where are the water bodies?",
            "image_ids": [opt1_id],
            "mode": "auto"
        })
        passed_3 = (st == 200 and res["task"] == "grounding" and res["status"] == "completed" and res["selected_model"]["id"] == "rs-grounding-focal")
        results[3] = log_test(3, "Visual Grounding Routing", passed_3, f"Task={res['task']}, Model={res['selected_model']['id']}")
    except Exception as e:
        results[3] = log_test(3, "Visual Grounding Routing", False, str(e))

    # 4. Change Detection Query Routing (2 images)
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "What changed between these two images?",
            "image_ids": [opt1_id, opt2_id],
            "mode": "auto"
        })
        passed_4 = (st == 200 and res["task"] == "change_detection" and res["status"] == "completed" and res["selected_model"]["id"] == "rs-change-diff")
        results[4] = log_test(4, "Change Detection Routing", passed_4, f"Task={res['task']}, Model={res['selected_model']['id']}")
    except Exception as e:
        results[4] = log_test(4, "Change Detection Routing", False, str(e))

    # 5. Optical + SAR Cross-Modal Routing (1 optical + 1 SAR)
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "Compare the optical and SAR images.",
            "image_ids": [opt1_id, sar_id],
            "mode": "auto"
        })
        passed_5 = (st == 200 and res["task"] == "optical_sar" and res["status"] == "completed" and res["selected_model"]["id"] == "rs-optical-sar-fusion")
        results[5] = log_test(5, "Optical + SAR Cross-Modal Routing", passed_5, f"Task={res['task']}, Model={res['selected_model']['id']}")
    except Exception as e:
        results[5] = log_test(5, "Optical + SAR Cross-Modal Routing", False, str(e))

    # 6. Scene Captioning Routing
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "Describe this scene and its land cover distribution.",
            "image_ids": [opt1_id],
            "mode": "auto"
        })
        passed_6 = (st == 200 and res["task"] == "captioning" and res["status"] == "completed" and res["selected_model"]["id"] == "rs-captioning-landcover")
        results[6] = log_test(6, "Scene Captioning Routing", passed_6, f"Task={res['task']}, Model={res['selected_model']['id']}")
    except Exception as e:
        results[6] = log_test(6, "Scene Captioning Routing", False, str(e))

    # 7. Context Rejection: 1 image with Change Detection query
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "What changed between these two images?",
            "image_ids": [opt1_id],
            "mode": "auto"
        })
        passed_7 = (st == 200 and res["status"] == "incompatible" and "Two images are required" in res["error_message"])
        results[7] = log_test(7, "Single Image + Change Query Rejection", passed_7, f"Status={res['status']}, Reason='{res['error_message']}'")
    except Exception as e:
        results[7] = log_test(7, "Single Image + Change Query Rejection", False, str(e))

    # 8. Context Rejection: Optical+SAR query missing SAR image
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "Compare optical and SAR data.",
            "image_ids": [opt1_id, opt2_id],
            "mode": "auto"
        })
        passed_8 = (st == 200 and res["status"] == "incompatible" and "Missing SAR" in res["error_message"])
        results[8] = log_test(8, "Optical+SAR Missing SAR Rejection", passed_8, f"Status={res['status']}, Reason='{res['error_message']}'")
    except Exception as e:
        results[8] = log_test(8, "Optical+SAR Missing SAR Rejection", False, str(e))

    # 9. Unsupported / Out-of-Scope Query Rejection
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "Predict tomorrow's weather from this satellite image.",
            "image_ids": [opt1_id],
            "mode": "auto"
        })
        passed_9 = (st == 200 and res["status"] == "unsupported" and "outside SatQuery AI's supported analysis tasks" in res["error_message"])
        results[9] = log_test(9, "Unsupported Query Rejection", passed_9, f"Status={res['status']}, Reason='{res['error_message']}'")
    except Exception as e:
        results[9] = log_test(9, "Unsupported Query Rejection", False, str(e))

    # 10. Manual Mode Enforcement
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "Where are the roads?",
            "image_ids": [opt1_id],
            "mode": "change_detection"
        })
        passed_10 = (st == 200 and res["status"] == "incompatible" and res["task"] == "change_detection" and "Two images are required" in res["error_message"])
        results[10] = log_test(10, "Manual Mode Validation Enforcement", passed_10, f"Task={res['task']}, Status={res['status']}")
    except Exception as e:
        results[10] = log_test(10, "Manual Mode Validation Enforcement", False, str(e))

    # 11. Observable Execution Trace
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "Highlight the transportation network.",
            "image_ids": [opt1_id],
            "mode": "auto"
        })
        trace = res["execution_trace"]
        has_6_steps = len(trace) == 6
        all_completed = all(step["status"] == "completed" for step in trace)
        passed_11 = (st == 200 and has_6_steps and all_completed)
        results[11] = log_test(11, "Observable Execution Trace", passed_11, f"{len(trace)}/6 steps completed cleanly")
    except Exception as e:
        results[11] = log_test(11, "Observable Execution Trace", False, str(e))

    # 12. Demo Execution & Honest Confidence
    try:
        st, res = post_json(f"{BASE_URL}/api/analyze", {
            "query": "Are there buildings here?",
            "image_ids": [opt1_id],
            "mode": "auto"
        })
        is_demo_backend = (res["backend"] == "demo")
        no_fabricated_conf = (res["result"]["confidence"] is None)
        passed_12 = (st == 200 and is_demo_backend and no_fabricated_conf)
        results[12] = log_test(12, "Demo Marker & Honest Confidence", passed_12, f"Backend={res['backend']}, Real Confidence={res['result']['confidence']}")
    except Exception as e:
        results[12] = log_test(12, "Demo Marker & Honest Confidence", False, str(e))

    # 13. SQLite Analysis Persistence & GET /api/analysis/{id}
    try:
        analysis_id = res["analysis_id"]
        st, rec = get_json(f"{BASE_URL}/api/analysis/{analysis_id}")
        passed_13 = (st == 200 and rec["analysis_id"] == analysis_id and rec["detected_task"] == "vqa")
        results[13] = log_test(13, "Analysis Record Persistence", passed_13, f"Retrieved record ID={rec['analysis_id']} Status={rec['status']}")
    except Exception as e:
        results[13] = log_test(13, "Analysis Record Persistence", False, str(e))

    # 14. Frontend Proxy Integration
    try:
        st, proxy_models = get_json(f"{FRONTEND_URL}/api/models")
        passed_14 = (st == 200 and len(proxy_models) >= 5)
        results[14] = log_test(14, "Frontend to Backend Routing Proxy", passed_14, f"Vite proxy successfully routed /api/models -> {len(proxy_models)} models")
    except Exception as e:
        results[14] = log_test(14, "Frontend to Backend Routing Proxy", False, str(e))

    print("==================================================")
    all_passed = all(results.values())
    print(f"OVERALL STATUS: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"Passed: {sum(1 for v in results.values() if v)} / {len(results)}")
    print("==================================================")
    return all_passed

if __name__ == "__main__":
    import sys
    success = run_phase4_verification()
    if not success:
        sys.exit(1)
