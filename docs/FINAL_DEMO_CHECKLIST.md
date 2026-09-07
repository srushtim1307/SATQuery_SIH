# SatQuery AI — Final SIH Demo Checklist & Acceptance Document

**Project**: SatQuery AI (ISRO Problem Statement `SIH26167`, Space Technology Theme)  
**Objective**: Interactive Vision-Language Assistant for Multimodal Remote-Sensing Satellite Image Analysis  
**Philosophy**: *"Don't choose the model. Just ask the question."*

---

## 1. Problem Statement & Mandatory Requirements Mapping

| SIH Requirement | Requirement Description | SatQuery AI Implementation | Status |
|---|---|---|---|
| **REQ-01** | Multi-Modal Satellite Ingestion | Supports GeoTIFF (.tif, .tiff), PNG, and JPEG. Header-only metadata extraction, format verification, and web preview generation. | **VERIFIED** |
| **REQ-02** | Natural Language Query Interface | ChatGPT-style clean conversational interface. Users type free-form queries without manually picking models or setting thresholds. | **VERIFIED** |
| **REQ-03** | Autonomous Task Classification & Routing | `TaskClassifier` dynamically classifies intent into VQA, Grounding, Change Detection, Captioning, or Optical+SAR fusion. | **VERIFIED** |
| **REQ-04** | Real Deep Learning Vision-Language Models | Integration of real neural architectures: OWL-ViT (Grounding), TinyCD (Change Detection), SmolVLM (VQA), BigEarthNet-19 (Captioning), CrossModalOpticalSARNet (Fusion). | **VERIFIED** |
| **REQ-05** | Observable Step-by-Step Execution Trace | Trace records 6 observable steps (`Understanding question` $\to$ `Checking compatibility` $\to$ `Identifying task` $\to$ `Selecting specialist` $\to$ `Analyzing` $\to$ `Preparing evidence`). | **VERIFIED** |
| **REQ-06** | Visual Grounding & Feature Localization | OWL-ViT open-vocabulary detector with normalized bounding box coordinates and class query confidence scores. | **VERIFIED** |
| **REQ-07** | Dense Bi-Temporal Change Detection | TinyCD neural change detection with dense change masks, alteration percentages, and coral-red overlay synthesis. | **VERIFIED** |
| **REQ-08** | Multi-Sensor Optical + SAR Fusion | Cross-attention network fusing multispectral reflectance and C-band microwave backscatter with physical dB analytics and edge correlation. | **VERIFIED** |
| **REQ-09** | Zero-Fabrication Metric Integrity | Strictly derives confidence scores from raw logits, mask probabilities, or token softmax. Demo outputs explicitly marked with confidence `null`. | **VERIFIED** |
| **REQ-10** | Database Persistence & Audit Retrieval | SQLite persistence via SQLAlchemy (`AnalysisRecord`). Full historical analysis dialogues restorable via `GET /api/history` and the History tab. | **VERIFIED** |

---

## 2. Model & Specialist Inventory

| Capability | Specialist Model ID | Architecture | Model Type | Checkpoint / Weights | Confidence Source |
|---|---|---|---|---|---|
| **Change Detection** | `rs-change-diff` | Sliced EfficientNet-B4 + MAMB + UpMask | `real` | `HZDR-FWGEL/UCD-MNCD256-TinyCD` (1.15 MB safetensors) | Mean sigmoid change mask pixel probability |
| **Visual Grounding** | `rs-grounding-focal` | Vision-Language Dual Transformer | `real` | `google/owlvit-base-patch32` (584 MB safetensors) | Sigmoid logit activation score |
| **Remote-Sensing VQA** | `rs-vqa-specialist` | SmolVLM (Idefics3 Vision-Language) | `real` | `HuggingFaceTB/SmolVLM-256M-Instruct` (489 MB safetensors) | Mean softmax probability of generated tokens |
| **Scene Captioning** | `rs-captioning-landcover` | MobileNetV3-Small (CORINE-19 Adapted) | `adapted` | `bigearthnet_adapted.pth` (5.99 MB, 92.89% val acc) | Sigmoid top-class activation probability |
| **Optical + SAR Fusion** | `rs-optical-sar-fusion` | CrossModalOpticalSARNet | `real` | `optical_sar_fusion.pth` (1.25 MB, 317k params) | Sobel edge correlation + Cross-attention coherence |

---

## 3. Remote Sensing Benchmark Status

Accessible programmatically via `GET /api/benchmarks`:

1. **BigEarthNet-19**:
   - **Status**: `evaluated`
   - **Metrics**: Validation Accuracy: 92.89%, Final Loss: 0.1604 across 5 training epochs.
   - **Source**: `backend/models/weights/training_log.json`.
2. **VRSBench**:
   - **Status**: `not_executed` ("Evaluation not yet executed. Grounding specialist operates using zero-shot vision-text embeddings.")
3. **RSVQA**:
   - **Status**: `not_executed` ("Evaluation not yet executed. Specialist executes autoregressive multi-modal generation.")
4. **CDVQA**:
   - **Status**: `not_executed` ("Evaluation not yet executed. Inference utilizes pretrained weights from HZDR-FWGEL/UCD-MNCD256-TinyCD.")

---

## 4. Known Limitations & Operating Environment

- **Python 3.14 Environment**:
  - Running on Windows 11 with Python 3.14.7.
  - Official PyTorch CUDA cp314 wheels are not yet distributed on PyPI for Windows; PyTorch executes in CPU mode.
  - Efficient architectures were chosen specifically so inference runs comfortably within seconds on CPU:
    - TinyCD: ~0.90s
    - BigEarthNet-19: ~0.35s
    - Optical+SAR Fusion: ~0.82s
    - OWL-ViT: ~5.2s
    - SmolVLM-256M: ~22-25s
- **Hardware Profile**:
  - NVIDIA RTX 3050 Laptop GPU (6 GB VRAM) supported; if CUDA is enabled via environment override `SATQUERY_DEVICE=cuda`, `ModelManager` leverages GPU acceleration automatically.
- **Single-Active LRU Policy**:
  - `ModelManager` enforces `max_active_models=1` to prevent RAM/VRAM exhaustion during multi-task sessions.

---

## 5. Exact Setup & Launch Instructions

### Terminal 1 — FastAPI Backend
```bash
# Navigate to project root
cd C:\Users\Srushti\OneDrive\Desktop\SATQuery_AI_SIH

# Start backend server on port 8000
python -m uvicorn app.main:app --port 8000 --app-dir backend --reload
```
- API Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/api/health`

### Terminal 2 — React Frontend
```bash
# Navigate to frontend directory
cd C:\Users\Srushti\OneDrive\Desktop\SATQuery_AI_SIH\frontend

# Start Vite dev server on port 5173
npm run dev -- --host 127.0.0.1 --port 5173
```
- Web Application: `http://127.0.0.1:5173`

---

## 6. Recommended 3-Minute Jury Demo Flow

### Minute 1: Introduction & Philosophical Proposition (30s)
1. Open `http://127.0.0.1:5173`.
2. Present the core idea: *"Traditional Earth Observation tools force operators to pick models, tune confidence thresholds, and write custom scripts. SatQuery AI enables: 'Don't choose the model. Just ask the question.'"*
3. Highlight the clean ChatGPT-style workspace, desert beige/navy palette, and Auto Detect mode.

### Minute 2: The Three Core Live Jury Demos (1.5 min)

#### Test 1 — Visual Grounding & Localization
- **Action**: Click **New Analysis**, upload an optical satellite scene containing coastal water/vegetation (or select suggestion).
- **Query**: `"Where are the water bodies in this image?"`
- **What to show the Jury**:
  - System trace opens: `Identifying analysis task` $\to$ `Task identified: Grounding`.
  - Specialist selected: `rs-grounding-focal (real)`.
  - Interactive bounding boxes drawn over water bodies with detector confidence (e.g. 88%).
  - Bounding box toggle (Overlays On/Off) and Full Frame mode.

#### Test 2 — Bi-Temporal Change Detection
- **Action**: Stage 2 images (T1 baseline and T2 current).
- **Query**: `"What changed between these two images?"`
- **What to show the Jury**:
  - System trace: routes to `change_detection` with `TinyCD Specialist (real)`.
  - Side-by-side comparison with coral-red neural change mask overlay.
  - Mathematical change percentage (e.g. 17.6%) and region count.
  - Zero fabricated numbers; confidence derived from change mask probabilities.

#### Test 3 — Optical + SAR Cross-Modal Synergy
- **Action**: Stage 1 optical multispectral image and 1 Sentinel-1 SAR image.
- **Query**: `"Identify built-up and water-covered regions using both images."`
- **What to show the Jury**:
  - System trace: routes to `optical_sar` with `CrossModalOpticalSARNet (real)`.
  - Dual cards showing Optical visible/NIR alongside SAR microwave radar backscatter.
  - Calibrated SAR backscatter metrics (mean -4.5 dB), identifying specular water reflection and urban double-bounce.
  - Structural edge correlation ($r_{\text{edge}}$) confirming cross-modal alignment.

### Minute 3: Auditability, History & Architecture (1 min)
1. Navigate to the **History** tab (`/history`).
2. Show the persistent session log: all previous queries are saved in SQLite with task badges and real/adapted backend indicators.
3. Click any past session to demonstrate full restoration of the conversation, answers, evidence masks, and execution trace.
4. Show the observable trace: pure engineering actions (`Task identified`, `Inputs verified`, `Specialist selected`) without exposing hidden reasoning.
5. Emphasize SIH compliance: zero fake metrics, graceful error handling for corrupted/invalid files, and single-active LRU memory safety.

---

## 7. Backup / Offline Procedure

If internet connectivity to HuggingFace is interrupted during the presentation:
1. All model checkpoints (`optical_sar_fusion.pth`, `bigearthnet_adapted.pth`) are stored **locally** on disk in `backend/models/weights/`.
2. Pre-cached HuggingFace models (`TinyCD`, `OWL-ViT`, `SmolVLM`) are already stored in the local `.cache/huggingface` hub and run 100% offline.
3. If an unforeseen file I/O error occurs, the built-in deterministic demo fallback engine ensures that the application never displays an unhandled HTTP 500 error.
