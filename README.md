# SatQuery AI — Vision-Language Assistant for Satellite Imagery

**ISRO Problem Statement `SIH26167` | Space Technology Theme**

> *"Don't choose the model. Just ask the question."*

**SatQuery AI** is an intelligent vision-language assistant built for multimodal remote-sensing satellite image analysis. It bridges the gap between complex Earth observation rasters (Optical, Multispectral, and SAR) and natural-language understanding through an agentic task orchestration pipeline, authentic deep learning specialist models, and zero-fabrication visual evidence synthesis.

---

## 🌟 Key Highlights

- **Autonomous Agentic Orchestration**: Intelligently classifies user queries, validates sensor compatibility, selects capable specialist models, and executes inference with a visible 6-step trace.
- **5 Real Deep Learning Specialists**:
  1. **OWL-ViT (`google/owlvit-base-patch32`)**: Open-vocabulary visual grounding and bounding coordinate localization.
  2. **TinyCD (EfficientNet-B4 + MAMB)**: Dense bi-temporal surface change detection and pixel-level alteration overlays.
  3. **SmolVLM-256M-Instruct**: Open-domain remote-sensing visual question answering (VQA) with token probability confidence.
  4. **BigEarthNet-19 Multi-Spectral (MobileNetV3-Small)**: Automatic CORINE land-cover classification and scene captioning.
  5. **CrossModalOpticalSARNet**: Dual-stream spatial cross-attention fusion of optical reflectance and Sentinel-1 SAR microwave backscatter.
- **Zero-Fabrication Confidence Guarantees**: Confidences are strictly derived from mathematically validated model logits, token softmax probabilities, mask averages, or sensor correlations. No random or fake numbers.
- **Memory-Safe Infrastructure**: `ModelManager` singleton enforces a dynamic single-active LRU cache policy (`max_active_models=1`) with automatic CPU/CUDA auto-detection to prevent VRAM exhaustion.
- **Modern Interactive Workspace**: Built with React 19, TypeScript, Tailwind CSS, Lucide icons, and interactive side-by-side evidence inspection viewers.

---

## 🏗️ Architecture

```
User Query + Satellite Imagery (GeoTIFF / PNG / JPEG)
         │
         ▼
[Frontend Workspace] (React 19 + TypeScript + Tailwind CSS)
         │
         ▼  POST /api/analyze
[FastAPI Orchestrator Layer]
  ├── Step 1: Parse Query Intent
  ├── Step 2: Validate Image Context & Geometry
  ├── Step 3: Classify Analysis Task (Grounding / Change / VQA / Optical+SAR / Captioning)
  ├── Step 4: Model Registry Dispatch
  └── Step 5 & 6: Execute Specialist Inference & Synthesize Evidence
         │
  [ModelManager Singleton] (LRU Memory Safety, Device Auto-Detect)
  ├── OWL-ViT (Grounding)
  ├── TinyCD (Change Detection)
  ├── SmolVLM (Remote-Sensing VQA)
  ├── BigEarthNet-19 (Scene Captioning)
  └── CrossModalOpticalSARNet (Optical + SAR Dual-Stream)
         │
         ▼
[Evidence Synthesis & Database Persistence]
  ├── Visual Overlays & Masks (/static/processed/)
  ├── Calibrated Confidence Metrics
  └── SQLite Persistence (ImageAsset & AnalysisRecord)
```

---

## 🚀 Quickstart Guide

### Prerequisites
- **Python 3.10+** (Tested on Python 3.14)
- **Node.js 18+** & **npm**
- **Git**

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd SATQuery_AI_SIH
```

### 2. Backend Setup
```bash
# Install backend Python dependencies
pip install -r backend/requirements.txt

# Start the FastAPI server on port 8000
python -m uvicorn app.main:app --port 8000 --app-dir backend --host 127.0.0.1 --reload
```
API Documentation will be available at: `http://127.0.0.1:8000/docs`

### 3. Frontend Setup
In a new terminal:
```bash
cd frontend

# Install node dependencies
npm install

# Start Vite development server on port 5173
npm run dev -- --host 127.0.0.1 --port 5173
```
Open your browser at: `http://127.0.0.1:5173`

---

## 🧪 Testing & Verification

SatQuery AI includes an exhaustive test suite covering all units, adapters, and end-to-end integration flows:

### Run Full PyTest Suite (55 Tests)
```bash
pytest backend/tests/ -v
```

### Run Master Acceptance Verification (11/11 Suites)
```bash
python scripts/verify_final_e2e.py
```

### Run Frontend Production Build
```bash
cd frontend
npm run build
```

---

## 📁 Repository Structure

```
SATQuery_AI_SIH/
├── backend/
│   ├── app/
│   │   ├── adapters/            # 5 specialist model adapters
│   │   ├── api/                 # FastAPI routes (health, upload, validation, analysis)
│   │   ├── core/                # Configuration and constants
│   │   ├── db/                  # SQLite schema (ImageAsset, AnalysisRecord)
│   │   ├── models/              # Neural network implementations & ModelManager
│   │   ├── schemas/             # Pydantic validation schemas
│   │   └── services/            # Ingestion, TaskClassifier, SatQueryAgent
│   ├── data/                    # Weights, demo data, uploads, and processed outputs
│   ├── tests/                   # 55 automated unit and integration tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/          # UI components (chat, upload, analysis viewers)
│   │   ├── views/               # MainWorkspaceView, NewAnalysisView, HistoryView
│   │   ├── services/            # API client and session management
│   │   └── types/               # TypeScript interfaces
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   ├── ARCHITECTURE.md          # In-depth architectural design specification
│   └── FINAL_DEMO_CHECKLIST.md  # Jury presentation demo guide
├── scripts/
│   ├── verify_final_e2e.py      # Master acceptance verification script
│   └── download_all_models.py   # Automated model weight downloader
├── .gitignore
└── README.md
```

---

## 🛡️ License

Developed for the **Smart India Hackathon (SIH 2026)** under the **Indian Space Research Organisation (ISRO)** problem statement `SIH26167`.
