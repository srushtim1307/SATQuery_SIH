export interface UploadedAsset {
  id: string;
  session_id?: string | null;
  filename: string;
  format: string;
  modality: 'optical' | 'sar' | 'multispectral' | 'unknown';
  width: number | null;
  height: number | null;
  bands: number;
  crs: string | null;
  resolution: string | null;
  acquisition_date: string | null;
  preview_url: string | null;
  status: string;
  warnings: string[];
  metadata: Record<string, any>;
  created_at?: string;
}

export interface ValidateResponse {
  valid: boolean;
  mode: string;
  count: number;
  images: UploadedAsset[];
  warnings: string[];
  errors: string[];
}

export async function uploadSatelliteImages(
  files: File[], 
  sessionId?: string,
  modalityHint?: string
): Promise<UploadedAsset[]> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });
  if (sessionId) {
    formData.append('session_id', sessionId);
  }
  if (modalityHint) {
    formData.append('modality_hint', modalityHint);
  }

  const res = await fetch('/api/uploads', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    if (res.status === 502 || res.status === 504) {
      throw new Error("Cannot connect to backend server on port 8000. Please ensure 'python -m uvicorn app.main:app --port 8000 --app-dir backend --reload' is running.");
    }
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || `Upload failed with status ${res.status}`);
  }

  return await res.json();
}

export async function validateUploadedInputs(
  imageIds: string[], 
  mode: string
): Promise<ValidateResponse> {
  const res = await fetch('/api/validate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_ids: imageIds,
      mode: mode,
    }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || `Validation failed with status ${res.status}`);
  }

  return await res.json();
}

export async function deleteUploadedAsset(assetId: string): Promise<void> {
  const res = await fetch(`/api/uploads/${assetId}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    console.warn(`Failed to delete asset ${assetId}: ${res.status}`);
  }
}

export interface AnalysisRequestPayload {
  query: string;
  image_ids: string[];
  mode?: string;
  session_id?: string;
}

export interface TraceStepPayload {
  name: string;
  status: 'completed' | 'in_progress' | 'pending' | 'failed';
  detail?: string;
}

export interface AnalysisResponsePayload {
  analysis_id: string;
  query: string;
  task: string;
  selected_model?: {
    id: string;
    name: string;
    version: string;
    task: string;
    backend: string;
  };
  routing?: {
    task: string;
    specialist: string;
    confidence: number;
    reason: string;
    required_inputs: string[];
    compatible: boolean;
  };
  execution_trace: TraceStepPayload[];
  result: Record<string, any>;
  status: 'completed' | 'incompatible' | 'unsupported' | 'failed';
  error_message?: string;
  backend: string;
}

export async function runAnalysis(payload: AnalysisRequestPayload): Promise<AnalysisResponsePayload> {
  const res = await fetch('/api/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errData.detail || `Analysis failed with status ${res.status}`);
  }

  return await res.json();
}

export async function getAnalysisRecord(analysisId: string): Promise<AnalysisResponsePayload> {
  const res = await fetch(`/api/analysis/${analysisId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch analysis record with status ${res.status}`);
  }
  return await res.json();
}

export interface PersistedAnalysisRecord extends AnalysisResponsePayload {
  detected_task?: string;
  input_asset_ids?: string[];
  input_assets?: Array<{
    id: string;
    filename: string;
    preview_url?: string;
    modality?: string;
    resolution?: string;
  }>;
  created_at?: string;
}

export async function getAnalysisHistory(): Promise<PersistedAnalysisRecord[]> {
  try {
    const res = await fetch('/api/history');
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('Could not load analysis history:', err);
    return [];
  }
}

