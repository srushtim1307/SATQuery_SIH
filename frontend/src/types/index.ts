export type AnalysisType = 
  | 'VQA' 
  | 'Grounding' 
  | 'VQA + Grounding' 
  | 'Change Detection' 
  | 'Optical + SAR' 
  | 'Captioning';

export interface ImageAttachment {
  name: string;
  detail: string;
  status: 'Ready' | 'Uploading' | 'Processing';
  thumbnailUrl?: string;
  size?: string;
  modality?: 'Optical' | 'SAR' | 'Multispectral' | 'DEM' | 'unknown';
  assetId?: string;
}

export interface BoundingBox {
  id: string;
  label: string;
  confidence: number;
  top: number;
  left: number;
  width: number;
  height: number;
}

export interface SingleGroundingEvidence {
  imageUrl: string;
  scaleLabel: string;
  satelliteLabel: string;
  boundingBoxes: BoundingBox[];
}

export interface ChangeComparisonEvidence {
  beforeImageUrl: string;
  afterImageUrl: string;
  beforeLabel: string;
  beforeDate: string;
  beforeGsd: string;
  afterLabel: string;
  afterDate: string;
  afterOverlayTag: string;
}

export interface OpticalSarEvidence {
  opticalImageUrl: string;
  sarImageUrl: string;
  opticalLabel: string;
  sarLabel: string;
  fusionSummary: string;
}

export interface AnalysisSession {
  id: string;
  title: string;
  query: string;
  analysisType: AnalysisType;
  modalityBadge: string;
  imageCount: number;
  date: string;
  confidence: number | null;
  pipeline: string;
  format: string;
  thumbnail: string;
  backend?: string;
  userMessage: {

    text: string;
    timestamp?: string;
    attachments: ImageAttachment[];
  };
  assistantResponse: {
    text: string;
    evidenceType: 'single_grounding' | 'change_comparison' | 'optical_sar';
    evidence: SingleGroundingEvidence | ChangeComparisonEvidence | OpticalSarEvidence;
    traceSteps: string[];
    detectedTask: string;
    inputDescription: string;
    selectedModel: string;
  };
}

export interface StagedFile {
  id: string;
  name: string;
  size: string;
  modalityTag: string;
  thumbnailUrl: string;
  ready: boolean;
  format?: string;
  width?: number | null;
  height?: number | null;
  bands?: number;
  crs?: string | null;
  resolution?: string | null;
  modality?: 'optical' | 'sar' | 'multispectral' | 'unknown';
  warnings?: string[];
}
