import type { AnalysisSession, StagedFile } from '../types';

export const INITIAL_SESSIONS: AnalysisSession[] = [
  {
    id: 'water-body-detection',
    title: 'Water Body Identification',
    query: 'Where are the water bodies in this image?',
    analysisType: 'VQA + Grounding',
    modalityBadge: 'S2',
    imageCount: 1,
    date: 'Sept 6, 2026',
    confidence: 0.92,
    pipeline: 'VQA + Grounding',
    format: 'Sentinel-2 Multispectral',
    thumbnail: '/demo/thumb_water_body.png',
    userMessage: {
      text: 'Where are the water bodies in this image?',
      timestamp: '10:42 AM',
      attachments: [
        {
          name: 'sentinel2_estuary_delta.tif',
          detail: 'Optical · Multispectral',
          status: 'Ready',
          modality: 'Multispectral'
        }
      ]
    },
    assistantResponse: {
      text: 'Two major water bodies are visible in the image. The larger water body is located in the upper-right region, while a smaller water body appears near the lower-left area. Multispectral signature analysis flags active tidal sediment outflow in the central estuary mouth.',
      evidenceType: 'single_grounding',
      evidence: {
        imageUrl: '/demo/sentinel2_estuary_delta.png',
        scaleLabel: '10 km',
        satelliteLabel: 'SENTINEL-2 L1C',
        boundingBoxes: [
          {
            id: 'box-1',
            label: 'Primary Estuary Water Body #1',
            confidence: 0.94,
            top: 28,
            left: 58,
            width: 30,
            height: 35
          },
          {
            id: 'box-2',
            label: 'Tributary Channel #2',
            confidence: 0.91,
            top: 42,
            left: 24,
            width: 25,
            height: 28
          }
        ]
      },
      traceSteps: [
        'Query understood: Water body localization',
        'Input validated: 1 Multispectral GeoTIFF image',
        'Task detected: Visual Grounding',
        'Specialist selected: Remote-Sensing Grounding Model',
        'Analysis completed: Spectral water-index applied',
        'Visual evidence prepared: Bounding boxes calculated'
      ],
      detectedTask: 'Visual Grounding',
      inputDescription: 'Sentinel-2 Multispectral Scene (10m GSD)',
      selectedModel: 'Remote-Sensing Grounding Model'
    }
  },
  {
    id: 'urban-expansion',
    title: 'Urban Expansion Analysis',
    query: 'What changed between these two images?',
    analysisType: 'Change Detection',
    modalityBadge: 'T1/T2',
    imageCount: 2,
    date: 'Sept 7, 2026',
    confidence: 0.89,
    pipeline: 'Change Detection Module • Multi-Temporal Radial Analysis',
    format: 'Optical 10m GSD',
    thumbnail: '/demo/thumb_urban_expansion.png',
    userMessage: {
      text: 'What changed between these two images?',
      timestamp: '09:15 AM',
      attachments: [
        {
          name: 'sentinel2_may2023.tif',
          detail: 'Optical · Baseline · Ready',
          status: 'Ready',
          modality: 'Optical'
        },
        {
          name: 'sentinel2_oct2024.tif',
          detail: 'Optical · Current · Ready',
          status: 'Ready',
          modality: 'Optical'
        }
      ]
    },
    assistantResponse: {
      text: 'Built-up urban area increased by 14.2% in the eastern sector. Significant arterial infrastructure expansion radiating from the metropolitan beltway outward into agricultural fringes.',
      evidenceType: 'change_comparison',
      evidence: {
        beforeImageUrl: '/demo/sentinel2_may2023.png',
        afterImageUrl: '/demo/sentinel2_oct2024.png',
        beforeLabel: 'REFERENCE ACQUISITION',
        beforeDate: 'May 2023 (T1)',
        beforeGsd: 'Optical 10m GSD',
        afterLabel: 'RECENT EVALUATION',
        afterDate: 'Oct 2024 (T2)',
        afterOverlayTag: '+14.2% Impervious Mask'
      },
      traceSteps: [
        'Query understood: Bi-temporal surface diff query',
        'Input validated: 2 spatially corresponding optical rasters',
        'Task detected: Bi-Temporal Change Detection',
        'Specialist selected: Remote-Sensing Change Model',
        'Analysis completed: Coregistered pixel difference computed',
        'Visual evidence prepared: Impervious growth mask overlaid'
      ],
      detectedTask: 'Change Detection',
      inputDescription: 'Bi-temporal satellite imagery (T1 May 2023 / T2 Oct 2024)',
      selectedModel: 'Remote-Sensing Change Model'
    }
  },
  {
    id: 'optical-sar-analysis',
    title: 'Optical + SAR Analysis',
    query: 'Identify built-up and water-covered regions.',
    analysisType: 'Optical + SAR',
    modalityBadge: 'SAR',
    imageCount: 2,
    date: 'Sept 5, 2026',
    confidence: 0.94,
    pipeline: 'Multi-Sensor Co-Registration & Fusion',
    format: 'Optical 10m + SAR C-Band',
    thumbnail: '/demo/thumb_optical_sar.png',
    userMessage: {
      text: 'Use the optical and SAR images together to identify built-up and water-covered regions.',
      timestamp: '02:30 PM',
      attachments: [
        {
          name: 'sentinel2_estuary_delta.tif',
          detail: 'Optical · Sentinel-2 L1C · Ready',
          status: 'Ready',
          modality: 'Optical'
        },
        {
          name: 'gulf_coastal_sar.tif',
          detail: 'SAR · Sentinel-1 C-Band · Ready',
          status: 'Ready',
          modality: 'SAR'
        }
      ]
    },
    assistantResponse: {
      text: 'Using both optical and SAR imagery, the system identifies built-up regions and water-covered areas. SAR backscatter provides radar penetration to isolate rough high-reflectance urban geometry from specular calm water bodies, providing high confidence regardless of haze.',
      evidenceType: 'optical_sar',
      evidence: {
        opticalImageUrl: '/demo/sentinel2_estuary_delta.png',
        sarImageUrl: '/demo/thumb_optical_sar.png',
        opticalLabel: 'SENTINEL-2 OPTICAL (10M)',
        sarLabel: 'SENTINEL-1 SAR (C-BAND VV/VH)',
        fusionSummary: 'Cross-modal dual-sensor fusion confirms 88% spatial correlation between optical reflectance and SAR dielectric roughness.'
      },
      traceSteps: [
        'Query understood: Dual-sensor joint analysis query',
        'Input validated: Co-registered Optical + SAR pair present',
        'Task detected: Optical-SAR Multi-Sensor Fusion',
        'Specialist selected: Optical + SAR Specialist Adapter',
        'Analysis completed: Combined spectral & backscatter signatures',
        'Visual evidence prepared: Dual-modality comparison rendered'
      ],
      detectedTask: 'Optical + SAR Fusion',
      inputDescription: 'Cross-Modal Pair: Optical (Sentinel-2) + SAR (Sentinel-1)',
      selectedModel: 'Optical + SAR Fusion Adapter'
    }
  },
  {
    id: 'agri-land-use',
    title: 'Agricultural Land Use Shift',
    query: 'Quantify crop canopy boundary differences between harvest cycles.',
    analysisType: 'Change Detection',
    modalityBadge: 'NDVI',
    imageCount: 2,
    date: 'Aug 29, 2026',
    confidence: 0.88,
    pipeline: 'Spectral Biomass Variance Tracker',
    format: 'Multi-Temporal NDVI 10m',
    thumbnail: '/demo/thumb_agri_ndvi.png',
    userMessage: {
      text: 'Quantify crop canopy boundary differences between harvest cycles.',
      timestamp: '04:10 PM',
      attachments: [
        {
          name: 'crop_canopy_season1.tif',
          detail: 'Multispectral · Baseline · Ready',
          status: 'Ready',
          modality: 'Multispectral'
        },
        {
          name: 'crop_canopy_season2.tif',
          detail: 'Multispectral · Current · Ready',
          status: 'Ready',
          modality: 'Multispectral'
        }
      ]
    },
    assistantResponse: {
      text: 'Biomass index analysis shows a seasonal shift in vegetative canopy density of 18.5% across the northern cultivation parcels following post-monsoon harvest rotation.',
      evidenceType: 'change_comparison',
      evidence: {
        beforeImageUrl: '/demo/thumb_agri_ndvi.png',
        afterImageUrl: '/demo/thumb_urban_expansion.png',
        beforeLabel: 'HARVEST CYCLE 1',
        beforeDate: 'Jun 2026 (T1)',
        beforeGsd: 'NDVI 10m',
        afterLabel: 'HARVEST CYCLE 2',
        afterDate: 'Aug 2026 (T2)',
        afterOverlayTag: 'Vegetation Canopy Shift'
      },
      traceSteps: [
        'Query understood: Crop canopy variance request',
        'Input validated: 2 bi-temporal multispectral scenes',
        'Task detected: Vegetation Change Detection',
        'Specialist selected: Remote-Sensing Change Model',
        'Analysis completed: NDVI differential calculated',
        'Visual evidence prepared: Canopy shift boundary rendered'
      ],
      detectedTask: 'Change Detection',
      inputDescription: 'Bi-temporal Multispectral Imagery (Canopy baseline & harvest)',
      selectedModel: 'Remote-Sensing Change Model'
    }
  },
  {
    id: 'coastal-flood',
    title: 'Coastal Flood Risk Assessment',
    query: 'Detect submerged infrastructure along the estuary delta.',
    analysisType: 'Grounding',
    modalityBadge: 'DEM',
    imageCount: 1,
    date: 'Aug 24, 2026',
    confidence: 0.91,
    pipeline: 'Topographic Hydrological Grounding',
    format: 'DEM + Multispectral 10m',
    thumbnail: '/demo/thumb_coastal_dem.png',
    userMessage: {
      text: 'Detect submerged infrastructure along the estuary delta.',
      timestamp: '11:05 AM',
      attachments: [
        {
          name: 'coastal_estuary_dem.tif',
          detail: 'DEM · Elevation Grid · Ready',
          status: 'Ready',
          modality: 'DEM'
        }
      ]
    },
    assistantResponse: {
      text: 'Grounding identified 3 low-lying drainage culverts and coastal arterial causeways vulnerable to high-tide submergence along the central estuary bank.',
      evidenceType: 'single_grounding',
      evidence: {
        imageUrl: '/demo/sentinel2_estuary_delta.png',
        scaleLabel: '5 km',
        satelliteLabel: 'DEM + SENTINEL-2',
        boundingBoxes: [
          {
            id: 'flood-box-1',
            label: 'Submerged Causeway #1',
            confidence: 0.93,
            top: 35,
            left: 30,
            width: 20,
            height: 20
          },
          {
            id: 'flood-box-2',
            label: 'Low-Lying Drainage Culvert #2',
            confidence: 0.89,
            top: 55,
            left: 50,
            width: 25,
            height: 22
          }
        ]
      },
      traceSteps: [
        'Query understood: Inundation and submerged infrastructure search',
        'Input validated: 1 Coastal DEM / Optical scene',
        'Task detected: Visual Grounding & Topography',
        'Specialist selected: Remote-Sensing Grounding Model',
        'Analysis completed: Low-lying elevation masking applied',
        'Visual evidence prepared: Vulnerable infrastructure bounded'
      ],
      detectedTask: 'Visual Grounding',
      inputDescription: 'Coastal Estuary DEM + Optical Scene',
      selectedModel: 'Remote-Sensing Grounding Model'
    }
  }
];

export const INITIAL_STAGED_FILES: StagedFile[] = [];

export const SUGGESTION_CHIPS = [
  { label: 'Describe this scene', icon: 'Sparkles', query: 'Describe the land-cover and major features visible in this image.' },
  { label: 'Where are the water bodies?', icon: 'Droplets', query: 'Where are the water bodies in this image?' },
  { label: 'What changed between these images?', icon: 'Clock', query: 'What changed between these two images?' },
  { label: 'Identify built-up areas', icon: 'Building2', query: 'Identify built-up and developed urban areas.' },
  { label: 'Compare optical and SAR imagery', icon: 'Layers', query: 'Use the optical and SAR images together to identify built-up and water-covered regions.' }
];
