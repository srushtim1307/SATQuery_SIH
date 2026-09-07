import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';
import { TopHeader } from './components/layout/TopHeader';
import { SettingsModal } from './components/settings/SettingsModal';
import { MainWorkspaceView } from './views/MainWorkspaceView';
import { NewAnalysisView } from './views/NewAnalysisView';
import { HistoryView } from './views/HistoryView';
import { INITIAL_SESSIONS, INITIAL_STAGED_FILES } from './services/demoData';
import { runAnalysis, getAnalysisHistory } from './services/api';
import type { PersistedAnalysisRecord } from './services/api';
import type { AnalysisSession, ImageAttachment, StagedFile } from './types';

function mapRecordToSession(record: PersistedAnalysisRecord): AnalysisSession {
  const task = record.detected_task || record.task || 'general';
  let analysisType: any = 'Scene Captioning';
  let evidenceType: 'single_grounding' | 'change_comparison' | 'optical_sar' = 'single_grounding';
  let modalityBadge = 'OPTICAL';
  
  if (task === 'change_detection') {
    analysisType = 'Bi-Temporal Change';
    evidenceType = 'change_comparison';
    modalityBadge = 'BI-TEMPORAL';
  } else if (task === 'grounding') {
    analysisType = 'Visual Grounding';
    evidenceType = 'single_grounding';
    modalityBadge = 'OPTICAL';
  } else if (task === 'vqa') {
    analysisType = 'Visual Question Answering (VQA)';
    evidenceType = 'single_grounding';
    modalityBadge = 'MULTISPECTRAL';
  } else if (task === 'optical_sar') {
    analysisType = 'Optical + SAR Fusion';
    evidenceType = 'optical_sar';
    modalityBadge = 'DUAL-SENSOR';
  } else if (task === 'captioning') {
    analysisType = 'Scene Description';
    evidenceType = 'single_grounding';
    modalityBadge = 'MULTISPECTRAL';
  }

  const firstAsset = record.input_assets?.[0];
  const secondAsset = record.input_assets?.[1];
  const thumbnail = firstAsset?.preview_url || '/demo/sentinel2_estuary_delta.png';

  const attachments: ImageAttachment[] = (record.input_assets && record.input_assets.length > 0)
    ? record.input_assets.map(a => ({
        name: a.filename,
        detail: `${a.modality?.toUpperCase() || 'OPTICAL'} · ${a.resolution || '10m GSD'}`,
        status: 'Ready' as const,
        modality: (a.modality === 'sar' ? 'SAR' : 'Optical') as any,
        thumbnailUrl: a.preview_url,
        assetId: a.id
      }))
    : [{
        name: 'satellite_raster.tif',
        detail: 'Optical · 10m GSD',
        status: 'Ready',
        modality: 'Optical',
        thumbnailUrl: thumbnail
      }];

  const firstEvidence = record.result?.evidence?.[0] as any;
  let evidence: any = {
    imageUrl: thumbnail,
    scaleLabel: '10 km',
    satelliteLabel: 'ORBITAL EO SENSOR',
    boundingBoxes: []
  };

  if (task === 'change_detection' && firstEvidence) {
    evidence = {
      beforeImageUrl: firstEvidence.beforeImageUrl || thumbnail,
      afterImageUrl: firstEvidence.afterImageUrl || firstEvidence.overlay_url || thumbnail,
      beforeLabel: firstEvidence.beforeLabel || 'T1 Baseline',
      beforeDate: firstEvidence.beforeDate || 'Baseline Acquisition',
      beforeGsd: firstEvidence.beforeGsd || '10m GSD',
      afterLabel: firstEvidence.afterLabel || 'T2 Current',
      afterDate: firstEvidence.afterDate || 'Recent Acquisition',
      afterOverlayTag: firstEvidence.afterOverlayTag || `TinyCD Mask (${firstEvidence.change_percentage ?? 0}%)`
    };
  } else if (task === 'grounding' && firstEvidence) {
    evidence = {
      imageUrl: firstEvidence.imageUrl || thumbnail,
      scaleLabel: firstEvidence.scaleLabel || '10m GSD',
      satelliteLabel: firstEvidence.satelliteLabel || 'Optical Imagery',
      boundingBoxes: firstEvidence.boundingBoxes || firstEvidence.bounding_boxes || []
    };
  } else if (task === 'optical_sar' && firstEvidence) {
    evidence = {
      opticalImageUrl: firstEvidence.opticalImageUrl || thumbnail,
      sarImageUrl: firstEvidence.sarImageUrl || secondAsset?.preview_url || thumbnail,
      opticalLabel: firstEvidence.opticalLabel || 'Multispectral Optical',
      sarLabel: firstEvidence.sarLabel || 'SAR Microwave Backscatter',
      fusionSummary: firstEvidence.fusionSummary || 'Cross-modal neural fusion completed.'
    };
  }

  const traceSteps = Array.isArray(record.execution_trace)
    ? record.execution_trace.map((t: any) => `${t.name}: ${t.status}${t.detail ? ' — ' + t.detail : ''}`)
    : [];

  const createdDate = record.created_at
    ? new Date(record.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    : 'Recent Analysis';

  return {
    id: record.analysis_id,
    title: record.query.slice(0, 32) + (record.query.length > 32 ? '...' : ''),
    query: record.query,
    analysisType,
    modalityBadge,
    imageCount: record.input_asset_ids?.length || 1,
    date: createdDate,
    confidence: record.result?.confidence ?? null,
    backend: record.backend || 'demo',
    pipeline: record.selected_model?.id || 'Auto-Routed',
    format: 'GeoTIFF / PNG',
    thumbnail,
    userMessage: {
      text: record.query,
      timestamp: record.created_at || undefined,
      attachments
    },
    assistantResponse: {
      text: record.result?.answer || record.error_message || 'Analysis completed.',
      evidenceType,
      evidence,
      traceSteps,
      detectedTask: task.toUpperCase(),
      inputDescription: attachments.map(a => a.name).join(', '),
      selectedModel: record.selected_model?.id || 'Auto-Routed Specialist'
    }
  };
}

function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();

  // Sessions state
  const [sessions, setSessions] = useState<AnalysisSession[]>(INITIAL_SESSIONS);
  const [stagedFiles, setStagedFiles] = useState<StagedFile[]>(INITIAL_STAGED_FILES);

  // Load real persisted analysis records from database history on mount
  useEffect(() => {
    getAnalysisHistory().then((historyRecords) => {
      if (historyRecords && historyRecords.length > 0) {
        const persistedSessions = historyRecords.map(mapRecordToSession);
        setSessions((prev) => {
          const existingIds = new Set(persistedSessions.map(s => s.id));
          const nonDupeInitial = prev.filter(s => !existingIds.has(s.id));
          return [...persistedSessions, ...nonDupeInitial];
        });
      }
    });
  }, []);
  
  // Settings & Navigation state
  const [selectedMode, setSelectedMode] = useState('Auto Detect (Recommended)');
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [showEvidence, setShowEvidence] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);

  // Extract active session ID if on /analysis/:id
  const match = location.pathname.match(/^\/analysis\/([^/]+)/);
  const activeSessionId = match ? match[1] : undefined;
  const activeSession = sessions.find((s) => s.id === activeSessionId);

  // Dynamic breadcrumb
  let breadcrumbSub = 'Vision-Language Assistant for Satellite Imagery';
  if (location.pathname === '/new-analysis') {
    breadcrumbSub = 'Start a new remote-sensing analysis';
  } else if (location.pathname === '/history') {
    breadcrumbSub = 'Historical remote-sensing analyses';
  }

  // Handle adding query from chat input
  const handleAddNewQuery = (query: string, attachments: ImageAttachment[]): string => {
    // Dynamically create new session for user query
    const newId = `analysis-${Date.now()}`;
    const newSession: AnalysisSession = {
      id: newId,
      title: query.slice(0, 32) + (query.length > 32 ? '...' : ''),
      query: query,
      analysisType: attachments.length > 1 ? 'Change Detection' : 'VQA + Grounding',
      modalityBadge: attachments[0]?.modality === 'SAR' ? 'SAR' : 'S2',
      imageCount: attachments.length || 1,
      date: 'Just now',
      confidence: null,
      pipeline: 'Auto-Routed Agent Pipeline',
      format: attachments[0]?.name?.endsWith('.tif') ? 'GeoTIFF / TIFF' : 'Satellite Raster',
      thumbnail: attachments[0]?.thumbnailUrl || '/demo/thumb_water_body.png',
      backend: 'real',
      userMessage: {
        text: query,
        timestamp: 'Just now',
        attachments: attachments
      },
      assistantResponse: {
        text: `SatQuery AI Agent is analyzing your query across ${attachments.length || 1} image asset(s)...`,
        evidenceType: attachments.length > 1 ? 'change_comparison' : 'single_grounding',
        evidence: {
          imageUrl: attachments[0]?.thumbnailUrl || '',
          scaleLabel: '10 km',
          satelliteLabel: attachments[0]?.name || 'ORBITAL EO SENSOR',
          boundingBoxes: []
        },
        traceSteps: [
          `Query: "${query}"`,
          'Understanding your question...',
          'Checking image compatibility...',
          'Identifying analysis task...',
          'Selecting specialist model...',
          'Analyzing imagery...',
          'Preparing visual evidence...'
        ],
        detectedTask: 'Orchestrating...',
        inputDescription: attachments.map(a => a.name).join(', ') || 'Uploaded Satellite Imagery',
        selectedModel: 'Selecting...'
      }
    };

    setSessions((prev) => [newSession, ...prev]);

    // Asynchronously call backend /api/analyze if real image assets are present
    const realAssetIds = attachments
      .map(a => a.assetId)
      .filter(id => Boolean(id)) as string[];

    if (realAssetIds.length > 0) {
      const modeParam = selectedMode.toLowerCase().includes('auto') ? 'auto' : selectedMode.toLowerCase();
      runAnalysis({
        query,
        image_ids: realAssetIds,
        mode: modeParam,
        session_id: newId
      }).then((apiRes) => {
        setSessions((prev) => prev.map((s) => {
          if (s.id !== newId) return s;
          const isError = apiRes.status === 'incompatible' || apiRes.status === 'unsupported' || apiRes.status === 'failed';
          const answerText = isError 
            ? (apiRes.error_message || 'Analysis could not proceed due to input constraints.') 
            : (apiRes.result?.answer || 'Analysis completed successfully.');
          
          const firstEvidence = apiRes.result?.evidence?.[0] as any;
          let updatedEvidenceType = s.assistantResponse.evidenceType;
          let updatedEvidence = s.assistantResponse.evidence;

          if (apiRes.task === 'change_detection' && firstEvidence) {
            updatedEvidenceType = 'change_comparison';
            updatedEvidence = {
              beforeImageUrl: firstEvidence.beforeImageUrl || s.userMessage.attachments?.[0]?.thumbnailUrl || '',
              afterImageUrl: firstEvidence.afterImageUrl || firstEvidence.overlay_url || s.userMessage.attachments?.[1]?.thumbnailUrl || '',
              beforeLabel: firstEvidence.beforeLabel || s.userMessage.attachments?.[0]?.name || 'T1 Baseline',
              beforeDate: firstEvidence.beforeDate || 'Baseline Acquisition',
              beforeGsd: firstEvidence.beforeGsd || '10m GSD',
              afterLabel: firstEvidence.afterLabel || s.userMessage.attachments?.[1]?.name || 'T2 Current',
              afterDate: firstEvidence.afterDate || 'Recent Acquisition',
              afterOverlayTag: firstEvidence.afterOverlayTag || `TinyCD Mask (${firstEvidence.change_percentage ?? 0}%)`
            };
          } else if (apiRes.task === 'grounding' && firstEvidence) {
            updatedEvidenceType = 'single_grounding';
            updatedEvidence = {
              imageUrl: firstEvidence.annotatedImageUrl || firstEvidence.imageUrl || s.userMessage.attachments?.[0]?.thumbnailUrl || '',
              scaleLabel: firstEvidence.scaleLabel || '10m GSD',
              satelliteLabel: firstEvidence.satelliteLabel || s.userMessage.attachments?.[0]?.name || 'Optical Imagery',
              boundingBoxes: firstEvidence.boundingBoxes || firstEvidence.bounding_boxes || []
            };
          } else if (apiRes.task === 'optical_sar' && firstEvidence) {
            updatedEvidenceType = 'optical_sar';
            updatedEvidence = {
              opticalImageUrl: firstEvidence.opticalImageUrl || s.userMessage.attachments?.[0]?.thumbnailUrl || '',
              sarImageUrl: firstEvidence.sarImageUrl || s.userMessage.attachments?.[1]?.thumbnailUrl || '',
              opticalLabel: firstEvidence.opticalLabel || 'Multispectral Optical',
              sarLabel: firstEvidence.sarLabel || 'SAR Microwave Backscatter',
              fusionSummary: firstEvidence.fusionSummary || 'Cross-modal neural fusion completed.'
            };
          } else if ((apiRes.task === 'vqa' || apiRes.task === 'captioning') && firstEvidence) {
            updatedEvidenceType = 'single_grounding';
            updatedEvidence = {
              imageUrl: firstEvidence.imageUrl || s.userMessage.attachments?.[0]?.thumbnailUrl || '',
              scaleLabel: firstEvidence.scaleLabel || '10m GSD',
              satelliteLabel: firstEvidence.satelliteLabel || s.userMessage.attachments?.[0]?.name || 'Optical Imagery',
              boundingBoxes: []
            };
          }

          return {
            ...s,
            backend: apiRes.backend,
            confidence: apiRes.result?.confidence ?? null,
            assistantResponse: {
              ...s.assistantResponse,
              text: answerText,
              evidenceType: updatedEvidenceType,
              evidence: updatedEvidence,
              traceSteps: apiRes.execution_trace.map(t => `${t.name}: ${t.status}${t.detail ? ' — ' + t.detail : ''}`),
              detectedTask: apiRes.task.toUpperCase(),
              selectedModel: apiRes.selected_model ? `${apiRes.selected_model.name} (${apiRes.selected_model.backend})` : 'None'
            }
          };
        }));
      }).catch((err) => {
        console.warn('Backend analyze call error:', err);
      });
    }

    return newId;
  };


  const handleContinueToAnalysis = (mode: string) => {
    if (stagedFiles.length === 0) return;

    // Map staged files into real image attachments with backend asset IDs
    const attachments: ImageAttachment[] = stagedFiles.map((sf) => ({
      name: sf.name,
      detail: sf.modalityTag || `${(sf.modality || 'OPTICAL').toUpperCase()} · Ready`,
      status: 'Ready',
      modality: sf.modality === 'sar' ? 'SAR' : (sf.modality === 'multispectral' ? 'Multispectral' : 'Optical'),
      thumbnailUrl: sf.thumbnailUrl,
      assetId: sf.id
    }));

    // Choose query based on uploaded file count and mode
    let defaultQuery = 'Describe the land-cover and major features visible in this image.';
    const normMode = mode.toLowerCase();
    if (stagedFiles.length === 2) {
      if (normMode.includes('change')) {
        defaultQuery = 'What changed between these two images?';
      } else if (normMode.includes('optical') || stagedFiles.some(f => f.modality === 'sar')) {
        defaultQuery = 'Use the optical and SAR images together to identify built-up and water-covered regions.';
      } else {
        defaultQuery = 'What changed between these two satellite images?';
      }
    } else {
      if (normMode.includes('water')) {
        defaultQuery = 'Where are the water bodies in this image?';
      } else if (normMode.includes('grounding')) {
        defaultQuery = 'Identify and locate prominent surface features in this image.';
      }
    }

    const newSessionId = handleAddNewQuery(defaultQuery, attachments);
    // Clear staging queue so future analyses begin clean
    setStagedFiles([]);
    (window as any).__SATQUERY_TRIGGER_PROCESS = true;
    navigate(`/analysis/${newSessionId}`);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#F2EFE7] text-[#0D1B2A] font-sans antialiased">
      {/* Shared Left Sidebar */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* Main Container */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Shared Top Header */}
        <TopHeader
          breadcrumbSub={breadcrumbSub}
          activeSessionTitle={activeSession?.title}
          selectedMode={selectedMode}
          onSelectMode={setSelectedMode}
        />

        {/* View Routing */}
        <main className="flex-1 flex flex-col h-full overflow-hidden">
          <Routes>
            <Route 
              path="/" 
              element={
                <MainWorkspaceView 
                  sessions={sessions} 
                  stagedFiles={stagedFiles}
                  onAddNewQuery={handleAddNewQuery} 
                  onRemoveStagedFile={(idx) => setStagedFiles((prev) => prev.filter((_, i) => i !== idx))}
                />
              } 
            />
            <Route 
              path="/analysis/:id" 
              element={
                <MainWorkspaceView 
                  sessions={sessions} 
                  stagedFiles={stagedFiles}
                  onAddNewQuery={handleAddNewQuery} 
                  onRemoveStagedFile={(idx) => setStagedFiles((prev) => prev.filter((_, i) => i !== idx))}
                />
              } 
            />
            <Route 
              path="/new-analysis" 
              element={
                <NewAnalysisView
                  stagedFiles={stagedFiles}
                  onSetStagedFiles={setStagedFiles}
                  onContinueToAnalysis={handleContinueToAnalysis}
                  activeRoutingMode={selectedMode}
                />
              } 
            />
            <Route 
              path="/history" 
              element={<HistoryView sessions={sessions} />} 
            />
          </Routes>
        </main>
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        defaultMode={selectedMode}
        onChangeDefaultMode={setSelectedMode}
        showEvidence={showEvidence}
        onToggleEvidence={() => setShowEvidence(!showEvidence)}
        showConfidence={showConfidence}
        onToggleConfidence={() => setShowConfidence(!showConfidence)}
      />
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AppLayout />
    </Router>
  );
}
