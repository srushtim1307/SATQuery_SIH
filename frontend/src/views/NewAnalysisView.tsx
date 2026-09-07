import React, { useRef, useState, useEffect } from 'react';
import { ArrowRight, CheckCircle2, Loader2, AlertCircle, AlertTriangle } from 'lucide-react';
import type { StagedFile } from '../types';
import { UploadModeCard } from '../components/upload/UploadModeCard';
import { StagedQueue } from '../components/upload/StagedQueue';
import { uploadSatelliteImages, validateUploadedInputs, deleteUploadedAsset } from '../services/api';

interface NewAnalysisViewProps {
  stagedFiles: StagedFile[];
  onSetStagedFiles: React.Dispatch<React.SetStateAction<StagedFile[]>>;
  onContinueToAnalysis: (selectedMode: string) => void;
  activeRoutingMode: string;
}

export const NewAnalysisView: React.FC<NewAnalysisViewProps> = ({
  stagedFiles,
  onSetStagedFiles,
  onContinueToAnalysis,
  activeRoutingMode
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [currentUploadHint, setCurrentUploadHint] = useState<string>('single');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Validation state from backend
  const [validationState, setValidationState] = useState<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }>({ valid: true, errors: [], warnings: [] });

  // Run backend validation whenever staged files or mode changes
  useEffect(() => {
    async function checkValidation() {
      if (stagedFiles.length === 0) {
        setValidationState({ valid: false, errors: [], warnings: [] });
        return;
      }

      // Filter for files with real backend IDs (hex string)
      const realIds = stagedFiles
        .filter((f) => f.id && !f.id.startsWith('file-'))
        .map((f) => f.id);

      if (realIds.length === stagedFiles.length && realIds.length > 0) {
        try {
          const res = await validateUploadedInputs(realIds, activeRoutingMode);
          setValidationState({
            valid: res.valid,
            errors: res.errors,
            warnings: res.warnings
          });
        } catch (err: any) {
          console.warn('Backend validation check:', err.message);
        }
      } else {
        // For pre-seeded demo benchmark items, perform client validation
        const count = stagedFiles.length;
        const normMode = activeRoutingMode.toLowerCase();
        const errors: string[] = [];
        const warnings: string[] = [];

        if (normMode.includes('change') && count !== 2) {
          errors.push('Two images are required for change analysis. Please upload a before and after image.');
        } else if (normMode.includes('optical') && normMode.includes('sar') && count !== 2) {
          errors.push('Optical + SAR analysis requires exactly two images: one optical image and one SAR image.');
        } else if (normMode.includes('grounding') && count !== 1) {
          errors.push('Grounding analysis requires exactly one satellite image.');
        }

        setValidationState({
          valid: errors.length === 0,
          errors,
          warnings
        });
      }
    }

    checkValidation();
  }, [stagedFiles, activeRoutingMode]);

  const handleTriggerUpload = (mode: 'single' | 'optical_sar' | 'bi_temporal') => {
    setCurrentUploadHint(mode);
    setUploadError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleFilesSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setUploadError(null);

    try {
      const fileList = Array.from(files);
      const hint = currentUploadHint === 'optical_sar' ? 'sar' : undefined;
      const uploadedAssets = await uploadSatelliteImages(fileList, undefined, hint);

      // Convert backend asset payload to StagedFile representation
      const newStaged: StagedFile[] = uploadedAssets.map((asset, idx) => {
        const rawSize = fileList[idx]?.size || 0;
        const sizeMb = (rawSize / (1024 * 1024)).toFixed(1);
        return {
          id: asset.id,
          name: asset.filename,
          size: `${sizeMb} MB`,
          modalityTag: asset.resolution 
            ? `${asset.format} · ${asset.resolution}`
            : `${asset.format} (${asset.width}x${asset.height}) · ${asset.bands} bands`,
          thumbnailUrl: asset.preview_url || '/demo/sentinel2_estuary_delta.png',
          ready: true,
          format: asset.format,
          width: asset.width,
          height: asset.height,
          bands: asset.bands,
          crs: asset.crs,
          resolution: asset.resolution,
          modality: asset.modality,
          warnings: asset.warnings
        };
      });

      onSetStagedFiles((prev) => [...prev, ...newStaged]);
    } catch (err: any) {
      console.error('Upload failed:', err);
      setUploadError(err.message || 'Failed to upload satellite image. Please ensure the file is a valid GeoTIFF, TIFF, PNG, or JPEG.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveFile = async (id: string) => {
    // If it's a real backend asset, delete on server
    if (id && !id.startsWith('file-')) {
      await deleteUploadedAsset(id);
    }
    onSetStagedFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const handleClearAll = async () => {
    for (const f of stagedFiles) {
      if (f.id && !f.id.startsWith('file-')) {
        await deleteUploadedAsset(f.id);
      }
    }
    onSetStagedFiles([]);
  };

  const handleUpdateModality = (id: string, modality: 'optical' | 'sar' | 'unknown') => {
    onSetStagedFiles((prev) => prev.map((f) => f.id === id ? { ...f, modality } : f));
  };

  const handleAddDemoSample = () => {
    onSetStagedFiles([
      {
        id: `file-demo-1`,
        name: 'sentinel2_estuary_delta.tif',
        size: '18.4 MB',
        modalityTag: 'GeoTIFF · 10.0m Ground Resolution · WGS 84 / UTM',
        thumbnailUrl: '/demo/sentinel2_estuary_delta.png',
        ready: true,
        format: 'GeoTIFF',
        width: 1024,
        height: 1024,
        bands: 4,
        crs: 'WGS 84 / UTM zone 43N',
        resolution: '10.0m',
        modality: 'optical'
      },
      {
        id: `file-demo-2`,
        name: 'gulf_coastal_sar.tif',
        size: '34.1 MB',
        modalityTag: 'GeoTIFF · 10.0m Ground Resolution · C-Band GRD',
        thumbnailUrl: '/demo/sar_thumb.png',
        ready: true,
        format: 'GeoTIFF',
        width: 1024,
        height: 1024,
        bands: 1,
        crs: 'WGS 84 / UTM zone 43N',
        resolution: '10.0m',
        modality: 'sar'
      }
    ]);
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[#F2EFE7] px-6 py-10 select-none">
      <div className="max-w-4xl mx-auto space-y-10">
        {/* Hidden Native File Input */}
        <input
          type="file"
          ref={fileInputRef}
          multiple
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={handleFilesSelected}
          className="hidden"
        />

        {/* Header Section */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white border border-[#E2DDD3] text-[11px] font-semibold tracking-wider text-[#4E6B7C] uppercase shadow-2xs">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-600"></span>
            <span>WORKSPACE STAGE 01</span>
          </div>

          <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-[#0D1B2A]">
            Upload your satellite imagery
          </h1>

          <p className="text-sm text-[#4E6B7C] max-w-md mx-auto">
            Upload an image or image pair, then ask your question in natural language.
          </p>
        </div>

        {/* Ingestion Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <UploadModeCard
            id="single"
            title="Single Image"
            subtitle="Optical · Multispectral · SAR"
            buttonLabel={isUploading && currentUploadHint === 'single' ? "Uploading..." : "Upload Image"}
            onSelect={() => handleTriggerUpload('single')}
          />

          <UploadModeCard
            id="optical_sar"
            title="Optical + SAR"
            subtitle="Co-registered dual-sensor fusion"
            buttonLabel={isUploading && currentUploadHint === 'optical_sar' ? "Uploading..." : "Upload Pair"}
            isRecommended={true}
            onSelect={() => handleTriggerUpload('optical_sar')}
          />

          <UploadModeCard
            id="bi_temporal"
            title="Before + After"
            subtitle="Bi-temporal surface comparison"
            buttonLabel={isUploading && currentUploadHint === 'bi_temporal' ? "Uploading..." : "Upload Images"}
            onSelect={() => handleTriggerUpload('bi_temporal')}
          />
        </div>

        {/* Upload Loading State */}
        {isUploading && (
          <div className="p-4 rounded-2xl bg-white border border-[#E2DDD3] shadow-xs flex items-center justify-center gap-3 text-xs text-[#0D1B2A]">
            <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
            <span>Processing and extracting satellite metadata with FastAPI ingestion service...</span>
          </div>
        )}

        {/* Upload Error Banner */}
        {uploadError && (
          <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold">Upload Validation Error</div>
              <div>{uploadError}</div>
            </div>
          </div>
        )}

        {/* Supported Formats Banner Pill */}
        <div className="p-3.5 rounded-2xl bg-white border border-[#E2DDD3] shadow-xs flex flex-col sm:flex-row items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2 text-[#0D1B2A] font-medium">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>Supported formats: <span className="font-semibold">GeoTIFF · TIFF · PNG · JPEG</span></span>
          </div>
          <div className="text-[11px] text-[#4E6B7C]">
            Auto-checks metadata & selects analysis pipeline
          </div>
        </div>

        {/* Staged Ingestion Queue */}
        <StagedQueue
          files={stagedFiles}
          onRemoveFile={handleRemoveFile}
          onClearAll={handleClearAll}
          onAddDemoSample={handleAddDemoSample}
          onUpdateModality={handleUpdateModality}
        />

        {/* Validation Errors or Warnings Banner */}
        {validationState.errors.length > 0 && stagedFiles.length > 0 && (
          <div className="p-4 rounded-2xl bg-rose-50 border border-rose-200 text-xs text-rose-800 space-y-1">
            <div className="font-semibold flex items-center gap-1.5">
              <AlertCircle className="w-3.5 h-3.5 text-rose-600" />
              <span>Input Configuration Incompatible</span>
            </div>
            {validationState.errors.map((err, i) => (
              <div key={i} className="pl-5 text-[11px]">{err}</div>
            ))}
          </div>
        )}

        {validationState.warnings.length > 0 && stagedFiles.length > 0 && (
          <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-xs text-amber-800 space-y-1">
            <div className="font-semibold flex items-center gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
              <span>Pipeline Note</span>
            </div>
            {validationState.warnings.map((warn, i) => (
              <div key={i} className="pl-5 text-[11px]">{warn}</div>
            ))}
          </div>
        )}

        {/* Action Button: Continue to Analysis */}
        <div className="pt-2 flex flex-col items-center space-y-2">
          <button
            onClick={() => onContinueToAnalysis(activeRoutingMode)}
            disabled={stagedFiles.length === 0 || !validationState.valid}
            className="flex items-center gap-2 px-8 py-3.5 rounded-full bg-[#0D1B2A] hover:bg-[#1E2E42] disabled:bg-[#D5CFBF] text-white text-sm font-semibold transition-all shadow-md active:scale-98 disabled:cursor-not-allowed cursor-pointer"
          >
            <span>Continue to Analysis</span>
            <ArrowRight className="w-4 h-4" />
          </button>
          <span className="text-[11px] text-[#8B98A5]">
            Press <span className="font-mono font-medium">Enter ↵</span> or click continue to start asking questions
          </span>
        </div>
      </div>
    </div>
  );
};
