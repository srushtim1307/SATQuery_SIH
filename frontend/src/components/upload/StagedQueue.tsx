import React from 'react';
import { Check, X, Trash2, UploadCloud, AlertTriangle } from 'lucide-react';
import type { StagedFile } from '../../types';

interface StagedQueueProps {
  files: StagedFile[];
  onRemoveFile: (id: string) => void;
  onClearAll: () => void;
  onAddDemoSample: () => void;
  onUpdateModality?: (id: string, modality: 'optical' | 'sar' | 'unknown') => void;
}

export const StagedQueue: React.FC<StagedQueueProps> = ({
  files,
  onRemoveFile,
  onClearAll,
  onAddDemoSample,
  onUpdateModality
}) => {
  return (
    <div className="space-y-4 select-none">
      {/* Queue Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="text-base font-bold text-[#0D1B2A]">Staged Ingestion Queue</h3>
          <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-white border border-[#E2DDD3] text-[#4E6B7C]">
            {files.length} / {files.length} Ready
          </span>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onAddDemoSample}
            className="text-xs text-blue-600 hover:text-blue-800 font-medium cursor-pointer"
          >
            + Add Demo Benchmark
          </button>
          {files.length > 0 && (
            <button
              onClick={onClearAll}
              className="flex items-center gap-1 text-xs text-[#8B98A5] hover:text-rose-600 transition-colors cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear All</span>
            </button>
          )}
        </div>
      </div>

      {/* Files List */}
      <div className="space-y-3">
        {files.map((file) => {
          const mod = file.modality || 'unknown';
          return (
            <div
              key={file.id}
              className="p-3.5 rounded-2xl bg-white border border-[#E2DDD3] shadow-xs hover:border-[#B8CEDD] transition-all space-y-2"
            >
              <div className="flex items-center justify-between gap-3.5">
                {/* Thumbnail and Info */}
                <div className="flex items-center gap-3.5 overflow-hidden">
                  <div className="relative w-14 h-14 rounded-xl overflow-hidden bg-[#0D1B2A] shrink-0 border border-[#D5CFBF]">
                    <img
                      src={file.thumbnailUrl}
                      alt={file.name}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute bottom-0 inset-x-0 bg-black/75 text-white text-[8px] font-mono text-center py-0.5 uppercase tracking-wider">
                      {mod === 'optical' ? 'OPT' : mod === 'sar' ? 'SAR' : 'RAW'}
                    </div>
                  </div>

                  <div className="overflow-hidden">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold font-mono text-[#0D1B2A] truncate max-w-xs">
                        {file.name}
                      </span>
                      <span className="text-[11px] text-[#8B98A5] font-normal">
                        {file.size}
                      </span>
                      {file.format && (
                        <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-[#F2EFE7] text-[#4E6B7C]">
                          {file.format}
                        </span>
                      )}
                    </div>

                    <p className="text-[11px] text-[#4E6B7C] truncate mt-0.5">
                      {file.modalityTag || `${file.width || '?'}x${file.height || '?'} · ${file.bands || 1} bands`}
                      {file.resolution ? ` · ${file.resolution}` : ''}
                      {file.crs ? ` · ${file.crs}` : ''}
                    </p>
                  </div>
                </div>

                {/* Status, Modality Confirmation & Action */}
                <div className="flex items-center gap-2.5 shrink-0">
                  {/* Modality tag / selector */}
                  {onUpdateModality ? (
                    <select
                      value={mod}
                      onChange={(e) => onUpdateModality(file.id, e.target.value as any)}
                      className="text-[11px] font-semibold bg-[#F9F8F5] border border-[#D5CFBF] rounded-lg px-2 py-1 text-[#0D1B2A] outline-none cursor-pointer"
                    >
                      <option value="optical">Optical</option>
                      <option value="sar">SAR</option>
                      <option value="unknown">Unknown</option>
                    </select>
                  ) : (
                    <span className="text-[11px] font-semibold px-2 py-0.5 rounded-lg bg-[#F9F8F5] text-[#0D1B2A] border border-[#E2DDD3]">
                      {mod.toUpperCase()}
                    </span>
                  )}

                  <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#EBF4EC] text-emerald-800 text-xs font-semibold border border-[#CDE5D1]">
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Ready</span>
                  </div>

                  <button
                    onClick={() => onRemoveFile(file.id)}
                    title="Remove file"
                    className="p-1.5 rounded-full text-[#8B98A5] hover:text-[#0D1B2A] hover:bg-[#F2EFE7] transition-colors cursor-pointer"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Warning note if any */}
              {file.warnings && file.warnings.length > 0 && (
                <div className="flex items-center gap-1.5 text-[10px] text-amber-700 bg-amber-50 px-2.5 py-1 rounded-lg border border-amber-200">
                  <AlertTriangle className="w-3 h-3 text-amber-600 shrink-0" />
                  <span>{file.warnings[0]}</span>
                </div>
              )}
            </div>
          );
        })}

        {files.length === 0 && (
          <div 
            onClick={onAddDemoSample}
            className="p-8 rounded-3xl border-2 border-dashed border-[#D5CFBF] bg-[#FAF8F3] text-center space-y-2 cursor-pointer hover:border-[#0D1B2A] transition-colors"
          >
            <UploadCloud className="w-8 h-8 text-[#8B98A5] mx-auto" />
            <div className="text-sm font-semibold text-[#0D1B2A]">No images staged yet</div>
            <div className="text-xs text-[#4E6B7C]">Click above or select a modality card to load satellite imagery</div>
          </div>
        )}
      </div>
    </div>
  );
};
