import React from 'react';
import { X, ShieldCheck, Sliders } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultMode: string;
  onChangeDefaultMode: (mode: string) => void;
  showEvidence: boolean;
  onToggleEvidence: () => void;
  showConfidence: boolean;
  onToggleConfidence: () => void;
  backendVersion?: string;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  defaultMode,
  onChangeDefaultMode,
  showEvidence,
  onToggleEvidence,
  showConfidence,
  onToggleConfidence,
  backendVersion = '1.0.0'
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xs select-none animation-in fade-in duration-200">
      <div className="bg-white rounded-3xl border border-[#E2DDD3] shadow-2xl max-w-lg w-full overflow-hidden animation-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-[#F0ECE1]">
          <div className="flex items-center gap-2.5">
            <Sliders className="w-5 h-5 text-[#0D1B2A]" />
            <h2 className="text-base font-bold text-[#0D1B2A]">Settings & System Configuration</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-full text-[#8B98A5] hover:text-[#0D1B2A] hover:bg-[#F2EFE7] transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-5 text-xs text-[#4E6B7C]">
          {/* Analysis Settings */}
          <div className="space-y-3">
            <div className="text-[10px] font-bold uppercase tracking-wider text-[#8B98A5]">
              Analysis Preferences
            </div>

            <div className="flex items-center justify-between p-3 rounded-2xl bg-[#F9F8F5] border border-[#E8E4DA]">
              <div>
                <div className="font-semibold text-[#0D1B2A]">Default Routing Mode</div>
                <div className="text-[11px] text-[#8B98A5]">Let agent pick specialists automatically</div>
              </div>
              <select
                value={defaultMode}
                onChange={(e) => onChangeDefaultMode(e.target.value)}
                className="bg-white border border-[#D5CFBF] rounded-xl px-2.5 py-1 text-xs text-[#0D1B2A] font-medium outline-none cursor-pointer"
              >
                <option value="Auto Detect (Recommended)">Auto Detect (Recommended)</option>
                <option value="VQA">VQA</option>
                <option value="Grounding">Grounding</option>
                <option value="Change Detection">Change Detection</option>
                <option value="Optical + SAR">Optical + SAR</option>
              </select>
            </div>

            <div className="flex items-center justify-between p-3 rounded-2xl bg-[#F9F8F5] border border-[#E8E4DA]">
              <div>
                <div className="font-semibold text-[#0D1B2A]">Visual Evidence Overlays</div>
                <div className="text-[11px] text-[#8B98A5]">Render bounding boxes and masks on imagery</div>
              </div>
              <button
                onClick={onToggleEvidence}
                className={`w-11 h-6 rounded-full transition-colors relative cursor-pointer ${
                  showEvidence ? 'bg-[#0D1B2A]' : 'bg-[#D5CFBF]'
                }`}
              >
                <div 
                  className={`w-4 h-4 rounded-full bg-white transition-transform absolute top-1 ${
                    showEvidence ? 'left-6' : 'left-1'
                  }`} 
                />
              </button>
            </div>

            <div className="flex items-center justify-between p-3 rounded-2xl bg-[#F9F8F5] border border-[#E8E4DA]">
              <div>
                <div className="font-semibold text-[#0D1B2A]">Confidence Metrics Display</div>
                <div className="text-[11px] text-[#8B98A5]">Show statistical inference confidence percentage</div>
              </div>
              <button
                onClick={onToggleConfidence}
                className={`w-11 h-6 rounded-full transition-colors relative cursor-pointer ${
                  showConfidence ? 'bg-[#0D1B2A]' : 'bg-[#D5CFBF]'
                }`}
              >
                <div 
                  className={`w-4 h-4 rounded-full bg-white transition-transform absolute top-1 ${
                    showConfidence ? 'left-6' : 'left-1'
                  }`} 
                />
              </button>
            </div>
          </div>

          {/* About Section */}
          <div className="space-y-3 pt-2">
            <div className="text-[10px] font-bold uppercase tracking-wider text-[#8B98A5]">
              Project Specification
            </div>
            
            <div className="p-4 rounded-2xl bg-[#F2EFE7] border border-[#E2DDD3] space-y-2 text-[11px]">
              <div className="flex items-center justify-between text-[#0D1B2A]">
                <span className="font-semibold">Smart India Hackathon 2026:</span>
                <span className="font-mono font-bold">SIH26167</span>
              </div>
              <div className="flex items-center justify-between text-[#0D1B2A]">
                <span className="font-semibold">Organization / Theme:</span>
                <span>ISRO / Space Technology</span>
              </div>
              <div className="flex items-center justify-between text-[#0D1B2A]">
                <span className="font-semibold">Backend Status:</span>
                <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span>FastAPI v{backendVersion} Connected</span>
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-[#F9F8F5] border-t border-[#F0ECE1] flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-full bg-[#0D1B2A] hover:bg-[#1E2E42] text-white text-xs font-semibold cursor-pointer shadow-xs"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
