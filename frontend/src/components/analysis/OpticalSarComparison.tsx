import React from 'react';
import { Layers, Radio, Sparkles } from 'lucide-react';
import type { OpticalSarEvidence } from '../../types';

interface OpticalSarComparisonProps {
  evidence: OpticalSarEvidence;
}

export const OpticalSarComparison: React.FC<OpticalSarComparisonProps> = ({ evidence }) => {
  return (
    <div className="space-y-4 mt-4 select-none">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-bold tracking-wider uppercase text-[#4E6B7C]">
            Dual-Sensor Optical + SAR Visual Evidence
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#EBF0F5] text-[#4E6B7C]">
            Cross-Modal Alignment
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Optical Sensor Card */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold tracking-wider">
            <span className="text-[#8B98A5] flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-blue-500" />
              <span>{evidence.opticalLabel}</span>
            </span>
            <span className="text-xs text-[#0D1B2A] font-mono">Visible + NIR</span>
          </div>

          <div className="relative aspect-[4/3] rounded-2xl overflow-hidden border border-[#D5CFBF] bg-[#0D1B2A] shadow-xs">
            <img 
              src={evidence.opticalImageUrl} 
              alt="Optical imagery"
              className="w-full h-full object-cover" 
            />
            <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-black/60 backdrop-blur-md text-white text-[10px] font-mono border border-white/10">
              Multispectral Optical
            </div>
          </div>
        </div>

        {/* SAR Sensor Card */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold tracking-wider">
            <span className="text-[#8B98A5] flex items-center gap-1.5">
              <Radio className="w-3.5 h-3.5 text-amber-500" />
              <span>{evidence.sarLabel}</span>
            </span>
            <span className="text-xs text-[#0D1B2A] font-mono">Radar Backscatter</span>
          </div>

          <div className="relative aspect-[4/3] rounded-2xl overflow-hidden border border-[#D5CFBF] bg-[#0D1B2A] shadow-xs">
            <img 
              src={evidence.sarImageUrl} 
              alt="SAR radar imagery"
              className="w-full h-full object-cover" 
            />
            <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-black/60 backdrop-blur-md text-white text-[10px] font-mono border border-white/10">
              Synthetic Aperture Radar
            </div>
          </div>
        </div>
      </div>

      {/* Fusion Summary Note */}
      <div className="p-3.5 rounded-2xl bg-[#F9F8F5] border border-[#E8E4DA] text-xs text-[#4E6B7C] flex items-start gap-2.5">
        <Layers className="w-4 h-4 text-[#0D1B2A] shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-[#0D1B2A]">Multi-Sensor Synergy: </span>
          <span>{evidence.fusionSummary}</span>
        </div>
      </div>
    </div>
  );
};
