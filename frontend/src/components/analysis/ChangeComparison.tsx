import React from 'react';
import { Clock } from 'lucide-react';
import type { ChangeComparisonEvidence } from '../../types';

interface ChangeComparisonProps {
  query: string;
  confidence: number | null;
  evidence: ChangeComparisonEvidence;
  summaryText: string;
}

export const ChangeComparison: React.FC<ChangeComparisonProps> = ({
  query,
  confidence,
  evidence,
  summaryText
}) => {
  return (
    <div className="mt-6 p-6 rounded-3xl bg-white border border-[#E2DDD3] shadow-sm space-y-5 select-none">
      {/* Module Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#F0ECE1] pb-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-[#0D1B2A] text-white flex items-center justify-center shrink-0">
            <Clock className="w-4 h-4 text-[#C8D9E6]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-[#0D1B2A]">Change Detection Module</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#EBF0F5] text-[#4E6B7C]">
                Multi-Temporal Radial Analysis
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-[#8B98A5] italic">"{query}"</span>
          <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-[#E9F4EE] text-emerald-800 border border-[#CEEBD9]">
            {confidence !== null ? `Confidence: ${Math.round(confidence * 100)}%` : 'Demo Execution'}
          </span>
        </div>
      </div>


      {/* Analysis Result Text */}
      <div className="text-sm text-[#0D1B2A] leading-relaxed">
        <span className="font-semibold">{summaryText.split('.')[0]}.</span>
        <span>{summaryText.substring(summaryText.indexOf('.') + 1)}</span>
      </div>

      {/* Side-by-Side Satellite Panels */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
        {/* Panel 1: Before (T1) */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold tracking-wider">
            <span className="text-[#8B98A5] uppercase">{evidence.beforeLabel}</span>
            <span className="text-[#0D1B2A] font-medium">{evidence.beforeDate}</span>
          </div>

          <div className="relative aspect-[4/3] rounded-2xl overflow-hidden border border-[#D5CFBF] bg-[#0D1B2A] shadow-xs group">
            <img 
              src={evidence.beforeImageUrl} 
              alt="Baseline satellite acquisition"
              className="w-full h-full object-cover group-hover:scale-102 transition-transform duration-300" 
            />
            <div className="absolute bottom-3 left-3 px-2.5 py-1 rounded-lg bg-black/65 backdrop-blur-md text-white text-[10px] font-mono border border-white/10">
              {evidence.beforeGsd}
            </div>
          </div>
        </div>

        {/* Panel 2: After (T2) */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold tracking-wider">
            <span className="text-[#8B98A5] uppercase">{evidence.afterLabel}</span>
            <span className="text-[#0D1B2A] font-medium">{evidence.afterDate}</span>
          </div>

          <div className="relative aspect-[4/3] rounded-2xl overflow-hidden border border-[#D5CFBF] bg-[#0D1B2A] shadow-xs group">
            <img 
              src={evidence.afterImageUrl} 
              alt="Recent evaluation satellite acquisition"
              className="w-full h-full object-cover group-hover:scale-102 transition-transform duration-300" 
            />
            
            {/* Change Mask Highlight Badge */}
            <div className="absolute bottom-3 left-3 flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-black/75 backdrop-blur-md text-white text-[10px] font-mono border border-white/10 shadow-sm">
              <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
              <span className="text-rose-200 font-semibold">{evidence.afterOverlayTag}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
