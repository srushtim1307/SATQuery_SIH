import React, { useState } from 'react';
import { ChevronRight, CheckCircle2 } from 'lucide-react';

interface AnalysisTraceProps {
  steps: string[];
}

export const AnalysisTrace: React.FC<AnalysisTraceProps> = ({ steps }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="pt-2 border-t border-[#F0ECE1] select-none">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-1.5 text-xs font-medium text-[#4E6B7C] hover:text-[#0D1B2A] transition-colors cursor-pointer group"
      >
        <ChevronRight 
          className={`w-3.5 h-3.5 text-[#8B98A5] transition-transform duration-200 group-hover:text-[#0D1B2A] ${
            isExpanded ? 'rotate-90' : ''
          }`} 
        />
        <span>How this was analyzed</span>
      </button>

      {isExpanded && (
        <div className="mt-3 p-4 rounded-2xl bg-[#F9F8F5] border border-[#E8E4DA] space-y-2 text-xs animation-in fade-in slide-in-from-top-1">
          <div className="text-[10px] font-bold uppercase tracking-wider text-[#8B98A5] pb-1">
            Observable Execution Trace
          </div>
          <div className="space-y-1.5 font-mono text-[11px] text-[#2D3E4F]">
            {steps.map((step, idx) => (
              <div key={idx} className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 mt-0.5 shrink-0" />
                <span>{step}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
