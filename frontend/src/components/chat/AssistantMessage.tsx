import React from 'react';
import { CheckCircle2, Sliders, Layers } from 'lucide-react';
import type { AnalysisSession, SingleGroundingEvidence, ChangeComparisonEvidence, OpticalSarEvidence } from '../../types';
import { EvidenceViewer } from '../analysis/EvidenceViewer';
import { ChangeComparison } from '../analysis/ChangeComparison';
import { OpticalSarComparison } from '../analysis/OpticalSarComparison';
import { AnalysisTrace } from '../analysis/AnalysisTrace';

interface AssistantMessageProps {
  session: AnalysisSession;
}

export const AssistantMessage: React.FC<AssistantMessageProps> = ({ session }) => {
  const { assistantResponse, confidence, pipeline, format, query } = session;

  return (
    <div className="flex flex-col items-start space-y-4 max-w-3xl w-full animation-in fade-in slide-in-from-bottom-2 duration-300">
      {/* Assistant Bubble Container */}
      <div className="w-full bg-white p-6 sm:p-7 rounded-3xl border border-[#E2DDD3] shadow-sm space-y-5">
        {/* Answer Text */}
        <p className="text-sm md:text-base text-[#0D1B2A] leading-relaxed font-normal">
          {assistantResponse.text}
        </p>

        {/* Visual Evidence Area */}
        {assistantResponse.evidenceType === 'single_grounding' && (
          <EvidenceViewer 
            evidence={assistantResponse.evidence as SingleGroundingEvidence} 
          />
        )}

        {assistantResponse.evidenceType === 'change_comparison' && (
          <ChangeComparison 
            query={query}
            confidence={confidence}
            evidence={assistantResponse.evidence as ChangeComparisonEvidence}
            summaryText={assistantResponse.text}
          />
        )}

        {assistantResponse.evidenceType === 'optical_sar' && (
          <OpticalSarComparison 
            evidence={assistantResponse.evidence as OpticalSarEvidence}
          />
        )}

        {/* Evaluation & Pipeline Metrics Badges */}
        <div className="flex flex-wrap items-center gap-2 pt-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#EBF0F5] text-[#0D1B2A] text-xs font-semibold border border-[#D5E0EA]">
            <CheckCircle2 className="w-3.5 h-3.5 text-blue-600" />
            <span>
              {session.backend === 'demo'
                ? 'Backend: Demo (Confidence unavailable)'
                : confidence !== null
                ? `Confidence: ${Math.round(confidence * 100)}%`
                : 'Confidence unavailable'}
            </span>
          </div>

          <div className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${
            session.backend === 'real'
              ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
              : session.backend === 'adapted'
              ? 'bg-blue-50 text-blue-800 border-blue-200'
              : 'bg-amber-50 text-amber-800 border-amber-200'
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              session.backend === 'real'
                ? 'bg-emerald-500'
                : session.backend === 'adapted'
                ? 'bg-blue-500'
                : 'bg-amber-500'
            }`}></span>
            <span>Backend: {session.backend ? session.backend.toUpperCase() : 'DEMO'}</span>
          </div>


          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#F2EFE7] text-[#4E6B7C] text-xs font-medium border border-[#E2DDD3]">
            <Sliders className="w-3.5 h-3.5" />
            <span>Pipeline: {pipeline}</span>
          </div>

          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#F2EFE7] text-[#4E6B7C] text-xs font-medium border border-[#E2DDD3]">
            <Layers className="w-3.5 h-3.5" />
            <span>Format: {format}</span>
          </div>
        </div>

        {/* Collapsible Observable Execution Trace */}
        <AnalysisTrace steps={assistantResponse.traceSteps} />
      </div>
    </div>
  );
};
