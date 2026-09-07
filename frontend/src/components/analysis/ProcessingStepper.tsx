import React, { useEffect, useState } from 'react';
import { CheckCircle2, Loader2, Circle, Layers, FileText, Cpu } from 'lucide-react';
import type { AnalysisSession } from '../../types';

interface ProcessingStepperProps {
  session: AnalysisSession;
  onComplete?: () => void;
}

const STEP_DEFINITIONS = [
  'Understanding your question',
  'Checking image compatibility',
  'Identifying analysis task',
  'Selecting specialist model',
  'Analyzing imagery',
  'Preparing visual evidence'
];

export const ProcessingStepper: React.FC<ProcessingStepperProps> = ({ session, onComplete }) => {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);

  useEffect(() => {
    // Progress through steps smoothly to demonstrate agentic behavior
    const interval = setInterval(() => {
      setCurrentStepIndex((prev) => {
        if (prev < STEP_DEFINITIONS.length - 1) {
          return prev + 1;
        } else {
          clearInterval(interval);
          if (onComplete) {
            setTimeout(onComplete, 800);
          }
          return prev;
        }
      });
    }, 900);

    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <div className="w-full max-w-3xl mx-auto space-y-6 select-none animation-in fade-in duration-300">
      {/* User Query Reference Bubble */}
      <div className="flex justify-end">
        <div className="max-w-xl bg-[#0D1B2A] text-white p-5 rounded-3xl shadow-md space-y-3">
          <div className="text-sm font-normal">"{session.query}"</div>
          
          {/* Attached Files Chips */}
          <div className="flex flex-wrap gap-2 pt-1 border-t border-white/10">
            {session.userMessage.attachments.map((file, idx) => (
              <div 
                key={idx} 
                className="flex items-center gap-2.5 px-3 py-1.5 rounded-xl bg-white/10 border border-white/15 text-xs text-white"
              >
                <div className="w-5 h-5 rounded bg-white/20 flex items-center justify-center text-[10px]">
                  tif
                </div>
                <div className="overflow-hidden">
                  <div className="font-mono text-[11px] truncate max-w-[160px]">{file.name}</div>
                  <div className="text-[10px] text-[#C8D9E6]">{file.detail}</div>
                </div>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  Ready
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Agent Processing Box */}
      <div className="p-8 rounded-3xl bg-white border border-[#E2DDD3] shadow-md space-y-6">
        {/* Stepper Header */}
        <div className="flex items-center gap-3">
          <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
          <h2 className="text-lg font-bold text-[#0D1B2A]">
            SatQuery AI is analyzing your imagery...
          </h2>
        </div>

        {/* Vertical Step Progress List */}
        <div className="space-y-3 pl-1">
          {STEP_DEFINITIONS.map((stepText, idx) => {
            const isCompleted = idx < currentStepIndex;
            const isCurrent = idx === currentStepIndex;
            const isPending = idx > currentStepIndex;

            return (
              <div 
                key={idx} 
                className={`flex items-center justify-between p-2.5 rounded-2xl transition-all ${
                  isCurrent ? 'bg-[#EBF0F5] border border-[#B8CEDD]' : 'bg-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  {isCompleted && (
                    <CheckCircle2 className="w-5 h-5 text-[#0D1B2A] fill-[#0D1B2A] stroke-white" />
                  )}
                  {isCurrent && (
                    <div className="w-5 h-5 rounded-full border-2 border-[#0D1B2A] flex items-center justify-center">
                      <div className="w-2.5 h-2.5 rounded-full bg-[#0D1B2A] animate-pulse"></div>
                    </div>
                  )}
                  {isPending && (
                    <Circle className="w-5 h-5 text-[#C5BFA7]" />
                  )}

                  <span className={`text-sm ${
                    isCompleted ? 'text-[#0D1B2A] font-medium' :
                    isCurrent ? 'text-[#0D1B2A] font-semibold' : 'text-[#8B98A5]'
                  }`}>
                    {stepText}
                  </span>
                </div>

                {isCurrent && (
                  <span className="text-[10px] font-semibold px-2.5 py-0.5 rounded-full bg-[#0D1B2A] text-white">
                    In progress...
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* Dynamic Context Summary Box */}
        <div className="p-5 rounded-2xl bg-[#F9F8F5] border border-[#E8E4DA] grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <div className="text-[10px] font-bold uppercase tracking-wider text-[#8B98A5] flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-blue-500" />
              <span>DETECTED TASK</span>
            </div>
            <div className="text-xs font-bold text-[#0D1B2A] mt-1">
              {session.assistantResponse.detectedTask}
            </div>
          </div>

          <div>
            <div className="text-[10px] font-bold uppercase tracking-wider text-[#8B98A5] flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-emerald-500" />
              <span>INPUT</span>
            </div>
            <div className="text-xs font-medium text-[#0D1B2A] mt-1 truncate">
              {session.assistantResponse.inputDescription}
            </div>
          </div>

          <div>
            <div className="text-[10px] font-bold uppercase tracking-wider text-[#8B98A5] flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-purple-500" />
              <span>SELECTED MODEL</span>
            </div>
            <div className="text-xs font-bold text-[#0D1B2A] mt-1">
              {session.assistantResponse.selectedModel}
            </div>
          </div>
        </div>

        {/* Footnote */}
        <div className="text-[11px] text-[#8B98A5] text-center">
          SatQuery automatically selects the appropriate specialist model based on your query and imagery.
        </div>
      </div>
    </div>
  );
};
