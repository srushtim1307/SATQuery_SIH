import React from 'react';
import { ArrowRight, Layers } from 'lucide-react';
import type { AnalysisSession } from '../../types';

interface HistoryCardProps {
  session: AnalysisSession;
  onOpen: (id: string) => void;
}

export const HistoryCard: React.FC<HistoryCardProps> = ({ session, onOpen }) => {
  return (
    <div
      onClick={() => onOpen(session.id)}
      className="p-5 rounded-3xl bg-white border border-[#E2DDD3] shadow-xs hover:border-[#0D1B2A] hover:shadow-md transition-all flex items-center justify-between gap-4 cursor-pointer select-none group"
    >
      {/* Thumbnail with overlay badge & Title */}
      <div className="flex items-center gap-5 overflow-hidden">
        {/* Thumbnail */}
        <div className="relative w-20 h-20 rounded-2xl overflow-hidden bg-[#0D1B2A] shrink-0 border border-[#D5CFBF]">
          <img
            src={session.thumbnail}
            alt={session.title}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
          />
          <div className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded bg-black/75 text-white font-mono text-[9px] font-bold">
            {session.modalityBadge}
          </div>
        </div>

        {/* Content details */}
        <div className="space-y-1.5 overflow-hidden">
          <h3 className="text-base font-bold text-[#0D1B2A] group-hover:text-blue-900 transition-colors truncate">
            {session.title}
          </h3>
          <p className="text-xs text-[#4E6B7C] italic truncate">
            "{session.query}"
          </p>

          <div className="flex items-center gap-2 pt-1 flex-wrap">
            <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-[#EBF0F5] text-[#0D1B2A]">
              {session.analysisType}
            </span>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider ${
              session.backend === 'real'
                ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                : session.backend === 'adapted'
                ? 'bg-blue-50 text-blue-800 border border-blue-200'
                : 'bg-amber-50 text-amber-800 border border-amber-200'
            }`}>
              {session.backend ? session.backend.toUpperCase() : 'DEMO'}
            </span>
            <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-[#F6F4EE] text-[#4E6B7C] flex items-center gap-1">
              <Layers className="w-3 h-3" />
              <span>{session.imageCount} {session.imageCount === 1 ? 'image' : 'images'}</span>
            </span>
          </div>
        </div>
      </div>

      {/* Date & Navigation Arrow */}
      <div className="flex items-center gap-4 shrink-0 pl-2">
        <span className="text-xs font-medium text-[#8B98A5] hidden sm:inline">
          {session.date}
        </span>
        <div className="w-10 h-10 rounded-full bg-[#F6F4EE] group-hover:bg-[#0D1B2A] group-hover:text-white text-[#0D1B2A] flex items-center justify-center transition-all shadow-2xs">
          <ArrowRight className="w-4 h-4" />
        </div>
      </div>
    </div>
  );
};
