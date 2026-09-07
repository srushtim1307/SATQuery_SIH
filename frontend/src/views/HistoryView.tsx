import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { AnalysisSession } from '../types';
import { HistoryFilters } from '../components/history/HistoryFilters';
import { HistoryCard } from '../components/history/HistoryCard';

interface HistoryViewProps {
  sessions: AnalysisSession[];
}

export const HistoryView: React.FC<HistoryViewProps> = ({ sessions }) => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('All');

  // Filter sessions based on query and filter pill
  const filteredSessions = sessions.filter((session) => {
    // 1. Tab filter
    const matchesFilter =
      selectedFilter === 'All' ||
      (selectedFilter === 'VQA' && session.analysisType.includes('VQA')) ||
      (selectedFilter === 'Grounding' && session.analysisType.includes('Grounding')) ||
      (selectedFilter === 'Change Detection' && session.analysisType.includes('Change')) ||
      (selectedFilter === 'Optical + SAR' && session.analysisType.includes('Optical'));

    // 2. Search query
    const queryLower = searchQuery.toLowerCase();
    const matchesSearch =
      !searchQuery ||
      session.title.toLowerCase().includes(queryLower) ||
      session.query.toLowerCase().includes(queryLower) ||
      session.analysisType.toLowerCase().includes(queryLower) ||
      session.modalityBadge.toLowerCase().includes(queryLower);

    return matchesFilter && matchesSearch;
  });

  return (
    <div className="flex-1 overflow-y-auto bg-[#F2EFE7] px-6 py-10 select-none">
      <div className="max-w-4xl mx-auto space-y-8">
        {/* Header */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-[11px] font-bold tracking-wider uppercase text-[#8B98A5]">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-600"></span>
            <span>SESSIONS & DIALOGUES</span>
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-[#0D1B2A]">
            Analysis History
          </h1>
          <p className="text-sm text-[#4E6B7C]">
            View and continue your previous satellite-image analyses.
          </p>
        </div>

        {/* Search & Filter Tabs */}
        <HistoryFilters
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          selectedFilter={selectedFilter}
          onFilterChange={setSelectedFilter}
        />

        {/* Sessions List */}
        <div className="space-y-3 pt-2">
          {filteredSessions.map((session) => (
            <HistoryCard
              key={session.id}
              session={session}
              onOpen={(id) => navigate(`/analysis/${id}`)}
            />
          ))}

          {filteredSessions.length === 0 && (
            <div className="p-10 rounded-3xl bg-white border border-[#E2DDD3] text-center space-y-2">
              <div className="text-sm font-semibold text-[#0D1B2A]">No matching analyses found</div>
              <div className="text-xs text-[#4E6B7C]">Try adjusting your search keywords or filter tab</div>
            </div>
          )}
        </div>

        {/* Keyboard shortcut footer */}
        <div className="text-[11px] text-[#8B98A5] text-center pt-4">
          Press <span className="font-mono font-medium">↵</span> to open selected session • <span className="font-mono font-medium">↑↓</span> to navigate
        </div>
      </div>
    </div>
  );
};
