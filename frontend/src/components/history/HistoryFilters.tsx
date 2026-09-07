import React from 'react';
import { Search } from 'lucide-react';

interface HistoryFiltersProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  selectedFilter: string;
  onFilterChange: (filter: string) => void;
}

const FILTER_TABS = ['All', 'VQA', 'Grounding', 'Change Detection', 'Optical + SAR'];

export const HistoryFilters: React.FC<HistoryFiltersProps> = ({
  searchQuery,
  onSearchChange,
  selectedFilter,
  onFilterChange
}) => {
  return (
    <div className="space-y-4 select-none">
      {/* Search Input Bar */}
      <div className="relative w-full">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#8B98A5]" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search analyses... (e.g. water bodies, urban, SAR)"
          className="w-full bg-white border border-[#D5CFBF] rounded-2xl pl-11 pr-14 py-3 text-sm text-[#0D1B2A] placeholder-[#8B98A5] outline-none focus:border-[#0D1B2A] shadow-xs transition-colors"
        />
        <div className="absolute right-3.5 top-1/2 -translate-y-1/2 px-2 py-0.5 rounded-md bg-[#F2EFE7] border border-[#E2DDD3] text-[10px] font-mono text-[#8B98A5]">
          ⌘K
        </div>
      </div>

      {/* Filter Tabs Chips */}
      <div className="flex flex-wrap items-center gap-2 pt-1">
        {FILTER_TABS.map((tab) => {
          const isSelected = selectedFilter === tab;
          return (
            <button
              key={tab}
              onClick={() => onFilterChange(tab)}
              className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer ${
                isSelected
                  ? 'bg-[#0D1B2A] text-white shadow-xs'
                  : 'bg-white text-[#4E6B7C] border border-[#E2DDD3] hover:border-[#8B98A5] hover:text-[#0D1B2A]'
              }`}
            >
              {tab}
            </button>
          );
        })}
      </div>
    </div>
  );
};
