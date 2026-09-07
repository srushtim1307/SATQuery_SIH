import React from 'react';
import { Sparkles, Droplets, Clock, Building2, Layers } from 'lucide-react';
import { SUGGESTION_CHIPS } from '../../services/demoData';

interface EmptyGreetingProps {
  onSelectSuggestion: (query: string) => void;
}

export const EmptyGreeting: React.FC<EmptyGreetingProps> = ({ onSelectSuggestion }) => {
  const getIcon = (iconName: string) => {
    switch (iconName) {
      case 'Sparkles': return <Sparkles className="w-3.5 h-3.5 text-blue-600" />;
      case 'Droplets': return <Droplets className="w-3.5 h-3.5 text-cyan-600" />;
      case 'Clock': return <Clock className="w-3.5 h-3.5 text-amber-600" />;
      case 'Building2': return <Building2 className="w-3.5 h-3.5 text-purple-600" />;
      case 'Layers': return <Layers className="w-3.5 h-3.5 text-emerald-600" />;
      default: return <Sparkles className="w-3.5 h-3.5 text-blue-600" />;
    }
  };

  return (
    <div className="max-w-2xl w-full text-center space-y-6 mx-auto py-8 select-none animation-in fade-in duration-300">
      {/* EO VLM Badge */}
      <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white border border-[#E2DDD3] text-[11px] font-semibold tracking-wider text-[#4E6B7C] uppercase shadow-xs">
        <span className="flex gap-1">
          <span className="w-2 h-2 rounded-full bg-blue-500"></span>
          <span className="w-2 h-2 rounded-full bg-amber-500"></span>
          <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span className="w-2 h-2 rounded-full bg-rose-500"></span>
        </span>
        <span>EARTH OBSERVATION VLM</span>
      </div>

      {/* Headline */}
      <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-[#0D1B2A] leading-tight">
        Ask questions about your<br />satellite images.
      </h1>

      {/* Subtitle */}
      <p className="text-sm md:text-base text-[#4E6B7C] max-w-lg mx-auto font-normal leading-relaxed">
        Upload orbital scenes and ask SatQuery AI to detect targets, quantify changes, and ground features using natural language.
      </p>

      {/* Quick Suggestion Chips */}
      <div className="flex flex-wrap items-center justify-center gap-2.5 pt-2 max-w-xl mx-auto">
        {SUGGESTION_CHIPS.map((chip, idx) => (
          <button
            key={idx}
            onClick={() => onSelectSuggestion(chip.query)}
            className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white border border-[#D5CFBF] hover:border-[#4E6B7C] hover:bg-[#F9F8F5] text-xs font-medium text-[#0D1B2A] transition-all shadow-2xs hover:shadow-xs active:scale-98 cursor-pointer"
          >
            {getIcon(chip.icon)}
            <span>{chip.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
