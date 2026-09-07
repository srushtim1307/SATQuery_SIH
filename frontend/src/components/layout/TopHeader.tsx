import React, { useState, useRef, useEffect } from 'react';
import { ChevronDown, Bell, Check } from 'lucide-react';

interface TopHeaderProps {
  breadcrumbSub?: string;
  activeSessionTitle?: string;
  selectedMode: string;
  onSelectMode: (mode: string) => void;
}

const MODES = [
  'Auto Detect (Recommended)',
  'VQA (Visual Question Answering)',
  'Grounding (Region Localization)',
  'Change Detection (Bi-Temporal)',
  'Optical + SAR (Sensor Fusion)'
];

export const TopHeader: React.FC<TopHeaderProps> = ({
  breadcrumbSub = 'Vision-Language Assistant for Satellite Imagery',
  activeSessionTitle,
  selectedMode,
  onSelectMode
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <header className="h-16 border-b border-[#E2DDD3] bg-white/85 backdrop-blur-md px-6 flex items-center justify-between shrink-0 shadow-xs z-10 select-none">
      {/* Breadcrumb Title */}
      <div className="flex items-center gap-2 text-xs md:text-sm">
        <span className="font-semibold text-[#0D1B2A]">SatQuery AI</span>
        <span className="text-[#8B98A5]">/</span>
        <span className="text-[#4E6B7C] font-normal truncate max-w-md">
          {activeSessionTitle ? (
            <>
              <span className="text-[#0D1B2A] font-medium">{activeSessionTitle}</span>
              <span className="text-[11px] text-[#8B98A5] ml-2 font-mono">Auto-Routed</span>
            </>
          ) : (
            breadcrumbSub
          )}
        </span>
      </div>

      {/* Controls: Mode Selector & Notification */}
      <div className="flex items-center gap-3 md:gap-4">
        {/* Mode Dropdown */}
        <div className="relative" ref={dropdownRef}>
          <button 
            onClick={() => setIsOpen(!isOpen)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-[#D5CFBF] bg-[#F9F8F5] text-xs font-medium text-[#0D1B2A] hover:bg-white hover:border-[#4E6B7C] transition-all shadow-2xs cursor-pointer"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0"></span>
            <span>Mode: <span className="font-semibold">{selectedMode}</span></span>
            <ChevronDown className={`w-3.5 h-3.5 text-[#4E6B7C] transition-transform ${isOpen ? 'rotate-180' : ''}`} />
          </button>

          {isOpen && (
            <div className="absolute right-0 mt-2 w-64 bg-white rounded-2xl shadow-xl border border-[#E2DDD3] p-1.5 z-50 text-xs animation-in fade-in slide-in-from-top-1">
              <div className="px-3 py-1.5 text-[10px] uppercase font-bold text-[#8B98A5] tracking-wider border-b border-[#F0ECE1]">
                Analysis Routing Mode
              </div>
              <div className="space-y-0.5 pt-1">
                {MODES.map((mode) => {
                  const isSelected = selectedMode === mode;
                  return (
                    <button
                      key={mode}
                      onClick={() => {
                        onSelectMode(mode);
                        setIsOpen(false);
                      }}
                      className={`w-full flex items-center justify-between px-3 py-2 rounded-xl transition-colors text-left cursor-pointer ${
                        isSelected 
                          ? 'bg-[#EBF0F5] font-semibold text-[#0D1B2A]' 
                          : 'text-[#4E6B7C] hover:bg-[#F9F8F5] hover:text-[#0D1B2A]'
                      }`}
                    >
                      <span>{mode}</span>
                      {isSelected && <Check className="w-3.5 h-3.5 text-blue-600 ml-2 shrink-0" />}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Notification Bell */}
        <button 
          title="Notifications"
          className="p-2 rounded-full text-[#4E6B7C] hover:text-[#0D1B2A] hover:bg-[#F2EFE7] transition-colors relative cursor-pointer"
        >
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-1.5 h-1.5 bg-blue-600 rounded-full"></span>
        </button>

        {/* Avatar */}
        <div className="w-8 h-8 rounded-full bg-[#0D1B2A] text-white flex items-center justify-center text-xs font-semibold shadow-2xs border border-[#E2DDD3]">
          AS
        </div>
      </div>
    </header>
  );
};
