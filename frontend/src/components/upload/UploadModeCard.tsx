import React from 'react';
import { Layers, SplitSquareVertical, Sparkles } from 'lucide-react';

interface UploadModeCardProps {
  id: 'single' | 'optical_sar' | 'bi_temporal';
  title: string;
  subtitle: string;
  buttonLabel: string;
  isRecommended?: boolean;
  onSelect: () => void;
}

export const UploadModeCard: React.FC<UploadModeCardProps> = ({
  id,
  title,
  subtitle,
  buttonLabel,
  isRecommended,
  onSelect
}) => {
  const getIcon = () => {
    switch (id) {
      case 'single':
        return (
          <div className="w-10 h-10 rounded-2xl bg-[#F6F4EE] flex items-center justify-center text-[#0D1B2A]">
            <Sparkles className="w-5 h-5 text-[#0D1B2A]" />
          </div>
        );
      case 'optical_sar':
        return (
          <div className="w-10 h-10 rounded-2xl bg-[#0D1B2A] flex items-center justify-center text-white">
            <Layers className="w-5 h-5 text-[#C8D9E6]" />
          </div>
        );
      case 'bi_temporal':
        return (
          <div className="w-10 h-10 rounded-2xl bg-[#F6F4EE] flex items-center justify-center text-[#0D1B2A]">
            <SplitSquareVertical className="w-5 h-5 text-[#0D1B2A]" />
          </div>
        );
    }
  };

  return (
    <div 
      className={`relative p-7 rounded-3xl bg-white transition-all flex flex-col justify-between select-none ${
        isRecommended
          ? 'border-2 border-[#0D1B2A] shadow-md ring-1 ring-[#0D1B2A]/5'
          : 'border border-[#E2DDD3] shadow-xs hover:border-[#4E6B7C]'
      }`}
    >
      {isRecommended && (
        <div className="absolute -top-3 right-6 bg-[#0D1B2A] text-white text-[10px] font-bold tracking-wider uppercase px-3 py-0.5 rounded-full shadow-xs">
          RECOMMENDED
        </div>
      )}

      <div className="space-y-4">
        {getIcon()}
        <div>
          <h3 className="text-base font-bold text-[#0D1B2A]">{title}</h3>
          <p className="text-xs text-[#4E6B7C] mt-1 font-normal leading-relaxed">{subtitle}</p>
        </div>
      </div>

      <div className="pt-6">
        <button
          onClick={onSelect}
          className={`w-full py-2.5 px-4 rounded-full text-xs font-semibold transition-all cursor-pointer ${
            isRecommended
              ? 'bg-[#0D1B2A] hover:bg-[#1E2E42] text-white shadow-xs'
              : 'bg-[#F6F4EE] hover:bg-[#EAE6DD] text-[#0D1B2A] border border-[#E2DDD3]'
          }`}
        >
          {buttonLabel}
        </button>
      </div>
    </div>
  );
};
