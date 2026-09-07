import React, { useState } from 'react';
import { Maximize2, Download, Layers } from 'lucide-react';
import type { SingleGroundingEvidence } from '../../types';

interface EvidenceViewerProps {
  evidence: SingleGroundingEvidence;
}

export const EvidenceViewer: React.FC<EvidenceViewerProps> = ({ evidence }) => {
  const [isFullFrame, setIsFullFrame] = useState(false);
  const [showMask, setShowMask] = useState(true);

  const handleDownloadMask = () => {
    alert('Simulating export of GeoTIFF Byte Mask (EPSG:4326, 10m resolution) for detected regions.');
  };

  return (
    <div className="space-y-2 mt-4 select-none">
      {/* Evidence Section Header & Actions */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-bold tracking-wider uppercase text-[#4E6B7C]">
            Visual Evidence
          </span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#EBF0F5] text-[#4E6B7C]">
            {evidence.boundingBoxes.length} Features Grounded
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowMask(!showMask)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors cursor-pointer ${
              showMask
                ? 'bg-[#EBF0F5] border-[#B8CEDD] text-[#0D1B2A]'
                : 'bg-white border-[#E2DDD3] text-[#4E6B7C] hover:bg-[#F9F8F5]'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Overlays {showMask ? 'On' : 'Off'}</span>
          </button>

          <button
            onClick={() => setIsFullFrame(!isFullFrame)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-white border border-[#E2DDD3] text-[#4E6B7C] hover:text-[#0D1B2A] hover:bg-[#F9F8F5] transition-colors cursor-pointer"
          >
            <Maximize2 className="w-3.5 h-3.5" />
            <span>Full Frame</span>
          </button>

          <button
            onClick={handleDownloadMask}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium bg-white border border-[#E2DDD3] text-[#4E6B7C] hover:text-[#0D1B2A] hover:bg-[#F9F8F5] transition-colors cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>GeoTIFF Mask</span>
          </button>
        </div>
      </div>

      {/* Satellite Imagery Display Container */}
      <div 
        className={`relative overflow-hidden rounded-2xl border border-[#D5CFBF] bg-[#0D1B2A] shadow-md transition-all ${
          isFullFrame ? 'fixed inset-4 z-50 rounded-3xl' : 'w-full aspect-[16/9]'
        }`}
      >
        <img
          src={evidence.imageUrl}
          alt="Satellite visual evidence"
          className="w-full h-full object-cover"
        />

        {/* Bounding Boxes Overlay */}
        {showMask && evidence.boundingBoxes.map((box) => (
          <div
            key={box.id}
            style={{
              top: `${box.top}%`,
              left: `${box.left}%`,
              width: `${box.width}%`,
              height: `${box.height}%`
            }}
            className="absolute rounded-xl border-2 border-blue-400 bg-blue-500/20 backdrop-blur-[1px] shadow-[0_0_15px_rgba(59,130,246,0.3)] transition-all hover:bg-blue-500/30 group cursor-pointer"
          >
            {/* Box Tag Label */}
            <div className="absolute -top-3 left-3 bg-white/95 backdrop-blur-sm text-[#0D1B2A] text-[10px] font-semibold px-2 py-0.5 rounded-md shadow-sm border border-blue-200 flex items-center gap-1.5 whitespace-nowrap">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-600"></span>
              <span>{box.label} [{Math.round(box.confidence * 100)}%]</span>
            </div>
          </div>
        ))}

        {/* Bottom Right Scale & Sensor Badge */}
        <div className="absolute bottom-3 right-3 flex items-center gap-2 bg-black/60 backdrop-blur-md px-3 py-1 rounded-lg border border-white/10 text-white text-[10px] font-mono tracking-wide">
          <span>{evidence.scaleLabel}</span>
          <span className="text-white/40">|</span>
          <span className="text-cyan-200 font-medium">{evidence.satelliteLabel}</span>
        </div>

        {/* Full Frame Close Button */}
        {isFullFrame && (
          <button
            onClick={() => setIsFullFrame(false)}
            className="absolute top-4 right-4 bg-black/70 hover:bg-black text-white px-3 py-1.5 rounded-full text-xs font-medium cursor-pointer"
          >
            Exit Full Frame
          </button>
        )}
      </div>
    </div>
  );
};
