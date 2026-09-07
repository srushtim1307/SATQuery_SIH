import React from 'react';
import type { ImageAttachment } from '../../types';

interface UserMessageProps {
  text: string;
  timestamp?: string;
  attachments?: ImageAttachment[];
}

export const UserMessage: React.FC<UserMessageProps> = ({
  text,
  timestamp = 'Just now',
  attachments = []
}) => {
  return (
    <div className="flex flex-col items-end space-y-2 select-none animation-in fade-in slide-in-from-bottom-2 duration-300">
      <div className="text-[11px] text-[#8B98A5] font-medium pr-1">
        You • {timestamp}
      </div>

      <div className="max-w-xl bg-[#0D1B2A] text-white p-5 rounded-3xl rounded-tr-md shadow-md space-y-3">
        <p className="text-sm font-normal leading-relaxed">{text}</p>

        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 pt-2 border-t border-white/10">
            {attachments.map((file, idx) => (
              <div 
                key={idx} 
                className="flex items-center gap-2.5 px-3 py-1.5 rounded-xl bg-white/10 border border-white/15 text-xs text-white"
              >
                <div className="w-5 h-5 rounded bg-white/20 flex items-center justify-center text-[10px] font-mono">
                  tif
                </div>
                <div className="overflow-hidden">
                  <div className="font-mono text-[11px] font-medium truncate max-w-[150px]">{file.name}</div>
                  <div className="text-[10px] text-[#C8D9E6]">{file.detail}</div>
                </div>
                <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  {file.status}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
