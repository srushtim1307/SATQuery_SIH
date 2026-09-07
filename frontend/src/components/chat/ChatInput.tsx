import React, { useState } from 'react';
import { Plus, ArrowUp, Loader2, X, FileCheck } from 'lucide-react';
import type { ImageAttachment } from '../../types';

interface ChatInputProps {
  onSendMessage: (query: string, attachments: ImageAttachment[]) => void;
  isProcessing?: boolean;
  onOpenUpload?: () => void;
  stagedFiles?: ImageAttachment[];
  onRemoveStagedFile?: (index: number) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isProcessing = false,
  onOpenUpload,
  stagedFiles = [],
  onRemoveStagedFile
}) => {
  const [inputText, setInputText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isProcessing) return;
    onSendMessage(inputText.trim(), stagedFiles);
    setInputText('');
  };

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-6 pt-2 select-none">
      <form 
        onSubmit={handleSubmit}
        className="w-full bg-white rounded-3xl border border-[#D5CFBF] shadow-lg p-2 transition-all focus-within:border-[#0D1B2A] focus-within:shadow-xl"
      >
        {/* Staged Image Attachment Chips */}
        {stagedFiles.length > 0 && (
          <div className="flex flex-wrap gap-2 px-3 pt-2 pb-1.5 border-b border-[#F0ECE1]">
            {stagedFiles.map((file, idx) => (
              <div 
                key={idx}
                className="flex items-center gap-2 px-2.5 py-1 rounded-xl bg-[#F6F4EE] border border-[#E2DDD3] text-xs text-[#0D1B2A]"
              >
                <FileCheck className="w-3.5 h-3.5 text-blue-600" />
                <span className="font-mono text-[11px] truncate max-w-[140px]">{file.name}</span>
                {onRemoveStagedFile && (
                  <button
                    type="button"
                    onClick={() => onRemoveStagedFile(idx)}
                    className="p-0.5 rounded-full hover:bg-black/10 text-[#8B98A5] hover:text-[#0D1B2A] transition-colors cursor-pointer"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Input Controls Row */}
        <div className="flex items-center gap-2 pl-2 pr-1 py-1">
          {/* Upload Button */}
          <button
            type="button"
            onClick={onOpenUpload}
            disabled={isProcessing}
            className="flex items-center gap-1.5 px-3 py-2 rounded-full hover:bg-[#F2EFE7] text-xs font-medium text-[#4E6B7C] hover:text-[#0D1B2A] transition-colors disabled:opacity-50 cursor-pointer shrink-0"
          >
            <Plus className="w-4 h-4" />
            <span>Upload</span>
          </button>

          {/* Text Input Field */}
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isProcessing}
            placeholder={isProcessing ? "SatQuery AI is processing your analysis..." : "Ask something about your image..."}
            className="flex-1 bg-transparent px-2 py-2 text-sm text-[#0D1B2A] placeholder-[#8B98A5] outline-none disabled:cursor-not-allowed font-normal"
          />

          {/* Submit Button */}
          <button
            type="submit"
            disabled={!inputText.trim() || isProcessing}
            className="w-10 h-10 rounded-full bg-[#0D1B2A] hover:bg-[#1E2E42] disabled:bg-[#D5CFBF] text-white flex items-center justify-center transition-all disabled:cursor-not-allowed cursor-pointer shrink-0 shadow-xs"
          >
            {isProcessing ? (
              <Loader2 className="w-4 h-4 text-white animate-spin" />
            ) : (
              <ArrowUp className="w-4 h-4" />
            )}
          </button>
        </div>
      </form>
      
      {/* Shortcut note */}
      <div className="text-[10px] text-center text-[#8B98A5] pt-2">
        Press <span className="font-mono font-semibold">Enter ↵</span> to submit query • SatQuery automatically auto-routes to the best specialist
      </div>
    </div>
  );
};
