import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import type { AnalysisSession, ImageAttachment, StagedFile } from '../types';
import { EmptyGreeting } from '../components/chat/EmptyGreeting';
import { UserMessage } from '../components/chat/UserMessage';
import { AssistantMessage } from '../components/chat/AssistantMessage';
import { ChatInput } from '../components/chat/ChatInput';
import { ProcessingStepper } from '../components/analysis/ProcessingStepper';

interface MainWorkspaceViewProps {
  sessions: AnalysisSession[];
  stagedFiles?: StagedFile[];
  onAddNewQuery: (query: string, attachments: ImageAttachment[]) => string;
  onRemoveStagedFile?: (index: number) => void;
}

export const MainWorkspaceView: React.FC<MainWorkspaceViewProps> = ({
  sessions,
  stagedFiles = [],
  onAddNewQuery,
  onRemoveStagedFile
}) => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  // Find active session
  const activeSession = id ? sessions.find((s) => s.id === id) : undefined;
  
  // Track active attachments for conversation context or staged files
  const activeAttachments: ImageAttachment[] = activeSession?.userMessage?.attachments?.length
    ? activeSession.userMessage.attachments
    : stagedFiles.map((sf) => ({
        name: sf.name,
        detail: sf.modalityTag || `${(sf.modality || 'OPTICAL').toUpperCase()} · Ready`,
        status: 'Ready' as const,
        modality: sf.modality === 'sar' ? ('SAR' as const) : (sf.modality === 'multispectral' ? ('Multispectral' as const) : ('Optical' as const)),
        thumbnailUrl: sf.thumbnailUrl,
        assetId: sf.id
      }));

  // Track processing state
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [pendingSession, setPendingSession] = useState<AnalysisSession | null>(null);

  useEffect(() => {
    // If navigating to a new query that is flagged for processing
    if (activeSession && (window as any).__SATQUERY_TRIGGER_PROCESS) {
      delete (window as any).__SATQUERY_TRIGGER_PROCESS;
      setPendingSession(activeSession);
      setIsProcessing(true);
    } else {
      setIsProcessing(false);
      setPendingSession(null);
    }
  }, [id, activeSession]);

  const handleSendMessage = (query: string, attachments: ImageAttachment[]) => {
    // Prioritize passed attachments, fallback to active session / staged files
    const finalAttachments = attachments.length > 0 ? attachments : activeAttachments;
    const newSessionId = onAddNewQuery(query, finalAttachments);
    (window as any).__SATQUERY_TRIGGER_PROCESS = true;
    navigate(`/analysis/${newSessionId}`);
  };

  const handleSelectSuggestion = (query: string) => {
    // Prioritize user's actual uploaded files if available
    let attachments: ImageAttachment[] = activeAttachments;
    if (attachments.length === 0) {
      if (query.toLowerCase().includes('water') || query.toLowerCase().includes('describe')) {
        attachments = [{
          name: 'sentinel2_estuary_delta.tif',
          detail: 'Optical (Sentinel-2 L1C) · 10m GSD',
          status: 'Ready',
          modality: 'Multispectral'
        }];
      } else if (query.toLowerCase().includes('change')) {
        attachments = [
          { name: 'sentinel2_may2023.tif', detail: 'Optical · Baseline · Ready', status: 'Ready', modality: 'Optical' },
          { name: 'sentinel2_oct2024.tif', detail: 'Optical · Current · Ready', status: 'Ready', modality: 'Optical' }
        ];
      } else if (query.toLowerCase().includes('sar') || query.toLowerCase().includes('built-up')) {
        attachments = [
          { name: 'sentinel2_estuary_delta.tif', detail: 'Optical · Sentinel-2 L1C · Ready', status: 'Ready', modality: 'Optical' },
          { name: 'gulf_coastal_sar.tif', detail: 'SAR · Sentinel-1 C-Band · Ready', status: 'Ready', modality: 'SAR' }
        ];
      }
    }

    handleSendMessage(query, attachments);
  };

  const handleProcessingComplete = () => {
    setIsProcessing(false);
    setPendingSession(null);
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-[#F2EFE7]">
      {/* Scrollable Conversation Workspace */}
      <div className="flex-1 overflow-y-auto px-4 py-8 flex flex-col items-center">
        <div className="max-w-3xl w-full flex-1 flex flex-col justify-center space-y-6">
          {/* 1. Empty State when no session is selected */}
          {!activeSession && !isProcessing && (
            <EmptyGreeting onSelectSuggestion={handleSelectSuggestion} />
          )}

          {/* 2. Processing Stepper (Teal reference UI) */}
          {isProcessing && pendingSession && (
            <ProcessingStepper 
              session={pendingSession} 
              onComplete={handleProcessingComplete} 
            />
          )}

          {/* 3. Completed Conversation Display */}
          {!isProcessing && activeSession && (
            <div className="space-y-6 w-full pb-6">
              {/* User Message */}
              <UserMessage
                text={activeSession.userMessage.text}
                timestamp={activeSession.userMessage.timestamp}
                attachments={activeSession.userMessage.attachments}
              />

              {/* Assistant Response with Evidence & Metrics */}
              <AssistantMessage session={activeSession} />
            </div>
          )}
        </div>
      </div>

      {/* Docked Chat Input Bar */}
      <ChatInput
        onSendMessage={handleSendMessage}
        isProcessing={isProcessing}
        onOpenUpload={() => navigate('/new-analysis')}
        stagedFiles={activeAttachments}
        onRemoveStagedFile={onRemoveStagedFile}
      />
    </div>
  );
};
