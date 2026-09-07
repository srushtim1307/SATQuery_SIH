import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Plus, 
  Settings as SettingsIcon, 
  Satellite, 
  Compass, 
  History
} from 'lucide-react';
import type { AnalysisSession } from '../../types';

interface SidebarProps {
  sessions: AnalysisSession[];
  activeSessionId?: string;
  onOpenSettings: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  sessions, 
  activeSessionId, 
  onOpenSettings 
}) => {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <aside className="w-64 bg-white border-r border-[#E2DDD3] flex flex-col justify-between p-4 select-none shrink-0 shadow-sm h-full z-20">
      <div className="space-y-6">
        {/* Brand & Product Identity */}
        <div 
          onClick={() => navigate('/')}
          className="flex items-center gap-3 px-1 pt-1 cursor-pointer group"
        >
          <div className="w-9 h-9 rounded-xl bg-[#0D1B2A] flex items-center justify-center text-white shadow-sm transition-transform group-hover:scale-105">
            <Satellite className="w-5 h-5 text-[#C8D9E6]" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-base tracking-tight text-[#0D1B2A]">SatQuery AI</span>
              <span className="w-2 h-2 rounded-full bg-[#3B82F6] animate-pulse"></span>
            </div>
            <p className="text-[11px] font-medium text-[#4E6B7C]">Autonomous EO Vision</p>
          </div>
        </div>

        {/* Primary Action: + New Analysis */}
        <button 
          onClick={() => navigate('/new-analysis')}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-full bg-[#0D1B2A] hover:bg-[#1E2E42] text-white text-sm font-medium transition-all shadow-sm active:scale-98 cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          <span>New Analysis</span>
        </button>

        {/* Quick Nav Links */}
        <div className="space-y-1">
          <button
            onClick={() => navigate('/')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-medium transition-colors cursor-pointer ${
              location.pathname === '/' && !activeSessionId
                ? 'bg-[#EBF0F5] text-[#0D1B2A] font-semibold'
                : 'text-[#4E6B7C] hover:bg-[#F9F8F5] hover:text-[#0D1B2A]'
            }`}
          >
            <Compass className="w-4 h-4" />
            <span>Query Studio</span>
          </button>
          
          <button
            onClick={() => navigate('/history')}
            className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-xs font-medium transition-colors cursor-pointer ${
              location.pathname === '/history'
                ? 'bg-[#EBF0F5] text-[#0D1B2A] font-semibold'
                : 'text-[#4E6B7C] hover:bg-[#F9F8F5] hover:text-[#0D1B2A]'
            }`}
          >
            <History className="w-4 h-4" />
            <span>Analysis History</span>
          </button>
        </div>

        {/* Recent Sessions List */}
        <div className="space-y-2 pt-1">
          <div className="flex items-center justify-between px-2 text-[11px] font-semibold text-[#8B98A5] tracking-wider uppercase">
            <span>Recent Sessions</span>
          </div>

          <div className="space-y-1">
            {sessions.slice(0, 4).map((session) => {
              const isActive = activeSessionId === session.id;
              const dotColor = 
                session.modalityBadge === 'S2' ? 'bg-emerald-500' :
                session.modalityBadge === 'T1/T2' ? 'bg-blue-500' :
                session.modalityBadge === 'SAR' ? 'bg-amber-500' : 'bg-purple-500';

              return (
                <div 
                  key={session.id}
                  onClick={() => navigate(`/analysis/${session.id}`)}
                  className={`flex items-center justify-between p-2 rounded-xl cursor-pointer text-sm group transition-all ${
                    isActive 
                      ? 'bg-[#EBF0F5] border border-[#D0DDE7] shadow-xs' 
                      : 'hover:bg-[#F6F4EE] border border-transparent'
                  }`}
                >
                  <div className="flex items-center gap-2 overflow-hidden">
                    <span className={`w-2 h-2 rounded-full ${dotColor} shrink-0`}></span>
                    <span className={`truncate text-xs font-medium ${isActive ? 'text-[#0D1B2A] font-semibold' : 'text-[#2D3E4F]'}`}>
                      {session.title}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white border border-[#E2DDD3] text-[#4E6B7C] shrink-0">
                    {session.modalityBadge}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="space-y-3 pt-4 border-t border-[#EAE6DD]">
        <button 
          onClick={onOpenSettings}
          className="flex items-center gap-2.5 px-2 py-1.5 w-full rounded-lg text-xs font-medium text-[#4E6B7C] hover:text-[#0D1B2A] hover:bg-[#F6F4EE] transition-colors cursor-pointer"
        >
          <SettingsIcon className="w-4 h-4" />
          <span>Settings & Keys</span>
        </button>

        {/* User Profile Card */}
        <div className="flex items-center gap-2.5 p-2 rounded-xl bg-[#F6F4EE] border border-[#E8E4DA]">
          <div className="w-8 h-8 rounded-full bg-[#0D1B2A] text-white flex items-center justify-center text-xs font-semibold shadow-xs">
            AS
          </div>
          <div className="overflow-hidden">
            <div className="text-xs font-semibold text-[#0D1B2A] truncate">Dr. Aya Sharma</div>
            <div className="text-[10px] text-[#4E6B7C] truncate">Principal Analyst</div>
          </div>
        </div>
      </div>
    </aside>
  );
};
