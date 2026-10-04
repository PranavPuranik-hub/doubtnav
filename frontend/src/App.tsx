import { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { 
  Activity, 
  Map, 
  Camera, 
  ShieldAlert, 
  AlertTriangle, 
  Play, 
  Gamepad2,
  CloudFog,
  Moon,
  Sun,
  Zap,
  Target,
  RefreshCw,
  Box,
  BarChart3,
  Video
} from 'lucide-react';
import { WS_BASE, API_ENDPOINTS } from './config';
import { Benchmark } from './Benchmark';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

// Utility for merging tailwind classes safely
function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

type Mode = 'video' | 'sim';
type Degradation = 'none' | 'night' | 'fog' | 'glare';

interface FrameData {
  overlay?: string;
  uncertainty_heatmap?: string;
  costmap?: string;
  confidence: number;
  mode: string;
  fps: number;
  trajectory: { x: number; y: number }[];
  event_log: { timestamp: number; message: string }[];
}

function Dashboard() {
  const [activeTab, setActiveTab] = useState<'camera' | 'uncertainty' | 'costmap'>('camera');
  const [mode, setMode] = useState<Mode>('video');
  const [clips, setClips] = useState<string[]>([]);
  const [selectedClip, setSelectedClip] = useState<string>('');
  const [degradation, setDegradation] = useState<Degradation>('none');
  const [frameData, setFrameData] = useState<FrameData | null>(null);
  const [connected, setConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  // Fetch clips on mount
  useEffect(() => {
    fetch(API_ENDPOINTS.clips)
      .then(r => r.json())
      .then(data => {
        if (data.clips && data.clips.length > 0) {
          setClips(data.clips);
          setSelectedClip(data.clips[0]);
        }
      })
      .catch(err => console.error("Failed to fetch clips:", err));
  }, []);

  // WebSocket connection
  useEffect(() => {
    function connect() {
      ws.current = new WebSocket(WS_BASE);
      ws.current.onopen = () => setConnected(true);
      ws.current.onclose = () => {
        setConnected(false);
        setTimeout(connect, 2000); // Reconnect loop
      };
      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setFrameData(data);
        } catch (e) {
          console.error("Failed to parse WS message", e);
        }
      };
    }
    connect();
    return () => {
      if (ws.current) {
        ws.current.onclose = null;
        ws.current.close();
      }
    };
  }, []);

  // Send session config when controls change
  useEffect(() => {
    if (mode === 'video' && !selectedClip) return;
    fetch(API_ENDPOINTS.session, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode, clip: selectedClip, degradation }),
    }).catch(err => console.error("Failed to set session:", err));
  }, [mode, selectedClip, degradation]);

  const handleSimAction = async (endpoint: string, payload: any = {}) => {
    try {
      await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    } catch (err) {
      console.error(`Failed to call ${endpoint}:`, err);
    }
  };

  const getModeColor = (m: string) => {
    if (m === 'NORMAL') return 'text-accent-teal border-accent-teal bg-accent-teal/10';
    if (m === 'CAUTIOUS') return 'text-warning-amber border-warning-amber bg-warning-amber/10';
    if (m === 'DEAD_RECKONING') return 'text-danger-coral border-danger-coral bg-danger-coral/10';
    return 'text-muted border-border bg-panel';
  };

  const activeImage = frameData
    ? activeTab === 'camera' ? frameData.overlay
      : activeTab === 'uncertainty' ? frameData.uncertainty_heatmap
      : frameData.costmap
    : null;

  return (
    <div className="flex flex-col h-screen bg-background text-text overflow-hidden">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-4 py-2 bg-panel border-b border-border shadow-subtle shrink-0">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-6 h-6 text-accent-teal" />
            <h1 className="text-lg font-bold">DoubtNav</h1>
          </div>
          <span className="text-sm text-muted hidden md:inline">A UGV that knows what it doesn't know</span>
        </div>
        <div className="flex items-center gap-6">
          <Link to="/benchmark" className="text-sm text-muted hover:text-text transition-colors flex items-center gap-1">
            <BarChart3 className="w-4 h-4" /> Benchmark
          </Link>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted">Status:</span>
            <div className="flex items-center gap-1">
              <div className={cn("w-2 h-2 rounded-full", connected ? "bg-accent-teal" : "bg-danger-coral")} />
              <span className="text-sm">{connected ? "Connected" : "Reconnecting..."}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted">FPS:</span>
            <span className="text-sm font-mono font-medium">{frameData?.fps.toFixed(1) || '--'}</span>
          </div>
          <div className={cn("px-3 py-1 rounded-full text-xs font-semibold border transition-colors duration-200", getModeColor(frameData?.mode || ''))}>
            {frameData?.mode || 'WAITING'}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex flex-1 overflow-hidden">
        {/* Left: Main View (65%) */}
        <section className="w-[65%] flex flex-col p-4 gap-4 border-r border-border">
          {/* Tabs */}
          <div className="flex gap-2">
            {[
              { id: 'camera', label: 'Camera', icon: Camera },
              { id: 'uncertainty', label: 'Uncertainty', icon: AlertTriangle },
              { id: 'costmap', label: 'Costmap', icon: Map }
            ].map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded transition-colors duration-200 text-sm font-medium",
                    activeTab === tab.id ? "bg-panel text-text shadow-subtle border border-border" : "text-muted hover:text-text hover:bg-panel/50"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              )
            })}
          </div>

          {/* Image Display */}
          <div className="flex-1 bg-panel rounded flex items-center justify-center overflow-hidden border border-border relative shadow-subtle">
            {!connected ? (
              <div className="flex flex-col items-center text-muted gap-2 animate-pulse">
                <Activity className="w-8 h-8" />
                <p>Waiting for connection...</p>
              </div>
            ) : !activeImage ? (
              <div className="flex flex-col items-center text-muted gap-2">
                <Video className="w-8 h-8" />
                <p>No feed available</p>
              </div>
            ) : (
              <img 
                src={`data:image/jpeg;base64,${activeImage}`} 
                alt={`${activeTab} view`} 
                className="w-full h-full object-contain"
              />
            )}
          </div>
        </section>

        {/* Right: Sidebar (35%) */}
        <section className="w-[35%] flex flex-col p-4 gap-4 bg-background overflow-y-auto">
          {/* Confidence Gauge */}
          <div className="bg-panel p-4 rounded border border-border shadow-subtle flex flex-col gap-2">
            <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">Localization Confidence</h3>
            <div className="flex items-center gap-4">
              <span className="text-2xl font-bold font-mono">
                {frameData ? (frameData.confidence * 100).toFixed(0) : '--'}%
              </span>
              <div className="flex-1 h-3 bg-background rounded-full overflow-hidden border border-border">
                <div 
                  className={cn(
                    "h-full transition-all duration-200 ease-out",
                    (frameData?.confidence ?? 1) >= 0.6 ? "bg-accent-teal" :
                    (frameData?.confidence ?? 1) >= 0.35 ? "bg-warning-amber" : "bg-danger-coral"
                  )}
                  style={{ width: `${Math.max(0, Math.min(100, (frameData?.confidence || 0) * 100))}%` }}
                />
              </div>
            </div>
          </div>

          {/* Trajectory Map (simplified visual) */}
          <div className="bg-panel p-4 rounded border border-border shadow-subtle flex flex-col gap-2 flex-1 min-h-[200px]">
             <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">Trajectory Map</h3>
             <div className="flex-1 bg-background rounded border border-border relative overflow-hidden">
                {frameData?.trajectory && frameData.trajectory.length > 0 ? (
                  (() => {
                    const currentPos = frameData.trajectory[frameData.trajectory.length - 1];
                    const range = 25; // 25 meters view range
                    const minX = currentPos.x - range;
                    const minY = currentPos.y - range;
                    return (
                      <svg className="absolute inset-0 w-full h-full" viewBox={`${minX} ${minY} ${range * 2} ${range * 2}`} preserveAspectRatio="xMidYMid meet">
                        {/* Origin */}
                        <circle cx="0" cy="0" r="0.8" fill="#8B96A5" />
                        {/* Path */}
                        <polyline 
                          points={frameData.trajectory.map(p => `${p.x},${p.y}`).join(' ')} 
                          fill="none" 
                          stroke="#3FB6A8" 
                          strokeWidth="0.6" 
                        />
                        {/* Current Pos */}
                        <circle 
                          cx={currentPos.x} 
                          cy={currentPos.y} 
                          r="1.5" 
                          fill="#3FB6A8" 
                        />
                      </svg>
                    );
                  })()
                ) : (
                  <div className="flex h-full items-center justify-center text-muted text-sm">Waiting for pose data...</div>
                )}
             </div>
          </div>

          {/* Event Timeline */}
          <div className="bg-panel p-4 rounded border border-border shadow-subtle flex flex-col gap-2 flex-1 max-h-[300px]">
            <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">Event Timeline</h3>
            <div className="flex-1 overflow-y-auto flex flex-col gap-2 font-mono text-xs">
               {frameData?.event_log && frameData.event_log.length > 0 ? (
                 frameData.event_log.slice().reverse().map((ev, i) => (
                   <div key={i} className="flex gap-2 p-2 rounded bg-background border border-border text-muted">
                      <span className="opacity-50">{(new Date(ev.timestamp * 1000)).toISOString().substr(11,8)}</span>
                      <span className={ev.message.includes("dead-reckoning") ? "text-danger-coral" : "text-text"}>{ev.message}</span>
                   </div>
                 ))
               ) : (
                 <div className="text-center text-muted mt-4">No events yet.</div>
               )}
            </div>
          </div>
        </section>
      </main>

      {/* Bottom Control Bar */}
      <footer className="bg-panel border-t border-border p-4 shadow-subtle shrink-0 flex items-center justify-between gap-4">
        
        {/* Mode & Source */}
        <div className="flex items-center gap-4">
          <div className="flex bg-background rounded p-1 border border-border">
            <button 
              onClick={() => setMode('video')}
              className={cn("px-3 py-1.5 rounded text-sm font-medium flex items-center gap-2 transition-colors", mode === 'video' ? 'bg-panel text-text shadow-sm' : 'text-muted hover:text-text')}
            >
              <Play className="w-4 h-4" /> Video Replay
            </button>
            <button 
              onClick={() => setMode('sim')}
              className={cn("px-3 py-1.5 rounded text-sm font-medium flex items-center gap-2 transition-colors", mode === 'sim' ? 'bg-panel text-text shadow-sm' : 'text-muted hover:text-text')}
            >
              <Gamepad2 className="w-4 h-4" /> Simulator
            </button>
          </div>

          {mode === 'video' && clips.length > 0 && (
            <select 
              value={selectedClip} 
              onChange={e => setSelectedClip(e.target.value)}
              className="bg-background border border-border text-text text-sm rounded px-3 py-2 outline-none focus:border-accent-teal transition-colors"
            >
              {clips.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          )}
        </div>

        {/* Degradation Toggles */}
        <div className="flex items-center gap-2 bg-background p-1 rounded border border-border">
           {[
             { id: 'none', icon: Sun, label: 'Clear' },
             { id: 'night', icon: Moon, label: 'Night' },
             { id: 'fog', icon: CloudFog, label: 'Fog' },
             { id: 'glare', icon: Zap, label: 'Glare' },
           ].map(deg => {
             const Icon = deg.icon;
             return (
               <button 
                 key={deg.id}
                 onClick={() => setDegradation(deg.id as Degradation)}
                 title={deg.label}
                 className={cn("p-2 rounded transition-colors", degradation === deg.id ? 'bg-panel text-accent-teal shadow-sm' : 'text-muted hover:text-text')}
               >
                 <Icon className="w-4 h-4" />
               </button>
             );
           })}
        </div>

        {/* Sim Controls */}
        <div className="flex items-center gap-2">
           <button 
             onClick={() => handleSimAction(API_ENDPOINTS.spawnObstacle)}
             disabled={mode !== 'sim'}
             className="px-4 py-2 bg-background border border-border rounded text-sm font-medium text-text hover:bg-panel hover:text-danger-coral disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
           >
             <Box className="w-4 h-4" /> Drop obstacle
           </button>
           <button 
             onClick={() => handleSimAction(API_ENDPOINTS.goal, { x: 10 + Math.random() * 40, y: 10 + Math.random() * 40 })}
             disabled={mode !== 'sim'}
             className="px-4 py-2 bg-background border border-border rounded text-sm font-medium text-text hover:bg-panel hover:text-accent-teal disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
           >
             <Target className="w-4 h-4" /> Set goal
           </button>
           <button 
             onClick={() => handleSimAction(API_ENDPOINTS.reset)}
             disabled={mode !== 'sim'}
             className="px-4 py-2 bg-background border border-border rounded text-sm font-medium text-text hover:bg-panel disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
           >
             <RefreshCw className="w-4 h-4" /> Reset
           </button>
        </div>

      </footer>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/benchmark" element={<Benchmark />} />
      </Routes>
    </BrowserRouter>
  );
}
