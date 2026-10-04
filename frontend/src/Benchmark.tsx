import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, BarChart3, CheckCircle2, AlertTriangle, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { API_ENDPOINTS } from './config';

interface BenchmarkResult {
  baseline: {
    success_rate: number;
    avg_collisions: number;
    avg_path_length: number;
  };
  doubtnav: {
    success_rate: number;
    avg_collisions: number;
    avg_path_length: number;
  };
}

export function Benchmark() {
  const [data, setData] = useState<BenchmarkResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(API_ENDPOINTS.benchmark)
      .then(r => r.json())
      .then(d => {
        setData(d);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load benchmark:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="flex h-screen bg-background items-center justify-center text-muted">
        Loading benchmark results...
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-screen bg-background flex-col gap-4 items-center justify-center text-danger-coral">
        <AlertCircle className="w-12 h-12" />
        <p>Failed to load benchmark data.</p>
        <Link to="/" className="text-accent-teal hover:underline flex items-center gap-2">
          <ArrowLeft className="w-4 h-4" /> Back to Dashboard
        </Link>
      </div>
    );
  }

  const chartData = [
    {
      name: 'Success Rate (%)',
      Baseline: data.baseline.success_rate,
      DoubtNav: data.doubtnav.success_rate,
    },
    {
      name: 'Avg Collisions',
      Baseline: data.baseline.avg_collisions * 10, // scaled for visibility if needed, but let's keep raw and rely on multiple charts or dual axis
      DoubtNav: data.doubtnav.avg_collisions * 10,
    }
  ];

  return (
    <div className="flex flex-col min-h-screen bg-background text-text">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-4 py-3 bg-panel border-b border-border shadow-subtle shrink-0">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-muted hover:text-text transition-colors p-1 rounded hover:bg-background border border-transparent hover:border-border">
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div className="flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-accent-teal" />
            <h1 className="text-lg font-bold">Benchmark Results</h1>
          </div>
        </div>
      </header>

      <main className="flex-1 p-8 max-w-5xl mx-auto w-full flex flex-col gap-8">
        
        <div className="flex flex-col gap-2">
           <h2 className="text-xl font-semibold">Simulator Benchmark (20 Seeds)</h2>
           <p className="text-muted text-sm">Comparison of standard navigation vs. DoubtNav's uncertainty-aware planning in degraded conditions.</p>
        </div>

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
           {/* DoubtNav */}
           <div className="bg-panel p-6 rounded-lg border border-accent-teal/30 shadow-subtle flex flex-col gap-4 relative overflow-hidden">
             <div className="absolute top-0 right-0 w-32 h-32 bg-accent-teal/5 rounded-bl-full -z-10" />
             <div className="flex items-center gap-2 text-accent-teal">
               <CheckCircle2 className="w-5 h-5" />
               <h3 className="font-bold text-lg">DoubtNav</h3>
             </div>
             <div className="grid grid-cols-3 gap-4">
                <div className="flex flex-col gap-1">
                   <span className="text-xs text-muted uppercase">Success Rate</span>
                   <span className="text-2xl font-mono">{data.doubtnav.success_rate.toFixed(1)}%</span>
                </div>
                <div className="flex flex-col gap-1">
                   <span className="text-xs text-muted uppercase">Collisions</span>
                   <span className="text-2xl font-mono">{data.doubtnav.avg_collisions.toFixed(2)}</span>
                </div>
                <div className="flex flex-col gap-1">
                   <span className="text-xs text-muted uppercase">Path Len</span>
                   <span className="text-2xl font-mono">{data.doubtnav.avg_path_length.toFixed(1)}m</span>
                </div>
             </div>
           </div>

           {/* Baseline */}
           <div className="bg-panel p-6 rounded-lg border border-border shadow-subtle flex flex-col gap-4 relative overflow-hidden">
             <div className="absolute top-0 right-0 w-32 h-32 bg-danger-coral/5 rounded-bl-full -z-10" />
             <div className="flex items-center gap-2 text-muted">
               <AlertTriangle className="w-5 h-5" />
               <h3 className="font-bold text-lg text-text">Baseline (Standard A*)</h3>
             </div>
             <div className="grid grid-cols-3 gap-4">
                <div className="flex flex-col gap-1">
                   <span className="text-xs text-muted uppercase">Success Rate</span>
                   <span className="text-2xl font-mono">{data.baseline.success_rate.toFixed(1)}%</span>
                </div>
                <div className="flex flex-col gap-1">
                   <span className="text-xs text-muted uppercase">Collisions</span>
                   <span className="text-2xl font-mono text-danger-coral">{data.baseline.avg_collisions.toFixed(2)}</span>
                </div>
                <div className="flex flex-col gap-1">
                   <span className="text-xs text-muted uppercase">Path Len</span>
                   <span className="text-2xl font-mono">{data.baseline.avg_path_length.toFixed(1)}m</span>
                </div>
             </div>
           </div>
        </div>

        {/* Chart */}
        <div className="bg-panel p-6 rounded-lg border border-border shadow-subtle flex flex-col gap-6 h-[400px]">
           <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">Performance Comparison</h3>
           <div className="flex-1">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#232B36" vertical={false} />
                 <XAxis dataKey="name" stroke="#8B96A5" tick={{ fill: '#8B96A5' }} />
                 <YAxis stroke="#8B96A5" tick={{ fill: '#8B96A5' }} />
                 <Tooltip 
                   cursor={{ fill: '#232B36', opacity: 0.4 }}
                   contentStyle={{ backgroundColor: '#161C24', borderColor: '#232B36', borderRadius: '8px', color: '#E6EAF0' }}
                 />
                 <Legend wrapperStyle={{ paddingTop: '20px' }} />
                 <Bar dataKey="Baseline" fill="#E07A7A" radius={[4, 4, 0, 0]} maxBarSize={60} />
                 <Bar dataKey="DoubtNav" fill="#3FB6A8" radius={[4, 4, 0, 0]} maxBarSize={60} />
               </BarChart>
             </ResponsiveContainer>
           </div>
        </div>
      </main>
    </div>
  );
}
