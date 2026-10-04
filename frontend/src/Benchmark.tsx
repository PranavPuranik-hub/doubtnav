import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, BarChart3, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { API_ENDPOINTS } from './config';

interface ScenarioResult {
  success_rate_pct: number;
  collision_count: number;
  stuck_count: number;
  avg_steps: number;
  avg_path_length_m: number;
}

interface FullBenchmark {
  sim: Record<string, {
    baseline: ScenarioResult;
    doubtnav: ScenarioResult;
  }>;
}

export function Benchmark() {
  const [data, setData] = useState<FullBenchmark | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(API_ENDPOINTS.benchmark)
      .then(r => r.json())
      .then(d => {
        // Handle wrapper or direct
        setData(d.sim ? d : { sim: d });
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

  if (!data || !data.sim) {
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

  // Chart data: extract success rate across all 4 scenarios
  const chartData = Object.entries(data.sim).map(([scenario, res]) => ({
    name: scenario.replace('_', ' ').toUpperCase(),
    Baseline: res.baseline.success_rate_pct,
    DoubtNav: res.doubtnav.success_rate_pct,
  }));

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
           <p className="text-muted text-sm">Comparison of standard navigation vs. DoubtNav's uncertainty-aware planning across 4 scenarios.</p>
        </div>

        {/* Known Limitations Note */}
        <div className="bg-warning-amber/10 border border-warning-amber/50 rounded-lg p-4 flex gap-3 text-sm text-warning-amber">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <p>
            <strong>Known limitations:</strong> In simulated night and fog, degradation changes only the uncertainty signal, not what the planner can see, so this test cannot show a benefit. DoubtNav is slower or less safe there; cause under investigation.
          </p>
        </div>

        {/* Chart */}
        <div className="bg-panel p-6 rounded-lg border border-border shadow-subtle flex flex-col gap-6 h-[400px]">
           <h3 className="text-sm font-semibold text-muted uppercase tracking-wider">Success Rate (%)</h3>
           <div className="flex-1">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                 <CartesianGrid strokeDasharray="3 3" stroke="#232B36" vertical={false} />
                 <XAxis dataKey="name" stroke="#8B96A5" tick={{ fill: '#8B96A5' }} />
                 <YAxis stroke="#8B96A5" tick={{ fill: '#8B96A5' }} domain={[0, 100]} />
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

        {/* Table */}
        <div className="bg-panel rounded-lg border border-border shadow-subtle overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-background border-b border-border text-muted">
              <tr>
                <th className="px-4 py-3 font-semibold">Scenario</th>
                <th className="px-4 py-3 font-semibold">System</th>
                <th className="px-4 py-3 font-semibold">Success %</th>
                <th className="px-4 py-3 font-semibold">Collisions</th>
                <th className="px-4 py-3 font-semibold">Stuck</th>
                <th className="px-4 py-3 font-semibold">Avg Path (m)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {Object.entries(data.sim).map(([scenario, res]) => (
                <React.Fragment key={scenario}>
                  <tr className="hover:bg-background/50 transition-colors">
                    <td className="px-4 py-3 capitalize" rowSpan={2}>{scenario.replace('_', ' ')}</td>
                    <td className="px-4 py-3 font-medium">Baseline</td>
                    <td className="px-4 py-3">{res.baseline.success_rate_pct.toFixed(1)}%</td>
                    <td className="px-4 py-3">{res.baseline.collision_count}</td>
                    <td className="px-4 py-3">{res.baseline.stuck_count}</td>
                    <td className="px-4 py-3">{res.baseline.avg_path_length_m.toFixed(1)}</td>
                  </tr>
                  <tr className="hover:bg-background/50 transition-colors">
                    <td className="px-4 py-3 font-medium text-accent-teal">DoubtNav</td>
                    <td className="px-4 py-3 text-accent-teal">{res.doubtnav.success_rate_pct.toFixed(1)}%</td>
                    <td className="px-4 py-3">{res.doubtnav.collision_count}</td>
                    <td className="px-4 py-3">{res.doubtnav.stuck_count}</td>
                    <td className="px-4 py-3">{res.doubtnav.avg_path_length_m.toFixed(1)}</td>
                  </tr>
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>

      </main>
    </div>
  );
}
