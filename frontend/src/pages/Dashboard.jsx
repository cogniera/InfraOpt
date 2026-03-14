import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Cpu, Zap, Clock, MemoryStick, TrendingUp } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Area, AreaChart } from 'recharts';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-bright" style={{ padding: '10px 14px', fontSize: 13 }}>
      <div style={{ color: '#818cf8' }}>Latency: <strong>{payload[0]?.value}ms</strong></div>
      {payload[1] && <div style={{ color: '#a78bfa' }}>Throughput: <strong>{payload[1]?.value} rps</strong></div>}
    </div>
  );
};

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [s, h] = await Promise.all([
          fetch('/api/stats').then(r => r.json()),
          fetch('/api/latency-history').then(r => r.json()),
        ]);
        setStats(s);
        setHistory(h.history.map((p, i) => ({ i: i + 1, latency: p.latency_ms, throughput: p.throughput })));
      } catch {
        // backend not running, show placeholder
        setStats({ total_requests: '—', avg_latency_ms: '—', throughput_rps: '—', gpu_utilization: '—', memory_used_gb: '—', uptime_hours: '—' });
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const statCards = stats ? [
    { label: 'Total Requests', value: stats.total_requests, icon: Activity, color: '#6366f1', delay: 0 },
    { label: 'Avg Latency', value: stats.avg_latency_ms, unit: 'ms', icon: Clock, color: '#8b5cf6', delay: 0.05 },
    { label: 'Throughput', value: stats.throughput_rps, unit: 'rps', icon: TrendingUp, color: '#06b6d4', delay: 0.1 },
    { label: 'GPU Util', value: stats.gpu_utilization, unit: '%', icon: Cpu, color: '#10b981', delay: 0.15 },
    { label: 'Memory Used', value: stats.memory_used_gb, unit: 'GB', icon: MemoryStick, color: '#f59e0b', delay: 0.2 },
    { label: 'Uptime', value: stats.uptime_hours, unit: 'hrs', icon: Zap, color: '#ec4899', delay: 0.25 },
  ] : [];

  return (
    <div>
      <PageHeader title="Dashboard" subtitle="Real-time inference performance metrics" />

      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: '#64748b', padding: '40px 0' }}>
          <div className="spinner" />
          Loading metrics...
        </div>
      ) : (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 16, marginBottom: 32 }}>
            {statCards.map(c => <StatCard key={c.label} {...c} />)}
          </div>

          {history.length > 0 && (
            <motion.div
              className="glass"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              style={{ padding: 24 }}
            >
              <div style={{ marginBottom: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontWeight: 600, color: '#e2e8f0', fontSize: 15 }}>Latency & Throughput</div>
                  <div style={{ fontSize: 12, color: '#475569', marginTop: 2 }}>Last 10 minutes · 30s intervals</div>
                </div>
                <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#64748b' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ width: 20, height: 2, background: '#818cf8', display: 'inline-block', borderRadius: 1 }} /> Latency
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ width: 20, height: 2, background: '#a78bfa', display: 'inline-block', borderRadius: 1 }} /> Throughput
                  </span>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={history} margin={{ top: 4, right: 4, bottom: 0, left: -16 }}>
                  <defs>
                    <linearGradient id="latGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#818cf8" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="thrGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="i" tick={{ fill: '#475569', fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fill: '#475569', fontSize: 11 }} tickLine={false} axisLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="latency" stroke="#818cf8" strokeWidth={2} fill="url(#latGrad)" dot={false} />
                  <Area type="monotone" dataKey="throughput" stroke="#a78bfa" strokeWidth={2} fill="url(#thrGrad)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </motion.div>
          )}
        </>
      )}
    </div>
  );
}
