import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Zap, TrendingUp, Shield } from 'lucide-react';
import PageHeader from '../components/PageHeader.jsx';

const STRATEGIES = [
  { id: 'speed', label: 'Speed', desc: 'Maximum throughput, minimal quality loss', icon: Zap, color: '#f59e0b' },
  { id: 'balanced', label: 'Balanced', desc: 'Optimal tradeoff between speed and quality', icon: TrendingUp, color: '#6366f1' },
  { id: 'quality', label: 'Quality', desc: 'Maximum output quality, minimal speedup', icon: Shield, color: '#10b981' },
];
const QUANT = [
  { id: '', label: 'None (FP32)' },
  { id: 'fp16', label: 'FP16 (Half Precision)' },
  { id: 'int8', label: 'INT8 (Quantized)' },
];

export default function Optimize() {
  const [strategy, setStrategy] = useState('balanced');
  const [quant, setQuant] = useState('fp16');
  const [batchSize, setBatchSize] = useState(4);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleApply = async () => {
    setLoading(true); setResult(null);
    try {
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy, batch_size: batchSize, quantization: quant || null }),
      });
      setResult(await res.json());
    } catch { setResult({ error: true }); }
    finally { setLoading(false); }
  };

  return (
    <div>
      <PageHeader title="Optimize" subtitle="Configure inference optimization strategy" />

      {/* Strategy picker */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>Strategy</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          {STRATEGIES.map(s => {
            const active = strategy === s.id;
            return (
              <motion.button
                key={s.id} onClick={() => setStrategy(s.id)}
                whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }}
                className={active ? 'glass-bright' : 'glass'}
                style={{
                  border: active ? `1px solid ${s.color}55` : '1px solid rgba(255,255,255,0.06)',
                  padding: '18px 20px', textAlign: 'left', cursor: 'pointer',
                  background: active ? `${s.color}11` : undefined,
                  boxShadow: active ? `0 0 20px ${s.color}22` : undefined,
                  transition: 'all 0.2s',
                }}
              >
                <s.icon size={20} color={active ? s.color : '#475569'} style={{ marginBottom: 10 }} />
                <div style={{ fontWeight: 600, color: active ? '#e2e8f0' : '#64748b', fontSize: 14, marginBottom: 4 }}>{s.label}</div>
                <div style={{ fontSize: 12, color: '#475569' }}>{s.desc}</div>
              </motion.button>
            );
          })}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
        <div className="glass" style={{ padding: 20 }}>
          <label style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>Quantization</label>
          <select value={quant} onChange={e => setQuant(e.target.value)}
            style={{ width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 14, outline: 'none' }}>
            {QUANT.map(q => <option key={q.id} value={q.id} style={{ background: '#0d1422' }}>{q.label}</option>)}
          </select>
        </div>
        <div className="glass" style={{ padding: 20 }}>
          <label style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>Batch Size: {batchSize}</label>
          <input type="range" min={1} max={32} step={1} value={batchSize} onChange={e => setBatchSize(+e.target.value)}
            style={{ width: '100%', accentColor: '#6366f1' }} />
        </div>
      </div>

      <motion.button className="btn-primary" onClick={handleApply} disabled={loading}
        whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
        style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {loading ? <div className="spinner" /> : <Zap size={15} />}
        {loading ? 'Applying...' : 'Apply Optimization'}
      </motion.button>

      {result && !result.error && (
        <motion.div className="glass" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: 24, padding: 24 }}>
          <div style={{ fontWeight: 600, color: '#e2e8f0', marginBottom: 16 }}>Projected Improvements</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
            {Object.entries(result.projected_improvements).map(([k, v]) => (
              <div key={k} style={{ background: 'rgba(0,0,0,0.2)', borderRadius: 10, padding: '16px 18px' }}>
                <div style={{ fontSize: 11, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>{k.replace(/_/g, ' ')}</div>
                <div style={{ fontWeight: 700, color: '#818cf8', fontSize: 18, fontFamily: 'JetBrains Mono, monospace' }}>{v}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 12, fontSize: 12, color: '#475569' }}>
            Strategy: <span style={{ color: '#818cf8' }}>{result.strategy}</span> · Batch: <span style={{ color: '#818cf8' }}>{result.batch_size}</span> · Quant: <span style={{ color: '#818cf8' }}>{result.quantization || 'none'}</span>
          </div>
        </motion.div>
      )}
    </div>
  );
}
