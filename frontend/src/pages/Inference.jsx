import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Send, Zap } from 'lucide-react';
import PageHeader from '../components/PageHeader.jsx';

const MODELS = ['llama-3-8b', 'mistral-7b', 'phi-3-mini', 'gemma-2b', 'qwen-14b'];

export default function Inference() {
  const [model, setModel] = useState('llama-3-8b');
  const [prompt, setPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    if (!prompt.trim()) return;
    setLoading(true); setResult(null); setError(null);
    try {
      const res = await fetch('/api/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, prompt, max_tokens: maxTokens, temperature }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <PageHeader title="Inference" subtitle="Run prompts against available models" />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div className="glass" style={{ padding: 20 }}>
          <label style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>Model</label>
          <select
            value={model} onChange={e => setModel(e.target.value)}
            style={{ width: '100%', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 14, outline: 'none' }}
          >
            {MODELS.map(m => <option key={m} value={m} style={{ background: '#0d1422' }}>{m}</option>)}
          </select>
        </div>
        <div className="glass" style={{ padding: 20 }}>
          <label style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>Max Tokens: {maxTokens}</label>
          <input type="range" min={64} max={1024} step={64} value={maxTokens} onChange={e => setMaxTokens(+e.target.value)}
            style={{ width: '100%', accentColor: '#6366f1' }} />
        </div>
      </div>
      <div className="glass" style={{ padding: 20, marginBottom: 16 }}>
        <label style={{ fontSize: 12, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 8 }}>Prompt</label>
        <textarea
          value={prompt} onChange={e => setPrompt(e.target.value)}
          placeholder="Enter your prompt here..."
          rows={5}
          style={{ width: '100%', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '12px 14px', color: '#e2e8f0', fontSize: 14, resize: 'vertical', outline: 'none', fontFamily: 'inherit', boxSizing: 'border-box' }}
        />
      </div>
      <motion.button
        className="btn-primary"
        onClick={handleRun}
        disabled={loading || !prompt.trim()}
        whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
        style={{ display: 'flex', alignItems: 'center', gap: 8, opacity: (!prompt.trim() || loading) ? 0.5 : 1 }}
      >
        {loading ? <div className="spinner" /> : <Send size={15} />}
        {loading ? 'Running...' : 'Run Inference'}
      </motion.button>

      {error && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ marginTop: 16, padding: '14px 18px', background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 10, color: '#f87171', fontSize: 14 }}>
          Error: {error}
        </motion.div>
      )}

      {result && (
        <motion.div className="glass" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: 20, padding: 24 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16, alignItems: 'flex-start' }}>
            <div style={{ fontWeight: 600, color: '#e2e8f0' }}>Output</div>
            <div style={{ display: 'flex', gap: 20, fontSize: 12, color: '#64748b' }}>
              <span><span style={{ color: '#818cf8' }}>Latency:</span> {result.latency_ms}ms</span>
              <span><span style={{ color: '#818cf8' }}>Tokens:</span> {result.tokens_generated}</span>
              <span><span style={{ color: '#818cf8' }}>Speed:</span> {result.tokens_per_second} tok/s</span>
            </div>
          </div>
          <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 10, padding: '16px 18px', fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#94a3b8', lineHeight: 1.7 }}>
            {result.output}
          </div>
        </motion.div>
      )}
    </div>
  );
}
