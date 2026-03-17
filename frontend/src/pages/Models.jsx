import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Cpu, Clock } from 'lucide-react';
import PageHeader from '../components/PageHeader.jsx';

export default function Models() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/models')
      .then(r => r.json())
      .then(d => setModels(d.models))
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <PageHeader title="Models" subtitle="Available inference models and their status" />
      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: '#64748b' }}>
          <div className="spinner" /> Loading models...
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {models.map((m, i) => (
            <motion.div
              key={m.id}
              className="glass"
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.07 }}
              style={{ padding: '18px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div style={{
                  width: 40, height: 40, borderRadius: 10,
                  background: 'linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15))',
                  border: '1px solid rgba(99,102,241,0.2)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <Cpu size={18} color="#818cf8" />
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: '#e2e8f0', fontSize: 15 }}>{m.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', marginTop: 2, fontFamily: 'JetBrains Mono, monospace' }}>{m.id}</div>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: '#475569', marginBottom: 2 }}>SIZE</div>
                  <div style={{ fontWeight: 600, color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>{m.size}</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: '#475569', marginBottom: 2, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <Clock size={10} /> AVG LATENCY
                  </div>
                  <div style={{ fontWeight: 600, color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>{m.latency_ms}ms</div>
                </div>
                <span className={m.status === 'ready' ? 'badge-ready' : 'badge-loading'}>{m.status}</span>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
