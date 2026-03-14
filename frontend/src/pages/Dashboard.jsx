import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, Database, Zap, TrendingDown, Layers, Cpu } from 'lucide-react';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const s = await fetch('/stats').then(r => r.json());
        setStats(s);
      } catch {
        setStats({
          total_requests: '—',
          cache_hit_rate: '—',
          average_savings_ratio: '—',
          total_tokens_saved: '—',
          slots_served_from_cache: '—',
          slots_served_from_inference: '—',
        });
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  function pct(v) {
    if (v === '—') return '—';
    return `${Math.round(v * 100)}%`;
  }

  const statCards = stats ? [
    { label: 'Total Requests',      value: stats.total_requests,                               icon: Activity,    color: '#6366f1', delay: 0    },
    { label: 'Cache Hit Rate',       value: pct(stats.cache_hit_rate),                          icon: Database,    color: '#10b981', delay: 0.05 },
    { label: 'Avg Token Savings',    value: pct(stats.average_savings_ratio),                   icon: TrendingDown,color: '#8b5cf6', delay: 0.1  },
    { label: 'Tokens Saved',         value: stats.total_tokens_saved,                           icon: Zap,         color: '#06b6d4', delay: 0.15 },
    { label: 'Slots from Cache',     value: stats.slots_served_from_cache,                      icon: Layers,      color: '#f59e0b', delay: 0.2  },
    { label: 'Slots from LLM',       value: stats.slots_served_from_inference,                  icon: Cpu,         color: '#ec4899', delay: 0.25 },
  ] : [];

  return (
    <div>
      <PageHeader title="Dashboard" subtitle="TemplateCache pipeline metrics" />

      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: '#64748b', padding: '40px 0' }}>
          <div className="spinner" />
          Loading metrics…
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 16 }}>
          {statCards.map(c => <StatCard key={c.label} {...c} />)}
        </div>
      )}

      {!loading && (
        <motion.div
          className="glass"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35 }}
          style={{ padding: 24, marginTop: 24 }}
        >
          <div style={{ fontWeight: 600, color: '#e2e8f0', fontSize: 15, marginBottom: 6 }}>
            How the cache works
          </div>
          <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.8 }}>
            Every query is embedded and compared against stored intent centroids via cosine similarity.
            On a <span style={{ color: '#10b981' }}>cache hit</span>, the matched template's slots are
            filled from Redis or targeted LLM calls — saving the majority of tokens.
            On a <span style={{ color: '#ef4444' }}>cache miss</span>, the full LLM response is generated
            and a new template + intent centroid are stored for future reuse.
          </div>
        </motion.div>
      )}
    </div>
  );
}
