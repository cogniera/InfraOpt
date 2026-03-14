import React from 'react';
import { motion } from 'framer-motion';

export default function StatCard({ label, value, unit, icon: Icon, color = '#6366f1', delay = 0 }) {
  return (
    <motion.div
      className="glass"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
      style={{ padding: '20px 24px', position: 'relative', overflow: 'hidden' }}
    >
      {/* Glow accent */}
      <div style={{
        position: 'absolute', top: -20, right: -20,
        width: 80, height: 80,
        background: `radial-gradient(circle, ${color}22, transparent 70%)`,
        borderRadius: '50%',
      }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div style={{ fontSize: 12, color: '#64748b', fontWeight: 500, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
            <span className="stat-number">{value}</span>
            {unit && <span style={{ fontSize: 13, color: '#475569', fontWeight: 500 }}>{unit}</span>}
          </div>
        </div>
        {Icon && (
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: `${color}1a`,
            border: `1px solid ${color}33`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={16} color={color} />
          </div>
        )}
      </div>
    </motion.div>
  );
}
