import React from 'react';
import { motion } from 'framer-motion';
import { LayoutDashboard, Cpu, Zap, Settings2, Activity } from 'lucide-react';

const NAV = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'models', label: 'Models', icon: Cpu },
  { id: 'inference', label: 'Inference', icon: Activity },
  { id: 'optimize', label: 'Optimize', icon: Zap },
];

export default function Sidebar({ active, setPage }) {
  return (
    <aside style={{
      width: 220,
      minHeight: '100vh',
      background: 'rgba(13, 20, 34, 0.8)',
      backdropFilter: 'blur(20px)',
      borderRight: '1px solid rgba(255,255,255,0.06)',
      display: 'flex',
      flexDirection: 'column',
      padding: '24px 16px',
      position: 'sticky',
      top: 0,
      zIndex: 10,
    }}>
      {/* Logo */}
      <div style={{ marginBottom: 40, paddingLeft: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32,
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            borderRadius: 8,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 20px rgba(99,102,241,0.4)',
          }}>
            <Zap size={16} color="white" />
          </div>
          <span style={{ fontWeight: 700, fontSize: 16, letterSpacing: '-0.3px', color: '#e2e8f0' }}>
            infer<span style={{ color: '#6366f1' }}>Opt</span>
          </span>
        </div>
        <div style={{ marginTop: 6, paddingLeft: 42, display: 'flex', alignItems: 'center', gap: 6 }}>
          <div className="pulse-dot" />
          <span style={{ fontSize: 11, color: '#64748b', fontFamily: 'JetBrains Mono, monospace' }}>v1.0.0 · live</span>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1 }}>
        <div style={{ fontSize: 10, color: '#475569', fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8, paddingLeft: 8 }}>
          Navigation
        </div>
        {NAV.map(({ id, label, icon: Icon }) => {
          const isActive = active === id;
          return (
            <motion.button
              key={id}
              onClick={() => setPage(id)}
              whileHover={{ x: 2 }}
              whileTap={{ scale: 0.97 }}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '10px 12px',
                borderRadius: 10,
                border: 'none',
                cursor: 'pointer',
                marginBottom: 4,
                background: isActive
                  ? 'linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15))'
                  : 'transparent',
                color: isActive ? '#818cf8' : '#64748b',
                fontWeight: isActive ? 600 : 400,
                fontSize: 14,
                transition: 'all 0.15s',
                position: 'relative',
                textAlign: 'left',
                borderLeft: isActive ? '2px solid #6366f1' : '2px solid transparent',
              }}
            >
              <Icon size={16} strokeWidth={isActive ? 2.2 : 1.8} />
              {label}
            </motion.button>
          );
        })}
      </nav>

      {/* Bottom */}
      <div style={{ fontSize: 11, color: '#334155', textAlign: 'center', lineHeight: 1.6 }}>
        <div>Node.js · FastAPI · Vite</div>
      </div>
    </aside>
  );
}
