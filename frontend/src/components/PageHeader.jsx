import React from 'react';

export default function PageHeader({ title, subtitle }) {
  return (
    <div style={{ marginBottom: 32 }}>
      <h1 style={{ fontSize: 26, fontWeight: 700, color: '#f1f5f9', margin: 0, letterSpacing: '-0.5px' }}>{title}</h1>
      {subtitle && <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>{subtitle}</p>}
    </div>
  );
}
