import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Zap, ArrowRight, Activity, Cpu, TrendingUp } from 'lucide-react';

// ─── Embedding Vector Field Canvas ────────────────────────────────────────────
function EmbeddingCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animId;
    let t = 0;

    let W = window.innerWidth;
    let H = window.innerHeight;

    const resize = () => {
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = W;
      canvas.height = H;
    };
    resize();
    window.addEventListener('resize', resize);

    // ── Smooth flow field angle at (x, y, t) ──────────────────────────────
    // Combines multiple sine/cosine waves to look like projected high-dim vectors
    function fieldAngle(x, y) {
      const nx = x * 0.0022;
      const ny = y * 0.0022;
      return (
        Math.sin(nx + t * 0.22) * Math.cos(ny * 1.4 - t * 0.16) * Math.PI * 1.6 +
        Math.cos(nx * 0.75 - ny * 0.55 + t * 0.11) * Math.PI * 0.9 +
        Math.sin(nx * 1.3 + ny * 0.9 + t * 0.07) * Math.PI * 0.45
      );
    }

    // ── Embedding clusters (semantic groups) ──────────────────────────────
    const PALETTE = [
      [99,  102, 241],  // indigo  — cluster 0
      [139, 92,  246],  // violet  — cluster 1
      [6,   182, 212],  // cyan    — cluster 2
      [16,  185, 129],  // emerald — cluster 3
      [236, 72,  153],  // pink    — cluster 4
    ];
    const DIM_LABELS = ['ε₀', 'ε₁', 'ε₂', 'ε₃', 'ε₄'];
    const N_CLUSTERS = PALETTE.length;

    const clusters = Array.from({ length: N_CLUSTERS }, (_, i) => ({
      x: W * (0.12 + (i / (N_CLUSTERS - 1)) * 0.76),
      y: H * (0.35 + Math.sin(i * 1.3 + 1) * 0.18),
      vx: (Math.random() - 0.5) * 0.25,
      vy: (Math.random() - 0.5) * 0.25,
      phase: Math.random() * Math.PI * 2,   // for pulsing glow
      color: PALETTE[i],
      label: DIM_LABELS[i],
    }));

    // ── Particles ─────────────────────────────────────────────────────────
    const N_PARTICLES = 320;

    function makeParticle(i) {
      const cluster = i % N_CLUSTERS;
      return {
        x: Math.random() * W,
        y: Math.random() * H,
        px: 0, py: 0,       // previous position (for line drawing)
        vx: 0, vy: 0,
        speed: Math.random() * 1.0 + 0.5,
        cluster,
        life: Math.floor(Math.random() * 180 + 80),
        maxLife: 260,
        size: Math.random() * 1.1 + 0.35,
        fresh: true,        // skip first-frame line draw
      };
    }

    const particles = Array.from({ length: N_PARTICLES }, (_, i) => makeParticle(i));

    // ── Arrow grid ────────────────────────────────────────────────────────
    const GRID_STEP = 48;
    const ARROW_LEN = 11;

    function drawArrow(x, y, angle, alpha) {
      const ex = x + Math.cos(angle) * ARROW_LEN;
      const ey = y + Math.sin(angle) * ARROW_LEN;
      const HL = 3.5;
      const HA = 0.45;

      ctx.beginPath();
      ctx.moveTo(x - Math.cos(angle) * 3, y - Math.sin(angle) * 3); // small tail
      ctx.lineTo(ex, ey);
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - HL * Math.cos(angle - HA), ey - HL * Math.sin(angle - HA));
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - HL * Math.cos(angle + HA), ey - HL * Math.sin(angle + HA));
      ctx.strokeStyle = `rgba(129, 140, 248, ${alpha})`;
      ctx.lineWidth = 0.55;
      ctx.stroke();
    }

    // ── Projection axis lines (faint orthogonal grid) ─────────────────────
    function drawAxes() {
      ctx.save();
      ctx.setLineDash([2, 18]);
      ctx.lineWidth = 0.4;
      // Horizontal lines every ~15% height
      for (let frac = 0.15; frac <= 0.85; frac += 0.15) {
        ctx.beginPath();
        ctx.moveTo(0, H * frac);
        ctx.lineTo(W, H * frac);
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.045)';
        ctx.stroke();
      }
      // Vertical lines every ~15% width
      for (let frac = 0.15; frac <= 0.85; frac += 0.15) {
        ctx.beginPath();
        ctx.moveTo(W * frac, 0);
        ctx.lineTo(W * frac, H);
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.04)';
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.restore();
    }

    // Mouse influence
    const mouse = { x: -9999, y: -9999 };
    const onMouseMove = e => { mouse.x = e.clientX; mouse.y = e.clientY; };
    window.addEventListener('mousemove', onMouseMove);

    // ── Main render loop ──────────────────────────────────────────────────
    function draw() {
      t += 0.007;

      // Persist trails — semi-transparent dark wash instead of clearRect
      ctx.fillStyle = 'rgba(8, 12, 20, 0.10)';
      ctx.fillRect(0, 0, W, H);

      // 1. Faint projection axes
      drawAxes();

      // 2. Vector field arrows
      const cols = Math.ceil(W / GRID_STEP) + 1;
      const rows = Math.ceil(H / GRID_STEP) + 1;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const ax = c * GRID_STEP + GRID_STEP / 2;
          const ay = r * GRID_STEP + GRID_STEP / 2;
          const angle = fieldAngle(ax, ay);
          // Fade arrows near mouse
          const mdx = ax - mouse.x;
          const mdy = ay - mouse.y;
          const md = Math.sqrt(mdx * mdx + mdy * mdy);
          const alpha = md < 130 ? 0.03 + (md / 130) * 0.055 : 0.07;
          drawArrow(ax, ay, angle, alpha);
        }
      }

      // 3. Cluster centroids — pulsing glows + labels
      for (const cl of clusters) {
        cl.x += cl.vx;
        cl.y += cl.vy;
        if (cl.x < 60 || cl.x > W - 60) cl.vx *= -1;
        if (cl.y < 60 || cl.y > H - 60) cl.vy *= -1;

        const pulse = 0.12 + 0.06 * Math.sin(t * 1.8 + cl.phase);
        const outerR = 70 + 14 * Math.sin(t * 1.4 + cl.phase);
        const [r, g, b] = cl.color;

        // Outer glow
        const grad = ctx.createRadialGradient(cl.x, cl.y, 0, cl.x, cl.y, outerR);
        grad.addColorStop(0, `rgba(${r},${g},${b},${pulse})`);
        grad.addColorStop(0.4, `rgba(${r},${g},${b},${pulse * 0.4})`);
        grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cl.x, cl.y, outerR, 0, Math.PI * 2);
        ctx.fill();

        // Core dot
        ctx.beginPath();
        ctx.arc(cl.x, cl.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},0.95)`;
        ctx.fill();

        // Dimension label
        ctx.font = '500 11px "JetBrains Mono", monospace';
        ctx.fillStyle = `rgba(${r},${g},${b},0.55)`;
        ctx.fillText(cl.label, cl.x + 8, cl.y - 8);
      }

      // 4. Particles flowing through the field
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.life--;

        const oob = p.x < -10 || p.x > W + 10 || p.y < -10 || p.y > H + 10;
        if (p.life <= 0 || oob) {
          particles[i] = makeParticle(i);
          continue;
        }

        p.px = p.x;
        p.py = p.y;

        // Flow field force
        const angle = fieldAngle(p.x, p.y);
        p.vx += Math.cos(angle) * 0.09;
        p.vy += Math.sin(angle) * 0.09;

        // Gentle pull toward own cluster centroid
        const cl = clusters[p.cluster];
        const cdx = cl.x - p.x;
        const cdy = cl.y - p.y;
        const cd = Math.sqrt(cdx * cdx + cdy * cdy);
        if (cd > 80) {
          p.vx += (cdx / cd) * 0.025;
          p.vy += (cdy / cd) * 0.025;
        }

        // Mouse repulsion / warp
        const mdx = p.x - mouse.x;
        const mdy = p.y - mouse.y;
        const md = Math.sqrt(mdx * mdx + mdy * mdy);
        if (md < 160) {
          const force = (1 - md / 160) * 1.1;
          p.vx += (mdx / md) * force;
          p.vy += (mdy / md) * force;
        }

        // Speed cap + damping
        const spd = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
        const maxSpd = p.speed * 2.2;
        if (spd > maxSpd) { p.vx = (p.vx / spd) * maxSpd; p.vy = (p.vy / spd) * maxSpd; }
        p.vx *= 0.91;
        p.vy *= 0.91;

        p.x += p.vx;
        p.y += p.vy;

        const lifeRatio = p.life / p.maxLife;
        const alpha = Math.min(lifeRatio * 4, 1) * 0.75;
        const [r, g, b] = PALETTE[p.cluster];

        if (!p.fresh) {
          // Draw line segment (trail)
          ctx.beginPath();
          ctx.moveTo(p.px, p.py);
          ctx.lineTo(p.x, p.y);
          ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
          ctx.lineWidth = p.size;
          ctx.stroke();
        }

        // Head dot (brighter)
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * 1.4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha * 0.9})`;
        ctx.fill();

        p.fresh = false;
      }

      animId = requestAnimationFrame(draw);
    }

    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', onMouseMove);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none' }}
    />
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────
const FEATURES = [
  { icon: Activity, label: 'Real-time Metrics',   desc: 'Live latency and throughput monitoring' },
  { icon: Cpu,      label: 'Multi-Model Support', desc: 'Llama, Mistral, Phi, Gemma and more'   },
  { icon: TrendingUp, label: 'Smart Optimization', desc: 'Speed, quality, and balanced strategies' },
];

export default function Landing({ onEnter }) {
  return (
    <div style={{ minHeight: '100vh', position: 'relative', overflow: 'hidden', background: '#080c14' }}>

      {/* Vector embedding field */}
      <EmbeddingCanvas />

      {/* Soft center radial glow */}
      <div style={{
        position: 'fixed', inset: 0, zIndex: 0, pointerEvents: 'none',
        background: 'radial-gradient(ellipse 55% 45% at 50% 42%, rgba(99,102,241,0.10) 0%, transparent 70%)',
      }} />

      {/* ── Nav ── */}
      <nav style={{
        position: 'relative', zIndex: 2,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '28px 48px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 9,
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 24px rgba(99,102,241,0.5)',
          }}>
            <Zap size={17} color="white" />
          </div>
          <span style={{ fontWeight: 700, fontSize: 18, letterSpacing: '-0.4px', color: '#f1f5f9' }}>
            infer<span style={{ color: '#818cf8' }}>Opt</span>
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }} />
          <span style={{ fontSize: 12, color: '#475569', fontFamily: 'JetBrains Mono, monospace' }}>
            All systems operational
          </span>
        </div>
      </nav>

      {/* ── Hero ── */}
      <div style={{
        position: 'relative', zIndex: 2,
        display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center',
        padding: '72px 24px 56px',
      }}>

        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 8,
            background: 'rgba(99,102,241,0.1)',
            border: '1px solid rgba(99,102,241,0.25)',
            borderRadius: 100, padding: '6px 16px',
            fontSize: 12, color: '#818cf8', fontWeight: 500, marginBottom: 32,
          }}
        >
          <Zap size={11} />
          AI Inference Optimization Platform
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.1 }}
          style={{
            fontSize: 'clamp(2.6rem, 6vw, 5rem)',
            fontWeight: 800, lineHeight: 1.08, letterSpacing: '-2px',
            margin: '0 0 24px', maxWidth: 780,
            background: 'linear-gradient(135deg, #f1f5f9 0%, #94a3b8 55%, #6366f1 100%)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
          }}
        >
          Inference at the Speed of Thought
        </motion.h1>

        {/* Subhead */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.2 }}
          style={{ fontSize: 18, color: '#64748b', maxWidth: 520, lineHeight: 1.7, margin: '0 0 48px' }}
        >
          Monitor, run, and optimize large language model inference with real-time metrics
          and intelligent tuning strategies.
        </motion.p>

        {/* CTA */}
        <motion.button
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4, delay: 0.3 }}
          whileHover={{ scale: 1.04, boxShadow: '0 0 44px rgba(99,102,241,0.6)' }}
          whileTap={{ scale: 0.97 }}
          onClick={onEnter}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 10,
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            border: 'none', borderRadius: 12, padding: '16px 32px',
            color: 'white', fontWeight: 600, fontSize: 16, cursor: 'pointer',
            boxShadow: '0 0 28px rgba(99,102,241,0.4)', transition: 'box-shadow 0.2s',
          }}
        >
          Launch Dashboard
          <ArrowRight size={18} />
        </motion.button>

        {/* Stack pills */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          style={{ display: 'flex', gap: 10, marginTop: 36, flexWrap: 'wrap', justifyContent: 'center' }}
        >
          {['Vite + React', 'Node.js', 'FastAPI', 'Docker'].map(t => (
            <span key={t} style={{
              background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 6, padding: '4px 12px',
              fontSize: 12, color: '#475569', fontFamily: 'JetBrains Mono, monospace',
            }}>{t}</span>
          ))}
        </motion.div>
      </div>

      {/* ── Feature cards ── */}
      <div style={{
        position: 'relative', zIndex: 2,
        display: 'flex', justifyContent: 'center', gap: 16,
        padding: '0 48px 80px', flexWrap: 'wrap',
      }}>
        {FEATURES.map(({ icon: Icon, label, desc }, i) => (
          <motion.div
            key={label}
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 + i * 0.1, duration: 0.4 }}
            style={{
              background: 'rgba(255,255,255,0.04)', backdropFilter: 'blur(12px)',
              border: '1px solid rgba(255,255,255,0.07)',
              borderRadius: 16, padding: '28px', width: 240, textAlign: 'left',
            }}
          >
            <div style={{
              width: 40, height: 40, borderRadius: 10, marginBottom: 16,
              background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.2)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Icon size={18} color="#818cf8" />
            </div>
            <div style={{ fontWeight: 600, color: '#e2e8f0', fontSize: 14, marginBottom: 6 }}>{label}</div>
            <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{desc}</div>
          </motion.div>
        ))}
      </div>

      {/* Bottom fade */}
      <div style={{
        position: 'fixed', bottom: 0, left: 0, right: 0, height: 130, zIndex: 1, pointerEvents: 'none',
        background: 'linear-gradient(to top, #080c14 20%, transparent)',
      }} />
    </div>
  );
}
