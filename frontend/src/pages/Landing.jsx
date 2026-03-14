import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Zap, ArrowRight, Activity, Cpu, TrendingUp } from 'lucide-react';

// ─── 3D Space Embedding Canvas ────────────────────────────────────────────────
function SpaceCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animId;
    let t = 0;
    let W = window.innerWidth;
    let H = window.innerHeight;

    // ── Rotation state ─────────────────────────────────────────────────────
    let rotX = 0.38;   // tilt down slightly so axes visible
    let rotY = 0;
    let isDragging = false;
    let lastMX = 0;
    let lastMY = 0;

    const onMouseDown = e => { isDragging = true; lastMX = e.clientX; lastMY = e.clientY; };
    const onMouseUp   = () => { isDragging = false; };
    const onMouseMove = e => {
      if (!isDragging) return;
      rotY += (e.clientX - lastMX) * 0.005;
      rotX  = Math.max(-0.65, Math.min(0.65, rotX + (e.clientY - lastMY) * 0.003));
      lastMX = e.clientX;
      lastMY = e.clientY;
    };
    window.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mouseup',   onMouseUp);
    window.addEventListener('mousemove', onMouseMove);

    // ── 3-D → screen projection ────────────────────────────────────────────
    function project(px, py, pz) {
      // rotate Y
      const cy = Math.cos(rotY), sy = Math.sin(rotY);
      const x1 =  px * cy + pz * sy;
      const z1 = -px * sy + pz * cy;
      // rotate X
      const cx = Math.cos(rotX), sx = Math.sin(rotX);
      const y2 =  py * cx - z1 * sx;
      const z2 =  py * sx + z1 * cx;
      // perspective
      const FOV   = 900;
      const scale = FOV / (FOV + z2 + 180);
      return { x: W / 2 + x1 * scale, y: H / 2 + y2 * scale, scale, z: z2 };
    }

    // ── Canvas resize ──────────────────────────────────────────────────────
    let bgStars = [];
    const initBgStars = () => {
      bgStars = Array.from({ length: 260 }, () => ({
        x: Math.random() * W,
        y: Math.random() * H,
        r: Math.random() * 0.75 + 0.15,
        a: Math.random() * 0.45 + 0.1,
        tp: Math.random() * Math.PI * 2,
        ts: Math.random() * 0.018 + 0.004,
      }));
    };
    const resize = () => {
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width  = W;
      canvas.height = H;
      initBgStars();
    };
    resize();
    window.addEventListener('resize', resize);

    // ── Embedding stars (3-D) ──────────────────────────────────────────────
    const WORD_POOL = [
      'tree','cats','bats','fire','moon','star','wave','data','node','flux',
      'beam','glow','code','rust','bold','fast','link','flow','dark','neon',
      'cube','void','echo','ring','dawn','dusk','rain','snow','mist','dust',
      'bird','wolf','fish','frog','deer','bull','hawk','lion','bear','crab',
      'rose','leaf','seed','root','bark','vine','reed','fern','moss','kelp',
      'gold','iron','salt','sand','lava','coal','jade','opal','ruby','onyx',
    ];

    // Spread evenly on a sphere + some inner ones
    const STAR_COLORS = [
      [220, 230, 255],  // blue-white
      [255, 245, 210],  // yellow-white
      [200, 220, 255],  // pale blue
      [255, 210, 190],  // orange-white
      [210, 255, 220],  // green-white
      [255, 190, 255],  // pink-white
      [180, 210, 255],  // ice blue
    ];

    const embStars = Array.from({ length: 90 }, () => {
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(2 * Math.random() - 1);
      const r     = 160 + Math.random() * 340;
      return {
        x:  r * Math.sin(phi) * Math.cos(theta),
        y:  r * Math.sin(phi) * Math.sin(theta),
        z:  r * Math.cos(phi),
        r:  Math.random() * 3.2 + 1.2,
        brightness: Math.random() * 0.45 + 0.55,
        color: STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)],
        tp:   Math.random() * Math.PI * 2,
        ts:   Math.random() * 0.025 + 0.008,
        word: WORD_POOL[Math.floor(Math.random() * WORD_POOL.length)],
      };
    });

    // ── Axes ───────────────────────────────────────────────────────────────
    const AXIS_LEN = 480;
    const axes = [
      { d: [1, 0, 0], color: [99,  102, 241], label: 'dim₀' },  // indigo  X
      { d: [0, 1, 0], color: [16,  185, 129], label: 'dim₁' },  // emerald Y
      { d: [0, 0, 1], color: [139,  92, 246], label: 'dim₂' },  // violet  Z
    ];

    // ── Sun + word-on-pulse ────────────────────────────────────────────────
    // Dramatic pulse: ball shrinks to ~35% then bursts back to full size
    // During the growing phase a 4-letter word fades in then burns away
    const PULSE_SPEED = 0.5;          // rad/s — full breath every ~12 s
    const WORDS       = ['EMBD', 'VECT', 'DIMS', 'INFT'];
    let   pulseCount  = 0;
    let   prevSinPos  = false;        // was sin positive last frame?

    function drawSun(ox, oy) {
      const sinVal    = Math.sin(t * PULSE_SPEED);
      const cosVal    = Math.cos(t * PULSE_SPEED);   // > 0 while sin is rising
      const sinPos    = sinVal > 0;

      // Count pulses (rising zero-cross = new word)
      if (sinPos && !prevSinPos) pulseCount++;
      prevSinPos = sinPos;

      // pulse: 0.32 (small) → 1.0 (full)
      const pulse = 0.32 + 0.68 * ((sinVal + 1) / 2);
      const coreR = 34 * pulse;

      // ── Word alpha ────────────────────────────────────────────────────
      // Show when growing (cosVal > 0) and ball is still on the small side
      // Fade in from sinVal=-0.6, peak near sinVal=0, fade out by sinVal=0.75
      const isGrowing = cosVal > 0;
      let wordAlpha = 0;
      if (isGrowing) {
        const progress = (sinVal + 1) / 2;            // 0 (small) → 1 (big)
        // bell curve: rises 0→0.45, peaks ~0.45, gone by 0.85
        wordAlpha = Math.max(0, Math.sin(progress * Math.PI * 1.15)) * 0.92;
      }

      // ── Corona layers ─────────────────────────────────────────────────
      const layers = [
        { r: coreR * 15, rgba: [255, 200,  80, 0.022] },
        { r: coreR * 10, rgba: [255, 180,  50, 0.048] },
        { r: coreR *  6, rgba: [255, 160,  30, 0.09 ] },
        { r: coreR *  3.5, rgba: [255, 220, 110, 0.18] },
        { r: coreR *  2, rgba: [255, 240, 160, 0.32 ] },
      ];
      for (const { r, rgba: [r_, g, b, a] } of layers) {
        const g2 = ctx.createRadialGradient(ox, oy, 0, ox, oy, r);
        g2.addColorStop(0, `rgba(${r_},${g},${b},${a})`);
        g2.addColorStop(1, `rgba(${r_},${g},${b},0)`);
        ctx.fillStyle = g2;
        ctx.beginPath();
        ctx.arc(ox, oy, r, 0, Math.PI * 2);
        ctx.fill();
      }

      // ── Core disc ─────────────────────────────────────────────────────
      const cg = ctx.createRadialGradient(ox, oy, 0, ox, oy, coreR);
      cg.addColorStop(0,    'rgba(255,255,245,1)');
      cg.addColorStop(0.25, 'rgba(255,245,160,0.97)');
      cg.addColorStop(0.6,  'rgba(255,180, 60,0.85)');
      cg.addColorStop(1,    'rgba(255,100, 10,0)');
      ctx.fillStyle = cg;
      ctx.beginPath();
      ctx.arc(ox, oy, coreR, 0, Math.PI * 2);
      ctx.fill();

      // ── 4-letter word burned into the expanding ball ───────────────────
      if (wordAlpha > 0.01) {
        const word     = WORDS[pulseCount % WORDS.length];
        const fontSize = Math.max(10, coreR * 0.72);
        ctx.save();
        ctx.font         = `900 ${fontSize}px "JetBrains Mono", monospace`;
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        // Dark core text so it pops against the bright sun
        ctx.fillStyle = `rgba(40, 10, 80, ${wordAlpha})`;
        ctx.fillText(word, ox, oy);
        // Thin bright outline so it's legible at small sizes too
        ctx.strokeStyle = `rgba(255, 240, 180, ${wordAlpha * 0.35})`;
        ctx.lineWidth   = 0.6;
        ctx.strokeText(word, ox, oy);
        ctx.restore();
      }
    }

    // ── Render ─────────────────────────────────────────────────────────────
    function draw() {
      t += 0.008;
      if (!isDragging) rotY += 0.0028;

      // Deep space background
      ctx.fillStyle = '#050810';
      ctx.fillRect(0, 0, W, H);

      // 1. Background star field
      for (const s of bgStars) {
        s.tp += s.ts;
        const alpha = s.a + 0.12 * Math.sin(s.tp);
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(210, 225, 255, ${alpha})`;
        ctx.fill();
      }

      const o = project(0, 0, 0);

      // 2. Axes (gradient, ticks, labels)
      for (const ax of axes) {
        const [dx, dy, dz] = ax.d;
        const [r, g, b]    = ax.color;
        const neg = project(-dx * AXIS_LEN, -dy * AXIS_LEN, -dz * AXIS_LEN);
        const pos = project( dx * AXIS_LEN,  dy * AXIS_LEN,  dz * AXIS_LEN);

        // Axis line with gradient brightness
        const axGr = ctx.createLinearGradient(neg.x, neg.y, pos.x, pos.y);
        axGr.addColorStop(0,   `rgba(${r},${g},${b},0.06)`);
        axGr.addColorStop(0.5, `rgba(${r},${g},${b},0.65)`);
        axGr.addColorStop(1,   `rgba(${r},${g},${b},0.06)`);
        ctx.beginPath();
        ctx.moveTo(neg.x, neg.y);
        ctx.lineTo(pos.x, pos.y);
        ctx.strokeStyle = axGr;
        ctx.lineWidth = 1.3;
        ctx.stroke();

        // Tick dots along positive arm
        for (let k = 1; k <= 4; k++) {
          const f  = k / 4;
          const tp = project(dx * AXIS_LEN * f, dy * AXIS_LEN * f, dz * AXIS_LEN * f);
          ctx.beginPath();
          ctx.arc(tp.x, tp.y, 2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r},${g},${b},0.45)`;
          ctx.fill();
        }

        // Label
        ctx.font = 'bold 12px "JetBrains Mono", monospace';
        ctx.fillStyle = `rgba(${r},${g},${b},0.85)`;
        ctx.fillText(ax.label, pos.x + 7, pos.y + 4);
      }

      // 3. Light beams — sun → each embedding star
      for (const s of embStars) {
        const sp = project(s.x, s.y, s.z);
        const [r, g, b] = s.color;

        const beamGr = ctx.createLinearGradient(o.x, o.y, sp.x, sp.y);
        beamGr.addColorStop(0,    'rgba(255, 230, 120, 0.22)');
        beamGr.addColorStop(0.25, `rgba(${r},${g},${b},0.10)`);
        beamGr.addColorStop(1,    `rgba(${r},${g},${b},0.0)`);

        ctx.beginPath();
        ctx.moveTo(o.x, o.y);
        ctx.lineTo(sp.x, sp.y);
        ctx.strokeStyle = beamGr;
        ctx.lineWidth   = 0.65;
        ctx.stroke();
      }

      // 4. Embedding stars — back-to-front depth sort
      const sorted = embStars
        .map(s => ({ s, p: project(s.x, s.y, s.z) }))
        .sort((a, b) => b.p.z - a.p.z);

      for (const { s, p } of sorted) {
        s.tp += s.ts;
        const twinkle = 1 + 0.28 * Math.sin(s.tp);
        const gr3d    = s.r * p.scale * twinkle * 5;
        const alpha   = s.brightness * (0.65 + 0.35 * p.scale);
        const [r, g, b] = s.color;

        // Glow halo
        const sg = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, gr3d * 2.8);
        sg.addColorStop(0,   `rgba(${r},${g},${b},${alpha * 0.85})`);
        sg.addColorStop(0.45,`rgba(${r},${g},${b},${alpha * 0.25})`);
        sg.addColorStop(1,   `rgba(${r},${g},${b},0)`);
        ctx.fillStyle = sg;
        ctx.beginPath();
        ctx.arc(p.x, p.y, gr3d * 2.8, 0, Math.PI * 2);
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(p.x, p.y, Math.max(0.8, gr3d * 0.7), 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${Math.min(1, alpha * 1.3)})`;
        ctx.fill();

        // Word at peak twinkle — sinVal near 1.0
        const sinVal  = Math.sin(s.tp);
        const wordAlpha = Math.max(0, (sinVal - 0.55) / 0.45) * alpha;
        if (wordAlpha > 0.01) {
          const fontSize = Math.max(8, gr3d * 1.1);
          ctx.save();
          ctx.font         = `700 ${fontSize}px "JetBrains Mono", monospace`;
          ctx.textAlign    = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle    = `rgba(${r},${g},${b},${wordAlpha})`;
          ctx.fillText(s.word, p.x, p.y - gr3d * 2.2);
          ctx.restore();
        }
      }

      // 5. Sun always on top
      drawSun(o.x, o.y);

      animId = requestAnimationFrame(draw);
    }
    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize',    resize);
      window.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mouseup',   onMouseUp);
      window.removeEventListener('mousemove', onMouseMove);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: 'fixed', inset: 0, zIndex: 0, cursor: 'grab' }}
    />
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────
const FEATURES = [
  { icon: Activity,    label: 'Real-time Metrics',   desc: 'Live latency and throughput monitoring'  },
  { icon: Cpu,         label: 'Multi-Model Support', desc: 'Llama, Mistral, Phi, Gemma and more'    },
  { icon: TrendingUp,  label: 'Smart Optimization',  desc: 'Speed, quality, and balanced strategies' },
];

export default function Landing({ onEnter }) {
  return (
    <div style={{ minHeight: '100vh', position: 'relative', overflow: 'hidden', background: '#050810', userSelect: 'none' }}>

      <SpaceCanvas />

      {/* Subtle dark vignette so text is readable over bright sun */}
      <div style={{
        position: 'fixed', inset: 0, zIndex: 1, pointerEvents: 'none',
        background: [
          'radial-gradient(ellipse 70% 60% at 50% 50%, transparent 30%, rgba(5,8,16,0.55) 100%)',
        ].join(','),
      }} />

      {/* ── Nav ── */}
      <nav style={{
        position: 'relative', zIndex: 3,
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

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {/* drag hint */}
          <span style={{ fontSize: 11, color: '#334155', fontFamily: 'JetBrains Mono, monospace' }}>
            drag to rotate
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }} />
            <span style={{ fontSize: 12, color: '#475569', fontFamily: 'JetBrains Mono, monospace' }}>
              All systems operational
            </span>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <div style={{
        position: 'relative', zIndex: 3,
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
            background: 'rgba(99,102,241,0.12)',
            border: '1px solid rgba(99,102,241,0.28)',
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
          whileHover={{ scale: 1.04, boxShadow: '0 0 44px rgba(99,102,241,0.65)' }}
          whileTap={{ scale: 0.97 }}
          onClick={onEnter}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 10,
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            border: 'none', borderRadius: 12, padding: '16px 32px',
            color: 'white', fontWeight: 600, fontSize: 16, cursor: 'pointer',
            boxShadow: '0 0 28px rgba(99,102,241,0.45)', transition: 'box-shadow 0.2s',
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
        position: 'relative', zIndex: 3,
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
              background: 'rgba(5,8,16,0.6)', backdropFilter: 'blur(16px)',
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
        position: 'fixed', bottom: 0, left: 0, right: 0, height: 100, zIndex: 2, pointerEvents: 'none',
        background: 'linear-gradient(to top, #050810 20%, transparent)',
      }} />
    </div>
  );
}
