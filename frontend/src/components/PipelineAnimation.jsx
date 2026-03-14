import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// ─── 3-D Camera ────────────────────────────────────────────────────────────────
const ROT_X = 0.22;
const ROT_Y = 0.15;
const FOV   = 1000;
const SCALE = 100;

function project(wx, wy, wz, W, H) {
  const cy = Math.cos(ROT_Y), sy = Math.sin(ROT_Y);
  const x1 =  wx * cy + wz * sy;
  const z1 = -wx * sy + wz * cy;
  const cx = Math.cos(ROT_X), sx = Math.sin(ROT_X);
  const y2 =  wy * cx - z1 * sx;
  const z2 =  wy * sx + z1 * cx;
  const sc = FOV / (FOV + z2 * SCALE);
  return { x: W / 2 + x1 * SCALE * sc, y: H / 2 + y2 * SCALE * sc, sc, z: z2 };
}

// ─── Node world positions ──────────────────────────────────────────────────────
// Vertical spine: input → split → embed → router
// Miss branch (right): llm
// Hit branch (left):   slot
const NP = {
  input:  [0,    -2.5,  0],
  split:  [0,    -1.5,  0],
  embed:  [0,    -0.4,  0],
  router: [0,     0.7,  0],
  llm:    [ 1.9,  1.9,  0],
  slot:   [-1.9,  1.9,  0],
};
const BASE_R = 68;
const TUBE_R = 8;

// ─── Stage timing (seconds) ───────────────────────────────────────────────────
function makeStages(cacheHit) {
  const s = {
    queryAppear:  [0.0,  1.1],
    flowSplit:    [1.1,  2.3],
    splitCheck:   [2.0,  3.1],
    flowEmbed:    [3.1,  4.3],
    flowRouter:   [4.3,  5.7],
    routerSearch: [5.2,  6.8],
    routerResult: [6.8,  7.6],
  };
  if (!cacheHit) {
    s.flowLLM = [7.6,  8.8];
    s.llmGen  = [8.6, 11.2];
    s.done    = 11.8;
  } else {
    s.flowSlot  = [7.6,  8.8];
    s.gapDetect = [8.8, 10.2];
    s.slotFill  = [10.2, 14.8];
    s.done      = 15.2;
  }
  return s;
}

// ─── Colour sets ──────────────────────────────────────────────────────────────
const COL = {
  neutral:  { glow:[99,102,241],  fill:'rgba(14,20,52,0.62)'  },
  generate: { glow:[251,146,60],  fill:'rgba(48,16,4,0.62)'   },
  slotNode: { glow:[56,189,248],  fill:'rgba(4,32,52,0.62)'   },
  routerHit:{ glow:[16,185,129],  fill:'rgba(4,36,22,0.65)'   },
  routerMiss:{ glow:[239,68,68],  fill:'rgba(48,8,8,0.65)'    },
};

// ─── Demo slot data (mirrors SlotEngine fill structure) ────────────────────────
const SLOTS = [
  { name:'slot_0', label:'topic',    value:'LLM inference',  source:'cache'     },
  { name:'slot_1', label:'metric',   value:'cost/token',     source:'inference' },
  { name:'slot_2', label:'method',   value:'quantization',   source:'cache'     },
  { name:'slot_3', label:'saving',   value:'~40%',           source:'inference' },
  { name:'slot_4', label:'scope',    value:'per request',    source:'cache'     },
];

// ─── Math helpers ─────────────────────────────────────────────────────────────
const lerp  = (a,b,t) => a+(b-a)*Math.max(0,Math.min(1,t));
const sp    = (t,s,e) => Math.max(0,Math.min(1,(t-s)/(e-s)));
const ease  = t => t<0.5?2*t*t:-1+(4-2*t)*t;
const g3    = ([r,g,b],a) => `rgba(${r},${g},${b},${a})`;
const lerpA = (a,b,t) => a.map((v,i)=>Math.round(lerp(v,b[i],t)));

function rr(ctx,x,y,w,h,r){
  if(ctx.roundRect){ctx.roundRect(x,y,w,h,r);return;}
  ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();
}

// ─── 3-D Sphere ───────────────────────────────────────────────────────────────
function drawSphere(ctx, x, y, r, col, label, sub, alpha=1) {
  const [gr,gg,gb]=col.glow;
  ctx.save(); ctx.globalAlpha=alpha;

  ctx.globalAlpha = alpha * 0.3;
  ctx.fillStyle='rgba(0,0,0,0.7)';
  ctx.beginPath(); ctx.ellipse(x+r*0.1, y+r*0.9, r*0.85, r*0.22, 0, 0, Math.PI*2); ctx.fill();
  ctx.globalAlpha = alpha;

  const corona=ctx.createRadialGradient(x,y,r*0.4,x,y,r*2.1);
  corona.addColorStop(0, g3([gr,gg,gb],0.22));
  corona.addColorStop(1, g3([gr,gg,gb],0));
  ctx.fillStyle=corona;
  ctx.beginPath(); ctx.arc(x,y,r*2.1,0,Math.PI*2); ctx.fill();

  const body=ctx.createRadialGradient(x-r*0.30, y-r*0.30, r*0.04, x+r*0.18, y+r*0.18, r*1.12);
  body.addColorStop(0,   'rgba(255,255,255,0.18)');
  body.addColorStop(0.22, g3([gr,gg,gb],0.40));
  body.addColorStop(0.65, g3([gr,gg,gb],0.20));
  body.addColorStop(1,    col.fill);
  ctx.fillStyle=body;
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();

  ctx.strokeStyle=g3([gr,gg,gb],0.30);
  ctx.lineWidth=1.1;
  ctx.beginPath(); ctx.ellipse(x,y, r*0.97, r*0.30, 0.12, 0, Math.PI*2); ctx.stroke();
  ctx.beginPath(); ctx.ellipse(x,y, r*0.30, r*0.97, 0.22, 0, Math.PI*2); ctx.stroke();

  ctx.strokeStyle=g3([gr,gg,gb],0.60);
  ctx.lineWidth=1.8;
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.stroke();

  const hl=ctx.createRadialGradient(x-r*0.33,y-r*0.33, 0, x-r*0.33,y-r*0.33, r*0.46);
  hl.addColorStop(0,'rgba(255,255,255,0.50)');
  hl.addColorStop(1,'rgba(255,255,255,0)');
  ctx.fillStyle=hl;
  ctx.beginPath(); ctx.arc(x-r*0.33,y-r*0.33, r*0.46, 0, Math.PI*2); ctx.fill();

  if(label){
    const fs=Math.max(11,r*0.26);
    ctx.font=`700 ${fs}px Inter,system-ui,sans-serif`;
    ctx.fillStyle='rgba(0,0,0,0.88)';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(label, x, sub ? y-r*0.14 : y);
  }
  if(sub){
    const fs2=Math.max(9,r*0.19);
    ctx.font=`400 ${fs2}px "JetBrains Mono",monospace`;
    ctx.fillStyle='rgba(0,0,0,0.60)';
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(sub, x, y+r*0.23);
  }
  ctx.restore();
}

// ─── 3-D Tube ─────────────────────────────────────────────────────────────────
function drawTube(ctx, p1, p2, glow, progress=1, alpha=1) {
  if(progress<=0) return;
  const [gr,gg,gb]=glow;
  const r1=TUBE_R*p1.sc, r2=TUBE_R*p2.sc;
  const ex=lerp(p1.x, p2.x, progress);
  const ey=lerp(p1.y, p2.y, progress);
  const er=lerp(r1,   r2,   progress);
  const dx=ex-p1.x, dy=ey-p1.y;
  const len=Math.sqrt(dx*dx+dy*dy);
  if(len<1) return;
  const nx=-dy/len, ny=dx/len;
  ctx.save(); ctx.globalAlpha=alpha;
  const fillGrad=ctx.createLinearGradient(p1.x+nx*r1, p1.y+ny*r1, p1.x-nx*r1, p1.y-ny*r1);
  fillGrad.addColorStop(0,   g3([gr,gg,gb],0.06));
  fillGrad.addColorStop(0.30, g3([gr,gg,gb],0.22));
  fillGrad.addColorStop(0.50, g3([gr,gg,gb],0.34));
  fillGrad.addColorStop(0.70, g3([gr,gg,gb],0.22));
  fillGrad.addColorStop(1,   g3([gr,gg,gb],0.06));
  ctx.fillStyle=fillGrad;
  ctx.beginPath();
  ctx.moveTo(p1.x+nx*r1, p1.y+ny*r1);
  ctx.lineTo(ex +nx*er,  ey +ny*er );
  ctx.lineTo(ex -nx*er,  ey -ny*er );
  ctx.lineTo(p1.x-nx*r1, p1.y-ny*r1);
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle=g3([gr,gg,gb],0.55);
  ctx.lineWidth=1.1;
  ctx.beginPath(); ctx.moveTo(p1.x+nx*r1,p1.y+ny*r1); ctx.lineTo(ex+nx*er,ey+ny*er); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(p1.x-nx*r1,p1.y-ny*r1); ctx.lineTo(ex-nx*er,ey-ny*er); ctx.stroke();
  ctx.restore();
}

// ─── Flowing particles ─────────────────────────────────────────────────────────
function drawTubeParticles(ctx, p1, p2, t, glow, alpha=1) {
  const [gr,gg,gb]=glow;
  ctx.save(); ctx.globalAlpha=alpha;
  for(let i=0;i<5;i++){
    const phase=((t*0.75+i/5)%1);
    const px=p1.x+(p2.x-p1.x)*phase;
    const py=p1.y+(p2.y-p1.y)*phase;
    const psc=lerp(p1.sc,p2.sc,phase);
    const sz=lerp(7,4,phase)*psc;
    const a=Math.sin(phase*Math.PI)*0.85;
    const g2=ctx.createRadialGradient(px,py,0,px,py,sz*2.2);
    g2.addColorStop(0,g3([gr,gg,gb],a));
    g2.addColorStop(1,g3([gr,gg,gb],0));
    ctx.fillStyle=g2;
    ctx.beginPath(); ctx.arc(px,py,sz*2.2,0,Math.PI*2); ctx.fill();
    ctx.fillStyle=g3([gr,gg,gb],a);
    ctx.beginPath(); ctx.arc(px,py,sz*0.45,0,Math.PI*2); ctx.fill();
  }
  ctx.restore();
}

// ─── Curved tube (for branch connectors) ──────────────────────────────────────
function drawCurvedTube(ctx, p1, p2, ctrl, glow, progress=1, alpha=1) {
  if(progress<=0) return;
  const [gr,gg,gb]=glow;
  ctx.save(); ctx.globalAlpha=alpha;
  const pts=[];
  const steps=30;
  for(let i=0;i<=steps*progress;i++){
    const t2=i/(steps);
    const bx=(1-t2)*(1-t2)*p1.x+2*(1-t2)*t2*ctrl.x+t2*t2*p2.x;
    const by=(1-t2)*(1-t2)*p1.y+2*(1-t2)*t2*ctrl.y+t2*t2*p2.y;
    pts.push({x:bx,y:by});
  }
  const grad=ctx.createLinearGradient(p1.x,p1.y,p2.x,p2.y);
  grad.addColorStop(0,g3([gr,gg,gb],0.12));
  grad.addColorStop(0.5,g3([gr,gg,gb],0.70));
  grad.addColorStop(1,g3([gr,gg,gb],0.12));
  ctx.strokeStyle=grad;
  ctx.lineWidth=TUBE_R*p1.sc*1.4;
  ctx.lineCap='round';
  ctx.beginPath();
  pts.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));
  ctx.stroke();
  ctx.strokeStyle=g3([gr,gg,gb],0.65);
  ctx.lineWidth=2;
  ctx.beginPath();
  pts.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));
  ctx.stroke();
  if(pts.length>0){
    const tip=pts[pts.length-1];
    const tg=ctx.createRadialGradient(tip.x,tip.y,0,tip.x,tip.y,10);
    tg.addColorStop(0,g3([gr,gg,gb],0.9));
    tg.addColorStop(1,g3([gr,gg,gb],0));
    ctx.fillStyle=tg;
    ctx.beginPath(); ctx.arc(tip.x,tip.y,10,0,Math.PI*2); ctx.fill();
  }
  ctx.restore();
}

// ─── Searching ring (orbits router sphere during lookup) ──────────────────────
function drawSearchRing(ctx, x, y, r, t) {
  for(let i=0;i<3;i++){
    const a=t*2.5+i*(Math.PI*2/3);
    const px=x+r*0.72*Math.cos(a);
    const py=y+r*0.72*Math.sin(a);
    const g2=ctx.createRadialGradient(px,py,0,px,py,6);
    g2.addColorStop(0,'rgba(99,102,241,0.9)');
    g2.addColorStop(1,'rgba(99,102,241,0)');
    ctx.fillStyle=g2;
    ctx.beginPath(); ctx.arc(px,py,6,0,Math.PI*2); ctx.fill();
  }
}

// ─── Generating animation (inside LLM sphere) ─────────────────────────────────
function drawGenerating(ctx, x, y, r, elapsed) {
  const [gr,gg,gb]=COL.generate.glow;
  ctx.save();
  ctx.font=`500 ${Math.max(9,r*0.21)}px Inter,sans-serif`;
  ctx.fillStyle='rgba(0,0,0,0.82)';
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText('Full LLM', x, y-r*0.28);
  ctx.fillText('generation', x, y);
  ctx.fillText('+ template', x, y+r*0.28);
  for(let i=0;i<3;i++){
    const by=y+r*0.58+Math.sin(elapsed*3.5+i*1.1)*4;
    const dg=ctx.createRadialGradient(x+(i-1)*13,by,0,x+(i-1)*13,by,5);
    dg.addColorStop(0,g3([gr,gg,gb],0.7));
    dg.addColorStop(1,g3([gr,gg,gb],0));
    ctx.fillStyle=dg;
    ctx.beginPath(); ctx.arc(x+(i-1)*13,by,5,0,Math.PI*2); ctx.fill();
  }
  ctx.restore();
}

// ─── Slot fill section (cache hit path) ───────────────────────────────────────
function drawSlotSection(ctx, W, H, t, stages) {
  const BASE_Y   = 3.2;
  const gapP     = sp(t, stages.gapDetect[0], stages.gapDetect[1]);
  const fillP    = sp(t, stages.slotFill[0],  stages.slotFill[1]);
  const fillCount = Math.floor(fillP * SLOTS.length);

  // Section title
  const titlePt = project(0, BASE_Y - 0.68, 0, W, H);
  ctx.save();
  ctx.globalAlpha = gapP;
  ctx.font = `600 ${Math.max(12, 13 * titlePt.sc)}px Inter, sans-serif`;
  ctx.fillStyle = 'rgba(56,189,248,0.85)';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const phase = t < stages.slotFill[0] ? 'Gap Detection' : 'Slot Engine — filling template';
  ctx.fillText(phase, titlePt.x, titlePt.y);
  ctx.restore();

  // Skeleton skeleton preview
  const skelPt = project(0, BASE_Y - 0.36, 0, W, H);
  ctx.save();
  ctx.globalAlpha = gapP * 0.5;
  ctx.font = `500 ${Math.max(9, 10 * skelPt.sc)}px "JetBrains Mono", monospace`;
  ctx.fillStyle = '#475569';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText('The [slot_0] [slot_1] via [slot_2] saves [slot_3] [slot_4]', skelPt.x, skelPt.y);
  ctx.restore();

  // Slot pills
  const spacing = 0.65;
  const startX  = -((SLOTS.length - 1) * spacing) / 2;

  SLOTS.forEach((slot, i) => {
    const wx = startX + i * spacing;
    const p  = project(wx, BASE_Y, 0, W, H);

    // Staggered reveal during gap detection
    const revealStart = stages.gapDetect[0] + (i / SLOTS.length) * (stages.gapDetect[1] - stages.gapDetect[0]);
    const revP = sp(t, revealStart, stages.gapDetect[1]);
    if (revP <= 0) return;

    const filled   = i < fillCount;
    const segStart = stages.slotFill[0] + (i / SLOTS.length) * (stages.slotFill[1] - stages.slotFill[0]);
    const fillAge  = ease(sp(t, segStart, segStart + 1.0));

    const isCache  = slot.source === 'cache';
    const bg    = filled ? (isCache ? 'rgba(4,38,28,0.95)'  : 'rgba(48,24,4,0.95)')  : 'rgba(14,20,52,0.90)';
    const bord  = filled ? (isCache ? 'rgba(16,185,129,0.7)': 'rgba(251,146,60,0.7)'): 'rgba(99,102,241,0.30)';
    const txCol = filled ? (isCache ? '#6ee7b7'             : '#fdba74')              : '#475569';

    const label = filled ? slot.value : slot.name;
    const fs    = Math.max(8, 10 * p.sc);
    const pillW = Math.max(44, (label.length * fs * 0.63 + 16)) * p.sc;
    const pillH = 24 * p.sc;

    ctx.save();
    ctx.globalAlpha = revP;

    // Glow burst on fill
    if (filled && fillAge < 0.7) {
      ctx.shadowColor = isCache ? 'rgba(16,185,129,0.8)' : 'rgba(251,146,60,0.8)';
      ctx.shadowBlur  = 20 * (1 - fillAge) * p.sc;
    }

    ctx.fillStyle = bg;
    ctx.beginPath(); rr(ctx, p.x - pillW/2, p.y - pillH/2, pillW, pillH, 5 * p.sc); ctx.fill();
    ctx.strokeStyle = bord; ctx.lineWidth = 1; ctx.shadowBlur = 0;
    ctx.beginPath(); rr(ctx, p.x - pillW/2, p.y - pillH/2, pillW, pillH, 5 * p.sc); ctx.stroke();

    ctx.font = `600 ${fs}px "JetBrains Mono", monospace`;
    ctx.fillStyle = txCol;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(label, p.x, p.y);

    // Source badge below
    if (filled && fillAge > 0.3) {
      const bdPt = project(wx, BASE_Y + 0.24, 0, W, H);
      ctx.globalAlpha = revP * Math.min(1, (fillAge - 0.3) / 0.4) * 0.85;
      ctx.font = `500 ${Math.max(7, 8 * bdPt.sc)}px Inter, sans-serif`;
      ctx.fillStyle = isCache ? 'rgba(16,185,129,0.8)' : 'rgba(251,146,60,0.8)';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(isCache ? '● cache' : '◈ llm', bdPt.x, bdPt.y);
    }

    // Slot label above
    const lblPt = project(wx, BASE_Y - 0.24, 0, W, H);
    ctx.globalAlpha = revP * 0.5;
    ctx.font = `400 ${Math.max(7, 8 * lblPt.sc)}px Inter, sans-serif`;
    ctx.fillStyle = '#475569';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(slot.label, lblPt.x, lblPt.y);

    ctx.restore();
  });
}

// ─── Floating token chip ──────────────────────────────────────────────────────
function drawToken3D(ctx, wx, wy, wz, W, H, text, type='normal', extraAlpha=1) {
  const p=project(wx,wy,wz,W,H);
  const styles={
    normal:  {bg:'rgba(18,28,68,0.95)',  border:'rgba(99,102,241,0.60)',  text:'#c7d2fe'},
    redacted:{bg:'rgba(48,8,8,0.95)',    border:'rgba(239,68,68,0.50)',   text:'#fca5a5'},
    id:      {bg:'rgba(8,38,28,0.95)',   border:'rgba(16,185,129,0.50)',  text:'#6ee7b7'},
    vector:  {bg:'rgba(28,8,48,0.95)',   border:'rgba(167,139,250,0.50)', text:'#ddd6fe'},
  };
  const st=styles[type]||styles.normal;
  const sc=p.sc;
  const fs=Math.max(8,11*sc);
  const W2=Math.max(32,(text.length*fs*0.62+14))*sc;
  const H2=22*sc;
  ctx.save();
  ctx.globalAlpha=extraAlpha;
  ctx.shadowColor=st.border; ctx.shadowBlur=6*sc;
  ctx.fillStyle=st.bg;
  ctx.beginPath(); rr(ctx,p.x-W2/2,p.y-H2/2,W2,H2,4*sc); ctx.fill();
  ctx.strokeStyle=st.border; ctx.lineWidth=1;
  ctx.shadowBlur=0;
  ctx.beginPath(); rr(ctx,p.x-W2/2,p.y-H2/2,W2,H2,4*sc); ctx.stroke();
  ctx.font=`600 ${fs}px "JetBrains Mono",monospace`;
  ctx.fillStyle=st.text; ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText(text,p.x,p.y);
  ctx.restore();
}

// ─── Bottom progress bar ──────────────────────────────────────────────────────
function drawProgress(ctx, W, H, t, total) {
  const p=Math.min(1,t/total);
  const bw=W*0.38,bh=3,bx=W/2-bw/2,by=H*0.94;
  ctx.save();
  ctx.fillStyle='rgba(255,255,255,0.06)';
  ctx.beginPath(); rr(ctx,bx,by,bw,bh,2); ctx.fill();
  const gr=ctx.createLinearGradient(bx,by,bx+bw*p,by);
  gr.addColorStop(0,'rgba(99,102,241,0.85)');
  gr.addColorStop(1,'rgba(139,92,246,0.95)');
  ctx.fillStyle=gr;
  ctx.beginPath(); rr(ctx,bx,by,bw*p,bh,2); ctx.fill();
  ctx.restore();
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function PipelineAnimation({ visible, cacheHit=false, onComplete }) {
  const canvasRef=useRef(null);

  useEffect(()=>{
    if(!visible) return;
    const canvas=canvasRef.current;
    const ctx=canvas.getContext('2d');
    let animId, startTime=null, completed=false;

    const resize=()=>{ canvas.width=window.innerWidth; canvas.height=window.innerHeight; };
    resize(); window.addEventListener('resize',resize);

    const S=makeStages(cacheHit);

    function N(W,H){ return Object.fromEntries(Object.entries(NP).map(([k,v])=>[k,project(...v,W,H)])); }

    function draw(ts){
      if(!startTime) startTime=ts;
      const t=(ts-startTime)/1000;
      if(!completed && t>=S.done){ completed=true; onComplete?.(); cancelAnimationFrame(animId); return; }

      const W=canvas.width, H=canvas.height;
      const n=N(W,H);
      const R=BASE_R;

      // ── Background ──────────────────────────────────────────────────────
      ctx.fillStyle='rgba(4,7,14,0.97)';
      ctx.fillRect(0,0,W,H);
      ctx.save();
      ctx.strokeStyle='rgba(99,102,241,0.04)';
      ctx.lineWidth=1;
      const GS=55;
      for(let x=0;x<W;x+=GS){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
      for(let y=0;y<H;y+=GS){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
      ctx.restore();

      // ── Scene title ──────────────────────────────────────────────────────
      ctx.save();
      ctx.globalAlpha=Math.min(1,t/0.7)*0.55;
      ctx.font=`500 ${Math.min(15,W*0.011)}px Inter,system-ui,sans-serif`;
      ctx.fillStyle='#94a3b8';
      ctx.textAlign='center'; ctx.textBaseline='top';
      ctx.fillText('inferOpt · TemplateCache Pipeline', W/2, H*0.038);
      ctx.restore();

      // ─ Lerp colour helper ───────────────────────────────────────────────
      function lerpCol(a,b,t2){
        return { glow: lerpA(a.glow,b.glow,t2), fill: a.fill };
      }

      // ── Router colour based on result ────────────────────────────────────
      let routerCol=COL.neutral;
      if(t>=S.routerResult[0]){
        const rp=ease(sp(t,S.routerResult[0],S.routerResult[1]));
        routerCol=lerpCol(COL.neutral, cacheHit ? COL.routerHit : COL.routerMiss, rp);
      }

      // ── Stage label ──────────────────────────────────────────────────────
      let stageLabel='';
      if     (t < S.flowSplit[0])      stageLabel='Receiving query';
      else if(t < S.splitCheck[0])     stageLabel='Checking for multiple sub-questions';
      else if(t < S.flowEmbed[0])      stageLabel='Query split analysis complete';
      else if(t < S.flowRouter[0])     stageLabel='Embedding query vector';
      else if(t < S.routerResult[0])   stageLabel='Searching intent centroids…';
      else if(!cacheHit && t<S.done-1) stageLabel='Cache miss — running full LLM + extracting template';
      else if(cacheHit && S.gapDetect && t>=S.gapDetect[0] && t<S.slotFill[0])
                                        stageLabel='Detecting gaps in cached template';
      else if(cacheHit && S.slotFill && t>=S.slotFill[0])
                                        stageLabel='SlotEngine — filling template from cache + LLM';

      // ── Tubes (behind spheres) ───────────────────────────────────────────
      // input → split
      drawTube(ctx, n.input, n.split, [99,102,241], sp(t,S.flowSplit[0],S.flowSplit[1]));
      if(t>=S.flowSplit[0]) drawTubeParticles(ctx,n.input,n.split,t,[99,102,241],sp(t,S.flowSplit[0],S.flowSplit[1]));

      // split → embed
      drawTube(ctx, n.split, n.embed, [99,102,241], sp(t,S.flowEmbed[0],S.flowEmbed[1]));
      if(t>=S.flowEmbed[0]) drawTubeParticles(ctx,n.split,n.embed,t,[99,102,241],sp(t,S.flowEmbed[0],S.flowEmbed[1]));

      // embed → router
      drawTube(ctx, n.embed, n.router, [99,102,241], sp(t,S.flowRouter[0],S.flowRouter[1]));
      if(t>=S.flowRouter[0]) drawTubeParticles(ctx,n.embed,n.router,t,[99,102,241],sp(t,S.flowRouter[0],S.flowRouter[1]));

      // Branch connectors
      if(!cacheHit && S.flowLLM){
        const cp={x:(n.router.x+n.llm.x)/2, y:n.router.y+30};
        drawCurvedTube(ctx, n.router, n.llm, cp, [239,68,68], sp(t,S.flowLLM[0],S.flowLLM[1]));
        if(t>=S.flowLLM[0]) drawTubeParticles(ctx,n.router,n.llm,t,[239,68,68],sp(t,S.flowLLM[0],S.flowLLM[1]));
      }
      if(cacheHit && S.flowSlot){
        const cp={x:(n.router.x+n.slot.x)/2, y:n.router.y+30};
        drawCurvedTube(ctx, n.router, n.slot, cp, [16,185,129], sp(t,S.flowSlot[0],S.flowSlot[1]));
        if(t>=S.flowSlot[0]) drawTubeParticles(ctx,n.router,n.slot,t,[16,185,129],sp(t,S.flowSlot[0],S.flowSlot[1]));
      }

      // ── Spheres ──────────────────────────────────────────────────────────

      // Input — orbiting query tokens
      const platA=ease(sp(t,0,0.9));
      if(platA>0){
        ['query','text','?'].forEach((tok,i)=>{
          const orbitAngle=t*0.6+i*(Math.PI*2/3);
          const ox=Math.cos(orbitAngle)*0.22;
          const oz=Math.sin(orbitAngle)*0.1;
          const fadeOut=1-sp(t,S.flowSplit[0],S.flowSplit[0]+0.55);
          drawToken3D(ctx,ox,-2.5+oz,oz,W,H,tok,'normal',platA*fadeOut);
        });
        const ip=project(0,-2.65,0,W,H);
        ctx.save(); ctx.globalAlpha=platA*(1-sp(t,S.flowSplit[0],S.flowSplit[0]+0.5));
        const pg=ctx.createRadialGradient(ip.x,ip.y+ip.sc*8,3,ip.x,ip.y+ip.sc*8,ip.sc*55);
        pg.addColorStop(0,'rgba(99,102,241,0.25)'); pg.addColorStop(1,'rgba(99,102,241,0)');
        ctx.fillStyle=pg;
        ctx.beginPath(); ctx.ellipse(ip.x,ip.y+ip.sc*10,ip.sc*55,ip.sc*14,0,0,Math.PI*2); ctx.fill();
        ctx.strokeStyle='rgba(99,102,241,0.28)'; ctx.lineWidth=1;
        ctx.beginPath(); ctx.ellipse(ip.x,ip.y+ip.sc*10,ip.sc*55,ip.sc*14,0,0,Math.PI*2); ctx.stroke();
        ctx.restore();
      }

      // Split sphere
      const splitA=ease(sp(t,1.0,1.9));
      if(splitA>0){
        drawSphere(ctx, n.split.x, n.split.y, R*n.split.sc, COL.neutral, 'Split', 'multi-query?', splitA);
        // Show sub-question chips during splitCheck
        if(t>=S.splitCheck[0] && t<S.flowEmbed[0]+0.3){
          const ip2=sp(t,S.splitCheck[0],S.splitCheck[1]);
          const fa=ip2*(1-sp(t,S.flowEmbed[0],S.flowEmbed[0]+0.3))*splitA;
          drawToken3D(ctx,-0.18,-1.5+0.15,0,W,H,'Q1','id',fa);
          drawToken3D(ctx, 0.18,-1.5+0.15,0,W,H,'Q2','vector',fa);
        }
      }

      // Embedding sphere
      const embA=ease(sp(t,3.0,3.9));
      if(embA>0){
        drawSphere(ctx, n.embed.x, n.embed.y, R*n.embed.sc, COL.neutral, 'Embed', 'query→vector', embA);
        if(t>=S.flowEmbed[0]+0.4 && t<S.flowRouter[0]+0.25){
          const ip2=sp(t,S.flowEmbed[0]+0.5,S.flowEmbed[1]);
          const fa=ip2*(1-sp(t,S.flowRouter[0],S.flowRouter[0]+0.3))*embA;
          ['0.23','-0.87','0.41'].forEach((v,i)=>drawToken3D(ctx,(i-1)*0.19,-0.4+0.15,0,W,H,v,'vector',fa));
        }
      }

      // Intent Router sphere
      const routerA=ease(sp(t,4.1,5.0));
      if(routerA>0){
        const routerLabel=t>=S.routerResult[0]?(cacheHit?'Hit ✓':'Miss ✗'):'Router';
        const routerSub=t>=S.routerSearch[0]&&t<S.routerResult[0]?'searching…':'cosine sim';
        drawSphere(ctx, n.router.x, n.router.y, R*n.router.sc, routerCol, routerLabel, routerSub, routerA);
        if(t>=S.routerSearch[0]&&t<S.routerResult[0]) drawSearchRing(ctx,n.router.x,n.router.y,R*n.router.sc,t);
      }

      // LLM sphere (cache miss)
      if(!cacheHit && S.flowLLM && t>=S.flowLLM[0]){
        const ga=ease(sp(t,S.flowLLM[0],S.flowLLM[0]+0.9));
        drawSphere(ctx,n.llm.x,n.llm.y,R*n.llm.sc,COL.generate,'LLM','full gen',ga);
        if(t>=S.llmGen[0]) drawGenerating(ctx,n.llm.x,n.llm.y,R*n.llm.sc,t-S.llmGen[0]);
      }

      // Slot Engine sphere (cache hit)
      if(cacheHit && S.flowSlot && t>=S.flowSlot[0]){
        const ca=ease(sp(t,S.flowSlot[0],S.flowSlot[0]+0.9));
        drawSphere(ctx,n.slot.x,n.slot.y,R*n.slot.sc,COL.slotNode,'SlotEngine','filling…',ca);
      }

      // Slot fill visualization
      if(cacheHit && S.gapDetect && t>=S.gapDetect[0]) drawSlotSection(ctx,W,H,t,S);

      // Stage label
      if(stageLabel){
        ctx.save();
        ctx.globalAlpha=0.62;
        ctx.font='500 13px Inter,system-ui,sans-serif';
        ctx.fillStyle='#94a3b8';
        ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText(stageLabel, W/2, H*0.90);
        ctx.restore();
      }

      drawProgress(ctx,W,H,t,S.done);
      animId=requestAnimationFrame(draw);
    }

    animId=requestAnimationFrame(draw);
    return ()=>{ cancelAnimationFrame(animId); window.removeEventListener('resize',resize); };
  },[visible,cacheHit,onComplete]);

  return (
    <AnimatePresence>
      {visible&&(
        <motion.div
          key="pipeline-anim"
          initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}}
          transition={{duration:0.35}}
          style={{position:'fixed',inset:0,zIndex:20}}
        >
          <canvas ref={canvasRef} style={{display:'block',width:'100%',height:'100%'}}/>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
