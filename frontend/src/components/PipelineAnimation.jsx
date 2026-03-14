import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// ─── 3-D Camera ────────────────────────────────────────────────────────────────
const ROT_X = 0.22;   // tilt down — lets you see the top of each sphere
const ROT_Y = 0.15;   // slight side rotation — depth visible
const FOV   = 1000;
const SCALE = 100;    // world-unit → pixel multiplier

function project(wx, wy, wz, W, H) {
  // Y-axis rotation
  const cy = Math.cos(ROT_Y), sy = Math.sin(ROT_Y);
  const x1 =  wx * cy + wz * sy;
  const z1 = -wx * sy + wz * cy;
  // X-axis rotation
  const cx = Math.cos(ROT_X), sx = Math.sin(ROT_X);
  const y2 =  wy * cx - z1 * sx;
  const z2 =  wy * sx + z1 * cx;
  // Perspective divide
  const sc = FOV / (FOV + z2 * SCALE);
  return { x: W / 2 + x1 * SCALE * sc, y: H / 2 + y2 * SCALE * sc, sc, z: z2 };
}

// ─── Node world positions (top → bottom along Y axis) ─────────────────────────
const NP = {
  input:      [0,      -2.2,  0   ],
  redact:     [0,      -1.3,  0   ],
  tokenize:   [0,      -0.4,  0   ],
  embed:      [0,       0.5,  0   ],
  db:         [0,       1.4,  0   ],
  generate:   [ 1.85,   2.3,  0   ],
  confidence: [-1.85,   2.3,  0   ],
};
const BASE_R  = 68;  // sphere radius at sc=1
const TUBE_R  = 8;   // tube half-width at sc=1

// ─── Stage timing (seconds) ───────────────────────────────────────────────────
function makeStages(cacheHit) {
  const s = {
    tokensAppear: [0.0,  1.2],
    flowRedact:   [1.2,  2.4],
    flowTok:      [2.4,  3.6],
    flowEmbed:    [3.6,  4.8],
    flowDB:       [4.8,  6.2],
    dbSearch:     [5.8,  7.2],
    dbResult:     [7.2,  8.0],
  };
  if (!cacheHit) {
    s.flowGen   = [8.0,  9.2];
    s.generating= [9.0, 11.5];
    s.done      = 11.8;
  } else {
    s.flowConf  = [8.0,  9.2];
    s.tokenLine = [9.2, 11.2];
    s.removeLow = [11.2,12.8];
    s.catFill   = [12.8,16.5];
    s.done      = 16.8;
  }
  return s;
}

// ─── Colour sets ──────────────────────────────────────────────────────────────
const COL = {
  neutral:    { glow:[99,102,241],  fill:'rgba(14,20,52,0.62)'  },
  generate:   { glow:[251,146,60],  fill:'rgba(48,16,4,0.62)'   },
  confidence: { glow:[56,189,248],  fill:'rgba(4,32,52,0.62)'   },
  dbHit:      { glow:[16,185,129],  fill:'rgba(4,36,22,0.65)'   },
  dbMiss:     { glow:[239,68,68],   fill:'rgba(48,8,8,0.65)'    },
};

// ─── Sample tokens for the cache-hit perplexity step ─────────────────────────
const TOKENS = [
  {w:'What',     s:0.94},{w:'is',      s:0.91},{w:'the',     s:0.87},
  {w:'fastest',  s:0.26},{w:'way',     s:0.83},{w:'to',      s:0.89},
  {w:'reduce',   s:0.31},{w:'cost',    s:0.92},{w:'of',      s:0.88},
  {w:'LLM',      s:0.23},{w:'inference',s:0.85},
];
const LOW_IDX = TOKENS.map((t,i)=>t.s<0.45?i:-1).filter(i=>i>=0);

// ─── Math helpers ─────────────────────────────────────────────────────────────
const lerp   = (a,b,t) => a+(b-a)*Math.max(0,Math.min(1,t));
const sp     = (t,s,e) => Math.max(0,Math.min(1,(t-s)/(e-s)));
const ease   = t => t<0.5?2*t*t:-1+(4-2*t)*t;
const g3     = ([r,g,b],a) => `rgba(${r},${g},${b},${a})`;
const lerpA  = (a,b,t) => a.map((v,i)=>Math.round(lerp(v,b[i],t)));

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

  // Drop shadow (flat ellipse on a virtual floor below sphere)
  ctx.globalAlpha = alpha * 0.3;
  ctx.fillStyle='rgba(0,0,0,0.7)';
  ctx.beginPath(); ctx.ellipse(x+r*0.1, y+r*0.9, r*0.85, r*0.22, 0, 0, Math.PI*2); ctx.fill();

  ctx.globalAlpha = alpha;

  // Outer corona glow
  const corona=ctx.createRadialGradient(x,y,r*0.4,x,y,r*2.1);
  corona.addColorStop(0, g3([gr,gg,gb],0.22));
  corona.addColorStop(1, g3([gr,gg,gb],0));
  ctx.fillStyle=corona;
  ctx.beginPath(); ctx.arc(x,y,r*2.1,0,Math.PI*2); ctx.fill();

  // Main sphere body — radial gradient offset to upper-left for lighting
  const body=ctx.createRadialGradient(x-r*0.30, y-r*0.30, r*0.04, x+r*0.18, y+r*0.18, r*1.12);
  body.addColorStop(0,   'rgba(255,255,255,0.18)');
  body.addColorStop(0.22, g3([gr,gg,gb],0.40));
  body.addColorStop(0.65, g3([gr,gg,gb],0.20));
  body.addColorStop(1,    col.fill);
  ctx.fillStyle=body;
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill();

  // Wireframe equator ring (foreshortened for 3-D feel)
  ctx.strokeStyle=g3([gr,gg,gb],0.30);
  ctx.lineWidth=1.1;
  ctx.beginPath(); ctx.ellipse(x,y, r*0.97, r*0.30, 0.12, 0, Math.PI*2); ctx.stroke();
  // Wireframe meridian ring
  ctx.beginPath(); ctx.ellipse(x,y, r*0.30, r*0.97, 0.22, 0, Math.PI*2); ctx.stroke();

  // Outer glowing rim
  ctx.strokeStyle=g3([gr,gg,gb],0.60);
  ctx.lineWidth=1.8;
  ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.stroke();

  // Specular highlight — bright spot upper-left
  const hl=ctx.createRadialGradient(x-r*0.33,y-r*0.33, 0, x-r*0.33,y-r*0.33, r*0.46);
  hl.addColorStop(0,'rgba(255,255,255,0.50)');
  hl.addColorStop(1,'rgba(255,255,255,0)');
  ctx.fillStyle=hl;
  ctx.beginPath(); ctx.arc(x-r*0.33,y-r*0.33, r*0.46, 0, Math.PI*2); ctx.fill();

  // Labels
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

// ─── 3-D Tube (perspective-correct cylinder between two screen points) ────────
function drawTube(ctx, p1, p2, glow, progress=1, alpha=1) {
  if(progress<=0) return;
  const [gr,gg,gb]=glow;
  const r1=TUBE_R*p1.sc, r2=TUBE_R*p2.sc;

  // Partial endpoint
  const ex=lerp(p1.x, p2.x, progress);
  const ey=lerp(p1.y, p2.y, progress);
  const er=lerp(r1,   r2,   progress);

  const dx=ex-p1.x, dy=ey-p1.y;
  const len=Math.sqrt(dx*dx+dy*dy);
  if(len<1) return;
  const nx=-dy/len, ny=dx/len;

  ctx.save(); ctx.globalAlpha=alpha;

  // Fill — cross-section gradient for cylindrical shading
  const fillGrad=ctx.createLinearGradient(
    p1.x+nx*r1, p1.y+ny*r1, p1.x-nx*r1, p1.y-ny*r1
  );
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
  ctx.closePath();
  ctx.fill();

  // Edge highlight lines
  ctx.strokeStyle=g3([gr,gg,gb],0.55);
  ctx.lineWidth=1.1;
  ctx.beginPath(); ctx.moveTo(p1.x+nx*r1,p1.y+ny*r1); ctx.lineTo(ex+nx*er,ey+ny*er); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(p1.x-nx*r1,p1.y-ny*r1); ctx.lineTo(ex-nx*er,ey-ny*er); ctx.stroke();

  ctx.restore();
}

// ─── Flowing particles along a tube ───────────────────────────────────────────
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

// ─── 3-D Curved tube (for the diagonal branch connectors) ─────────────────────
function drawCurvedTube(ctx, p1, p2, ctrl, glow, progress=1, alpha=1) {
  if(progress<=0) return;
  const [gr,gg,gb]=glow;
  ctx.save(); ctx.globalAlpha=alpha;

  // Sample the bezier for the partial active portion
  const pts=[];
  const steps=30;
  for(let i=0;i<=steps*progress;i++){
    const t2=i/(steps);
    const bx=(1-t2)*(1-t2)*p1.x+2*(1-t2)*t2*ctrl.x+t2*t2*p2.x;
    const by=(1-t2)*(1-t2)*p1.y+2*(1-t2)*t2*ctrl.y+t2*t2*p2.y;
    pts.push({x:bx,y:by});
  }

  // Draw glowing path
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

  // Core bright line
  ctx.strokeStyle=g3([gr,gg,gb],0.65);
  ctx.lineWidth=2;
  ctx.beginPath();
  pts.forEach((p,i)=>i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));
  ctx.stroke();

  // Tip glow
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

// ─── Floating 3-D token chip (positioned by world coords) ─────────────────────
function drawToken3D(ctx, wx, wy, wz, W, H, text, type='normal', extraAlpha=1) {
  const p=project(wx,wy,wz,W,H);
  const styles={
    normal:  {bg:'rgba(18,28,68,0.95)',  border:'rgba(99,102,241,0.60)',  text:'#c7d2fe'},
    redacted:{bg:'rgba(48,8,8,0.95)',    border:'rgba(239,68,68,0.50)',   text:'#fca5a5'},
    id:      {bg:'rgba(8,38,28,0.95)',   border:'rgba(16,185,129,0.50)',  text:'#6ee7b7'},
    vector:  {bg:'rgba(28,8,48,0.95)',   border:'rgba(167,139,250,0.50)', text:'#ddd6fe'},
    lowConf: {bg:'rgba(48,24,4,0.95)',   border:'rgba(251,146,60,0.55)',  text:'#fdba74'},
    filled:  {bg:'rgba(4,38,28,0.95)',   border:'rgba(16,185,129,0.70)',  text:'#6ee7b7'},
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

// ─── Searching ring orbiting the DB sphere ────────────────────────────────────
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

// ─── Generating animation inside the generation sphere ────────────────────────
function drawGenerating(ctx, x, y, r, elapsed) {
  const [gr,gg,gb]=COL.generate.glow;
  ctx.save();
  ctx.font=`500 ${Math.max(9,r*0.21)}px Inter,sans-serif`;
  ctx.fillStyle='rgba(0,0,0,0.82)';
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText('Generating',  x, y-r*0.16);
  ctx.fillText('response...', x, y+r*0.16);
  for(let i=0;i<3;i++){
    const by=y+r*0.48+Math.sin(elapsed*3.5+i*1.1)*4;
    ctx.fillStyle='rgba(0,0,0,0.70)';
    ctx.beginPath(); ctx.arc(x+(i-1)*13,by,3.2,0,Math.PI*2); ctx.fill();
  }
  ctx.restore();
}

// ─── Cat character (N-gram model) ─────────────────────────────────────────────
function drawCat(ctx, x, y, sc2, mouthOpen, pawRaise) {
  const s=sc2;
  ctx.save(); ctx.translate(x,y);

  // Ground shadow
  ctx.fillStyle='rgba(0,0,0,0.22)';
  ctx.beginPath(); ctx.ellipse(4*s,15*s,26*s,6*s,0,0,Math.PI*2); ctx.fill();

  // Tail
  ctx.strokeStyle='#c8a87a'; ctx.lineWidth=4.5*s; ctx.lineCap='round';
  ctx.beginPath(); ctx.moveTo(-20*s,5*s); ctx.quadraticCurveTo(-38*s,-6*s,-28*s,-22*s); ctx.stroke();

  // Body
  ctx.fillStyle='#d4b483';
  ctx.beginPath(); ctx.ellipse(0,2*s,20*s,14*s,0,0,Math.PI*2); ctx.fill();

  // Back paw
  ctx.fillStyle='#c8a87a';
  ctx.beginPath(); ctx.ellipse(-8*s,13*s,9*s,5.5*s,0.3,0,Math.PI*2); ctx.fill();

  // Head
  ctx.fillStyle='#dfc090';
  ctx.beginPath(); ctx.arc(24*s,-10*s,14*s,0,Math.PI*2); ctx.fill();

  // Ears outer
  ctx.fillStyle='#c8a87a';
  [[17,-23,21,-10,27,-23],[27,-24,31,-11,37,-24]].forEach(([ax,ay,bx,by,cx2,cy2])=>{
    ctx.beginPath(); ctx.moveTo(ax*s,ay*s); ctx.lineTo(bx*s,by*s); ctx.lineTo(cx2*s,cy2*s); ctx.closePath(); ctx.fill();
  });
  // Ears inner
  ctx.fillStyle='#e8a8a8';
  [[19,-21,22,-12,26,-21],[29,-22,32,-12,36,-22]].forEach(([ax,ay,bx,by,cx2,cy2])=>{
    ctx.beginPath(); ctx.moveTo(ax*s,ay*s); ctx.lineTo(bx*s,by*s); ctx.lineTo(cx2*s,cy2*s); ctx.closePath(); ctx.fill();
  });

  // Eyes
  ctx.fillStyle='#2d2d2d';
  ctx.beginPath(); ctx.ellipse(19*s,-12*s,2.8*s,3.2*s,0,0,Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.ellipse(29*s,-12*s,2.8*s,3.2*s,0,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='rgba(255,255,255,0.9)';
  ctx.beginPath(); ctx.arc(20.2*s,-13.5*s,1*s,0,Math.PI*2); ctx.fill();
  ctx.beginPath(); ctx.arc(30.2*s,-13.5*s,1*s,0,Math.PI*2); ctx.fill();

  // Nose
  ctx.fillStyle='#e87a7a';
  ctx.beginPath(); ctx.moveTo(24*s,-8.5*s); ctx.lineTo(22*s,-7*s); ctx.lineTo(26*s,-7*s); ctx.closePath(); ctx.fill();

  // Mouth
  ctx.strokeStyle='#a06060'; ctx.lineWidth=1.2*s;
  if(mouthOpen){
    ctx.fillStyle='rgba(80,20,20,0.7)';
    ctx.beginPath(); ctx.arc(24*s,-5.5*s,3*s,0.1,Math.PI-0.1); ctx.fill();
  } else {
    ctx.beginPath(); ctx.moveTo(21.5*s,-6.5*s); ctx.quadraticCurveTo(24*s,-4.5*s,26.5*s,-6.5*s); ctx.stroke();
  }

  // Whiskers
  ctx.strokeStyle='rgba(230,230,230,0.8)'; ctx.lineWidth=0.9*s;
  [[8,-9,18,-8],[8,-7,18,-7],[8,-5,18,-6],[30,-8,42,-9],[30,-7,42,-7],[30,-6,42,-5]].forEach(([ax,ay,bx,by])=>{
    ctx.beginPath(); ctx.moveTo(ax*s,ay*s); ctx.lineTo(bx*s,by*s); ctx.stroke();
  });

  // Raised paw
  const py=lerp(6*s,-10*s,pawRaise);
  ctx.fillStyle='#dfc090';
  ctx.beginPath(); ctx.ellipse(12*s,py,8*s,5.5*s,-0.4,0,Math.PI*2); ctx.fill();
  ctx.fillStyle='#c07060';
  for(let i=0;i<3;i++){ctx.beginPath();ctx.arc((9+i*3)*s,py+4*s,1.5*s,0,Math.PI*2);ctx.fill();}

  ctx.restore();
}

// ─── Perplexity token line + cat (all in 3-D projected space) ────────────────
function drawPerplexitySection(ctx, W, H, t, stages) {
  const BASE_Y=3.1;
  const SPACING=0.52;
  const startX=-((TOKENS.length-1)*SPACING)/2;

  const lineReveal = sp(t, stages.tokenLine[0], stages.tokenLine[1]);
  const removeP    = sp(t, stages.removeLow[0],  stages.removeLow[1]);
  const catP       = sp(t, stages.catFill[0],    stages.catFill[1]);
  const catActive  = t>=stages.catFill[0] && catP<1;
  const catFillCount=Math.floor(catP*LOW_IDX.length);

  // Title label
  const titleProj=project(0,BASE_Y-0.55,0,W,H);
  ctx.save();
  ctx.globalAlpha=lineReveal;
  ctx.font=`600 ${Math.max(12,13*titleProj.sc)}px Inter,sans-serif`;
  ctx.fillStyle='rgba(56,189,248,0.85)';
  ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText('Token Confidence Estimation', titleProj.x, titleProj.y);
  ctx.restore();

  TOKENS.forEach((tok,i)=>{
    if(i/TOKENS.length>lineReveal) return;
    const wx=startX+i*SPACING;
    const p=project(wx,BASE_Y,0,W,H);
    const isLow=tok.s<0.45;
    const fillIdx=LOW_IDX.indexOf(i);
    const isFilled=fillIdx>=0 && fillIdx<catFillCount;
    const revP=sp(t,stages.tokenLine[0]+(i/TOKENS.length)*(stages.tokenLine[1]-stages.tokenLine[0]), stages.tokenLine[1]);

    // Perplexity bar above token
    const barProj=project(wx,BASE_Y-0.32,0,W,H);
    const barH=24*barProj.sc, barW=7*barProj.sc;
    ctx.save();
    ctx.globalAlpha=lineReveal*0.85;
    ctx.fillStyle='rgba(255,255,255,0.06)';
    ctx.beginPath(); rr(ctx,barProj.x-barW/2,barProj.y-barH,barW,barH,2); ctx.fill();
    const barFill=barH*tok.s;
    ctx.fillStyle=isLow?'rgba(251,146,60,0.75)':'rgba(16,185,129,0.65)';
    ctx.beginPath(); rr(ctx,barProj.x-barW/2,barProj.y-barFill,barW,barFill,2); ctx.fill();
    // Score text
    ctx.font=`600 ${Math.max(8,9*barProj.sc)}px "JetBrains Mono",monospace`;
    ctx.fillStyle=isLow?'rgba(251,146,60,0.9)':'rgba(16,185,129,0.8)';
    ctx.textAlign='center'; ctx.textBaseline='bottom';
    ctx.fillText(tok.s.toFixed(2),barProj.x,barProj.y-barH-2);
    ctx.restore();

    // Token chip
    if(isLow && !isFilled){
      const remA=1-ease(removeP);
      if(remA<0.03){
        // Empty slot dashes
        ctx.save();
        ctx.globalAlpha=0.22;
        ctx.strokeStyle='rgba(251,146,60,0.5)'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
        const W2=32*p.sc, H2=20*p.sc;
        ctx.beginPath(); rr(ctx,p.x-W2/2,p.y-H2/2,W2,H2,3); ctx.stroke();
        ctx.setLineDash([]); ctx.restore();
      } else {
        drawToken3D(ctx,wx,BASE_Y,0,W,H,tok.w,'lowConf',remA*revP);
      }
    } else if(isFilled){
      const segStart=stages.catFill[0]+fillIdx*(stages.catFill[1]-stages.catFill[0])/LOW_IDX.length;
      const segEnd=stages.catFill[0]+(fillIdx+1)*(stages.catFill[1]-stages.catFill[0])/LOW_IDX.length;
      const fillA=Math.min(1,sp(t,segStart,segEnd)*3);
      drawToken3D(ctx,wx,BASE_Y,0,W,H,tok.w,'filled',fillA);
    } else {
      drawToken3D(ctx,wx,BASE_Y,0,W,H,tok.w,'normal',revP);
    }
  });

  // Cat
  if(catActive){
    const filledSoFar=catFillCount;
    const nextLow=filledSoFar<LOW_IDX.length ? LOW_IDX[filledSoFar] : TOKENS.length-1;
    const prevLow=filledSoFar>0 ? LOW_IDX[filledSoFar-1] : -1;
    const prevX=prevLow>=0 ? startX+prevLow*SPACING : startX-SPACING*1.5;
    const nextX=startX+nextLow*SPACING;
    const segDur=(stages.catFill[1]-stages.catFill[0])/Math.max(LOW_IDX.length,1);
    const segT=(t-stages.catFill[0])%segDur/segDur;
    const catWX=lerp(prevX,nextX,ease(Math.min(segT*1.8,1)));
    const cp=project(catWX,BASE_Y-0.28,0.3,W,H);
    const isPlacing=LOW_IDX.some(i=>Math.abs(catWX-(startX+i*SPACING))<0.1);
    const pawR=isPlacing?Math.abs(Math.sin(t*6)):0;
    drawCat(ctx,cp.x,cp.y,cp.sc*0.95,isPlacing,pawR);

    // "N-gram model" label above cat
    ctx.save();
    ctx.globalAlpha=0.75;
    ctx.font=`600 ${Math.max(10,11*cp.sc)}px Inter,sans-serif`;
    ctx.fillStyle='rgba(16,185,129,0.8)';
    ctx.textAlign='center'; ctx.textBaseline='bottom';
    ctx.fillText('N-gram model',cp.x,cp.y-BASE_R*cp.sc-8);
    ctx.restore();
  }
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

    // Pre-project all node positions each frame (W/H can change)
    function N(W,H){ return Object.fromEntries(Object.entries(NP).map(([k,v])=>[k,project(...v,W,H)])); }

    function draw(ts){
      if(!startTime) startTime=ts;
      const t=(ts-startTime)/1000;
      if(!completed && t>=S.done){ completed=true; onComplete?.(); cancelAnimationFrame(animId); return; }

      const W=canvas.width, H=canvas.height;
      const n=N(W,H);
      const R=BASE_R; // sphere base radius (perspective scales it per-node)

      // ── Background ──────────────────────────────────────────────────────
      ctx.fillStyle='rgba(4,7,14,0.97)';
      ctx.fillRect(0,0,W,H);

      // Subtle perspective grid on a virtual floor
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
      ctx.fillText('inferOpt · Semantic Cache Pipeline', W/2, H*0.038);
      ctx.restore();

      // ─ Helper: lerp color sets ──────────────────────────────────────────
      function lerpCol(a,b,t2){
        return { glow: lerpA(a.glow,b.glow,t2), fill: a.fill };
      }

      // ── Determine DB colour ──────────────────────────────────────────────
      let dbCol=COL.neutral;
      if(t>=S.dbResult[0]){
        const rp=ease(sp(t,S.dbResult[0],S.dbResult[1]));
        dbCol=lerpCol(COL.neutral, cacheHit?COL.dbHit:COL.dbMiss, rp);
      }

      // ── Stage label ──────────────────────────────────────────────────────
      let stageLabel='';
      if     (t < S.flowRedact[0])    stageLabel='Receiving tokens';
      else if(t < S.flowTok[0])       stageLabel='Redacting private information';
      else if(t < S.flowEmbed[0])     stageLabel='Converting to token IDs';
      else if(t < S.flowDB[0])        stageLabel='Generating embedding vector';
      else if(t < S.dbResult[0])      stageLabel='Searching semantic cache…';
      else if(!cacheHit && t<S.done-1)stageLabel='Cache miss — running full inference';
      else if(cacheHit && S.tokenLine && t>=S.tokenLine[0] && t<S.removeLow[0])
                                       stageLabel='Cache hit — estimating token confidence';
      else if(cacheHit && S.removeLow && t>=S.removeLow[0] && t<S.catFill[0])
                                       stageLabel='Removing low-confidence tokens';
      else if(cacheHit && S.catFill && t>=S.catFill[0])
                                       stageLabel='N-gram model filling token gaps';

      // ── Tubes (drawn behind spheres) ────────────────────────────────────
      // input → redact
      drawTube(ctx, n.input, n.redact, [99,102,241], sp(t,S.flowRedact[0],S.flowRedact[1]));
      if(t>=S.flowRedact[0]) drawTubeParticles(ctx,n.input,n.redact,t,[99,102,241],sp(t,S.flowRedact[0],S.flowRedact[1]));

      // redact → tokenize
      drawTube(ctx, n.redact, n.tokenize, [99,102,241], sp(t,S.flowTok[0],S.flowTok[1]));
      if(t>=S.flowTok[0]) drawTubeParticles(ctx,n.redact,n.tokenize,t,[99,102,241],sp(t,S.flowTok[0],S.flowTok[1]));

      // tokenize → embed
      drawTube(ctx, n.tokenize, n.embed, [99,102,241], sp(t,S.flowEmbed[0],S.flowEmbed[1]));
      if(t>=S.flowEmbed[0]) drawTubeParticles(ctx,n.tokenize,n.embed,t,[99,102,241],sp(t,S.flowEmbed[0],S.flowEmbed[1]));

      // embed → db
      drawTube(ctx, n.embed, n.db, [99,102,241], sp(t,S.flowDB[0],S.flowDB[1]));
      if(t>=S.flowDB[0]) drawTubeParticles(ctx,n.embed,n.db,t,[99,102,241],sp(t,S.flowDB[0],S.flowDB[1]));

      // Branch connectors (curved)
      if(!cacheHit && S.flowGen){
        const cp={x:(n.db.x+n.generate.x)/2, y:n.db.y+30};
        drawCurvedTube(ctx, n.db, n.generate, cp, [239,68,68], sp(t,S.flowGen[0],S.flowGen[1]));
        if(t>=S.flowGen[0]) drawTubeParticles(ctx,n.db,n.generate,t,[239,68,68],sp(t,S.flowGen[0],S.flowGen[1]));
      }
      if(cacheHit && S.flowConf){
        const cp={x:(n.db.x+n.confidence.x)/2, y:n.db.y+30};
        drawCurvedTube(ctx, n.db, n.confidence, cp, [16,185,129], sp(t,S.flowConf[0],S.flowConf[1]));
        if(t>=S.flowConf[0]) drawTubeParticles(ctx,n.db,n.confidence,t,[16,185,129],sp(t,S.flowConf[0],S.flowConf[1]));
      }

      // ── Spheres (drawn front to back, simple top-to-bottom order) ────────
      // Input platform tokens
      const platA=ease(sp(t,0,0.9));
      if(platA>0){
        ['tok','en','s'].forEach((tok,i)=>{
          const off=(i-1)*0.28;
          const orbitAngle=t*0.6+i*(Math.PI*2/3);
          const ox=Math.cos(orbitAngle)*0.22;
          const oz=Math.sin(orbitAngle)*0.1;
          const fadeOut=1-sp(t,S.flowRedact[0],S.flowRedact[0]+0.55);
          drawToken3D(ctx,ox,-2.2+oz,oz,W,H,tok,'normal',platA*fadeOut);
        });
        // Platform disc
        const ip=project(0,-2.35,0,W,H);
        ctx.save(); ctx.globalAlpha=platA*(1-sp(t,S.flowRedact[0],S.flowRedact[0]+0.5));
        const pg=ctx.createRadialGradient(ip.x,ip.y+ip.sc*8,3,ip.x,ip.y+ip.sc*8,ip.sc*55);
        pg.addColorStop(0,'rgba(99,102,241,0.25)'); pg.addColorStop(1,'rgba(99,102,241,0)');
        ctx.fillStyle=pg;
        ctx.beginPath(); ctx.ellipse(ip.x,ip.y+ip.sc*10,ip.sc*55,ip.sc*14,0,0,Math.PI*2); ctx.fill();
        ctx.strokeStyle='rgba(99,102,241,0.28)'; ctx.lineWidth=1;
        ctx.beginPath(); ctx.ellipse(ip.x,ip.y+ip.sc*10,ip.sc*55,ip.sc*14,0,0,Math.PI*2); ctx.stroke();
        ctx.restore();
      }

      // Redaction sphere
      const redA=ease(sp(t,1.0,1.8));
      if(redA>0){
        drawSphere(ctx, n.redact.x, n.redact.y, R*n.redact.sc, COL.neutral, 'Redaction', 'privacy filter', redA);
        if(t>=S.flowRedact[0]+0.35 && t<S.flowTok[0]+0.25){
          const ip2=sp(t,S.flowRedact[0]+0.45,S.flowRedact[1]);
          const fa=ip2*(1-sp(t,S.flowTok[0],S.flowTok[0]+0.3))*redA;
          drawToken3D(ctx,-0.14,-1.3+0.14,0,W,H,'tok','normal',fa);
          drawToken3D(ctx, 0.0, -1.3+0.14,0,W,H,'████','redacted',fa);
          drawToken3D(ctx, 0.14,-1.3+0.14,0,W,H,'██','redacted',fa);
        }
      }

      // Tokenizer sphere
      const tokA=ease(sp(t,2.2,3.0));
      if(tokA>0){
        drawSphere(ctx, n.tokenize.x, n.tokenize.y, R*n.tokenize.sc, COL.neutral, 'Tokenizer', 'text → ids', tokA);
        if(t>=S.flowTok[0]+0.4 && t<S.flowEmbed[0]+0.25){
          const ip2=sp(t,S.flowTok[0]+0.5,S.flowTok[1]);
          const fa=ip2*(1-sp(t,S.flowEmbed[0],S.flowEmbed[0]+0.3))*tokA;
          ['1024','5821','392'].forEach((id,i)=>drawToken3D(ctx,(i-1)*0.17,-0.4+0.14,0,W,H,id,'id',fa));
        }
      }

      // Embedding sphere
      const embA=ease(sp(t,3.4,4.3));
      if(embA>0){
        drawSphere(ctx, n.embed.x, n.embed.y, R*n.embed.sc, COL.neutral, 'Embedding', 'ids → vector', embA);
        if(t>=S.flowEmbed[0]+0.4 && t<S.flowDB[0]+0.25){
          const ip2=sp(t,S.flowEmbed[0]+0.5,S.flowEmbed[1]);
          const fa=ip2*(1-sp(t,S.flowDB[0],S.flowDB[0]+0.3))*embA;
          ['0.23','-0.87','0.41'].forEach((v,i)=>drawToken3D(ctx,(i-1)*0.19,0.5+0.14,0,W,H,v,'vector',fa));
        }
      }

      // DB sphere
      const dbA=ease(sp(t,4.6,5.5));
      if(dbA>0){
        const dbLabel=t>=S.dbResult[0]?(cacheHit?'Cache Hit ✓':'Cache Miss ✗'):'Cache DB';
        const dbSub=t>=S.dbSearch[0]&&t<S.dbResult[0]?'searching…':'cosine similarity';
        drawSphere(ctx, n.db.x, n.db.y, R*n.db.sc, dbCol, dbLabel, dbSub, dbA);
        if(t>=S.dbSearch[0]&&t<S.dbResult[0]) drawSearchRing(ctx,n.db.x,n.db.y,R*n.db.sc,t);
      }

      // Generation sphere (cache miss)
      if(!cacheHit && S.flowGen && t>=S.flowGen[0]){
        const ga=ease(sp(t,S.flowGen[0],S.flowGen[0]+0.9));
        drawSphere(ctx,n.generate.x,n.generate.y,R*n.generate.sc,COL.generate,'Generating','answer',ga);
        if(t>=S.generating[0]) drawGenerating(ctx,n.generate.x,n.generate.y,R*n.generate.sc,t-S.generating[0]);
      }

      // Confidence sphere (cache hit)
      if(cacheHit && S.flowConf && t>=S.flowConf[0]){
        const ca=ease(sp(t,S.flowConf[0],S.flowConf[0]+0.9));
        drawSphere(ctx,n.confidence.x,n.confidence.y,R*n.confidence.sc,COL.confidence,'Confidence','estimating…',ca);
      }

      // Perplexity line + cat
      if(cacheHit && S.tokenLine && t>=S.tokenLine[0]) drawPerplexitySection(ctx,W,H,t,S);

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
