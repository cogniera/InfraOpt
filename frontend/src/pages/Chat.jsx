import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Plus, MessageSquare, Trash2, Send, Zap,
  ChevronDown, Sparkles, Copy, Check,
} from 'lucide-react';
import PipelineAnimation from '../components/PipelineAnimation.jsx';

// ── Helpers ───────────────────────────────────────────────────────────────────
function uid() { return Math.random().toString(36).slice(2, 10); }

function timeAgo(ts) {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60)  return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function loadChats() {
  try { return JSON.parse(localStorage.getItem('inferopt-chats')) || []; }
  catch { return []; }
}
function saveChats(c) {
  try { localStorage.setItem('inferopt-chats', JSON.stringify(c)); } catch {}
}

const MODELS = [
  { id: 'llama-3-8b',  label: 'Llama 3 8B'  },
  { id: 'mistral-7b',  label: 'Mistral 7B'  },
  { id: 'phi-3-mini',  label: 'Phi-3 Mini'  },
  { id: 'qwen-14b',    label: 'Qwen 14B'    },
];

const SUGGESTIONS = [
  'Explain transformer attention in simple terms',
  'What is the difference between GGUF and GPTQ quantization?',
  'How do I reduce LLM inference latency?',
  'Compare Mistral 7B and Llama 3 8B for coding tasks',
];

// ── Stitch visualization panel ────────────────────────────────────────────────
function SlotTable({ fills, sources }) {
  const names = Object.keys(fills || {});
  if (!names.length) return null;
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginTop: 6 }}>
      <thead>
        <tr>
          {['Slot', 'Value', 'Source'].map(h => (
            <th key={h} style={{ textAlign: 'left', color: '#475569', fontWeight: 500, padding: '3px 8px', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {names.map(name => {
          const isSupp = name.startsWith('supplement_');
          const src = sources?.[name];
          return (
            <tr key={name}>
              <td style={{ padding: '3px 8px', fontFamily: 'JetBrains Mono, monospace', color: isSupp ? '#fbbf24' : '#60a5fa', fontSize: 11 }}>[{name}]</td>
              <td style={{ padding: '3px 8px', color: '#e2e8f0', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{fills[name]}</td>
              <td style={{ padding: '3px 8px', fontSize: 11, color: src === 'cache' ? '#4ade80' : '#fbbf24' }}>
                {src === 'cache' ? '✓ CACHE' : '⚡ LLM'}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function StitchPanel({ stitch, intentId }) {
  const [open, setOpen] = useState(false);
  if (!stitch) return null;

  const sectionStyle = { marginBottom: 12 };
  const labelStyle = { fontSize: 10, fontWeight: 600, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 };
  const codeBox = { background: 'rgba(0,0,0,0.35)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 6, padding: '8px 12px', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, lineHeight: 1.6, whiteSpace: 'pre-wrap', overflowX: 'auto' };

  const slotNames = Object.keys(stitch.slot_fills || {});
  const regularSlots = slotNames.filter(n => !n.startsWith('supplement_'));
  const suppSlots = slotNames.filter(n => n.startsWith('supplement_'));
  const gaps = stitch.gaps_detected || [];

  // Highlight [slot_name] markers in skeleton text
  function highlightSkeleton(text, filled = false) {
    if (!text) return null;
    const parts = text.split(/(\[[^\]]+\])/g);
    return parts.map((part, i) => {
      const match = part.match(/^\[([^\]]+)\]$/);
      if (!match) return <span key={i} style={{ color: '#9ca3af' }}>{part}</span>;
      const name = match[1];
      const isSupp = name.startsWith('supplement_');
      const fillVal = stitch.slot_fills?.[name];
      if (filled && fillVal) {
        return <span key={i} style={{ background: isSupp ? 'rgba(146,64,14,0.5)' : 'rgba(20,83,45,0.6)', color: isSupp ? '#fbbf24' : '#4ade80', padding: '1px 5px', borderRadius: 3 }}>{fillVal}</span>;
      }
      return <span key={i} style={{ background: isSupp ? 'rgba(146,64,14,0.3)' : 'rgba(30,58,95,0.6)', color: isSupp ? '#fbbf24' : '#60a5fa', padding: '1px 5px', borderRadius: 3, fontWeight: 600 }}>[{name}]</span>;
    });
  }

  return (
    <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.07)', paddingTop: 8 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#60a5fa', fontSize: 12, padding: 0, textDecoration: 'underline' }}
      >
        {open ? '▼ Hide stitching' : '▶ Show stitching'}
      </button>

      {open && (
        <div style={{ marginTop: 10, fontSize: 13 }}>

          {/* Multi-query breakdown */}
          {stitch.multi_query && stitch.sub_results?.length > 0 && (
            <div style={sectionStyle}>
              <div style={labelStyle}>🔀 Multi-Query Routing</div>
              {stitch.sub_results.map((sr, i) => (
                <div key={i} style={{ padding: '3px 0', fontSize: 12 }}>
                  {sr.cache_hit ? '🟢' : '🔴'} <span style={{ color: '#e2e8f0' }}>{sr.sub_query}</span>
                  {' — '}
                  <span style={{ color: sr.cache_hit ? '#4ade80' : '#fbbf24', fontWeight: 600 }}>
                    {sr.cache_hit ? 'CACHE HIT' : 'LLM GENERATED'}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Intent ID */}
          {intentId && (
            <div style={sectionStyle}>
              <div style={labelStyle}>🎯 Matched Intent</div>
              <code style={{ color: '#60a5fa', fontSize: 11 }}>{intentId}</code>
            </div>
          )}

          {/* No-slot case */}
          {!stitch.has_slots && gaps.length === 0 && (
            <div style={sectionStyle}>
              <div style={labelStyle}>📋 Template (no slots)</div>
              <div style={codeBox}>{highlightSkeleton(stitch.skeleton)}</div>
              <div style={{ marginTop: 6, fontSize: 11, color: '#94a3b8', background: 'rgba(30,41,59,0.5)', padding: '3px 10px', borderRadius: 4, display: 'inline-block' }}>
                ✓ Entire response served from cache — no LLM calls needed
              </div>
            </div>
          )}

          {/* Gaps */}
          {gaps.length > 0 && (
            <div style={sectionStyle}>
              <div style={labelStyle}>🔍 Gaps Detected</div>
              <div style={{ fontSize: 12, color: '#fbbf24', marginBottom: 4 }}>Query aspects not covered by cache:</div>
              <ul style={{ margin: '0 0 0 16px', fontSize: 12, color: '#fbbf24' }}>
                {gaps.map((g, i) => <li key={i}>"{g}"</li>)}
              </ul>
            </div>
          )}

          {/* Skeleton */}
          {stitch.has_slots && (
            <div style={sectionStyle}>
              <div style={labelStyle}>📋 Cached Template</div>
              <div style={codeBox}>{highlightSkeleton(stitch.skeleton, false)}</div>
            </div>
          )}

          {/* Supplement slots */}
          {suppSlots.length > 0 && (
            <div style={sectionStyle}>
              <div style={labelStyle}>➕ Supplement Slots (LLM-generated for gaps)</div>
              <div style={codeBox}>
                {suppSlots.map(n => (
                  <span key={n} style={{ background: 'rgba(146,64,14,0.4)', color: '#fbbf24', padding: '1px 5px', borderRadius: 3, marginRight: 6 }}>[{n}]</span>
                ))}
              </div>
            </div>
          )}

          {/* Slot fills table */}
          {slotNames.length > 0 && (
            <div style={sectionStyle}>
              <div style={labelStyle}>🔧 Slot Fills</div>
              <SlotTable fills={stitch.slot_fills} sources={stitch.slot_sources} />
            </div>
          )}

          {/* Stitch arrow */}
          {slotNames.length > 0 && (
            <div style={{ textAlign: 'center', color: '#334155', fontSize: 16, padding: '4px 0' }}>↓ stitch ↓</div>
          )}

          {/* Stitched result */}
          {slotNames.length > 0 && (
            <div style={sectionStyle}>
              <div style={labelStyle}>✅ Stitched Result</div>
              <div style={{ ...codeBox, border: '1px solid rgba(20,83,45,0.5)', background: 'rgba(4,26,10,0.6)' }}>
                {highlightSkeleton(stitch.skeleton, true)}
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  );
}

// ── Copy button for messages ──────────────────────────────────────────────────
function CopyBtn({ text }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };
  return (
    <button onClick={copy} style={{
      background: 'none', border: 'none', cursor: 'pointer',
      color: '#475569', padding: 4, borderRadius: 4,
      display: 'flex', alignItems: 'center',
      transition: 'color 0.15s',
    }}
    onMouseEnter={e => e.currentTarget.style.color = '#94a3b8'}
    onMouseLeave={e => e.currentTarget.style.color = '#475569'}
    >
      {copied ? <Check size={13} color="#10b981" /> : <Copy size={13} />}
    </button>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────
function Message({ msg }) {
  const isUser = msg.role === 'user';
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22 }}
      style={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        marginBottom: 20,
        gap: 12,
        alignItems: 'flex-start',
      }}
    >
      {/* Assistant avatar */}
      {!isUser && (
        <div style={{
          width: 30, height: 30, borderRadius: 8, flexShrink: 0, marginTop: 2,
          background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 0 14px rgba(99,102,241,0.35)',
        }}>
          <Zap size={13} color="white" />
        </div>
      )}

      <div style={{ maxWidth: '72%', minWidth: 0 }}>
        {/* Bubble */}
        <div style={{
          padding: '12px 16px',
          borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
          background: isUser
            ? 'linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.2))'
            : 'rgba(255,255,255,0.05)',
          border: isUser
            ? '1px solid rgba(99,102,241,0.3)'
            : '1px solid rgba(255,255,255,0.07)',
          color: '#e2e8f0',
          fontSize: 14,
          lineHeight: 1.7,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}>
          {msg.content}
        </div>

        {/* Stitch panel — only on assistant messages with stitch data */}
        {!isUser && msg.stitch && (
          <StitchPanel stitch={msg.stitch} intentId={msg.intentId} />
        )}

        {/* Meta row */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, marginTop: 4, flexWrap: 'wrap',
          justifyContent: isUser ? 'flex-end' : 'flex-start',
        }}>
          <span style={{ fontSize: 11, color: '#334155' }}>{timeAgo(msg.ts)}</span>
          {!isUser && msg.cacheHit != null && (
            <span style={{
              fontSize: 10, fontFamily: 'JetBrains Mono, monospace', fontWeight: 600,
              padding: '1px 6px', borderRadius: 4,
              background: msg.cacheHit ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.10)',
              color: msg.cacheHit ? '#10b981' : '#ef4444',
              border: `1px solid ${msg.cacheHit ? 'rgba(16,185,129,0.25)' : 'rgba(239,68,68,0.2)'}`,
            }}>
              {msg.cacheHit ? '✓ cache hit' : '✗ cache miss'}
            </span>
          )}
          {!isUser && msg.savingsRatio != null && (
            <span style={{ fontSize: 10, color: '#6366f1', fontFamily: 'JetBrains Mono, monospace' }}>
              {Math.round(msg.savingsRatio * 100)}% saved
            </span>
          )}
          {!isUser && msg.slotsFromCache > 0 && (
            <span style={{ fontSize: 10, color: '#475569', fontFamily: 'JetBrains Mono, monospace' }}>
              {msg.slotsFromCache}↑cache {msg.slotsFromInference}↑llm
            </span>
          )}
          {!isUser && <CopyBtn text={msg.content} />}
        </div>
      </div>

      {/* User avatar */}
      {isUser && (
        <div style={{
          width: 30, height: 30, borderRadius: 8, flexShrink: 0, marginTop: 2,
          background: 'rgba(255,255,255,0.07)',
          border: '1px solid rgba(255,255,255,0.1)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, color: '#64748b', fontWeight: 600,
        }}>
          U
        </div>
      )}
    </motion.div>
  );
}

// ── Left sidebar ──────────────────────────────────────────────────────────────
function ChatSidebar({ chats, activeChatId, onSelect, onNew, onDelete }) {
  return (
    <aside style={{
      width: 268,
      minWidth: 268,
      height: '100vh',
      background: 'rgba(5, 8, 16, 0.97)',
      borderRight: '1px solid rgba(255,255,255,0.06)',
      display: 'flex',
      flexDirection: 'column',
      position: 'sticky',
      top: 0,
      overflow: 'hidden',
    }}>
      {/* Logo */}
      <div style={{
        padding: '20px 16px 14px',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8,
            background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 16px rgba(99,102,241,0.4)',
          }}>
            <Zap size={14} color="white" />
          </div>
          <span style={{ fontWeight: 700, fontSize: 16, color: '#f1f5f9', letterSpacing: '-0.3px' }}>
            infer<span style={{ color: '#818cf8' }}>Opt</span>
          </span>
        </div>

        {/* New chat */}
        <motion.button
          whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.97 }}
          onClick={onNew}
          style={{
            width: '100%', display: 'flex', alignItems: 'center', gap: 8,
            background: 'linear-gradient(135deg, rgba(99,102,241,0.18), rgba(139,92,246,0.12))',
            border: '1px solid rgba(99,102,241,0.25)',
            borderRadius: 10, padding: '9px 14px',
            color: '#818cf8', fontSize: 13, fontWeight: 500, cursor: 'pointer',
          }}
        >
          <Plus size={15} />
          New Chat
        </motion.button>
      </div>

      {/* Chat list */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 8px' }}>
        {chats.length === 0 && (
          <div style={{ padding: '32px 16px', textAlign: 'center', color: '#334155', fontSize: 13 }}>
            No conversations yet
          </div>
        )}
        {[...chats].reverse().map(chat => {
          const active = chat.id === activeChatId;
          return (
            <motion.div
              key={chat.id}
              whileHover={{ x: 2 }}
              onClick={() => onSelect(chat.id)}
              style={{
                display: 'flex', alignItems: 'center', gap: 9,
                padding: '9px 10px', borderRadius: 9, marginBottom: 2,
                cursor: 'pointer',
                background: active ? 'rgba(99,102,241,0.14)' : 'transparent',
                border: active ? '1px solid rgba(99,102,241,0.2)' : '1px solid transparent',
                transition: 'all 0.15s',
              }}
            >
              <MessageSquare size={13} color={active ? '#818cf8' : '#334155'} style={{ flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: 13, color: active ? '#e2e8f0' : '#64748b',
                  fontWeight: active ? 500 : 400,
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {chat.title}
                </div>
                <div style={{ fontSize: 11, color: '#1e293b', marginTop: 1 }}>
                  {timeAgo(chat.updatedAt)} · {chat.messages.length} msg{chat.messages.length !== 1 ? 's' : ''}
                </div>
              </div>
              <button
                onClick={e => { e.stopPropagation(); onDelete(chat.id); }}
                style={{
                  background: 'none', border: 'none', cursor: 'pointer',
                  color: '#1e293b', padding: 3, borderRadius: 4, flexShrink: 0,
                  display: 'flex', alignItems: 'center',
                  transition: 'color 0.15s',
                }}
                onMouseEnter={e => e.currentTarget.style.color = '#ef4444'}
                onMouseLeave={e => e.currentTarget.style.color = '#1e293b'}
              >
                <Trash2 size={12} />
              </button>
            </motion.div>
          );
        })}
      </div>

      {/* Footer */}
      <div style={{
        padding: '12px 16px',
        borderTop: '1px solid rgba(255,255,255,0.04)',
        fontSize: 11, color: '#1e293b', fontFamily: 'JetBrains Mono, monospace',
      }}>
        TemplateCache · Redis · Ollama
      </div>
    </aside>
  );
}

// ── Model selector ─────────────────────────────────────────────────────────────
function ModelSelector({ value, onChange }) {
  const [open, setOpen] = useState(false);
  const current = MODELS.find(m => m.id === value) || MODELS[0];
  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          display: 'flex', alignItems: 'center', gap: 6,
          background: 'rgba(255,255,255,0.05)',
          border: '1px solid rgba(255,255,255,0.1)',
          borderRadius: 8, padding: '5px 10px',
          color: '#94a3b8', fontSize: 12, cursor: 'pointer',
          fontFamily: 'JetBrains Mono, monospace',
        }}
      >
        {current.label}
        <ChevronDown size={12} style={{ transition: 'transform 0.2s', transform: open ? 'rotate(180deg)' : 'none' }} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.96 }}
            transition={{ duration: 0.12 }}
            style={{
              position: 'absolute', bottom: '110%', left: 0,
              background: '#0d1422',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: 10, overflow: 'hidden', minWidth: 160,
              boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
              zIndex: 50,
            }}
          >
            {MODELS.map(m => (
              <div
                key={m.id}
                onClick={() => { onChange(m.id); setOpen(false); }}
                style={{
                  padding: '9px 14px', fontSize: 12, cursor: 'pointer',
                  color: m.id === value ? '#818cf8' : '#64748b',
                  background: m.id === value ? 'rgba(99,102,241,0.1)' : 'transparent',
                  fontFamily: 'JetBrains Mono, monospace',
                  transition: 'all 0.1s',
                }}
                onMouseEnter={e => { if (m.id !== value) e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; }}
                onMouseLeave={e => { if (m.id !== value) e.currentTarget.style.background = 'transparent'; }}
              >
                {m.label}
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────
export default function Chat() {
  const [chats, setChats]             = useState(loadChats);
  const [activeChatId, setActiveChatId] = useState(() => loadChats()[0]?.id ?? null);
  const [input, setInput]             = useState('');
  const [loading, setLoading]         = useState(false);
  const [model, setModel]             = useState('llama-3-8b');
  const [showPipeline, setShowPipeline] = useState(false);
  const [pipelineCacheHit, setPipelineCacheHit] = useState(false);
  const pendingMsg    = useRef(null);   // buffer API response during animation
  const pendingChatId = useRef(null);
  const animDone      = useRef(false);  // true once pipeline animation completes
  const messagesEndRef = useRef(null);
  const textareaRef    = useRef(null);

  const activeChat = chats.find(c => c.id === activeChatId) ?? null;

  // Persist
  useEffect(() => saveChats(chats), [chats]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeChat?.messages, loading]);

  // Auto-resize textarea
  const resizeTextarea = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  };

  const newChat = useCallback(() => {
    const chat = { id: uid(), title: 'New conversation', messages: [], createdAt: Date.now(), updatedAt: Date.now() };
    setChats(prev => [...prev, chat]);
    setActiveChatId(chat.id);
  }, []);

  const deleteChat = useCallback((id) => {
    setChats(prev => prev.filter(c => c.id !== id));
    setActiveChatId(prev => prev === id ? null : prev);
  }, []);

  // Apply buffered response to chat — safe to call from either animation or API resolver
  const applyPendingMsg = useCallback(() => {
    if (!pendingMsg.current || !pendingChatId.current) return;
    const msg    = pendingMsg.current;
    const chatId = pendingChatId.current;
    pendingMsg.current    = null;
    pendingChatId.current = null;
    animDone.current      = false;
    setShowPipeline(false);
    setLoading(false);
    setChats(prev => prev.map(c =>
      c.id === chatId
        ? { ...c, messages: [...c.messages, msg], updatedAt: Date.now() }
        : c
    ));
  }, []);

  // Called when the pipeline animation finishes
  const onPipelineComplete = useCallback(() => {
    animDone.current = true;
    if (pendingMsg.current) {
      // API already resolved — apply immediately
      applyPendingMsg();
    }
    // else: API still in flight — applyPendingMsg will be called when it resolves
  }, [applyPendingMsg]);

  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || loading) return;

    // Ensure active chat exists
    let chatId = activeChatId;
    let existingMessages = [];
    if (!chatId || !chats.find(c => c.id === chatId)) {
      const chat = { id: uid(), title: text.slice(0, 40), messages: [], createdAt: Date.now(), updatedAt: Date.now() };
      setChats(prev => [...prev, chat]);
      setActiveChatId(chat.id);
      chatId = chat.id;
    } else {
      existingMessages = chats.find(c => c.id === chatId)?.messages ?? [];
    }

    const userMsg = { id: uid(), role: 'user', content: text, ts: Date.now() };

    setChats(prev => prev.map(c => {
      if (c.id !== chatId) return c;
      return {
        ...c,
        messages: [...c.messages, userMsg],
        title: c.messages.length === 0 ? text.slice(0, 45) : c.title,
        updatedAt: Date.now(),
      };
    }));

    // Start animation immediately
    animDone.current = false;
    setPipelineCacheHit(false);
    setShowPipeline(true);
    setLoading(true);
    pendingChatId.current = chatId;

    // Fire API call — buffer result; whichever finishes last (API or animation) applies it
    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: text }),
      });
      const data = await res.json();
      setPipelineCacheHit(!!data.cache_hit);
      pendingMsg.current = {
        id: uid(), role: 'assistant',
        content: data.response ?? data.error ?? 'No response.',
        cacheHit: !!data.cache_hit,
        savingsRatio: data.savings_ratio ?? null,
        slotsFromCache: data.slots_from_cache ?? 0,
        slotsFromInference: data.slots_from_inference ?? 0,
        intentId: data.intent_id ?? null,
        stitch: data.stitch ?? null,
        ts: Date.now(),
      };
    } catch (err) {
      pendingMsg.current = {
        id: uid(), role: 'assistant',
        content: `Error connecting to backend: ${err.message}`,
        ts: Date.now(),
      };
    }

    // If animation already finished by the time API resolved, apply now
    if (animDone.current) {
      applyPendingMsg();
    }
    // Otherwise onPipelineComplete will call applyPendingMsg when animation ends
  }, [activeChatId, chats, loading, applyPendingMsg]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim()) { sendMessage(input); setInput(''); }
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#080c14', overflow: 'hidden' }}>

      <PipelineAnimation
        visible={showPipeline}
        cacheHit={pipelineCacheHit}
        onComplete={onPipelineComplete}
      />

      {/* ── Left sidebar ── */}
      <ChatSidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelect={setActiveChatId}
        onNew={newChat}
        onDelete={deleteChat}
      />

      {/* ── Main chat area ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, position: 'relative' }}>

        {/* Top bar */}
        <div style={{
          padding: '14px 24px',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: 'rgba(8,12,20,0.8)', backdropFilter: 'blur(12px)',
          zIndex: 2,
        }}>
          <div style={{ fontSize: 14, fontWeight: 500, color: '#94a3b8' }}>
            {activeChat ? activeChat.title : 'inferOpt Chat'}
          </div>
          <ModelSelector value={model} onChange={setModel} />
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '32px 10%' }}>
          {!activeChat || activeChat.messages.length === 0 ? (
            /* ── Empty state ── */
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: '12vh' }}>
              <div style={{
                width: 52, height: 52, borderRadius: 14, marginBottom: 20,
                background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                boxShadow: '0 0 32px rgba(99,102,241,0.35)',
              }}>
                <Sparkles size={24} color="white" />
              </div>
              <h2 style={{ fontSize: 22, fontWeight: 700, color: '#e2e8f0', margin: '0 0 8px', letterSpacing: '-0.4px' }}>
                What would you like to explore?
              </h2>
              <p style={{ fontSize: 14, color: '#475569', marginBottom: 32 }}>
                Ask anything about inference, models, or optimization.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, width: '100%', maxWidth: 560 }}>
                {SUGGESTIONS.map(s => (
                  <motion.button
                    key={s}
                    whileHover={{ scale: 1.02, borderColor: 'rgba(99,102,241,0.4)' }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => { sendMessage(s); }}
                    style={{
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.08)',
                      borderRadius: 10, padding: '12px 14px',
                      color: '#64748b', fontSize: 13, cursor: 'pointer',
                      textAlign: 'left', lineHeight: 1.5,
                      transition: 'all 0.15s',
                    }}
                  >
                    {s}
                  </motion.button>
                ))}
              </div>
            </div>
          ) : (
            /* ── Message list ── */
            <>
              {activeChat.messages.map((msg) => (
                <Message key={msg.id} msg={msg} />
              ))}
            </>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* ── Input area ── */}
        <div style={{
          padding: '16px 10%',
          background: 'rgba(8,12,20,0.9)',
          borderTop: '1px solid rgba(255,255,255,0.05)',
        }}>
          <div style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 14,
            padding: '12px 14px 10px',
            display: 'flex', alignItems: 'flex-end', gap: 10,
            transition: 'border-color 0.2s',
          }}
          onFocusCapture={e => e.currentTarget.style.borderColor = 'rgba(99,102,241,0.45)'}
          onBlurCapture={e  => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => { setInput(e.target.value); resizeTextarea(); }}
              onKeyDown={handleKeyDown}
              placeholder="Message inferOpt…"
              rows={1}
              style={{
                flex: 1, background: 'none', border: 'none', outline: 'none', resize: 'none',
                color: '#e2e8f0', fontSize: 14, lineHeight: 1.6,
                fontFamily: 'Inter, system-ui, sans-serif',
                maxHeight: 200, overflowY: 'auto',
              }}
            />
            <motion.button
              whileHover={{ scale: 1.08 }} whileTap={{ scale: 0.92 }}
              onClick={() => { if (input.trim()) { sendMessage(input); setInput(''); textareaRef.current.style.height = 'auto'; } }}
              disabled={loading || !input.trim()}
              style={{
                width: 34, height: 34, borderRadius: 9, flexShrink: 0,
                background: input.trim() && !loading
                  ? 'linear-gradient(135deg, #6366f1, #8b5cf6)'
                  : 'rgba(255,255,255,0.06)',
                border: 'none', cursor: input.trim() && !loading ? 'pointer' : 'default',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                transition: 'background 0.2s',
                boxShadow: input.trim() && !loading ? '0 0 16px rgba(99,102,241,0.4)' : 'none',
              }}
            >
              <Send size={15} color={input.trim() && !loading ? 'white' : '#334155'} />
            </motion.button>
          </div>
          <div style={{ marginTop: 8, textAlign: 'center', fontSize: 11, color: '#1e293b' }}>
            Enter to send · Shift+Enter for new line
          </div>
        </div>
      </div>
    </div>
  );
}
