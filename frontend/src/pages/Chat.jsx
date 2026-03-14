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

        {/* Meta row */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, marginTop: 4,
          justifyContent: isUser ? 'flex-end' : 'flex-start',
        }}>
          <span style={{ fontSize: 11, color: '#334155' }}>{timeAgo(msg.ts)}</span>
          {!isUser && msg.model && (
            <span style={{ fontSize: 11, color: '#334155', fontFamily: 'JetBrains Mono, monospace' }}>
              · {msg.model}
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
        Node.js · FastAPI · Vite
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
  const pendingMsg   = useRef(null);   // buffer API response during animation
  const pendingChatId = useRef(null);
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

  // Called when the pipeline animation finishes — apply buffered response
  const onPipelineComplete = useCallback(() => {
    setShowPipeline(false);
    setLoading(false);
    if (pendingMsg.current && pendingChatId.current) {
      const { msg, chatId } = { msg: pendingMsg.current, chatId: pendingChatId.current };
      pendingMsg.current    = null;
      pendingChatId.current = null;
      setChats(prev => prev.map(c =>
        c.id === chatId
          ? { ...c, messages: [...c.messages, msg], updatedAt: Date.now() }
          : c
      ));
    }
  }, []);

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

    // Start animation (random cache-hit demo 30% of the time)
    const cacheHit = Math.random() < 0.3;
    setPipelineCacheHit(cacheHit);
    setShowPipeline(true);
    setLoading(true);
    pendingChatId.current = chatId;

    // Fire API call in parallel — buffer result for when animation ends
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: [...existingMessages, userMsg].map(m => ({ role: m.role, content: m.content })),
        }),
      });
      const data = await res.json();
      pendingMsg.current = {
        id: uid(), role: 'assistant',
        content: data.content ?? data.error ?? 'No response.',
        model: data.model,
        ts: Date.now(),
      };
    } catch (err) {
      pendingMsg.current = {
        id: uid(), role: 'assistant',
        content: `Error: ${err.message}`,
        ts: Date.now(),
      };
    }
    // Response is now buffered — onPipelineComplete will apply it
  }, [activeChatId, chats, loading, model]);

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
