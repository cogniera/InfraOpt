import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Landing from './pages/Landing.jsx';
import Chat    from './pages/Chat.jsx';

export default function App() {
  const [view, setView] = useState('landing');

  return (
    <AnimatePresence mode="wait">
      {view === 'landing' ? (
        <motion.div
          key="landing"
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4 }}
        >
          <Landing onEnter={() => setView('chat')} />
        </motion.div>
      ) : (
        <motion.div
          key="chat"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.35 }}
        >
          <Chat />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
