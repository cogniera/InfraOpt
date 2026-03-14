import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Sidebar from './components/Sidebar.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Models from './pages/Models.jsx';
import Inference from './pages/Inference.jsx';
import Optimize from './pages/Optimize.jsx';
import Landing from './pages/Landing.jsx';

const PAGES = { dashboard: Dashboard, models: Models, inference: Inference, optimize: Optimize };

export default function App() {
  const [page, setPage] = useState('landing');

  if (page === 'landing') {
    return (
      <AnimatePresence mode="wait">
        <motion.div
          key="landing"
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4 }}
        >
          <Landing onEnter={() => setPage('dashboard')} />
        </motion.div>
      </AnimatePresence>
    );
  }

  const Page = PAGES[page];

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key="app"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4 }}
        style={{ display: 'flex', minHeight: '100vh', position: 'relative' }}
      >
        <div className="bg-gradient-mesh" />
        <Sidebar active={page} setPage={setPage} />
        <main style={{ flex: 1, padding: '32px', position: 'relative', zIndex: 1, overflowY: 'auto' }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={page}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
            >
              <Page />
            </motion.div>
          </AnimatePresence>
        </main>
      </motion.div>
    </AnimatePresence>
  );
}
