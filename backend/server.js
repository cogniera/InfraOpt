import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import cors from 'cors';
import morgan from 'morgan';
import 'dotenv/config';

const app = express();
const PORT = process.env.PORT || 3000;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:4173', 'http://localhost'] }));
app.use(morgan('dev'));
app.use(express.json());

// Health check for Node layer
app.get('/node-health', (req, res) => {
  res.json({ status: 'ok', service: 'inferOpt Node Proxy', fastapi: FASTAPI_URL });
});

// Proxy all /api and /health routes to FastAPI
app.use(
  ['/api', '/health'],
  createProxyMiddleware({
    target: FASTAPI_URL,
    changeOrigin: true,
    on: {
      error: (err, req, res) => {
        console.error('[Proxy Error]', err.message);
        res.status(502).json({ error: 'FastAPI backend unavailable', detail: err.message });
      },
    },
  })
);

app.listen(PORT, () => {
  console.log(`\n  inferOpt Node proxy → http://localhost:${PORT}`);
  console.log(`  Forwarding /api/* → ${FASTAPI_URL}\n`);
});
