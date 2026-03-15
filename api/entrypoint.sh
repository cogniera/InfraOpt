#!/bin/bash
set -e

echo "=== TemplateCache API Startup ==="

# 1. Wait for Redis to accept connections
echo "[1/4] Waiting for Redis at ${REDIS_HOST:-localhost}:${REDIS_PORT:-6379}..."
python -c "
import redis, os, time
r = redis.Redis(host=os.getenv('REDIS_HOST','localhost'), port=int(os.getenv('REDIS_PORT','6379')))
for i in range(30):
    try:
        r.ping()
        print('  → Redis is ready')
        break
    except redis.ConnectionError:
        time.sleep(1)
else:
    print('  ✗ Redis not reachable after 30s')
    exit(1)
"

# 2. Flush old cache
echo "[2/4] Flushing Redis cache..."
python -c "
import redis, os
r = redis.Redis(host=os.getenv('REDIS_HOST','localhost'), port=int(os.getenv('REDIS_PORT','6379')), db=int(os.getenv('REDIS_DB','0')))
r.flushdb()
print('  → Redis flushed')
"

# 3. Pre-download embedding model if using local embeddings
echo "[3/4] Seeding Redis cache (this may take a few minutes on first run)..."
python seed_cache.py
echo "  → Cache seeded successfully"

# 4. Start the API server
echo "[4/4] Starting uvicorn..."
exec python -m uvicorn templatecache.demo.app:app --host 0.0.0.0 --port 8000
