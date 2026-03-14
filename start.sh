#!/usr/bin/env bash
set -e

# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Loaded .env"
fi

# Find the right Python — prefer the one that has uvicorn installed
if /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -c "import uvicorn" 2>/dev/null; then
    PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
elif python3 -c "import uvicorn" 2>/dev/null; then
    PYTHON="python3"
else
    echo "✗ uvicorn not found. Install it with: pip3 install uvicorn"
    exit 1
fi
echo "✓ Using Python: $PYTHON"

# Check for API key (skip if using local LLM)
if [ "$USE_LOCAL_LLM" = "true" ]; then
    echo "✓ USE_LOCAL_LLM=true — using Ollama (no API key needed)"
else
    if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your-gemini-api-key-here" ]; then
        echo "✗ Please set your GOOGLE_API_KEY in the .env file"
        exit 1
    fi
    echo "✓ GOOGLE_API_KEY is set"
fi

if [ "$USE_LOCAL_EMBEDDINGS" = "true" ]; then
    echo "✓ USE_LOCAL_EMBEDDINGS=true — using local sentence-transformers"
fi

echo "Starting TemplateCache on http://localhost:8000 ..."
$PYTHON -m uvicorn templatecache.demo.app:app --host 0.0.0.0 --port 8000

