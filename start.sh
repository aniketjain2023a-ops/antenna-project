#!/usr/bin/env bash
# Build the React website and start the FastAPI server, which serves both the API and the site.
# Run from the project folder with the Python virtual environment activated.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d frontend/node_modules ]; then
  (cd frontend && npm install)
fi
(cd frontend && npm run build)
echo "MetaOpt-AI is starting at http://127.0.0.1:8000"
python3 -m backend
