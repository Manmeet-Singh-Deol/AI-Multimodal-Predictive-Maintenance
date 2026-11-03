#!/bin/bash
set -e

echo "Starting FastAPI Backend on port 8000..."
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

echo "Waiting for backend initialization..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "Backend is healthy on port 8000!"
        break
    fi
    sleep 1
done

echo "Starting Streamlit Frontend on port 7860..."
exec python -m streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
