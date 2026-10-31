# =========================================================================
# Dockerfile for AI Predictive Maintenance Assistant (FastAPI Backend)
# Optimized for 100% Free Hosting on Hugging Face Spaces / Render
# =========================================================================

FROM python:3.11-slim

WORKDIR /app

# Install system audio and build dependencies for librosa and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     libsndfile1     curl     && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY . .

# Create persistent storage directories
RUN mkdir -p data/raw data/processed data/manuals data/chroma_db models

# Port 7860 is the default port for Hugging Face Spaces (or use $PORT)
EXPOSE 7860

# Launch Uvicorn server on port 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
