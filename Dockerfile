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
RUN mkdir -p data/raw data/processed data/manuals data/chroma_db models && \
    chmod -R 777 data models

# Make launcher executable
RUN chmod +x start.sh

# Port 7860 is the default port for Hugging Face Spaces (or use $PORT)
EXPOSE 7860

# Launch both FastAPI backend (port 8000) and Streamlit frontend (port 7860)
CMD ["./start.sh"]
