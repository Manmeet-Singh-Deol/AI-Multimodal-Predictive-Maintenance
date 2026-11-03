---
title: AI Multimodal Predictive Maintenance
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI-Powered Multimodal Industrial Machine Troubleshooting & Predictive Maintenance Assistant

An intelligent system that combines **predictive maintenance ML models** with a **multimodal AI troubleshooting chatbot** to help technicians diagnose and prevent industrial machine failures.

## Features

- **Real-Time Monitoring Dashboard** — Live sensor data visualization with anomaly alerts
- **AI Troubleshooting Chatbot** — Ask questions, upload images, or use voice to troubleshoot
- **Predictive Analytics** — Fault classification, anomaly detection, and Remaining Useful Life (RUL) estimation
- **Knowledge Base** — RAG-powered search over machine manuals and maintenance documentation
- **Multimodal Input** — Supports text, images (defect detection), audio (machine sounds), and voice commands

## Tech Stack

| Component | Technology |
|:---|:---|
| ML Framework | PyTorch, scikit-learn |
| LLM | Google Gemini API |
| RAG | LangChain + ChromaDB |
| Backend | FastAPI |
| Frontend | Streamlit + Plotly |
| Audio | librosa, OpenAI Whisper |
| Database | SQLite, ChromaDB |

## Quick Start

### 1. Clone & Setup
```bash
cd "Major Project"
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Download Datasets
```bash
python -m src.data_pipeline.data_loader --download
```

### 4. Train Models
```bash
python -m src.ml_models.trainer --train-all
```

### 5. Index Knowledge Base
```bash
python -m src.rag_engine.document_processor --index data/manuals/
```

### 6. Start the Application
```bash
# Terminal 1: Start Backend
uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Start Frontend
streamlit run frontend/app.py
```

## Project Structure

```
Major Project/
├── config/             # Configuration settings
├── src/
│   ├── data_pipeline/  # Data ingestion & processing
│   ├── ml_models/      # Predictive maintenance ML models
│   ├── rag_engine/     # RAG pipeline for manual search
│   ├── multimodal/     # Image, audio, voice processing
│   ├── chatbot/        # AI troubleshooting agent
│   └── api/            # FastAPI backend
├── frontend/           # Streamlit dashboard
├── data/               # Datasets & manuals
├── models/             # Saved trained models
├── notebooks/          # Jupyter exploration notebooks
└── tests/              # Unit tests
```

## Datasets

- **NASA C-MAPSS** — Turbofan engine degradation for RUL estimation
- **Synthetic Generator** — Custom multi-sensor data with fault injection

## Architecture

The system uses a layered architecture:
1. **Data Layer** — Sensor data ingestion, feature engineering
2. **ML Layer** — Anomaly detection, fault classification, RUL prediction
3. **RAG Layer** — Document retrieval from maintenance manuals
4. **Agent Layer** — LangChain orchestrator with multimodal tools
5. **API Layer** — FastAPI REST endpoints
6. **UI Layer** — Streamlit multi-page dashboard

## License

This project is developed for academic purposes as a college major project.
