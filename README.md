# Kalki Nexus: Modular Multi-Agent AI Operating System (v1.0.0)

Kalki Nexus is an enterprise-grade, LangGraph-orchestrated multi-agent AI framework with dynamic routing, tool execution, persistent layered memory, RAG knowledge retrieval, background job scheduling, and Hermes Discord integration.

---

## 🏗️ Architecture Overview

```
                      [ User Input / Discord Message ]
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │   Supervisor Agent  │
                           └──────────┬──────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 │          Parallel Specialist Fan-out   │
                 ▼                                         ▼
       ┌──────────────────┐                      ┌──────────────────┐
       │   Python Agent   │                      │   Research Agent │
       └────────┬─────────┘                      └────────┬─────────┘
                │                                         │
                │ (Delegate/Tools)                        │ (RAG Context)
                ▼                                         ▼
       ┌──────────────────┐                      ┌──────────────────┐
       │   Quant Agent    │                      │ RAG Retriever    │
       └────────┬─────────┘                      └────────┬─────────┘
                │                                         │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │  Result Aggregator  │
                          └──────────┬──────────┘
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
                ┌───────────────┐         ┌───────────────┐
                │  Final Answer │         │  Error Node   │
                └───────────────┘         └───────────────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ FallbackAgent │
                                          └───────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- Docker (optional, for Docker tools)
- SQLite3

### 2. Installation
```bash
git clone https://github.com/SadamAnjaneyulu/kalki-nexus.git
cd kalki-nexus
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your credentials:

```bash
cp .env.example .env
nano .env
```

#### Example `.env` Configuration (NVIDIA NIM):
```env
MODEL_PROVIDER=nvidia
MODEL=meta/llama-3.3-70b-instruct
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1

# Optional Integrations
DISCORD_TOKEN=your-discord-bot-token
TAVILY_API_KEY=your-tavily-key-optional
```

---

## 🧪 Running the Test Suite

```bash
source .venv/bin/activate
pytest
```

Output:
```
====================== 28 passed in 0.45s ======================
```

---

## 📚 Knowledge Base Ingestion (RAG)

Populate your RAG knowledge base with documentation, papers, or project notes:

```bash
python scripts/index_docs.py ./README.md kalki_knowledge
```

`ResearchAgent` automatically queries the `kalki_knowledge` collection before calling the LLM, grounding all responses in your indexed documentation.

---

## ⏰ Background Job Scheduler

Run the application with the background job scheduler enabled:

```bash
python app.py --with-scheduler
```

Built-in background jobs:
- **Health Check**: Graph compilation ping every 30 minutes.
- **Daily Market Scan**: Weekdays at 06:30 UTC.
- **Daily GitHub Activity Summary**: Every day at 08:00 UTC.

---

## 🤖 Hermes Discord Integration

To run Kalki Nexus as a live Discord bot:

```bash
python app.py --discord
```

Hermes will respond to channel messages, automatically chunking long responses (>2000 characters) and executing agent workflows.

---

## ⚙️ Production Deployment (Systemd on Linux/Azure VM)

### 1. Install Systemd Service
```bash
sudo cp deploy/kalki-nexus.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable kalki-nexus
sudo systemctl start kalki-nexus
```

### 2. Inspect Logs & Status
```bash
sudo systemctl status kalki-nexus
tail -f logs/kalki.log
```

---

## 💾 Database Backups

Backup `kalki_memory.db` and `kalki_rag.db` at any time:

```bash
python scripts/backup_db.py
```
Backups are saved to `backups/YYYYMMDD_HHMMSS/`.
