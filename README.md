# 5G RCA RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** application for 5G network Root Cause Analysis (RCA). Built on the [TeleLogs dataset](https://huggingface.co/datasets/netop/TeleLogs) — a benchmark for LLM-based fault diagnosis in 5G wireless networks.

The system provides two interfaces: a **natural language query endpoint** for network engineers asking troubleshooting questions, and an **automated telemetry endpoint** for systems feeding raw drive-test measurements directly.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│          Natural Language Query  │  Raw Telemetry Tables        │
└──────────────────┬───────────────┴──────────────┬──────────────┘
                   │                               │
                   ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI APPLICATION                        │
│   POST /diagnose/query          POST /diagnose/telemetry        │
│   → Streaming text response     → Structured JSON response      │
│                                                                 │
│   Input Validation & Guardrails │ Rate Limiting (10 req/min)    │
└──────────────────────────────────────────┬──────────────────────┘
                                           │
                   ┌───────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL PIPELINE                         │
│                                                                 │
│  1. Multi-Query Expansion                                       │
│     GPT-4o-mini rephrases query into 3-4 variants              │
│                      │                                          │
│                       ▼                                         │
│  2. Hybrid Vector Search                                        │
│     Each variant → OpenAI text-embedding-3-large (1024d)       │
│     → Pinecone ANN search → top-20 candidates                  │
│                      │                                          │
│                       ▼                                         │
│  3. Cohere Reranking                                            │
│     Cross-encoder scores all 20 → returns top-5                │
└──────────────────────────────────────────┬──────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GENERATION PIPELINE                         │
│                                                                 │
│  Top-5 documents formatted with [Source N - Root Cause: CX]    │
│                      │                                          │
│                       ▼                                         │
│  GPT-4o generates grounded answer with citations               │
│                      │                                          │
│                       ▼                                         │
│  Redis Semantic Cache (score_threshold=0.2)                    │
│  → Cache hit: return instantly                                  │
│  → Cache miss: full pipeline execution                         │
└─────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RESPONSE                                  │
│                                                                 │
│  /diagnose/query     → Streaming tokens (SSE)                  │
│  /diagnose/telemetry → Structured JSON                         │
│  {                                                              │
│    "root_cause_code": "C7",                                     │
│    "description":     "...",                                    │
│    "confidence":      0.91,                                     │
│    "timestamp":       "2025-09-14T14:00:00",                   │
│    "citations":       [...],                                    │
│    "input_summary":   {...}                                     │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 8 Root Causes

The system diagnoses throughput degradation (below 600 Mbps) against 8 predefined root causes from the TeleLogs benchmark:

| Code | Root Cause | Category |
|------|-----------|----------|
| C1 | Serving cell downtilt too large — weak far-end coverage | Coverage |
| C2 | Coverage distance exceeds 1km — overshooting | Coverage |
| C3 | Neighbouring cell provides higher throughput | Resource |
| C4 | Non-colocated co-frequency neighbours — interference | Interference |
| C5 | Frequent handovers degrade performance | Mobility |
| C6 | Serving and neighbour cell share same PCI mod 30 | Interference |
| C7 | Vehicle speed exceeds 40 km/h — Doppler effect | Mobility |
| C8 | Average scheduled RBs below 160 | Resource |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o (generation) · GPT-4o-mini (multi-query) |
| Embeddings | OpenAI text-embedding-3-large (1024d) |
| Vector Store | Pinecone Serverless |
| Reranking | Cohere rerank-english-v3.0 |
| Orchestration | LangChain LCEL |
| API | FastAPI + Uvicorn |
| Caching | Redis Semantic Cache |
| Observability | LangFuse |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```
5g-rca-app/
├── data/
│   └── raw/
│       ├── train.json          # 2,400 training scenarios
│       └── test.json           # 864 test scenarios
│       └── load_dataset.py     # loading dataset script from Huggingface 
├── ingestion/
│   ├── formatter.py            # JSON → LangChain Documents
│   └── pipeline.py             # Embed + upsert to Pinecone
├── rag/
│   ├── retriever.py            # Multi-query + Cohere rerank
│   ├── chain.py                # LCEL RAG chain
│   └── prompts.py              # System prompts
├── api/
│   ├── main.py                 # FastAPI endpoints
│   └── schemas.py              # Pydantic models
├── evaluation/
│   ├── evaluate.py             # TeleLogs benchmark evaluation
│   ├── collect_results.py      # Collect RAG outputs for RAGAS
│   ├── run_ragas.py            # Custom RAGAS evaluation
│   └── ragas_results.json      # Evaluation results
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── .env.example                # Environment variable template
├── docker-compose.yml          # Docker stack
├── Dockerfile                  # API container
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- API keys for: OpenAI · Pinecone · Cohere 

### 1. Clone the repository

```bash
git clone https://github.com/bassemessam1/5g-rca-app.git
cd 5g-rca-app
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Fill in your API keys in `.env`:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=telelogs-rag
PINECONE_NAMESPACE=telelogs-v1
COHERE_API_KEY=...
```

### 3. Download the dataset

```python
python ./data/raw/load_dataset.py
```


### 4. Create Pinecone index

Create a serverless index in your Pinecone console:
- **Name:** `telelogs-rag`
- **Dimensions:** `1024`
- **Metric:** `cosine`
- **Cloud:** AWS us-east-1

### 5. Run the ingestion pipeline

```bash
pip install -r requirements.txt
python ingestion/pipeline.py
```

This embeds all 2,400 training documents and upserts them into Pinecone.

### 6. Start the application

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

---

## API Reference

### Health Check

```bash
GET /health
```

```json
{"status": "ok"}
```

---

### Natural Language Query

```bash
POST /diagnose/query
```

**Request:**
```json
{
  "question": "Why does throughput drop when the vehicle is moving fast?"
}
```

**Response:** Streaming text with cited root causes.

```
Based on [Source 1 - Root Cause: C7], throughput degradation at high 
vehicle speeds is caused by the Doppler effect...
```

---

### Automated Telemetry Diagnosis

```bash
POST /diagnose/telemetry
```

**Request:**
```json
{
  "engineering_parameters": "gNodeB ID|Cell ID|...\n0034038|4|...",
  "drive_test_data": "Timestamp|Longitude|...\n2025-05-07 15:23:52|..."
}
```

**Response:**
```json
{
  "root_cause_code": "C2",
  "description": "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
  "confidence": 0.87,
  "timestamp": "2025-09-14T14:04:23",
  "citations": [
    {"source": 1, "root_cause_code": "C2", "relevance_score": 0.91}
  ],
  "input_summary": {
    "engineering_parameters_preview": "gNodeB ID|Cell ID|Longitude...",
    "drive_test_preview": "Timestamp|Longitude|Latitude..."
  }
}
```

---

## Evaluation Results

### TeleLogs Benchmark (Structured Telemetry)

| Metric | Score |
|--------|-------|
| Answer Rate | 100% |
| Accuracy | ~20% |

> **Finding:** TeleLogs reuses the same telemetry scenario across questions — making it a reasoning benchmark, not a retrieval benchmark. RAG cannot distinguish between causes when all queries are identical. Fine-tuning is the appropriate approach for this evaluation.

### Natural Language Query Evaluation (Custom RAGAS)

Evaluated on 23 natural language questions covering all 8 root causes:

| Metric | Score | Threshold |
|--------|-------|-----------|
| Faithfulness | **1.000** | > 0.80 |
| Answer Relevancy | **0.957** | > 0.75 |

> **Finding:** The system performs excellently on natural language queries — every answer is fully grounded in retrieved context with no hallucination.

---

## Key Learnings

**1. RAG is not always the right tool**
Structured telemetry data with consistent patterns is better served by fine-tuning than retrieval. RAG excels at natural language knowledge retrieval — not structured data classification.

**2. Caching requires careful management**
Redis semantic cache dramatically improves production performance but corrupts evaluation results. Always disable caching during evaluation and flush after pipeline changes.

**3. Query-document format alignment is critical**
The format of your queries must match the format of your documents. Embedding a raw data table against prose documents produces poor retrieval because the representations live in different parts of the vector space.

---

## What's Next

This project is the first part of a two-part series:

- **Part 1 (this project):** RAG-based RCA system
- **Part 2 (coming soon):** Fine-tuning a local LLM on TeleLogs and comparing accuracy against the RAG approach

---

## Dataset

This project uses the [TeleLogs dataset](https://huggingface.co/datasets/netop/TeleLogs) by the NetOp Team at Huawei Paris Research Center, released under the MIT License.

```bibtex
@article{sana2025reasoning,
  title={Reasoning Language Models for Root Cause Analysis in 5G Wireless Networks},
  author={Mohamed Sana and Nicola Piovesan and Antonio De Domenico and 
          Yibin Kang and Haozhe Zhang and Merouane Debbah and Fadhel Ayed},
  year={2025},
  eprint={arXiv:2507.21974}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
