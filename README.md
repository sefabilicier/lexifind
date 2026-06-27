# LexiFind — Legal Platform

> **Production-grade Retrieval-Augmented Generation system for legal document analysis.**
> Built on hybrid vector search, knowledge graphs, multi-pipeline routing, and enterprise-grade security.

---

## Table of Contents

1. [What is LexiFind?](#1-what-is-lexifind)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [System Architecture](#3-system-architecture)
4. [RAG Pipeline Modes](#4-rag-pipeline-modes)
5. [Knowledge Graph Layer](#5-knowledge-graph-layer)
6. [Retrieval Architecture](#6-retrieval-architecture)
7. [Security Architecture](#7-security-architecture)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Tech Stack](#9-tech-stack)
10. [Ports & Services](#10-ports--services)
11. [Project Structure](#11-project-structure)
12. [Getting Started](#12-getting-started)
13. [API Reference](#13-api-reference)
14. [Configuration](#14-configuration)

---

## 1. What is LexiFind?

LexiFind is a **Retrieval-Augmented Generation (RAG) platform** designed for legal document analysis. It goes far beyond a basic question-answering system: it is a multi-pipeline, multi-strategy intelligent retrieval engine capable of reasoning across documents, correcting its own retrieval failures, building knowledge graphs from raw legal text, and evaluating its own output quality.

### Core Capabilities

- **Five distinct RAG pipeline modes** — from simple lookup to multi-step agentic reasoning
- **Hybrid retrieval** — dense (semantic) + sparse (lexical) search fused via Reciprocal Rank Fusion
- **Knowledge graph construction** — entity and relationship extraction from legal documents
- **Automatic pipeline routing** — query complexity classification selects the best pipeline
- **Self-correction** — Corrective RAG detects and recovers from low-quality retrieval
- **LLM-as-judge evaluation** — RAGAS-inspired quality metrics on every pipeline
- **Enterprise security** — prompt injection guard, rate limiting, in-memory API key rotation
- **Structured observability** — JSON-structured logging across every pipeline stage

---

## 2. Theoretical Foundation

### 2.1 Retrieval-Augmented Generation (RAG)

RAG was formally introduced by Lewis et al. (2020) in *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*. The core insight is that large language models (LLMs) encode parametric knowledge in weights — but this knowledge is static, potentially outdated, and not traceable to sources. RAG separates **knowledge storage** (a retrieval index) from **knowledge application** (the LLM), enabling:

- Grounded, citation-backed answers
- Up-to-date information without model retraining
- Reduced hallucination via context anchoring
- Domain specialization without fine-tuning

<img width="8270" height="5845" alt="Image" src="https://github.com/user-attachments/assets/9e4359b8-6313-4b36-9f7c-8ddab1a4984c" />

IBM's Think framework categorizes RAG systems into three maturity levels: **Naive RAG** (basic retrieve-then-generate), **Advanced RAG** (pre/post retrieval optimization), and **Modular/Agentic RAG** (dynamic, tool-using pipelines). LexiFind implements all three levels.

### 2.2 Hybrid Search — Dense + Sparse

Pure dense (vector) search excels at semantic similarity but can miss exact legal terms. Pure sparse (keyword) search catches exact matches but fails on paraphrase. IBM Think and AWS RAG best practices both identify **hybrid search as the production baseline**.

- **Dense retrieval**: Approximate Nearest Neighbor search over BGE-M3 embeddings (1024-dim). Captures *"what does this mean?"*
- **Sparse retrieval**: SPLADE-style lexical weights from BGE-M3. Captures *"what exact words appear?"*
- **Fusion**: Reciprocal Rank Fusion (Cormack et al., 2009) merges ranked lists without score normalization

BGE-M3 is uniquely positioned to power both dense and sparse retrieval from a single model pass, eliminating the need for two separate embedding systems.

### 2.3 Reranking

Retrieval is fast but approximate. Reranking is slow but precise. The two-stage paradigm (retrieve many, rerank to few) is now standard in production RAG:

```
Stage 1 — Retrieval (fast, approximate):
  Query → ANN Search → Top-K candidates (K=10-20)
  Latency: ~50ms | Method: cosine similarity

Stage 2 — Reranking (slow, precise):
  Query × Candidate → Cross-encoder score → Top-N (N=3)
  Latency: ~500ms | Method: BGE-Reranker-v2-M3 (cross-attention)
```

The cross-encoder jointly encodes query and document, enabling full attention between them — far more accurate than independent encoding, at higher cost.

### 2.4 Agentic RAG & LangGraph

ReAct (Yao et al., 2022) introduced the **Reasoning + Acting** paradigm: an LLM interleaves reasoning steps with tool use, enabling iterative problem solving. LangGraph operationalizes this as a stateful graph where nodes are functions and edges are conditional transitions.

LexiFind's Agentic RAG implements a Plan-Retrieve-Evaluate loop:

<img width="5850" height="4140" alt="Image" src="https://github.com/user-attachments/assets/da5c53ab-7a00-4035-9afc-85171e5b7f60" />
This enables multi-hop reasoning: decomposing a complex question into sub-questions, retrieving independently, and aggregating — something single-pass RAG cannot do.

### 2.5 Corrective RAG (CRAG)

Yan et al. (2024) in *"Corrective Retrieval Augmented Generation"* identified that naive RAG blindly trusts its retrieved chunks — even when they are irrelevant. CRAG adds a **grading step** that evaluates each retrieved chunk for relevance, then applies a correction strategy:

<img width="5334" height="4811" alt="Image" src="https://github.com/user-attachments/assets/989e79e3-c28c-4e45-bcae-b4324856081e" />

This self-correction loop recovers from retrieval failures without human intervention.

### 2.6 Graph RAG

Microsoft Research (Edge et al., 2024) demonstrated that pure vector search misses **relational context**: *"Article 5 requires X"* when the query mentions X but not Article 5. Graph RAG builds a knowledge graph from documents and uses it to expand retrieval beyond semantic similarity.

LexiFind implements the local-to-global Graph RAG approach:
- **Local**: entity-centric retrieval via graph traversal
- **Global**: community-level thematic summaries via Louvain detection

### 2.7 LLM-as-Judge Evaluation (RAGAS)

Es et al. (2023) introduced RAGAS (*"Automated Evaluation of RAG Pipelines"*) — a reference-free evaluation framework using an LLM as the judge. LexiFind implements four core metrics:

| Metric | Measures | Question asked |
|---|---|---|
| **Faithfulness** | Hallucination | Is the answer grounded in the context? |
| **Answer Relevancy** | Response quality | Does the answer address the question? |
| **Context Precision** | Retrieval signal/noise | Are the retrieved chunks actually useful? |
| **Context Recall** | Coverage | Does the context cover the ground truth? |

---

## 3. System Architecture

### 3.1 High-Level Overview

<img width="8270" height="5845" alt="Image" src="https://github.com/user-attachments/assets/faa06925-04b3-4145-ad4b-9c5a61cc2c0f" />

### 3.2 Ingestion Pipeline Flow

<img width="5845" height="4135" alt="Image" src="https://github.com/user-attachments/assets/ba74fb1f-cce5-4076-929f-667a4a290e00" />

### 3.3 Query Processing Flow

```
User Query
    │
    ▼
[Prompt Guard]  ──── BLOCKED ──→  400 Bad Request
    │ SAFE
    ▼
[Query Router]  →  Classify:  simple | complex | multi_hop | uncertain
    │               Route to: naive   | advanced | agentic   | corrective
    │
    ├──────────────────────────────────────────────────────────┐
    │                                                          │
    ▼                                                          ▼
[Naive RAG]                                           [Advanced RAG]
    │                                                          │
    ├─ Embed query (BGE-M3)                    ├─ Rewrite query (8B LLM)
    ├─ Dense retrieve (Qdrant)                 ├─ Embed: original + rewritten
    ├─ Sparse retrieve (Qdrant)                ├─ Retrieve both
    ├─ RRF Fusion                              ├─ Deduplicate
    ├─ Rerank (BGE-Reranker)                   ├─ RRF Fusion + Rerank
    └─ Generate (Llama-3.3-70B)               └─ Generate (Llama-3.3-70B)

    ▼                                                          ▼
[Agentic RAG]                                         [Corrective RAG]
    │                                                          │
    ├─ Plan: decompose → sub-questions         ├─ Retrieve chunks
    ├─ Retrieve per sub-question               ├─ Grade each chunk (8B LLM)
    ├─ Evaluate: sufficient? (reranker score)  │   relevant/ambiguous/irrelevant
    ├─ If NO: refine + re-retrieve             ├─ All irrelevant? → rewrite + retry
    └─ If YES: Rerank → Generate               └─ Filter → Rerank → Generate

    ▼
[Graph RAG]
    │
    ├─ Vector retrieval (hybrid)
    ├─ Entity recognition in query
    ├─ Graph traversal: neighbors (depth=2)
    ├─ Community summary injection
    └─ Merged context → Rerank → Generate

    │
    ▼
[Output Content Filter]  →  PII check, hallucination signals, citation grounding
    │
    ▼
[Response]
    {answer, citations, usage, pipeline, security, metadata}
```

### 3.4 LangGraph Agentic RAG State Machine

<img width="4680" height="4338" alt="Image" src="https://github.com/user-attachments/assets/9d81349a-a2bc-4348-a7b3-7c7ee4dea1d3" />

---

## 4. RAG Pipeline Modes

### 4.1 Pipeline Comparison

| Mode | Trigger | LLM Calls | Latency | Best For |
|---|---|---|---|---|
| **Naive** | `simple` query | 1 | ~2s | Single-hop factual lookup |
| **Advanced** | `complex` query | 2 | ~4s | Multi-concept, broad queries |
| **Agentic** | `multi_hop` query | 3-7 | ~10s | Cross-document reasoning |
| **Corrective** | `uncertain` query | 3-8 | ~8s | Ambiguous or low-confidence queries |
| **Graph** | explicit | 2 | ~5s | Relationship-heavy queries |
| **Auto** | any | varies | varies | Default — router decides |

### 4.2 Pipeline Selection Logic (QueryRouter)

<img width="5845" height="4135" alt="Image" src="https://github.com/user-attachments/assets/efce0645-7a9c-44df-875b-c9dae74137fc" />

---

## 5. Knowledge Graph Layer

### 5.1 Graph Construction Pipeline

<img width="5845" height="4135" alt="Image" src="https://github.com/user-attachments/assets/23bef723-6b18-4388-a353-623b416e2d26" />

### 5.2 Graph RAG Retrieval Expansion

<img width="4676" height="4950" alt="Image" src="https://github.com/user-attachments/assets/6767cf97-4f8b-4286-baf8-2ef96c18de80" />

---

## 6. Retrieval Architecture

### 6.1 BGE-M3: One Model, Three Retrieval Modes

BGE-M3 (BAAI, 2024) is uniquely capable of producing dense, sparse, and ColBERT-style embeddings from a single forward pass. LexiFind uses dense + sparse:

<img width="5845" height="4135" alt="Image" src="https://github.com/user-attachments/assets/a47f11c7-de22-4f6f-b164-09221dfa205e" />

### 6.2 Reciprocal Rank Fusion

RRF (Cormack et al., 2009) merges ranked lists without requiring score normalization. Score for each document:

```
RRF(d) = α × 1/(k + rank_dense(d))
        + (1-α) × 1/(k + rank_sparse(d))

Where:
  k   = 60  (RRF constant from original paper)
  α   = 0.5 (configurable — HYBRID_ALPHA in config)
  rank = position in each ranked list (1-indexed)
```

Documents appearing in both lists are boosted. Documents in only one list still score positively.

### 6.3 Full Retrieval Stack

<img width="5845" height="8270" alt="Image" src="https://github.com/user-attachments/assets/eb83ac88-c51a-448c-b890-56c4efd46bcd" />

---

## 7. Security Architecture

### 7.1 Security Layers

<img width="5845" height="8270" alt="Image" src="https://github.com/user-attachments/assets/28977481-ac11-492a-b9cc-1e6f03996b1f" />

### 7.2 API Key Lifecycle

```
Server Startup
    │
    ▼
secrets.token_urlsafe(32) → "lf_v1-{token}"
    │
    ├─ Stored in: memory only (APIKeyManager singleton)
    ├─ Printed to: terminal (one-time banner)
    ├─ Validated via: secrets.compare_digest() (timing-safe)
    └─ Expires when: process exits or restarts

Security properties:
  ├─ 256-bit entropy (cryptographically secure)
  ├─ Constant-time comparison (no timing attacks)
  ├─ Zero persistence (no .env, no DB, no logs)
  └─ Automatic rotation on every restart
```

---

## 8. Evaluation Framework

### 8.1 RAGAS Metrics — LLM-as-Judge

<img width="5377" height="4654" alt="Image" src="https://github.com/user-attachments/assets/9c264565-e297-4acf-a145-118e0e63dd54" />

### 8.2 Golden Set Structure

8 curated question-answer pairs, categorized by domain:

| Category | Count | Pipelines Tested |
|---|---|---|
| `technical_requirements` | 3 | naive |
| `pipeline_requirements` | 2 | advanced |
| `security_requirements` | 1 | advanced |
| `evaluation_requirements` | 1 | advanced |
| `graph_requirements` | 1 | graph |

---

## 9. Tech Stack

### Core Framework

| Component | Technology | Rationale |
|---|---|---|
| **API Framework** | FastAPI 0.115 | Async, type-safe, auto-docs |
| **LLM** | Groq API (Llama-3.3-70B) | Free tier, 70B quality |
| **Fast LLM** | Groq API (Llama-3.1-8B) | Classifier, rewriter, grader |
| **Embeddings** | BAAI/bge-m3 (local) | Multilingual, dense+sparse |
| **Reranker** | BAAI/bge-reranker-v2-m3 | Cross-encoder precision |
| **Vector DB** | Qdrant (Docker) | Dense + sparse native |
| **Graph** | NetworkX + python-louvain | In-memory, zero infra |
| **Agent Framework** | LangGraph | Stateful multi-step agents |
| **Rate Limiting** | slowapi | Starlette-native |
| **Logging** | structlog | JSON-structured |

### Document Processing

| Component | Technology |
|---|---|
| PDF parsing | PyMuPDF (fitz) |
| DOCX parsing | python-docx |
| HTML parsing | BeautifulSoup4 |
| Retry/backoff | tenacity |

### Zero-Cost Stack

Every component in LexiFind is **free to run**:
- Groq free tier: 6,000 TPM, 500K tokens/day on Llama-3.3-70B
- BGE-M3: local HuggingFace download, no API cost
- Qdrant: self-hosted Docker container
- NetworkX: in-memory Python library

---

## 10. Ports & Services

| Port | Service | Protocol | Description |
|---|---|---|---|
| **8000** | FastAPI Application | HTTP | Main API server — all endpoints |
| **6333** | Qdrant REST API | HTTP | Vector DB — search, upsert, collections |
| **6334** | Qdrant gRPC | gRPC | Qdrant high-performance interface |

### Endpoint Map

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/health` | ❌ Public | Health check |
| `GET` | `/docs` | ❌ Public | Swagger UI |
| `POST` | `/api/documents/ingest` | ✅ Required | Upload + ingest document |
| `POST` | `/api/documents/graph/build` | ✅ Required | Build knowledge graph from all chunks |
| `POST` | `/api/query` | ✅ Required | Query RAG pipeline |
| `POST` | `/api/evaluate/` | ✅ Required | Run full RAGAS evaluation |
| `POST` | `/api/evaluate/single` | ✅ Required | Evaluate single Q&A pair |
| `GET` | `/api/evaluate/report` | ✅ Required | Retrieve latest evaluation report |

---

## 11. Project Structure

```
lexi-find/
│
├── docker-compose.yml         # Qdrant container (run from WSL)
├── requirements.txt           # All Python dependencies
├── .env.example               # Environment template
├── .gitignore
├── README.md
│
├── scripts/
│   ├── setup.ps1              # Windows: deps + model download
│   ├── setup.sh               # Linux/Mac: deps + model download
│   └── generate_key.py        # API key generator (manual use)
│
├── data/
│   ├── graph_cache.json       # Persisted NetworkX graph
│   └── evaluation_report.json # Latest RAGAS evaluation results
│
└── app/
    ├── main.py                # FastAPI app, middleware, lifespan
    ├── config.py              # Pydantic Settings (centralized config)
    ├── dependencies.py        # DI container
    │
    ├── api/
    │   ├── routes/
    │   │   ├── documents.py   # POST /ingest, POST /graph/build
    │   │   ├── query.py       # POST /query (all pipelines)
    │   │   └── evaluation.py  # POST /evaluate, GET /report
    │   └── middleware/
    │       ├── auth.py        # API key middleware
    │       └── rate_limit.py  # slowapi limiter
    │
    ├── ingestion/
    │   ├── parsers/
    │   │   ├── base.py        # Abstract parser interface
    │   │   ├── pdf_parser.py  # PyMuPDF
    │   │   ├── docx_parser.py # python-docx
    │   │   └── html_parser.py # BeautifulSoup4
    │   ├── chunkers/
    │   │   ├── base.py        # Abstract chunker interface
    │   │   ├── fixed_size.py  # Character-based fixed chunks
    │   │   ├── recursive.py   # Hierarchy-aware splitting (default)
    │   │   └── semantic.py    # Embedding similarity breakpoints
    │   ├── embedder.py        # BGE-M3 singleton wrapper
    │   └── pipeline.py        # Parse→Chunk→Embed→Upsert orchestrator
    │
    ├── retrieval/
    │   ├── dense_retriever.py  # Qdrant ANN search (cosine)
    │   ├── sparse_retriever.py # Qdrant sparse search (lexical)
    │   ├── hybrid_retriever.py # RRF fusion (dense + sparse)
    │   └── reranker.py         # BGE-Reranker-v2-M3 cross-encoder
    │
    ├── pipeline/
    │   ├── router.py           # Query classifier → pipeline selector
    │   ├── naive_rag.py        # Retrieve → Rerank → Generate
    │   ├── advanced_rag.py     # Rewrite → Multi-retrieve → Generate
    │   ├── agentic_rag.py      # LangGraph Plan→Retrieve→Evaluate loop
    │   ├── corrective_rag.py   # CRAG: grade → self-correct → generate
    │   └── graph_rag.py        # Vector + graph traversal → generate
    │
    ├── graph/
    │   ├── entity_extractor.py  # LLM-based NER (legal entities)
    │   ├── builder.py           # NetworkX DiGraph construction + cache
    │   └── community_detector.py# Louvain community detection
    │
    ├── generation/
    │   ├── generator.py         # Groq LLM wrapper + retry
    │   └── citation_builder.py  # Source citation formatter
    │
    ├── security/
    │   ├── api_key_manager.py   # In-memory key generation + validation
    │   ├── prompt_guard.py      # Rule + LLM injection detection
    │   └── content_filter.py    # Output PII + hallucination filter
    │
    ├── evaluation/
    │   ├── golden_set.json      # 8 ground-truth Q&A pairs
    │   ├── evaluator.py         # RAGAS LLM-as-judge (4 metrics)
    │   └── runner.py            # Full evaluation orchestrator
    │
    └── observability/
        └── logger.py            # structlog JSON/pretty logger
```

---

## 12. Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop (for Qdrant via WSL)
- WSL2 (Windows) or native Docker (Linux/Mac)
- Groq API key — free at [console.groq.com](https://console.groq.com)

### 1. Clone & Configure

```bash
git clone <your-repo-url>
cd lexi-find
copy .env.example .env      # Windows
# cp .env.example .env      # Linux/Mac
```

Edit `.env` — only two fields are required:
```env
GROQ_API_KEY=your_groq_api_key_here
JWT_SECRET_KEY=any_long_random_string_here
```

### 2. Setup — Install Dependencies + Download Models

```powershell
# Windows
.venv\Scripts\activate
.\scripts\setup.ps1
```

```bash
# Linux / Mac
source .venv/bin/activate
bash scripts/setup.sh
```

This installs all Python packages, downloads BGE-M3 (~570MB) and BGE-Reranker-v2-M3 (~570MB) from HuggingFace.

### 3. Start Qdrant (WSL / Linux)

```bash
# From WSL terminal, navigate to project root
cd /mnt/d/lexi-find
docker compose up -d

# Verify Qdrant is healthy
curl http://localhost:6333/healthz
```

### 4. Start the API Server

```powershell
uvicorn app.main:app --reload --port 8000
```

On startup, you will see:

```
============================================================
  🔑  LexiFind — API Key Generated
============================================================

  Key:  lf_v1-Kj8mN2xPqR5vW9yA3cE7hL1nT4uD6fB0sG...

  Use this key in every request:
  X-API-Key: lf_v1-Kj8mN2xPqR5vW9yA3cE7hL1nT4uD6fB0sG...

  ⚠️  Key lives in memory only. Restart = new key.
============================================================
```

Copy this key — you will need it for every API request.

### 5. Ingest Your First Document

```bash
curl -X POST http://localhost:8000/api/documents/ingest \
  -H "X-API-Key: lf_v1-your-key-here" \
  -F "file=@/path/to/your/legal_document.pdf"
```

### 6. Build the Knowledge Graph

```bash
curl -X POST http://localhost:8000/api/documents/graph/build \
  -H "X-API-Key: lf_v1-your-key-here"
```

### 7. Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "X-API-Key: lf_v1-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the security requirements?", "pipeline": "auto"}'
```

### 8. Evaluate Pipeline Quality

```bash
curl -X POST http://localhost:8000/api/evaluate/ \
  -H "X-API-Key: lf_v1-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"question_ids": null}'
```

---

## 13. API Reference

### POST `/api/query`

The main query endpoint. All pipeline modes are available.

**Request:**
```json
{
  "query": "What security measures must be implemented?",
  "pipeline": "auto",
  "top_k": 10,
  "top_n": 3
}
```

**Pipeline options:** `auto | naive | advanced | agentic | corrective | graph`

**Response:**
```json
{
  "answer": "Based on the provided documents, the required security measures include...",
  "citations": [
    {
      "ref": 1,
      "source": "legal_document.pdf",
      "chunk_index": 5,
      "reranker_score": 0.8923
    }
  ],
  "usage": {
    "prompt_tokens": 1692,
    "completion_tokens": 91,
    "total_tokens": 1783
  },
  "pipeline": "advanced_rag",
  "metadata": {
    "original_query": "What security measures must be implemented?",
    "rewritten_query": "security requirements obligations legal document",
    "chunks_retrieved": 18,
    "chunks_used": 3,
    "security": {
      "output_safe": true,
      "warnings": []
    }
  }
}
```

### POST `/api/documents/ingest`

**Supported formats:** PDF, DOCX, DOC, HTML, HTM

**Request:** `multipart/form-data` with `file` field.

**Response:**
```json
{
  "file": "legal_document.pdf",
  "chunks": 47,
  "point_ids": ["uuid1", "uuid2", "..."],
  "original_filename": "legal_document.pdf"
}
```

### POST `/api/evaluate/single`

Evaluate any Q&A pair instantly without running a full pipeline.

**Request:**
```json
{
  "question": "What are the pipeline modes?",
  "answer": "The system supports five pipeline modes: Naive, Advanced, Agentic, Corrective, and Graph RAG.",
  "ground_truth": "Five RAG pipeline types: Naive RAG, Advanced RAG, Agentic RAG, CRAG, Graph RAG.",
  "context_chunks": ["The platform implements five distinct RAG pipeline modes..."]
}
```

**Response:**
```json
{
  "faithfulness": 0.95,
  "answer_relevancy": 0.97,
  "context_precision": 0.90,
  "context_recall": 0.92,
  "overall": 0.935
}
```

---

## 14. Configuration

All configuration is managed via `app/config.py` (Pydantic Settings) and loaded from `.env`.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API key |
| `GROQ_PRIMARY_MODEL` | `llama-3.3-70b-versatile` | Main generation model |
| `GROQ_FAST_MODEL` | `llama-3.1-8b-instant` | Fast ops (classify, rewrite, grade) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Local embedding model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant REST port |
| `QDRANT_COLLECTION` | `lexi_find` | Collection name |
| `TOP_K_RETRIEVAL` | `10` | Chunks fetched per retrieval |
| `FINAL_N_RERANK` | `3` | Chunks passed to LLM |
| `HYBRID_ALPHA` | `0.5` | Dense/sparse balance (0=sparse, 1=dense) |
| `CHUNK_SIZE` | `512` | Chunk size in tokens (approx) |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests per IP per minute |
| `MAX_AGENT_ITERATIONS` | `5` | Max Agentic RAG loop iterations |
| `LOG_FORMAT` | `json` | `json` (production) or `pretty` (dev) |

---

## References

- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* — RAG foundation
- Yao et al. (2022) — *ReAct: Synergizing Reasoning and Acting in Language Models* — Agentic basis
- Cormack et al. (2009) — *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods* — RRF
- Yan et al. (2024) — *Corrective Retrieval Augmented Generation* — CRAG
- Edge et al. (2024) — *From Local to Global: A Graph RAG Approach* — Microsoft GraphRAG
- Es et al. (2023) — *RAGAS: Automated Evaluation of RAG Pipelines* — Evaluation framework
- IBM Think RAG Series — Production RAG architecture best practices
- AWS RAG Best Practices — Hybrid search as production baseline
- OWASP LLM Top 10 (2023) — LLM security: prompt injection, output handling
- BAAI/bge-m3 (2024) — *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity*

---

*LexiFind — Built with precision, grounded in research, designed for production.*