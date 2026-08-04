# PaperLens - Research Paper Analyzer

A focused retrieval-augmented generation (RAG) project. Upload one research PDF, inspect its detected structure, ask questions, and receive answers grounded in passages with explicit page citations.

## What is genuinely implemented

- Single-PDF replacement workflow and page-aware text extraction with PyMuPDF
- Font-size, bold-style, numbering, and common-heading based section detection
- `Document → Section → Chunk` hierarchy persisted alongside the vector index
- Overlapping chunking that preserves paper, page, section name, and section level
- Sentence Transformer embeddings stored in a FAISS cosine-similarity index
- Semantic retrieval within the uploaded paper
- Grounded answer generation using Gemini or Groq
- Clickable inline citations mapped to `Citation [n] — paper.pdf — Page n` source cards
- Deterministic title, author, page-count, and abstract answers 


##
Metadata (`paper`, `page`, `section_name`, `section_level`, `chunk_id`, and text) is stored beside the FAISS vectors and returned with every retrieval result.

## Project structure

```text
backend/
  app/api.py          FastAPI routes
  app/chunking.py     recursive overlapping chunker
  app/pdf.py          metadata, layout, and section extraction
  app/retrieval.py    embeddings, FAISS, persistence, search
  app/generation.py   Gemini/Groq prompting and fallback
  tests/              automated unit tests
frontend/
  src/                React interface
```

## Run locally

Requirements: Python 3.10+, Node 18+, npm.

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate         
pip install -r requirements.txt
cp ../.env.example ../.env
uvicorn app.api:app --reload --port 8000
```


### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### 3. Tests

```bash
cd backend
pytest -q
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload or replace the single PDF |
| `GET` | `/api/documents` | Inspect the indexed paper |
| `DELETE` | `/api/documents` | Clear the paper and index |
| `POST` | `/api/ask` | Retrieve passages and generate a cited answer |
| `GET` | `/health` | Health check |



## Limitations

- Accepts text-based PDF files only.
- Heading detection is layout-aware but heuristic because publishers use inconsistent PDF structures.
- This version uses dense retrieval only. It does not claim hybrid BM25 retrieval or measured accuracy gains.
- Multi-paper comparison is intentionally not supported.

## Screenshots
<img width="1114" height="795" alt="image" src="https://github.com/user-attachments/assets/c96cdf32-c369-4ba4-9683-de91915074f0" />
<img width="1114" height="377" alt="image" src="https://github.com/user-attachments/assets/6108dc3c-513d-4db2-bd06-200e9f64175a" />
<img width="1114" height="771" alt="image" src="https://github.com/user-attachments/assets/c5693ac0-c722-42e4-872e-9dc2a6001b26" />



