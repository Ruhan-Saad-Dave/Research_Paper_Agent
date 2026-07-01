# Project Overview — Research Paper Q&A Agent

## What You Are Building

A **Research Paper Q&A Agent** that can:
1. Accept one or more PDF research papers as input.
2. Ingest and index the content of those papers.
3. Answer natural-language questions about the papers using the indexed content.
4. Cite specific passages from the papers in its answers.

This is a classic **Retrieval-Augmented Generation (RAG)** application, orchestrated
with **LangGraph**, powered by a **LangChain** pipeline, and optionally accelerated by
**PyTorch**-backed embedding models.

---

## Why This Project Is Ideal for Learning

| Technology | What You Practice |
|---|---|
| **PyTorch** | Running a local embedding model (sentence-transformers) to turn text into vectors |
| **LangChain** | Document loaders, text splitters, vector stores, prompt templates, and LLM chains |
| **LangGraph** | Stateful multi-step agent graph: ingest → retrieve → generate → cite |
| **RAG** | The full pipeline: chunk, embed, index, retrieve, augment, generate |

Each technology is used for a real purpose — not as a toy example.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                  (CLI or simple Gradio UI)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │ question
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph Agent                             │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐  │
│   │  Router  │───▶│ Retrieve │───▶│ Generate │───▶│  Cite  │  │
│   └──────────┘    └──────────┘    └────────┬─┘    └────────┘  │
│                        ▲                   │                    │
│                        │    grade/rerank   │                    │
│                        └───────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                         │ retrieve
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Vector Store                               │
│            (FAISS — stored locally on disk)                     │
│                                                                 │
│   Chunks ──[PyTorch embedding model]──▶ Vectors                 │
└─────────────────────────────────────────────────────────────────┘
                         ▲ ingest
                         │
┌─────────────────────────────────────────────────────────────────┐
│              PDF Ingestion Pipeline (LangChain)                 │
│   PDF ──▶ PyPDFLoader ──▶ RecursiveCharacterTextSplitter ──▶   │
│           HuggingFaceEmbeddings ──▶ FAISS.add_documents()      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Deliverables (What You Will Have at the End)

- `ingest.py` — loads PDFs, chunks text, embeds, saves vector store
- `agent.py` — LangGraph graph definition with retrieve/generate/cite nodes
- `main.py` — entry point, wires everything together, runs the CLI loop
- `docs/` — all learning and reference documentation (this folder)
- `papers/` — folder where you drop PDF research papers

---

## Non-Goals

- This project does **not** use a cloud vector database (Pinecone, Weaviate, etc.).
- It does **not** fine-tune a language model.
- It does **not** build a production-grade API.

These can be added later once the core pipeline is solid.
