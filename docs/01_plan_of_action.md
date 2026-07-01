# Plan of Action

Work through the phases in order. Each phase has a clear goal and a small set of files to
write. Do not skip ahead — later phases build on earlier ones.

---

## Phase 0 — Environment Setup
**Goal:** Get a working Python environment with all dependencies installed.

### Tasks
- [ ] Create a virtual environment: `python -m venv venv`
- [ ] Activate it: `venv\Scripts\activate` (Windows)
- [ ] Create `requirements.txt` with the libraries listed in `docs/02_learning_concepts.md`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify PyTorch is installed: run `python -c "import torch; print(torch.__version__)"`

### What You Learn
- Python virtual environments and dependency management.

---

## Phase 1 — PDF Ingestion Pipeline
**Goal:** Write `ingest.py` that reads a PDF and stores its text as embeddings in FAISS.

### Tasks
- [ ] Use `PyPDFLoader` from LangChain to load a PDF from the `papers/` folder.
- [ ] Use `RecursiveCharacterTextSplitter` to chunk the text.
- [ ] Use `HuggingFaceEmbeddings` (backed by a sentence-transformers model via PyTorch) to
      embed the chunks.
- [ ] Save the FAISS index to disk with `FAISS.save_local()`.

### Milestone
Run `python ingest.py papers/sample.pdf` and confirm a `vectorstore/` folder is created.

### What You Learn
- How RAG ingestion works end-to-end.
- What a text embedding is and why chunking matters.
- How PyTorch runs under the hood of a HuggingFace model.

---

## Phase 2 — Basic Retrieval Chain
**Goal:** Write a simple `retrieve.py` (or add to `ingest.py`) that loads the FAISS index
and answers a question using a basic LangChain chain — no agent yet.

### Tasks
- [ ] Load the saved FAISS index.
- [ ] Build a `RetrievalQA` chain with a prompt template.
- [ ] Pass a hard-coded question and print the answer + source chunks.

### Milestone
Running `python retrieve.py "What is the main contribution of the paper?"` returns a
grounded answer with citations.

### What You Learn
- LangChain chains, prompt templates, and retrievers.
- The retrieve → augment → generate loop in isolation.

---

## Phase 3 — LangGraph Agent
**Goal:** Replace the simple chain with a stateful LangGraph graph.

### Tasks
- [ ] Define a `State` TypedDict with fields: `question`, `documents`, `answer`.
- [ ] Write three node functions: `retrieve_node`, `generate_node`, `cite_node`.
- [ ] Wire them into a `StateGraph` with edges.
- [ ] Add a conditional edge: if retrieved docs are empty, route to a "no results" node.
- [ ] Compile and run the graph.

### Milestone
The graph handles both the happy path (docs found → answer → cite) and the fallback path
(no docs → "I could not find relevant information").

### What You Learn
- LangGraph state machines, nodes, edges, and conditional routing.
- How agents differ from simple chains.

---

## Phase 4 — Polish & CLI
**Goal:** Combine everything into `main.py` with a clean command-line interface.

### Tasks
- [ ] Accept a PDF path as a CLI argument; call `ingest.py` logic if the index doesn't exist.
- [ ] Enter an interactive question loop.
- [ ] Display the answer and the top-3 source chunks with page numbers.
- [ ] Add a `--reset` flag to delete and re-ingest the vector store.

### Milestone (Final)
You can run `python main.py papers/attention.pdf` and have a full conversation with the paper.

---

## Phase 5 (Optional Extensions)
Once Phase 4 is working, try any of these:
- Add a Gradio web UI.
- Support multiple PDFs in one session.
- Add a re-ranking step inside the LangGraph graph.
- Swap FAISS for ChromaDB.
- Replace HuggingFace embeddings with the OpenAI embeddings API.
