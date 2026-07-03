# Core Concepts & Learning Guide

This document explains every major concept you will encounter. Read the relevant section
before starting each phase. Come back here when something in the code confuses you.

---

## 1. RAG — Retrieval-Augmented Generation

### The Problem RAG Solves
Large Language Models (LLMs) are trained on a fixed dataset. They don't know about your
specific research paper. If you just ask "What does this paper say?", the model will
hallucinate or say it doesn't know.

### The RAG Idea
Instead of asking the LLM to remember the paper, you:
1. **Retrieve** the most relevant passages from the paper (using a vector search).
2. **Augment** the LLM's prompt with those passages.
3. **Generate** an answer grounded in the actual text.

The LLM is no longer a knowledge store — it is a reasoning engine over retrieved evidence.

### The Three Steps in Code
```
question ──▶ embed question ──▶ similarity search in FAISS ──▶ top-k chunks
                                                                    │
                                                                    ▼
                                          prompt = "Answer using: {chunks}\nQuestion: {q}"
                                                                    │
                                                                    ▼
                                                               LLM answer
```

---

## 2. PyTorch — What You Actually Use It For

You will not write raw PyTorch training loops. Instead, PyTorch runs **under the hood**
of the `sentence-transformers` library when you compute embeddings.

### Key Concept: Tensors
A tensor is a multi-dimensional array — the same idea as a NumPy array, but it can run on
a GPU and supports automatic differentiation.

```python
import torch
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2x2 tensor
print(t.shape)   # torch.Size([2, 2])
```

### Key Concept: Embeddings
An embedding model is a neural network that turns text into a fixed-size vector of floats.
Similar texts have vectors that point in similar directions (measured by cosine similarity).

```python
# sentence-transformers wraps a PyTorch model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode("Neural networks are great")
print(vec.shape)  # (384,)  — a 384-dimensional vector
```

### What to Pay Attention To
- `device` — whether the model runs on CPU (`"cpu"`) or GPU (`"cuda"`).
- `torch.no_grad()` — turns off gradient tracking during inference (faster, less memory).

---

## 3. LangChain — The Building Blocks

LangChain provides pre-built components that snap together like Lego bricks.

### Document Loaders
Load raw content from files into `Document` objects (text + metadata).
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("papers/paper.pdf")
docs = loader.load()          # list of Document objects
print(docs[0].page_content)   # raw text of page 1
print(docs[0].metadata)       # {"source": "...", "page": 0}
```

### Text Splitters
Split large documents into smaller chunks so that each chunk fits in the LLM context window and is semantically focused.

#### 1. CharacterTextSplitter vs. RecursiveCharacterTextSplitter

* **`CharacterTextSplitter`**: 
  * **Mechanism**: Splits strictly on a single defined separator (default is `"\n\n"`). It does not recursively split further if a chunk is larger than the target `chunk_size` unless that separator is found.
  * **Use Case**: Best for simple documents with a predictable, uniform structure (e.g., records separated by custom tags, tab-separated values, or double newlines).
* **`RecursiveCharacterTextSplitter` (Recommended for general text)**:
  * **Mechanism**: Attempts to split by a list of separators in order (default: `["\n\n", "\n", " ", ""]`). It recursively moves down the list to split text until individual chunks are below `chunk_size`, trying to keep paragraphs, sentences, and words intact.
  * **Use Case**: Excellent for articles, research papers, and books where you want to maintain semantic coherence.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = splitter.split_documents(docs)
```
`chunk_overlap` lets adjacent chunks share some context so semantic meaning isn't lost at boundaries.

#### 2. Structured Document Splitters

When dealing with structured formats, plain text splitting can destroy the hierarchy or syntax. Use these specialized splitters instead:

* **MarkdownHeaderTextSplitter / HTMLHeaderTextSplitter**:
  * Splits documents by header tags (e.g., `#`, `##` or `<h1>`, `<h2>`).
  * Instead of just splitting text, they add the header paths as metadata (e.g., `{"Header 1": "Intro", "Header 2": "Background"}`) to each chunk, helping the LLM keep track of the section context.
* **RecursiveJsonSplitter**:
  * Splits nested JSON structures while maintaining valid JSON format in each chunk.
  * Crucial for RAG over structured API responses or configurations.
* **Language-Specific Splitters** (e.g., Python, C++, HTML, etc.):
  * Splits code according to syntax (class definitions, function scopes).
  * Ideal for building code-understanding agents.

### Embeddings
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```
This wraps the sentence-transformers PyTorch model in the LangChain interface.

### Vector Stores (FAISS)
FAISS (Facebook AI Similarity Search) stores vectors and answers nearest-neighbor queries
extremely fast, entirely in memory / on disk — no server needed.
```python
from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore/")      # save to disk
db = FAISS.load_local("vectorstore/", embeddings)  # load later
```

### Retrievers
A retriever wraps a vector store and exposes a `get_relevant_documents(query)` method.
```python
retriever = db.as_retriever(search_kwargs={"k": 4})  # top 4 chunks
results = retriever.invoke("What is attention?")
```

### Prompt Templates
Structure the prompt you send to the LLM.
```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
Context: {context}
Question: {question}
""")
```

### Chains (LCEL)
LangChain Expression Language (LCEL) chains components with the `|` pipe operator.
```python
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": chunks_text, "question": "What is attention?"})
```

---

## 4. LangGraph — Stateful Agent Graphs

### Why Not Just Use a Chain?
A chain is a straight pipeline: A → B → C. An agent needs to make decisions:
"Did I find enough information? Should I retry? Should I take a different path?"
LangGraph models this as a **graph** where nodes are functions and edges are transitions.

### Core Concepts

**State** — a typed dictionary that all nodes read from and write to.
```python
from typing import TypedDict, List
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    documents: List[Document]
    answer: str
```

**Nodes** — Python functions that take the current state and return an updated state.
```python
def retrieve_node(state: State) -> State:
    docs = retriever.invoke(state["question"])
    return {"documents": docs}   # only return what changed
```

**Edges** — define what runs after each node.
```python
from langgraph.graph import StateGraph, END

graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge("retrieve", "generate")  # retrieve always goes to generate
graph.add_edge("generate", END)
graph.set_entry_point("retrieve")
app = graph.compile()
```

**Conditional Edges** — branch based on the state.
```python
def route(state: State) -> str:
    if not state["documents"]:
        return "no_results"
    return "generate"

graph.add_conditional_edges("retrieve", route, {
    "generate": "generate",
    "no_results": "no_results_node"
})
```

---

## 5. Required Libraries

Add these to your `requirements.txt`:

```
torch                          # PyTorch (CPU version is fine to start)
sentence-transformers          # Embedding models backed by PyTorch
langchain                      # Core LangChain
langchain-community            # Loaders, FAISS wrapper, HuggingFace embeddings
langchain-core                 # LCEL, base classes
langchain-openai               # OpenAI LLM (or use langchain-ollama for local LLMs)
langgraph                      # Agent graph framework
faiss-cpu                      # FAISS vector store (CPU build)
pypdf                          # PDF parsing backend for PyPDFLoader
python-dotenv                  # Load API keys from .env file
```

> **Note:** If you have an NVIDIA GPU, replace `faiss-cpu` with `faiss-gpu` and install
> the CUDA version of PyTorch from pytorch.org.

---

## 6. Concept Map

```
Your Question
     │
     ▼
[Embed with PyTorch sentence-transformer model]
     │ query vector
     ▼
[FAISS similarity search]  ◀── built from chunks embedded during ingest
     │ top-k Document chunks
     ▼
[LangGraph: retrieve_node → generate_node → cite_node]
     │ augmented prompt
     ▼
[LLM via LangChain chain]
     │
     ▼
Answer + citations
```
