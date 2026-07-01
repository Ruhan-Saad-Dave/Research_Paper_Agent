# Project File Structure

This is the target structure you will build toward. You don't need all these files on day
one — create them as you reach each phase in the plan.

```
Research_Paper_Agent/
│
├── CLAUDE.md                  # Claude behavior rules (already created)
├── README.md                  # Brief project description
├── requirements.txt           # Python dependencies  [Phase 0]
├── .env                       # API keys (never commit this)  [Phase 0]
│
├── papers/                    # Drop your PDF research papers here
│   └── .gitkeep
│
├── vectorstore/               # FAISS index saved to disk (auto-created by ingest.py)
│   └── .gitkeep
│
├── ingest.py                  # PDF → chunks → embeddings → FAISS  [Phase 1]
├── retrieve.py                # Simple retrieval chain (no agent)  [Phase 2]
├── agent.py                   # LangGraph graph definition  [Phase 3]
└── main.py                    # Entry point, CLI loop  [Phase 4]
│
└── docs/                      # All documentation (this folder)
    ├── 00_project_overview.md
    ├── 01_plan_of_action.md
    ├── 02_learning_concepts.md
    └── 03_file_structure.md   ← you are here
```

---

## File Responsibilities

### `ingest.py`
- Takes a PDF path as input.
- Loads, splits, embeds, and saves to `vectorstore/`.
- Can be run standalone: `python ingest.py papers/paper.pdf`

### `retrieve.py`  *(temporary — replaced by agent in Phase 3)*
- Loads `vectorstore/`.
- Builds a `RetrievalQA` chain.
- Accepts a question from the command line and prints the answer.

### `agent.py`
- Defines the `State` TypedDict.
- Defines all LangGraph node functions.
- Builds and compiles the `StateGraph`.
- Exports the compiled `app` for use by `main.py`.

### `main.py`
- Parses command-line arguments.
- Calls `ingest.py` logic if needed.
- Imports `app` from `agent.py`.
- Runs the interactive question-answer loop.

---

## What NOT to Put in Git

Add a `.gitignore` with at least:
```
venv/
vectorstore/
.env
__pycache__/
*.pyc
```

Never commit your `.env` file — it contains your API key.
