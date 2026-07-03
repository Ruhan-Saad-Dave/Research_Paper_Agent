# PDF Splitting Strategies: Page-Level vs. Structure-Aware

When building a RAG pipeline for research papers, how you chunk the PDFs directly impacts retrieval accuracy. Below are the two primary approaches, their implementations, and their trade-offs.

---

## Approach 1: Page-Level Chunking (Standard & Simple)

This approach uses a standard PDF parser to extract text page-by-page. The resulting documents are then split using a recursive text splitter.

### How it Works
1. `PyPDFLoader` loads the PDF, yielding one `Document` per page.
2. Each page document contains metadata: `{"source": "path/to/paper.pdf", "page": <page_num>}`.
3. `RecursiveCharacterTextSplitter` breaks down each page into smaller chunks, automatically carrying over the source and page number metadata.

### Implementation Example

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf_page_level(file_path: str):
    # 1. Load the PDF page by page
    loader = PyPDFLoader(file_path)
    pages = loader.load() # Returns list of Documents
    
    # 2. Initialize recursive splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    
    # 3. Split the pages into chunks
    chunks = text_splitter.split_documents(pages)
    
    # 4. Preview metadata propagation
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i} ---")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content Preview: {chunk.page_content[:100]}...\n")
        
    return chunks
```

### Trade-offs
* **Pros:**
  * Very easy to implement.
  * No complex external layout parsing dependencies.
  * Tells you exactly what page of the paper the information is on (great for citations).
* **Cons:**
  * **No Section Context:** The model doesn't know if a chunk belongs to the *Introduction*, *Methodology*, or *Conclusion*.
  * **Page Boundary Splits:** If a paragraph spans from the bottom of page 2 to the top of page 3, it is loaded as two separate documents and might get split in a way that breaks semantic flow.

---

## Approach 2: Structure-Aware Header Chunking (Advanced)

This approach converts the layout of the PDF into formatted Markdown first, then splits hierarchically using headers.

### How it Works
1. A layout-aware parser (e.g., `pymupdf4llm`) parses the PDF's visual layout (titles, headers, tables) and converts it to a unified Markdown string.
2. `MarkdownHeaderTextSplitter` splits this Markdown string by header levels (`#`, `##`, `###`), attaching the section titles to the metadata.
3. `RecursiveCharacterTextSplitter` splits those sections into model-sized chunks, preserving the header metadata.

### Implementation Example

First, ensure you have the parser library installed:
```bash
pip install pymupdf4llm
```

Here is the implementation:

```python
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def process_pdf_structure_aware(file_path: str):
    # 1. Convert PDF layout to clean Markdown text
    # pymupdf4llm preserves tables, bold headers, and lists
    markdown_text = pymupdf4llm.to_markdown(file_path)
    
    # 2. Define markdown headers to split on
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3")
    ]
    
    # 3. Split strictly by header structure
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    section_docs = header_splitter.split_text(markdown_text)
    
    # 4. Recursively split section documents into retrieval-sized chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(section_docs)
    
    # 5. Preview metadata propagation
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i} ---")
        print(f"Metadata: {chunk.metadata}")
        print(f"Content Preview: {chunk.page_content[:100]}...\n")
        
    return chunks
```

### Trade-offs
* **Pros:**
  * **Semantic Context:** Every chunk retains its section hierarchy (e.g. `{'Header_1': '3. Proposed Method', 'Header_2': '3.2 Loss Function'}`).
  * **Table Preservation:** Markdown parsers convert PDF tables into markdown table strings, preserving data structure.
  * **No Page Boundary Splits:** Text flows naturally across what used to be physical page breaks.
* **Cons:**
  * Higher computational overhead (parsing layout takes longer than plain text extraction).
  * Requires extra dependencies.
  * You lose the physical "page number" citation unless the parser specifically embeds page tags.

---

## Approach 3: The Hybrid Stateful Header Tracking (Best of Both Worlds)

If you need the RAG agent to provide a precise citation (e.g. *"Found in Page 4, Section 3.2"*), you can use a stateful parsing loop that tracks the page numbers and active headers simultaneously.

### How it Works
1. Use `pymupdf4llm.to_markdown(file_path, page_chunks=True)` to get page-by-page Markdown text.
2. Keep a running state of the current active headers (`Header_1`, `Header_2`, etc.) as you loop through the document.
3. For each page, parse the text line-by-line to see if a new header is introduced. If so, update the active header state.
4. Split the page's text into small chunks and inject both the page number and the active header state into each chunk's metadata.

### Implementation Example

```python
import re
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf_hybrid(file_path: str):
    # 1. Extract markdown text page-by-page (1-based page numbers in metadata)
    pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    
    # State tracker to remember what section we are in
    current_headers = {
        "Header_1": "Abstract",
        "Header_2": "",
        "Header_3": ""
    }
    
    # Regex to detect markdown headers
    h1_re = re.compile(r"^#\s+(.+)$")
    h2_re = re.compile(r"^##\s+(.+)$")
    h3_re = re.compile(r"^###\s+(.+)$")
    
    final_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["metadata"]["page_number"]
        
        # Scan page text line-by-line to update current headers
        for line in text.split("\n"):
            line = line.strip()
            if m := h1_re.match(line):
                current_headers["Header_1"] = m.group(1)
                current_headers["Header_2"] = ""  # Reset subheaders
                current_headers["Header_3"] = ""
            elif m := h2_re.match(line):
                current_headers["Header_2"] = m.group(1)
                current_headers["Header_3"] = ""
            elif m := h3_re.match(line):
                current_headers["Header_3"] = m.group(1)
                
        # Wrap page text in a Document object
        page_doc = Document(
            page_content=text,
            metadata={"page": page_num, "source": file_path}
        )
        
        # Split page into standard chunks
        page_splits = text_splitter.split_documents([page_doc])
        
        # Enrich metadata of each chunk with active headers
        for split in page_splits:
            if current_headers["Header_1"]:
                split.metadata["Header_1"] = current_headers["Header_1"]
            if current_headers["Header_2"]:
                split.metadata["Header_2"] = current_headers["Header_2"]
            if current_headers["Header_3"]:
                split.metadata["Header_3"] = current_headers["Header_3"]
            final_chunks.append(split)
            
    return final_chunks
```

### Trade-offs
* **Pros:**
  * **Absolute Context:** Every chunk contains both page numbers and section headers in its metadata.
  * **Precise Citations:** The agent can reference the page number AND the exact section name.
* **Cons:**
  * Needs custom header-tracking logic (regex/state machine).
  * If a header spans across pages, the tracker successfully remembers the header, but a section header that starts at the absolute bottom of a page might associate the page before it with that header.

---

## Comparison Matrix

| Feature | Approach 1: Page-Level | Approach 2: Structure-Aware | Approach 3: Hybrid Stateful |
| :--- | :--- | :--- | :--- |
| **Complexity** | Low | Medium-High | High |
| **Parsing Speed** | Fast | Moderate | Moderate |
| **Metadata Details** | Page number only | Section titles only | **Page number & Section titles** |
| **Best For** | Standard search engines | Question-answering over section layouts | Premium RAG agents requiring highly precise citations |
| **Table Quality** | Very Poor | Good | Good |
