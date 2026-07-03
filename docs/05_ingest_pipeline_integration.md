# Ingest Pipeline Integration Guide

This guide explains how to connect your PDF loading method in `ingest.py` with your text splitters in `main.py` (or ingestion scripts) to preserve page numbers and prepare documents for vector storage.

---

## 1. The Components

* **`ingest.py`**: Contains `ingest_pdf()`, which initializes `PyPDFLoader` and returns a lazy document generator (`lazy_load()`).
* **`main.py`**: The entry point where you pull everything together—loading papers, splitting them into chunks, and (eventually) embedding them into FAISS.

---

## 2. Integration Pattern

Because `lazy_load()` yields documents one-by-one to conserve memory, you must consume the generator (typically by converting it to a `list`) before passing it to the text splitter.

Here is the complete integration code:

```python
from ingest import ingest_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    # 1. Path to the research paper
    pdf_path = "papers/example_paper.pdf"
    
    print(f"Loading PDF from: {pdf_path}")
    
    # 2. Call the loader and convert the lazy generator to a list
    # loader.lazy_load() yields Document objects one-by-one.
    # We turn it into a list because the splitter needs the full list of pages.
    pages = list(ingest_pdf(pdf_path))
    print(f"Loaded {len(pages)} pages successfully.")

    # 3. Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Maximum character length per chunk
        chunk_overlap=100,    # Word overlap at chunk boundaries
        length_function=len
    )

    # 4. Split the page documents
    # The splitter processes each page and inherits the original page metadata
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} pages into {len(chunks)} chunks.")

    # 5. Verify metadata propagation
    if chunks:
        print("\n--- Example Chunk Verification ---")
        print(f"Source Document: {chunks[0].metadata['source']}")
        print(f"Source Page Num: {chunks[0].metadata['page']} (0-indexed)")
        print(f"Content Preview:\n{chunks[0].page_content[:150]}...")

if __name__ == "__main__":
    main()
```

---

## 3. How Metadata Behaves Under the Hood

When you load a document using `PyPDFLoader`, LangChain sets the metadata automatically:
```python
Document(
    page_content="...",
    metadata={
        "source": "papers/example_paper.pdf",
        "page": 0
    }
)
```

When you call `text_splitter.split_documents(pages)`:
1. The splitter splits `page_content` into smaller parts (chunks) based on the separator list.
2. For every split chunk generated from a page, the splitter copies the `metadata` dictionary of the parent page.
3. This guarantees that your final vector database contains chunks that know exactly which document and page they came from, allowing the Q&A agent to cite references.
