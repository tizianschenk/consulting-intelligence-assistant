# consulting-intelligence-assistant
A production-style RAG + Agentic AI application for analyzing  consulting documents, built with LangChain, LangGraph, and Groq.

## Architecture
- **LLM**: Llama 3.3 70B via Groq API (free tier)
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2` (local, free)
- **Vector Store**: ChromaDB (local persistence)
- **Agent**: LangGraph multi-node agent with routing and self-evaluation
- **UI**: Streamlit

## Setup
```bash
git clone https://github.com/tizianschenk/consulting-intelligence-assistant
cd consulting-intelligence-assistant
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # Add GROQ_API_KEY
streamlit run app.py
```

### Ingestion pipeline

The ingestion pipeline converts raw PDF documents into searchable vector embeddings using a local sentence-transformer model called all-MiniLM-L6-v2, which maps text into a 384-dimensional mathematical space where similar meanings cluster together. I chose ChromaDB for local persistence because it requires zero infrastructure setup, and the architecture is identical to a production system — you'd simply swap ChromaDB for Pinecone without changing any other code."

## Read a PDF file like this in the terminal ( I use zsh): 
```python -c "
from ingestion.pdf_loader import load_pdf

docs = load_pdf('your_file_name.pdf')

# Look at the first page
print('--- PAGE 1 TEXT (first 500 chars) ---')
print(docs[0].page_content[:500])

print()
print('--- METADATA ---')
print(docs[0].metadata)

print()
print(f'--- TOTAL PAGES LOADED: {len(docs)} ---')
"
```

## Try the chunker functions in the terminal
```python -c "
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_documents, inspect_chunks

docs = load_pdf('your_file_name.pdf')
chunks = chunk_documents(docs)
inspect_chunks(chunks, n=3)
"
```
## Ask the PDF a quesion in the terminal like this

```python -c "
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import VectorStore

docs = load_pdf('your_file_name.pdf')
chunks = chunk_documents(docs)
store = VectorStore()
store.add_documents(chunks)

print()
print('--- SEARCHING: what is your question you want to ask the pdf? ---')
results = store.search('what is your question you want to ask the pdf?', n_results=2)

for i, doc in enumerate(results):
    print(f'Result {i+1} (similarity: {doc.metadata[\"similarity_score\"]})')
    print(doc.page_content[:300])
    print()
"
```