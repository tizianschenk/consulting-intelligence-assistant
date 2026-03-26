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