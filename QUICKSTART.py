"""
Quick Start Guide for AI Academic Assistant

Follow these steps to get started:
"""

# STEP 1: INSTALL DEPENDENCIES
# ============================
# pip install -r requirements.txt

# STEP 2: SETUP OLLAMA (For Local LLM)
# ====================================
# Download from: https://ollama.ai
# 
# In terminal 1 - Start Ollama:
#   ollama serve
#
# In terminal 2 - Download a model:
#   ollama pull llama2
#   # or: ollama pull mistral
#   # or: ollama pull neural-chat

# STEP 3: CONFIGURE (OPTIONAL)
# =============================
# Edit .env file:
#   LLM_TYPE=ollama
#   LLM_MODEL=llama2
#   OLLAMA_BASE_URL=http://localhost:11434

# STEP 4: RUN THE APP
# ===================
# streamlit run app.py

# STEP 5: USE THE APP
# ===================
# 1. Go to http://localhost:8501
# 2. Upload PDFs in "📤 Upload Documents"
# 3. Ask questions in "📝 Chat"
# 4. View stats in "📊 Knowledge Base"

print("""
╔════════════════════════════════════════════════════════════╗
║  🚀 AI Academic Assistant - Quick Start                   ║
╚════════════════════════════════════════════════════════════╝

✅ INSTALLATION CHECKLIST:

1. [ ] Install dependencies:
       pip install -r requirements.txt

2. [ ] Install & start Ollama:
       - Download: https://ollama.ai
       - Terminal 1: ollama serve
       - Terminal 2: ollama pull llama2

3. [ ] Configure (optional):
       cp .env.example .env
       # Edit .env if needed

4. [ ] Run the app:
       streamlit run app.py

5. [ ] Open browser:
       http://localhost:8501

═════════════════════════════════════════════════════════════

📂 PROJECT STRUCTURE:

app.py                 → Main Streamlit application
config.py              → Configuration settings
requirements.txt       → Python dependencies
.env                   → Environment variables (create from .env.example)

utils/
  ├── pdf_reader.py    → PDF extraction
  ├── text_splitter.py → Text chunking
  ├── embeddings.py    → Embedding generation
  ├── llm_model.py     → LLM interfaces
  └── rag_pipeline.py  → RAG orchestration

uploads/               → Your uploaded PDFs
vectorstore/           → FAISS vector index

═════════════════════════════════════════════════════════════

🎯 USAGE:

1. Upload PDFs with your study materials
2. Ask questions about the materials
3. Get AI-powered answers with sources
4. Check confidence scores and citations

═════════════════════════════════════════════════════════════

💡 TIPS:

- For faster responses: Use 'mistral' model
- For better accuracy: Use 'llama2' (default)
- If slow: Reduce RAG_TOP_K in config.py
- If memory issues: Use CPU FAISS (default)

═════════════════════════════════════════════════════════════

❓ TROUBLESHOOTING:

Problem: "Ollama connection error"
Solution: Make sure Ollama is running (ollama serve)

Problem: "Slow responses"
Solution: Use mistral model or reduce chunk count

Problem: "Out of memory"
Solution: Reduce LLM_MODEL context or use smaller embeddings

═════════════════════════════════════════════════════════════

📚 Learn More:
- Ollama: https://ollama.ai
- Streamlit: https://streamlit.io
- FAISS: https://faiss.ai
- Sentence Transformers: https://www.sbert.net

═════════════════════════════════════════════════════════════
""")
