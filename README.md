# 📚 AI Academic Assistant

An intelligent RAG (Retrieval-Augmented Generation) system that helps students learn from their study materials using AI-powered question answering.

## 🎯 Features

✨ **Core Features:**
- 📤 **PDF Upload & Processing** - Upload study materials (lecture notes, textbooks, etc.)
- 🔍 **Semantic Search** - Intelligent document retrieval using embeddings
- 🤖 **AI-Powered Answers** - Get answers from your materials using Llama3
- 💬 **Chat Interface** - Interactive conversation with your study assistant
- 📚 **Source Citations** - See which documents your answers come from
- 📊 **Confidence Scores** - Understand answer reliability
- 🔄 **Context Preservation** - Overlapping chunks maintain meaning

## 🏗️ Architecture

### System Flow (9 Phases)

```
1️⃣  USER UPLOADS PDF
    ↓
2️⃣  PDF READER EXTRACTS TEXT
    ↓
3️⃣  TEXT SPLITTING (CHUNKING)
    ↓
4️⃣  EMBEDDINGS GENERATION
    ↓
5️⃣  FAISS VECTOR DATABASE
    ↓
6️⃣  USER ASKS QUESTION
    ↓
7️⃣  RETRIEVER (Smart Librarian)
    ↓
8️⃣  LLAMA3 GENERATION
    ↓
9️⃣  FINAL RESPONSE
```

### Technology Stack

- **Frontend:** Streamlit (Interactive Web UI)
- **PDF Processing:** pdfplumber, PyPDF2
- **Text Embedding:** Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database:** FAISS (Fast similarity search)
- **LLM:** Llama3 (via Ollama) or OpenAI API
- **Backend:** Python with modular architecture

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama installed ([Download](https://ollama.ai)) - for local Llama3
  - OR OpenAI API key for cloud-based LLM

### Installation

1. **Clone/Setup the project:**
```bash
cd AI_Academic_Assistant
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Setup configuration:**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Start Ollama (if using local LLM):**
```bash
ollama serve
# In another terminal:
ollama pull llama2
# or: ollama pull mistral
# or: ollama pull neural-chat
```

6. **Run the application:**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Documents
1. Go to "📤 Upload Documents" tab
2. Click "Choose a PDF file"
3. Select your study material (lecture notes, textbook chapters, etc.)
4. Wait for processing (extraction → chunking → embedding)

### 2. Ask Questions
1. Go to "📝 Chat" tab
2. Type your question: "What is deadlock?"
3. Click "🚀 Ask"
4. Get AI-powered answer with sources

### 3. View Statistics
1. Go to "📊 Knowledge Base" tab
2. See document count, embedding details, upload history
3. Clear chat history if needed

## 🛠️ Project Structure

```
AI_Academic_Assistant/
│
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore file
│
├── utils/
│   ├── __init__.py
│   ├── pdf_reader.py      # PDF extraction
│   ├── text_splitter.py   # Text chunking
│   ├── embeddings.py      # Embedding generation
│   ├── llm_model.py       # LLM interfaces
│   └── rag_pipeline.py    # RAG orchestration
│
├── uploads/               # Uploaded PDFs storage
├── vectorstore/           # FAISS index storage
├── data/                  # Data files
├── models/                # Model cache
└── README.md              # This file
```

## 🔧 Configuration

Edit `config.py` or `.env` to customize:

```python
# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast & lightweight
# EMBEDDING_MODEL = "all-mpnet-base-v2"  # More accurate but slower

# LLM Settings
LLM_TYPE = "ollama"  # or "openai"
LLM_MODEL = "llama2"  # or "mistral", "neural-chat"
LLM_TEMPERATURE = 0.7  # 0 = deterministic, 1 = creative

# RAG Settings
RAG_TOP_K = 5           # Retrieve 5 most relevant chunks
RAG_CHUNK_SIZE = 512    # Chunk size in characters
RAG_OVERLAP = 100       # Overlap between chunks
```

## 📊 Supported Models

### Embedding Models
- `all-MiniLM-L6-v2` ⭐ **Default** (Fast, 384-dim, good quality)
- `all-mpnet-base-v2` (More accurate, 768-dim, slower)
- `distiluse-base-multilingual-cased-v2` (Multilingual)

### LLM Models (Ollama)
- `llama2` ⭐ **Default** (General purpose)
- `mistral` (Faster, good quality)
- `neural-chat` (Conversational)
- `dolphin-mixtral` (Advanced reasoning)

### LLM Models (OpenAI)
- `gpt-3.5-turbo` (Fast, cost-effective)
- `gpt-4` (Most capable)

## 🎓 How RAG Works

### Without RAG:
```
LLM alone has general knowledge
→ May give outdated or irrelevant answers
→ Doesn't know your specific notes
```

### With RAG:
```
1. Your PDF uploaded
2. Text extracted and chunked
3. Converted to semantic embeddings
4. Stored in FAISS index
5. Question also converted to embedding
6. FAISS finds most similar chunks
7. LLM reads relevant chunks + question
8. Generates accurate, sourced answer
```

**Result:** Personalized, context-aware answers! 🎯

## 💡 Tips for Best Results

1. **Quality Materials:** Upload clear, well-structured PDFs
2. **Specific Questions:** Ask detailed questions for better answers
3. **Chunk Size:** Default 512 chars good balance (adjust in config.py)
4. **Confidence:** Higher score = more reliable answer
5. **Multiple Uploads:** Build knowledge base with multiple materials
6. **Check Sources:** Always verify retrieved documents

## 🐛 Troubleshooting

### Ollama Connection Error
```bash
# Make sure Ollama is running:
ollama serve

# Pull a model:
ollama pull llama2
```

### CUDA/GPU Issues
```bash
# Use CPU version of FAISS:
pip install faiss-cpu  # (Already in requirements.txt)
```

### Out of Memory
```python
# Reduce in config.py:
LLM_TEMPERATURE = 0.3  # Lower = simpler answers
RAG_TOP_K = 3          # Retrieve fewer chunks
RAG_CHUNK_SIZE = 256   # Smaller chunks
```

### Slow Response
```python
# Use faster models in config.py:
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fastest
LLM_MODEL = "mistral"  # Faster than llama2
```

## 📈 Performance Optimization

| Component | Optimization |
|-----------|---------------|
| Embeddings | Use MiniLM (384-dim, fastest) |
| Vector Search | FAISS with GPU support (optional) |
| LLM | Mistral > Llama2 > Mixtral (speed-wise) |
| Chunks | 256-1024 chars optimal |

## 🔐 Security & Privacy

- ✅ All processing done locally (if using Ollama)
- ✅ PDFs stored in `uploads/` directory
- ✅ Vector index stored in `vectorstore/`
- ✅ No data sent to external servers (except OpenAI if used)
- ✅ Chat history stored in session memory only

## 📝 Example Use Cases

1. **Study for Exams** → Upload lecture notes → Ask practice questions
2. **Clarify Concepts** → Upload textbook → Ask for explanations
3. **Research** → Upload multiple papers → Cross-reference topics
4. **Homework Help** → Upload course materials → Ask specific questions

## 🚀 Advanced Features (Future)

- [ ] Quiz generation from materials
- [ ] Multi-language support
- [ ] Document summarization
- [ ] Citation formatting (APA, MLA)
- [ ] Offline mode with pre-downloaded models
- [ ] Collaborative knowledge bases
- [ ] Performance analytics

## 📚 Learning Resources

- [RAG Explained](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [Ollama Models](https://ollama.ai/library)
- [Streamlit Docs](https://docs.streamlit.io/)

## 🤝 Contributing

Contributions welcome! Areas:
- Better prompting strategies
- Additional LLM support
- UI/UX improvements
- Performance optimization
- Documentation

## 📄 License

This project is provided as-is for educational purposes.

## ⚡ Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama serve

# Run the app
streamlit run app.py

# Clear cache
rm -rf vectorstore/ uploads/
```

---

**Made with ❤️ for students and learners** 📚✨
