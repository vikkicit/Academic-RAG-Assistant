"""
AI Academic Assistant - Stable Version with Lazy Initialization

Architecture: RAG components load ONLY when needed.
- Streamlit starts instantly (no heavy imports at startup)
- Components initialize on-demand (upload, query, test)
- Full error handling with user feedback
- CPU-only mode (no GPU/CUDA)
"""

import os
import streamlit as st
import logging
from datetime import datetime
from typing import Optional, Dict, List

# ========== CRITICAL: FORCE CPU MODE ==========
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# ============================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure directories
UPLOAD_FOLDER = "uploads"
VECTORSTORE_FOLDER = "vectorstore"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .success-box { background-color: #d4edda; padding: 1rem; border-radius: 0.5rem; }
    .source-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
    </style>
""", unsafe_allow_html=True)

# ========== LAZY LOADING FUNCTIONS ==========
# These ONLY execute when explicitly called

@st.cache_resource
def load_embeddings_manager():
    """Lazy load embeddings (only when first needed)."""
    try:
        from utils.embeddings import EmbeddingsManager
        logger.info("Loading embeddings manager...")
        manager = EmbeddingsManager(model_name="all-MiniLM-L6-v2")
        logger.info("✅ Embeddings manager loaded")
        return manager
    except Exception as e:
        logger.error(f"❌ Error loading embeddings: {e}")
        st.error(f"❌ Embeddings Error: {e}")
        return None


@st.cache_resource
def load_rag_pipeline():
    """Lazy load RAG pipeline (only when first needed)."""
    try:
        from utils.rag_pipeline import RAGPipeline
        logger.info("Initializing RAG pipeline...")
        
        pipeline = RAGPipeline(
            embeddings_model="all-MiniLM-L6-v2",
            llm_type="ollama",
            llm_model="tinyllama",  # Using tinyllama for stability
            vectorstore_path=VECTORSTORE_FOLDER
        )
        
        # Try to load existing FAISS index
        try:
            if pipeline.load_index():
                st.success(f"✅ Loaded existing index with {pipeline.get_stats()['total_documents']} documents")
            else:
                st.info("📌 No existing index found. Upload documents to create one.")
        except Exception as e:
            logger.warning(f"⚠️ Could not load FAISS index: {e}")
            st.warning(f"⚠️ FAISS Warning: {e}")
        
        logger.info("✅ RAG pipeline loaded")
        return pipeline
    
    except Exception as e:
        logger.error(f"❌ Error loading RAG pipeline: {e}")
        st.error(f"❌ RAG Pipeline Error: {e}")
        return None


def get_uploaded_files() -> List[str]:
    """Get list of uploaded PDFs."""
    if os.path.exists(UPLOAD_FOLDER):
        return [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
    return []


def process_pdf_to_rag(file_path: str, pipeline) -> Dict:
    """Process PDF: extract → chunk → embed → store."""
    try:
        from utils.pdf_reader import PDFReader
        from utils.text_splitter import TextSplitter
        
        logger.info(f"Processing PDF: {file_path}")
        
        # Step 1: Extract text
        pdf_reader = PDFReader()
        text, metadata = pdf_reader.extract_text_from_pdf(file_path)
        logger.info(f"✅ Extracted {len(text)} characters")
        
        # Step 2: Split into chunks
        splitter = TextSplitter(chunk_size=512, overlap=100)
        chunks = splitter.split_by_paragraphs(text)
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        # Step 3: Create metadata
        chunk_metadata = [
            {
                "source": metadata.get("file_name", "unknown"),
                "page_count": metadata.get("total_pages", "?"),
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        
        # Step 4: Add to RAG pipeline
        pipeline.add_documents(chunks, chunk_metadata)
        pipeline.save_index()
        logger.info(f"✅ Documents added to RAG pipeline")
        
        return {
            "success": True,
            "message": f"✅ Processed {metadata.get('file_name')}",
            "chunks": len(chunks),
            "file_size_kb": metadata.get("file_size_kb", 0)
        }
    
    except Exception as e:
        logger.error(f"❌ Error processing PDF: {e}")
        return {
            "success": False,
            "message": f"❌ Error: {str(e)}"
        }


# ========== MAIN UI ==========

st.markdown("# 📚 AI Academic Assistant")
st.markdown("*Your personalized RAG-powered academic learning companion*")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_loaded" not in st.session_state:
    st.session_state.rag_loaded = False

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Status indicator
    if st.session_state.rag_loaded:
        st.success("✅ RAG Ready")
    else:
        st.info("⏳ RAG Not Loaded (will load on first use)")
    
    st.divider()
    
    # Tab selection
    tab = st.radio(
        "Navigation:",
        ["📝 Chat", "📤 Upload Documents", "📊 Info"],
        label_visibility="collapsed"
    )

# ========== TAB 1: CHAT ==========
if tab == "📝 Chat":
    st.markdown("## Ask Questions About Your Documents")
    
    uploaded_files = get_uploaded_files()
    
    if not uploaded_files:
        st.warning("📌 Please upload PDF documents first!")
        st.info("Go to **📤 Upload Documents** tab to get started.")
    else:
        # Load RAG only when user is in chat tab
        if not st.session_state.rag_loaded:
            with st.spinner("⏳ Loading RAG system..."):
                pipeline = load_rag_pipeline()
                if pipeline:
                    st.session_state.rag_loaded = True
                    st.session_state.pipeline = pipeline
                else:
                    st.error("❌ Failed to load RAG system")
                    st.stop()
        
        pipeline = st.session_state.pipeline
        stats = pipeline.get_stats()
        
        # Display documents count
        st.info(f"📚 Knowledge base: {stats['total_documents']} documents")
        
        # Display chat history
        st.markdown("### Conversation")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
        
        # Input area
        st.divider()
        
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            user_question = st.text_input(
                "Ask a question:",
                placeholder="E.g., What is deadlock in OS?",
                key="user_input"
            )
        
        with col2:
            submit_button = st.button("🚀 Ask", use_container_width=True)
        
        # Process question
        if submit_button and user_question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            with st.spinner("🤔 Generating answer..."):
                try:
                    result = pipeline.generate_answer(
                        query=user_question,
                        top_k=5,
                        temperature=0.7
                    )
                    
                    answer = result["answer"]
                    confidence = result["confidence"]
                    sources = result["sources"]
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })
                    
                    # Display answer
                    st.success("✅ Answer Generated!")
                    st.markdown("### Answer")
                    st.write(answer)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        confidence_pct = f"{confidence:.0%}"
                        st.metric("Confidence", confidence_pct)
                    with col2:
                        st.metric("Sources", result['retrieved_docs'])
                    
                    # Show sources
                    if sources:
                        st.markdown("### 📚 Sources")
                        unique_sources = list(set(sources))
                        for source in unique_sources:
                            st.markdown(f"<div class='source-box'>📄 {source}</div>", unsafe_allow_html=True)
                    
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
                    logger.error(f"Error: {e}")

# ========== TAB 2: UPLOAD ==========
elif tab == "📤 Upload Documents":
    st.markdown("## Upload Study Materials")
    
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file:",
            type="pdf",
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Save file
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ Saved: {uploaded_file.name}")
            
            # Load RAG pipeline for processing
            with st.spinner("⏳ Loading RAG system..."):
                pipeline = load_rag_pipeline()
            
            if pipeline:
                # Process PDF
                with st.spinner("⚙️ Processing PDF..."):
                    result = process_pdf_to_rag(file_path, pipeline)
                
                if result["success"]:
                    st.success(result["message"])
                    st.info(f"📊 Created {result['chunks']} chunks")
                    st.balloons()
                    st.session_state.rag_loaded = True
                    st.session_state.pipeline = pipeline
                else:
                    st.error(result["message"])
            else:
                st.error("❌ Failed to load RAG system for processing")
    
    with col2:
        st.markdown("### 📋 Uploaded Files")
        files = get_uploaded_files()
        
        if files:
            for file in files:
                file_path = os.path.join(UPLOAD_FOLDER, file)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                st.caption(f"📄 {file} ({file_size_mb:.2f} MB)")
        else:
            st.caption("No files uploaded yet")

# ========== TAB 3: INFO ==========
elif tab == "📊 Info":
    st.markdown("## 📊 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("CPU Mode", "✅ Enabled")
    with col2:
        st.metric("CUDA", "❌ Disabled")
    with col3:
        st.metric("Files Uploaded", len(get_uploaded_files()))
    
    st.divider()
    
    st.markdown("### 🧠 RAG Architecture")
    st.write("""
    **Pipeline Components:**
    1. **PDF Reader** - Extract text from PDFs
    2. **Text Splitter** - Chunk text with overlap
    3. **Embeddings** - Sentence Transformers (CPU mode)
    4. **FAISS** - Vector similarity search
    5. **LLM** - Ollama Llama3 (local)
    
    **Processing Flow:**
    PDF → Extract → Chunk → Embed → FAISS → Retrieve → LLM → Answer
    """)
    
    st.divider()
    
    st.markdown("### 🛠️ Troubleshooting")
    
    if st.button("🔄 Clear All Data"):
        import shutil
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        if os.path.exists(VECTORSTORE_FOLDER):
            shutil.rmtree(VECTORSTORE_FOLDER)
            os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)
        st.session_state.chat_history = []
        st.session_state.rag_loaded = False
        st.success("✅ All data cleared")
        st.rerun()
    
    if st.button("🧪 Test Components"):
        st.info("Testing embeddings...")
        emb = load_embeddings_manager()
        if emb:
            test_embedding = emb.generate_single_embedding("test")
            st.success(f"✅ Embeddings working (dimension: {len(test_embedding)})")
        
        st.info("Testing RAG pipeline...")
        pipeline = load_rag_pipeline()
        if pipeline:
            stats = pipeline.get_stats()
            st.success(f"✅ RAG pipeline working ({stats['total_documents']} docs)")
