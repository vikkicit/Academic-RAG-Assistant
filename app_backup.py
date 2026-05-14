"""
AI Academic Assistant - Streamlit Main Application

A RAG-based academic assistant that helps students learn from their study materials.
Features:
- PDF Upload and Processing
- Semantic Search using FAISS
- AI-Powered Answers with Llama3
- Chat History Management
- Source Citations
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

# ========== CRITICAL: FORCE CPU MODE ==========
# Disable CUDA/GPU to avoid crashes
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# ============================================

import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config - MUST be first
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

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
    }
    .confidence-high {
        background-color: #90EE90;
        color: #006400;
    }
    .confidence-medium {
        background-color: #FFD700;
        color: #000;
    }
    .confidence-low {
        background-color: #FFB6C6;
        color: #8B0000;
    }
    </style>
""", unsafe_allow_html=True)

# Import custom modules - with try/except for better error handling
try:
    from utils.pdf_reader import PDFReader
    from utils.text_splitter import TextSplitter
    from utils.rag_pipeline import RAGPipeline
except ImportError as e:
    st.error(f"❌ Error importing modules: {e}")
    st.info("Make sure all files in utils/ folder are present")
    st.stop()


@st.cache_resource
def initialize_rag_pipeline():
    """Initialize RAG pipeline (cached)."""
    try:
        st.info("⏳ Initializing RAG Pipeline...")
        
        pipeline = RAGPipeline(
            embeddings_model="all-MiniLM-L6-v2",
            llm_type="ollama",
            llm_model="llama2",
            vectorstore_path=VECTORSTORE_FOLDER
        )
        
        # Try to load existing index
        pipeline.load_index()
        st.success("✅ RAG Pipeline initialized!")
        return pipeline
    
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        st.warning(f"⚠️ RAG Pipeline initialization failed: {e}")
        st.info("📌 Make sure Ollama is running: `ollama serve`")
        return None


def get_uploaded_files() -> List[str]:
    """Get list of uploaded PDF files."""
    if os.path.exists(UPLOAD_FOLDER):
        return [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.pdf')]
    return []


def process_uploaded_pdf(file_path: str, pipeline: RAGPipeline) -> Dict:
    """Process uploaded PDF and add to knowledge base."""
    try:
        logger.info(f"Processing PDF: {file_path}")
        
        # Extract text from PDF
        pdf_reader = PDFReader()
        text, metadata = pdf_reader.extract_text_from_pdf(file_path)
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        
        # Split into chunks
        splitter = TextSplitter(chunk_size=512, overlap=100)
        chunks = splitter.split_by_paragraphs(text)
        
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Create metadata for each chunk
        chunk_metadata = [
            {
                "source": metadata.get("file_name", "unknown"),
                "page_count": metadata.get("total_pages", "?"),
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        
        # Add to knowledge base
        pipeline.add_documents(chunks, chunk_metadata)
        pipeline.save_index()
        
        return {
            "success": True,
            "message": f"Successfully processed {metadata.get('file_name')}",
            "chunks": len(chunks),
            "file_size_kb": metadata.get("file_size_kb", 0)
        }
    
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return {
            "success": False,
            "message": f"Error processing PDF: {str(e)}"
        }


def get_confidence_badge(confidence: float) -> str:
    """Get HTML badge for confidence score."""
    if confidence >= 0.7:
        badge_class = "confidence-high"
        text = f"High Confidence ({confidence:.0%})"
    elif confidence >= 0.5:
        badge_class = "confidence-medium"
        text = f"Medium Confidence ({confidence:.0%})"
    else:
        badge_class = "confidence-low"
        text = f"Low Confidence ({confidence:.0%})"
    
    return f'<span class="confidence-badge {badge_class}">{text}</span>'


def main():
    """Main application."""
    
    # Header
    st.markdown("# 📚 AI Academic Assistant")
    st.markdown("*Your personalized learning companion powered by AI*")
    
    # Initialize session state
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = initialize_rag_pipeline()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    pipeline = st.session_state.rag_pipeline
    
    if pipeline is None:
        st.error("❌ Failed to initialize RAG pipeline. Please check your configuration.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Display stats
        stats = pipeline.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats["total_documents"])
        with col2:
            st.metric("Model Dim", stats["embedding_dimension"])
        
        st.divider()
        
        # Tab navigation
        tab = st.radio(
            "Select Tab:",
            ["📝 Chat", "📤 Upload Documents", "📊 Knowledge Base"],
            label_visibility="collapsed"
        )
    
    # Main content area
    if tab == "📝 Chat":
        st.markdown("## Ask Your Study Questions")
        
        if stats["total_documents"] == 0:
            st.warning("📌 Please upload study materials first to get started!")
        else:
            # Chat interface
            st.markdown("### Conversation")
            
            # Display chat history
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
                    "Ask a question about your study materials:",
                    placeholder="E.g., What is deadlock in operating systems?",
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
                
                with st.spinner("🤔 Thinking..."):
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
                        
                        # Display confidence and sources
                        col1, col2 = st.columns([0.3, 0.7])
                        
                        with col1:
                            st.markdown(get_confidence_badge(confidence), unsafe_allow_html=True)
                        
                        with col2:
                            st.caption(f"📌 Retrieved from {result['retrieved_docs']} sources")
                        
                        # Show sources
                        if sources:
                            st.markdown("### 📚 Sources")
                            unique_sources = list(set(sources))
                            for source in unique_sources:
                                st.markdown(f"<div class='source-box'>📄 {source}</div>", unsafe_allow_html=True)
                        
                        # Rerun to show updated chat
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
                        logger.error(f"Error: {e}")
    
    elif tab == "📤 Upload Documents":
        st.markdown("## Upload Your Study Materials")
        
        col1, col2 = st.columns([0.6, 0.4])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type="pdf",
                accept_multiple_files=False
            )
            
            if uploaded_file is not None:
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                
                with st.spinner("📥 Uploading..."):
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                with st.spinner("⚙️ Processing PDF..."):
                    result = process_uploaded_pdf(file_path, pipeline)
                
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    st.info(f"📊 Processed into {result['chunks']} chunks")
                    st.balloons()
                else:
                    st.error(f"❌ {result['message']}")
        
        with col2:
            st.markdown("### 📋 Uploaded Files")
            uploaded_files = get_uploaded_files()
            
            if uploaded_files:
                for file in uploaded_files:
                    file_path = os.path.join(UPLOAD_FOLDER, file)
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    st.caption(f"📄 {file} ({file_size_mb:.2f} MB)")
            else:
                st.caption("No files uploaded yet")
    
    elif tab == "📊 Knowledge Base":
        st.markdown("## Knowledge Base Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", stats["total_documents"])
        with col2:
            st.metric("Embedding Dim", stats["embedding_dimension"])
        with col3:
            st.metric("Model", stats["embedding_model"].split("/")[-1])
        with col4:
            st.metric("Chat Messages", len(st.session_state.chat_history))
        
        st.divider()
        
        st.markdown("### 📁 Uploaded Documents")
        uploaded_files = get_uploaded_files()
        
        if uploaded_files:
            df_data = []
            for file in uploaded_files:
                file_path = os.path.join(UPLOAD_FOLDER, file)
                file_size_kb = os.path.getsize(file_path) / 1024
                upload_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                df_data.append({
                    "File Name": file,
                    "Size (KB)": f"{file_size_kb:.2f}",
                    "Uploaded": upload_time.strftime("%Y-%m-%d %H:%M")
                })
            
            import pandas as pd
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        st.markdown("### 🧹 Maintenance")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("✅ Chat history cleared")
            st.rerun()


if __name__ == "__main__":
    main()
