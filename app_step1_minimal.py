"""
AI Academic Assistant - Minimal Frontend Test (STEP 1)

Purpose: Test Streamlit stability BEFORE adding RAG components.
No embeddings, FAISS, or Ollama initialization during startup.
"""

import os
import streamlit as st

# Force CPU mode globally
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# MINIMAL UI - FRONTEND ONLY (NO RAG YET)
# ============================================

st.markdown("# 📚 AI Academic Assistant")
st.markdown("*Stable RAG system for academic learning*")

# Test message
st.success("✅ Frontend is working correctly!")
st.info("📌 RAG components will be initialized on demand (not at startup)")

# Simple sidebar
with st.sidebar:
    st.markdown("## 🧪 Debug Info")
    st.write("**Status:** Frontend loaded ✅")
    st.write("**Rendering:** Stable")
    st.write("**CUDA:** Disabled (CPU mode)")

# Simple test input
st.divider()
st.markdown("## 🧪 Component Test")

col1, col2 = st.columns(2)
with col1:
    st.button("Test Button")
with col2:
    test_input = st.text_input("Test Input:", "Hello")

st.write(f"Input received: {test_input}")

st.success("✅ All Streamlit components working!")
