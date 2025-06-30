import streamlit as st
import os
import json
import time
from typing import List, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to path to import our RAG pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline_clean import RAGPipeline, RAGConfig
import PyPDF2

# Page configuration
st.set_page_config(
    page_title="CPU-Optimized RAG Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

.source-box {
    background-color: #e8f4fd;
    padding: 1rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
    border-radius: 0.25rem;
}

.response-box {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize RAG pipeline with caching"""
    config = RAGConfig()
    pipeline = RAGPipeline(config)
      # Try to load existing index
    if not pipeline.vector_store.load():
        # If no index exists, create sample documents and index them
        with st.spinner("Initializing RAG pipeline with sample documents..."):
            documents = pipeline.create_sample_documents()
            pipeline.index_documents(documents)
    else:
        pipeline.is_indexed = True
        logger.info("Loaded existing vector store index")
    
    return pipeline

def display_performance_metrics(result: Dict):
    """Display performance metrics"""
    col1, col2, col3 = st.columns(3)
    
    processing_time = result.get("processing_time", 0)
    num_retrieved = len(result.get("retrieved_documents", []))
    
    with col1:
        st.metric("Response Time", f"{processing_time:.2f}s")
    
    with col2:
        st.metric("Documents Retrieved", num_retrieved)
    
    with col3:
        efficiency = "Excellent" if processing_time < 2 else "Good" if processing_time < 5 else "Fair"
        st.metric("Efficiency", efficiency)

def display_retrieved_documents(retrieved_docs: List[Dict]):
    """Display retrieved documents with relevance scores"""
    st.subheader("📚 Retrieved Documents")
    
    for i, doc in enumerate(retrieved_docs):
        score = doc.get("score", 0)
        metadata = doc.get("metadata", {})
        with st.expander(f"Document {i+1} - Score: {score:.3f}"):
            st.markdown(f"**Source:** {metadata.get('title', 'Unknown')}")
            st.markdown(f"**Topics:** {', '.join(metadata.get('topics', ['No topics']))}")
            st.markdown("**Content:**")
            st.markdown(doc.get("text", ""), help="The relevant passage from the document")
            st.markdown(f"**Content:** {doc['text']}")
            
            # Display metadata
            metadata = doc['metadata']
            if metadata:
                st.json(metadata)

def create_performance_chart(response_times: List[float]):
    """Create performance visualization"""
    if len(response_times) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=response_times,
            mode='lines+markers',
            name='Response Time',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Response Time Performance",
            xaxis_title="Query Number",
            yaxis_title="Response Time (seconds)",
            height=300
        )
        return fig
    return None

def process_uploaded_file(file) -> Dict:
    """Process an uploaded file and return document dict"""
    try:
        # Read file content
        content = file.read()
        
        # Try to decode if it's bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
            
        return {
            "title": file.name,
            "content": content,
            "source": file.name
        }
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {str(e)}")
        raise e

def upload_section():
    """Document upload section"""
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents (txt, pdf, md)",
        accept_multiple_files=True,
        type=['txt', 'md', 'pdf']  
    )
    
    if uploaded_files:
        if st.sidebar.button("Process Uploaded Documents"):
            with st.spinner("Processing uploaded documents..."):
                try:
                    documents = []
                    for file in uploaded_files:
                        # Handle PDF files
                        if file.name.lower().endswith('.pdf'):
                            try:
                                pdf_reader = PyPDF2.PdfReader(file)
                                text = ""
                                for page in pdf_reader.pages:
                                    text += page.extract_text() or ""
                                doc = {
                                    "title": file.name,
                                    "content": text,
                                    "source": file.name
                                }
                            except Exception as pdf_e:
                                st.sidebar.error(f"Failed to process PDF {file.name}: {pdf_e}")
                                continue
                        else:
                            doc = process_uploaded_file(file)
                        documents.append(doc)
                    
                    pipeline = initialize_rag_pipeline()
                    pipeline.index_documents(documents)
                    
                    st.sidebar.success(f"Successfully processed {len(documents)} documents!")
                    st.cache_resource.clear()
                except Exception as e:
                    st.sidebar.error(f"Error processing documents: {str(e)}")
                    logger.exception("Error processing uploaded documents")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 CPU-Optimized RAG Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("**Retrieval-Augmented Generation optimized for AMD Ryzen processors**")
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = initialize_rag_pipeline()
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model configuration
    st.sidebar.subheader("Model Settings")
    top_k = st.sidebar.slider("Top-K Retrieval", 1, 10, st.session_state.rag_pipeline.config.top_k_retrieval)
    temperature = st.sidebar.slider("Generation Temperature", 0.1, 1.0, st.session_state.rag_pipeline.config.temperature, 0.1)
    
    # Update configuration if changed
    if top_k != st.session_state.rag_pipeline.config.top_k_retrieval:
        st.session_state.rag_pipeline.config.top_k_retrieval = top_k
    
    if temperature != st.session_state.rag_pipeline.config.temperature:
        st.session_state.rag_pipeline.config.temperature = temperature
    
    # System information
    st.sidebar.subheader("💻 System Info")
    st.sidebar.info(f"CPU Threads: {os.cpu_count()}")
    st.sidebar.info(f"Embedding Model: {st.session_state.rag_pipeline.config.embedding_model}")
    st.sidebar.info(f"LLM Model: {st.session_state.rag_pipeline.config.llm_model}")
    
    # Document upload section
    upload_section()
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask a Question")
        
        # Query input
        query = st.text_area(
            "Enter your question:",
            placeholder="What would you like to know about machine learning, NLP, or computer vision?",
            height=100
        )
        
        # Query button
        if st.button("🔍 Generate Response", type="primary"):
            if query.strip():
                with st.spinner("Processing your query..."):
                    try:
                        result = st.session_state.rag_pipeline.generate_response(query)
                        
                        # Store results
                        st.session_state.query_history.append(result)
                        st.session_state.response_times.append(result['processing_time'])
                        
                        # Display response
                        st.markdown('<div class="response-box">', unsafe_allow_html=True)
                        st.markdown("### 🤖 Response")
                        st.markdown(result['response'])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Performance metrics
                        display_performance_metrics(
                            result['processing_time'],
                            len(result['retrieved_documents'])
                        )
                        
                        # Retrieved documents
                        display_retrieved_documents(result['retrieved_documents'])
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.subheader("📊 Performance Analytics")
        
        if st.session_state.response_times:
            # Performance chart
            chart = create_performance_chart(st.session_state.response_times)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Statistics
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            min_time = min(st.session_state.response_times)
            max_time = max(st.session_state.response_times)
            
            st.markdown("**Performance Statistics:**")
            st.markdown(f"- Average: {avg_time:.2f}s")
            st.markdown(f"- Fastest: {min_time:.2f}s")
            st.markdown(f"- Slowest: {max_time:.2f}s")
            st.markdown(f"- Total Queries: {len(st.session_state.response_times)}")
        else:
            st.info("No queries processed yet. Ask a question to see performance metrics!")
    
    # Document management section
    st.subheader("📁 Document Management")
    
    tab1, tab2, tab3 = st.tabs(["Upload Documents", "Current Index", "Sample Queries"])
    
    with tab1:
        st.markdown("**Upload your own documents to expand the knowledge base:**")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['txt', 'md', 'pdf']
        )
        
        if uploaded_files:
            if st.button("Index Uploaded Documents"):
                with st.spinner("Processing uploaded documents..."):
                    documents = []
                    for file in uploaded_files:
                        content = str(file.read(), "utf-8")
                        documents.append({
                            "title": file.name,
                            "content": content,
                            "source": file.name
                        })
                    
                    # Re-index with new documents
                    st.session_state.rag_pipeline.index_documents(documents)
                    st.success(f"Successfully indexed {len(documents)} documents!")
    
    with tab2:
        st.markdown("**Current Knowledge Base:**")
        if hasattr(st.session_state.rag_pipeline.vector_store, 'documents'):
            num_docs = len(st.session_state.rag_pipeline.vector_store.documents)
            st.info(f"Total indexed chunks: {num_docs}")
            
            # Show sample documents
            if num_docs > 0:
                sample_docs = st.session_state.rag_pipeline.vector_store.documents[:5]
                for i, doc in enumerate(sample_docs):
                    with st.expander(f"Sample Chunk {i+1}"):
                        st.text(doc[:200] + "..." if len(doc) > 200 else doc)
    
    with tab3:
        st.markdown("**Try these sample queries:**")
        
        sample_queries = [
            "What is machine learning and how does it work?",
            "Explain the difference between supervised and unsupervised learning",
            "What are the main applications of natural language processing?",
            "How does computer vision work in autonomous vehicles?",
            "What is the role of neural networks in AI?"
        ]
        
        for i, sample_query in enumerate(sample_queries):
            if st.button(f"🔍 {sample_query}", key=f"sample_{i}"):
                st.session_state.sample_query = sample_query
                st.rerun()
        
        # Handle sample query selection
        if hasattr(st.session_state, 'sample_query'):
            st.text_area("Selected Query:", value=st.session_state.sample_query, key="selected_query")
    
    # Query history
    if st.session_state.query_history:
        st.subheader("📝 Query History")
        
        with st.expander("View Previous Queries"):
            for i, result in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5
                st.markdown(f"**Query {len(st.session_state.query_history)-i}:** {result['query']}")
                st.markdown(f"**Response:** {result['response'][:200]}...")
                st.markdown(f"**Time:** {result['processing_time']:.2f}s")
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit | Optimized for AMD Ryzen processors | "
        "CPU-only inference pipeline"
    )

def main():
    """Main Streamlit application"""
    st.markdown("<h1 class='main-header'>🤖 CPU-Optimized RAG Pipeline</h1>", unsafe_allow_html=True)
    
    # Initialize pipeline
    pipeline = initialize_rag_pipeline()
    
    # Add document upload section in sidebar
    upload_section()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, pipeline.config.temperature, 0.1,
                                  help="Higher values make the output more random")
    top_k = st.sidebar.slider("Top K Documents", 1, 10, pipeline.config.top_k_retrieval,
                             help="Number of documents to retrieve")
    max_tokens = st.sidebar.slider("Max Tokens", 100, 2048, pipeline.config.max_new_tokens,
                                  help="Maximum number of tokens in the response")
    
    # Update pipeline config
    pipeline.config.temperature = temperature
    pipeline.config.top_k_retrieval = top_k
    pipeline.config.max_new_tokens = max_tokens
    
    # Main interface
    query = st.text_area("🔍 Ask a question", height=100,
                        help="Enter your question here. The model will search through the documents to find relevant information.")
    
    if st.button("🚀 Generate Response", type="primary"):
        if not query:
            st.warning("Please enter a question")
            return
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Generate response
                result = pipeline.generate_response(query)
                
                # Display metrics
                display_performance_metrics(result)
                
                # Display response
                st.markdown("### 💡 Response")
                st.markdown("<div class='response-box'>", unsafe_allow_html=True)
                st.write(result["response"])
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display source documents
                if result.get("retrieved_documents"):
                    display_retrieved_documents(result["retrieved_documents"])
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.exception("Error generating response")

if __name__ == "__main__":
    main()