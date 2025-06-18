"""
CPU-Optimized RAG Pipeline Implementation
Optimized for technical documentation and guides
"""

import os
import logging
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import pickle
import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Completely suppress llama model loader output
llama_logger = logging.getLogger('llama_cpp')
llama_logger.setLevel(logging.CRITICAL)
llama_logger.propagate = False  # Prevent propagation to root logger

@dataclass
class RAGConfig:
    """Configuration class for RAG pipeline"""        
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model_path: str = "./models/llama-3.2-1b.gguf"
    max_seq_length: int = 512
    top_k_retrieval: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    temperature: float = 0.7
    max_new_tokens: int = 1024  # Increased to allow for more detailed responses
    cpu_threads: int = os.cpu_count()
    context_window: int = 2048
    batch_size: int = 8
    index_path: str = "./data/faiss_index"
    documents_path: str = "./data/documents.pkl"

class DocumentProcessor:
    """Handles document chunking and preprocessing"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._current_metadata = {}
    
    def process_text(self, text: str) -> str:
        """Process text content with section detection"""
        # Clean up common artifacts
        text = text.replace('\x0c', '\n')  # Form feed
        text = text.replace('\xa0', ' ')   # Non-breaking space
        text = text.replace('•', '\n• ')   # Bullet points
        
        # Split into lines and clean
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Fix broken words and lines
            if line.endswith('-'):
                lines.append(line[:-1])
            else:
                lines.append(line + ' ')
        
        # Join lines but preserve paragraph breaks
        paragraphs = []
        current_para = []
        
        for line in lines:
            if not line.strip():
                if current_para:
                    paragraphs.append(''.join(current_para).strip())
                    current_para = []
                continue
            
            # Check if this line starts a new section
            if (line.isupper() or 
                any(line.startswith(prefix) for prefix in ['Section', 'Part', 'Chapter', 'Unit']) or
                line.endswith(':') or
                (len(line.strip()) <= 50 and any(char in line for char in ['•', '-', '=']))):
                
                if current_para:
                    paragraphs.append(''.join(current_para).strip())
                    current_para = []
                
                paragraphs.append(f"\n=== {line.strip()} ===\n")
                
                # Track section as topic
                topic = line.strip(':#-=• ')
                if len(topic) <= 100:
                    self._current_metadata.setdefault('topics', set()).add(topic)
            else:
                current_para.append(line)
        
        # Add final paragraph
        if current_para:
            paragraphs.append(''.join(current_para).strip())
        
        # Convert topics set to sorted list
        if 'topics' in self._current_metadata:
            self._current_metadata['topics'] = sorted(list(self._current_metadata['topics']))
        
        return '\n\n'.join(paragraphs)
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Split text into overlapping chunks"""
        # Handle short texts
        if len(text) <= self.chunk_size:
            return [{
                "text": text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_id": 0,
                    "total_chunks": 1,
                    "is_single_chunk": True
                }
            }]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at section boundary or paragraph
            if end < len(text):
                # First try to find a section boundary
                section_end = text.find("\n===", start, end)
                if section_end != -1:
                    end = section_end
                else:
                    # Try to find paragraph break
                    para_end = text.rfind("\n\n", start, end)
                    if para_end != -1:
                        end = para_end
                    else:
                        # Fall back to word boundary
                        while end > start and text[end] != ' ':
                            end -= 1
                        if end == start:
                            end = start + self.chunk_size
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **(metadata or {}),
                        "chunk_id": len(chunks),
                        "start_pos": start,
                        "end_pos": end
                    }
                })
            
            start = end - self.chunk_overlap
        
        # Enhance chunk metadata
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk["metadata"].update({
                "total_chunks": total_chunks,
                "chunk_sequence": i + 1,
                "is_first_chunk": i == 0,
                "is_last_chunk": i == total_chunks - 1
            })
        
        return chunks

    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process documents into chunks"""
        all_chunks = []
        doc_counter = 0
        
        logger.info(f"Processing {len(documents)} documents")
        
        for doc in documents:
            self._current_metadata = {}
            
            text = doc.get("content", "")
            if not text.strip():
                continue
                
            # Clean and process text
            text = self.process_text(text)
            
            # Create metadata
            metadata = {
                "doc_id": f"doc_{doc_counter}",
                "title": doc.get("title", f"Document_{doc_counter}"),
                "source": doc.get("source", "unknown"),
                "content_length": len(text),
                "date_added": time.strftime("%Y-%m-%d %H:%M:%S"),
                "topics": self._current_metadata.get("topics", [])
            }
            
            # Process into chunks
            chunks = self.chunk_text(text, metadata)
            logger.info(f"Document '{metadata['title']}' processed into {len(chunks)} chunks")
            
            all_chunks.extend(chunks)
            doc_counter += 1
        
        return all_chunks

class VectorStore:
    """FAISS-based vector store"""
    
    def __init__(self, embedding_dim: int, index_path: str = "./data/faiss_index"):
        self.embedding_dim = embedding_dim
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents = []
        self.doc_metadata = []
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """Add documents to the index"""
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.embedding_dim}")
        
        self.index.add(embeddings)
        self.documents.extend([doc["text"] for doc in documents])
        self.doc_metadata.extend([doc["metadata"] for doc in documents])
        
        logger.info(f"Added {len(documents)} documents. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        # Ensure query shape
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        if len(indices) > 0 and len(indices[0]) > 0:
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    results.append({
                        "text": self.documents[idx],
                        "metadata": self.doc_metadata[idx],
                        "score": float(score)
                    })
        
        return results
    
    def save(self):
        """Save index and metadata"""
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        with open(self.index_path / "documents.pkl", "wb") as f:
            pickle.dump({"documents": self.documents, "metadata": self.doc_metadata}, f)
    
    def load(self):
        """Load index and metadata"""
        index_file = self.index_path / "index.faiss"
        doc_file = self.index_path / "documents.pkl"
        
        if index_file.exists() and doc_file.exists():
            self.index = faiss.read_index(str(index_file))
            with open(doc_file, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.doc_metadata = data["metadata"]
            return True
        return False

class RAGPipeline:
    """Main RAG Pipeline"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.embedding_model}")
        self.embedding_model = SentenceTransformer(config.embedding_model)
        torch.set_num_threads(os.cpu_count())
        self.embedding_model.eval()
        
        # Initialize vector store
        self.vector_store = VectorStore(
            embedding_dim=self.embedding_model.get_sentence_embedding_dimension(),
            index_path=config.index_path
        )
          # Initialize LLM
        logger.info(f"Loading LLM: {config.llm_model_path}")
        self.llm = Llama(
            model_path=config.llm_model_path,
            n_ctx=config.context_window,
            n_threads=config.cpu_threads,
            verbose=False,  # Disable verbose output
            logits_all=False  # Disable additional logging
        )
        
        self.is_indexed = False
    
    def index_documents(self, documents: List[Dict]) -> None:
        """Index documents"""
        logger.info("Starting document indexing...")
        start_time = time.time()
        
        # Process documents
        chunks = self.document_processor.process_documents(documents)
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            device='cpu',
            normalize_embeddings=True
        )
        
        # Add to vector store
        self.vector_store.add_documents(embeddings, chunks)
        self.vector_store.save()
        
        self.is_indexed = True
        logger.info(f"Indexing completed in {time.time() - start_time:.2f} seconds")
    
    def retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents"""
        if not self.is_indexed:
            raise ValueError("Pipeline not indexed")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Get initial results
        results = self.vector_store.search(
            query_embedding,
            k=self.config.top_k_retrieval
        )
        
        if not results:
            return []
            
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Use adaptive threshold
        max_score = max(r["score"] for r in results)
        threshold = max(0.3, max_score * 0.5)  # At least 50% of max score or 0.3
        
        # Filter by threshold
        results = [r for r in results if r["score"] >= threshold]
        
        return results[:self.config.top_k_retrieval]
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response for query"""
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query)
        
        if not retrieved_docs:
            return {
                "query": query,
                "response": "I cannot answer this question as I don't have any relevant information in the provided context.",
                "retrieved_documents": [],
                "processing_time": time.time() - start_time,
                "sources": []
            }
        
        # Construct context from retrieved documents
        context_parts = []
        sources = []
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            context_parts.append(f"""
=== {metadata.get('title', 'Untitled')} ===
{doc['text']}
""".strip())
            
            # Build source information
            sources.append({
                "doc_id": metadata.get("doc_id", "unknown"),
                "title": metadata.get("title", "Untitled"),
                "source": metadata.get("source", "unknown"),
                "content_length": len(doc["text"]),
                "date_added": metadata.get("date_added", "unknown"),
                "topics": list(metadata.get("topics", [])),
                "chunk_id": metadata.get("chunk_id", 0),
                "total_chunks": metadata.get("total_chunks", 1),
                "is_single_chunk": metadata.get("total_chunks", 1) == 1
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate prompt with strict context adherence
        prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a strict context-based assistant. Your responses MUST follow these rules:
1. ONLY use information that is EXPLICITLY stated in the given context
2. DO NOT add ANY external knowledge, examples, or explanations
3. DO NOT infer, extrapolate, or make assumptions beyond the context
4. If the context doesn't contain enough information, say 'The provided context does not contain enough information to answer this question'
5. Present information exactly as it appears in the context
6. Keep your response focused and directly tied to the context
7. If asked about specific topics, only mention those explicitly listed in the context"""
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide an answer using ONLY the information from the above context, with no additional knowledge or inference:"
                }
            ]
        }
        
        # Generate response
        response = self.llm.create_chat_completion(
            messages=prompt["messages"],
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature
        )
        
        answer = response["choices"][0]["message"]["content"].strip()
        
        return {
            "query": query,
            "response": answer,
            "retrieved_documents": retrieved_docs,
            "processing_time": time.time() - start_time,
            "sources": [doc["metadata"] for doc in retrieved_docs]
        }
    
    def create_sample_documents(self) -> List[Dict]:
        """Create sample technical documents for testing"""
        return [
            {
                "title": "Introduction to Machine Learning",
                "content": """=== Basic Concepts ===
Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from data.

=== Types of Machine Learning ===
1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

=== Common Applications ===
• Classification
• Regression
• Clustering
• Dimensionality Reduction""",
                "source": "ml_guide.txt"
            },
            {
                "title": "Data Science Fundamentals",
                "content": """=== Data Analysis ===
Data analysis involves inspecting, cleaning, and modeling data to discover useful information.

=== Statistical Methods ===
• Descriptive Statistics
• Inferential Statistics
• Hypothesis Testing

=== Data Visualization ===
• Charts and Graphs
• Interactive Dashboards
• Statistical Plots""",
                "source": "ds_basics.txt"
            }
        ]
