"""
FastAPI interface for the RAG pipeline
"""

from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, AsyncGenerator
import uvicorn
import logging
import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from pathlib import Path
import shutil
import tempfile
import os
from typing import Union

from rag_pipeline_clean import RAGConfig, RAGPipeline

# Optional PDF support
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PyPDF2 not installed. PDF support will not be available.")

# Initialize FastAPI app
app = FastAPI(title="Technical Documentation RAG API",
             description="API for document ingestion and question answering using RAG")

# Initialize RAG pipeline with default config and sample documents
config = RAGConfig()
pipeline = RAGPipeline(config)

# Create and index sample documents
sample_docs = pipeline.create_sample_documents()
pipeline.index_documents(sample_docs)
logger.info("Initialized pipeline with sample documents")

class Query(BaseModel):
    """Query model for question answering"""
    question: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.7  # Controls response creativity (0.0 to 1.0)
    max_tokens: Optional[int] = 1024  # Maximum number of tokens in the response

class Answer(BaseModel):
    """Response model for answers"""
    answer: str
    sources: List[dict]
    processing_time: float

async def process_pdf_in_chunks(pdf_path: str, chunk_size: int = 2) -> AsyncGenerator[Dict, None]:
    """Process PDF file page by page in chunks with minimal memory usage"""
    try:
        import PyPDF2
        import gc
        
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            current_chunk = []
            current_size = 0
            
            for page_num in range(total_pages):
                try:
                    # Process one page at a time
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if not page_text.strip():
                        continue
                        
                    formatted_page = f"[Page {page_num + 1}]\n{page_text}"
                    current_chunk.append(formatted_page)
                    current_size += 1
                    
                    # Yield chunk when size limit reached
                    if current_size >= chunk_size:
                        doc = {
                            "title": f"Pages {page_num - current_size + 2}-{page_num + 1}",
                            "content": "\n\n".join(current_chunk),
                            "source": f"pages_{page_num - current_size + 2}_{page_num + 1}"
                        }
                        yield doc
                        
                        # Clear memory
                        current_chunk = []
                        current_size = 0
                        gc.collect()
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
            
            # Yield remaining pages
            if current_chunk:
                doc = {
                    "title": f"Pages {total_pages - current_size + 1}-{total_pages}",
                    "content": "\n\n".join(current_chunk),
                    "source": f"pages_{total_pages - current_size + 1}_{total_pages}"
                }
                yield doc
                
        gc.collect()
        
    except ImportError:
        raise HTTPException(status_code=400, detail="PDF support not available. Please install PyPDF2.")

async def process_text_in_chunks(file_path: str, chunk_size: int = 1024 * 64) -> AsyncGenerator[Dict, None]:
    """Process text file in smaller chunks to manage memory"""
    part = 1
    
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            content = f.read(chunk_size)
            if not content:
                break
                
            if content.strip():
                yield {
                    "title": f"Part {part}",
                    "content": content,
                    "source": f"part_{part}"
                }
                part += 1
            
            # Force garbage collection
            gc.collect()

@app.post("/upload", status_code=201)
async def upload_document(file: UploadFile):
    """Upload and process a new document in chunks with memory optimization"""
    if not file.filename.endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported")
    
    try:
        # Create temp file
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Copy file in chunks to avoid memory issues
            chunk_size = 8192  # 8KB chunks
            while chunk := await file.read(chunk_size):
                tmp.write(chunk)
            tmp_path = Path(tmp.name)
        
        # Read the file content
        if suffix == '.pdf':
            try:
                import PyPDF2
                with open(tmp_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    content = '\n\n'.join(page.extract_text() for page in pdf_reader.pages)
            except ImportError:
                raise HTTPException(status_code=400, detail="PDF support not available. Please install PyPDF2.")
        else:  # .txt file
            with open(tmp_path, 'r', encoding='utf-8') as txt_file:
                content = txt_file.read()
        
        # Create document in the required format
        document = {
            "title": file.filename,
            "content": content,
            "source": file.filename
        }
          # Index the document
        pipeline.index_documents([document])
        
        # Cleanup
        os.unlink(tmp_path)
        
        return {"message": f"Successfully processed {file.filename}", "tokens": len(content.split())}
        tmp_path.unlink()
        
        return {"message": f"Successfully processed {file.filename}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=Answer)
async def query(query: Query):
    """Query the RAG pipeline"""
    try:        # Temporarily adjust pipeline parameters if provided
        original_temperature = pipeline.config.temperature
        original_max_tokens = pipeline.config.max_new_tokens
        
        if query.temperature is not None:
            pipeline.config.temperature = query.temperature
        if query.max_tokens is not None:
            pipeline.config.max_new_tokens = query.max_tokens
        
        try:
            result = pipeline.generate_response(query.question)
            return Answer(
                answer=result["response"],
                sources=result["sources"],
                processing_time=result["processing_time"]
            )
        finally:
            # Restore original parameters
            pipeline.config.temperature = original_temperature
            pipeline.config.max_new_tokens = original_max_tokens
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
