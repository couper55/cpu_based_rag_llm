# First index your documents if not already done
from rag_pipeline_clean import RAGPipeline, RAGConfig

config = RAGConfig()
pipeline = RAGPipeline(config)
documents = pipeline.create_sample_documents()  # or your own documents
pipeline.index_documents(documents)