"""
RAG Pipeline Benchmark Suite
Evaluates:
1. Response Accuracy
2. Retrieval Quality
3. Latency & Performance
4. CPU Usage and Memory Consumption
"""

import time
import psutil
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from rag_pipeline_clean import RAGPipeline, RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
class BenchmarkMetrics:
    """Metrics collected during benchmarking"""
    query: str
    response_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    num_relevant_docs: int
    top_doc_score: float
    response_length: int
    source_coverage: float  # How many sources were used in the response
    cpu_info: Dict
    timestamp: str

class RAGBenchmarker:
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test queries with ground truth for accuracy evaluation
        self.test_queries = [
            {
                "query": "What are the main types of machine learning?",
                "expected_concepts": ["supervised learning", "unsupervised learning", "reinforcement learning"],
                "category": "ml_fundamentals"
            },
            {
                "query": "What statistical methods are used in data science?",
                "expected_concepts": ["descriptive statistics", "inferential statistics", "hypothesis testing"],
                "category": "statistics"
            },
            {
                "query": "What visualization techniques are available?",
                "expected_concepts": ["charts", "graphs", "interactive dashboards", "statistical plots"],
                "category": "visualization"
            },
            {
                "query": "What are common machine learning applications?",
                "expected_concepts": ["classification", "regression", "clustering", "dimensionality reduction"],
                "category": "ml_applications"
            }
        ]

    def get_cpu_info(self) -> Dict:
        """Get detailed CPU information"""
        cpu_info = {
            "processor_count": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "max_frequency": psutil.cpu_freq().max if hasattr(psutil.cpu_freq(), 'max') else None,
            "current_frequency": psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else None,
            "architecture": platform.machine(),
            "model": platform.processor()
        }
        return cpu_info

    def calculate_source_coverage(self, response: str, sources: List[Dict]) -> float:
        """Calculate how many source documents were used in the response"""
        source_matches = 0
        response_lower = response.lower()
        
        for source in sources:
            source_content = source.get("text", "").lower()
            # Check if any significant phrases from the source appear in the response
            source_phrases = [p.strip() for p in source_content.split('.') if len(p.strip()) > 20]
            
            for phrase in source_phrases:
                if any(part in response_lower for part in phrase.split()):
                    source_matches += 1
                    break
        
        return source_matches / len(sources) if sources else 0

    def run_single_query_benchmark(self, pipeline: RAGPipeline, query_info: Dict) -> BenchmarkMetrics:
        """Run benchmark for a single query"""
        query = query_info["query"]
        
        # Start monitoring resources
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.time()
        cpu_percent_start = psutil.cpu_percent()        # Execute query
        result = pipeline.generate_response(query)  # The pipeline now handles the parameters internally

        # Calculate metrics
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)
        cpu_percent_end = psutil.cpu_percent()

        response_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = (cpu_percent_end + cpu_percent_start) / 2

        # Calculate retrieval quality metrics
        top_doc_score = max([doc.get("score", 0) for doc in result.get("retrieved_documents", [])], default=0)
        source_coverage = self.calculate_source_coverage(
            result.get("answer", ""),
            result.get("retrieved_documents", [])
        )

        return BenchmarkMetrics(
            query=query,
            response_time=response_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            num_relevant_docs=len(result.get("retrieved_documents", [])),
            top_doc_score=top_doc_score,
            response_length=len(result.get("answer", "")),
            source_coverage=source_coverage,
            cpu_info=self.get_cpu_info(),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def run_comprehensive_benchmark(self, pipeline: RAGPipeline) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        logger.info("Starting comprehensive RAG pipeline benchmark...")
        
        results = []
        for query_info in self.test_queries:
            logger.info(f"Testing query: {query_info['query']}")
            
            # Run query multiple times for stable measurements
            query_metrics = []
            for i in range(3):  # Run each query 3 times
                metrics = self.run_single_query_benchmark(pipeline, query_info)
                query_metrics.append(metrics)
                time.sleep(1)  # Brief pause between runs
            
            # Average the metrics
            avg_metrics = {
                "query": query_info["query"],
                "category": query_info["category"],
                "avg_response_time": statistics.mean([m.response_time for m in query_metrics]),
                "avg_memory_usage": statistics.mean([m.memory_usage_mb for m in query_metrics]),
                "avg_cpu_usage": statistics.mean([m.cpu_usage_percent for m in query_metrics]),
                "avg_source_coverage": statistics.mean([m.source_coverage for m in query_metrics]),
                "avg_num_relevant_docs": statistics.mean([m.num_relevant_docs for m in query_metrics]),
                "avg_top_doc_score": statistics.mean([m.top_doc_score for m in query_metrics])
            }
            
            results.append(avg_metrics)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        self.generate_benchmark_plots(results, timestamp)
        
        return results

    def generate_benchmark_plots(self, results: List[Dict], timestamp: str):
        """Generate visualization plots for benchmark results"""
        df = pd.DataFrame(results)
          # Set up the plotting style
        plt.style.use('default')  # Using default matplotlib style
        sns.set_theme()  # Apply seaborn theme on top
        
        # 1. Response Time by Category
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='category', y='avg_response_time')
        plt.title('Average Response Time by Query Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'response_times_{timestamp}.png')
        plt.close()
        
        # 2. Resource Usage Overview
        plt.figure(figsize=(12, 6))
        resource_data = pd.melt(df, 
                              id_vars=['category'],
                              value_vars=['avg_memory_usage', 'avg_cpu_usage'])
        sns.boxplot(data=resource_data, x='category', y='value', hue='variable')
        plt.title('Resource Usage by Query Category')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'resource_usage_{timestamp}.png')
        plt.close()
        
        # 3. Retrieval Quality Metrics
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='avg_top_doc_score', y='avg_source_coverage', 
                       size='avg_num_relevant_docs', hue='category')
        plt.title('Retrieval Quality Metrics')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'retrieval_quality_{timestamp}.png')
        plt.close()

def main():
    # Initialize RAG pipeline
    config = RAGConfig()
    pipeline = RAGPipeline(config)
    
    # Create and index sample documents
    logger.info("Creating and indexing sample documents...")
    sample_docs = [
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
    
    # Index the documents
    try:
        pipeline.index_documents(sample_docs)
        logger.info("Documents indexed successfully")
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return
    
    # Create benchmarker
    benchmarker = RAGBenchmarker(output_dir="./benchmark_results")
    
    # Run comprehensive benchmark
    results = benchmarker.run_comprehensive_benchmark(pipeline)
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 50)
    for result in results:
        print(f"\nQuery Category: {result['category']}")
        print(f"Average Response Time: {result['avg_response_time']:.2f} seconds")
        print(f"Average Memory Usage: {result['avg_memory_usage']:.2f} MB")
        print(f"Average CPU Usage: {result['avg_cpu_usage']:.2f}%")
        print(f"Source Coverage: {result['avg_source_coverage']:.2%}")
        print(f"Retrieval Quality Score: {result['avg_top_doc_score']:.3f}")

if __name__ == "__main__":
    main()
