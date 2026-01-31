"""
LLM Evaluation Pipeline using RAGAS Framework
==============================================

This module provides comprehensive evaluation capabilities for LLM and RAG systems
using the RAGAS (Retrieval Augmented Generation Assessment) framework.

Metrics included:
- Faithfulness: Factual consistency with context
- Answer Relevancy: How relevant the answer is to the question
- Context Precision: Precision of retrieved context
- Context Recall: Recall of relevant context
- Answer Similarity: Semantic similarity with ground truth
- Answer Correctness: Overall correctness score

Usage:
    from llm_eval_ragas import RAGASEvaluator
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(config="eval_config.yaml")
    
    # Run evaluation
    results = evaluator.evaluate_from_file("test_data.json")
    
    # Generate report
    evaluator.generate_report(results)
"""

from .ragas_evaluator import RAGASEvaluator

__all__ = ["RAGASEvaluator"]
__version__ = "1.0.0"
