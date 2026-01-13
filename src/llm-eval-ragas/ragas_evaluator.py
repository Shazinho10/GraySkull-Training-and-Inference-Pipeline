"""
RAGAS LLM Evaluation Pipeline
Evaluates LLM responses using RAGAS framework with Groq API
"""

import os
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

# Load environment variables
load_dotenv()


class RAGASEvaluator:
    """RAGAS-based LLM evaluation pipeline."""
    
    def __init__(self, config_path: str = "eval_config.yaml"):
        """
        Initialize the RAGAS evaluator with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings()
        self.metrics = self._setup_metrics()
        self.output_dir = Path(self.config['output']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(__file__).parent / config_path
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_llm(self) -> ChatGroq:
        """Setup Groq LLM for evaluation."""
        api_key_env = self.config['model']['api_key_env']
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(
                f"API key not found. Please set {api_key_env} in your .env file."
            )
        
        return ChatGroq(
            model=self.config['model']['name'],
            temperature=self.config['model']['temperature'],
            groq_api_key=api_key,
        )
    
    def _setup_embeddings(self) -> HuggingFaceEmbeddings:
        """Setup embeddings model."""
        return HuggingFaceEmbeddings(
            model_name=self.config['embeddings']['model']
        )
    
    def _setup_metrics(self) -> List:
        """Setup evaluation metrics based on configuration."""
        metrics_config = self.config['metrics']
        metrics = []
        
        if metrics_config.get('faithfulness', False):
            metrics.append(faithfulness)
        if metrics_config.get('answer_relevancy', False):
            metrics.append(answer_relevancy)
        if metrics_config.get('context_precision', False):
            metrics.append(context_precision)
        if metrics_config.get('context_recall', False):
            metrics.append(context_recall)
        if metrics_config.get('answer_similarity', False):
            metrics.append(answer_similarity)
        if metrics_config.get('answer_correctness', False):
            metrics.append(answer_correctness)
        
        if not metrics:
            raise ValueError("At least one metric must be enabled in the configuration.")
        
        return metrics
    
    def _load_data(self) -> Dataset:
        """
        Load evaluation data from file.
        
        Returns:
            Dataset in RAGAS format
        """
        data_config = self.config['data']
        input_file = data_config.get('input_file')
        
        # If no input file specified, use sample data
        if not input_file:
            print("No input file specified. Using sample data from sample_data.json")
            input_file = Path(__file__).parent / "sample_data.json"
            data_format = "json"
        else:
            input_file = Path(input_file)
            data_format = data_config.get('input_format', 'csv').lower()
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load data based on format
        if data_format == 'json':
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif data_format == 'csv':
            df = pd.read_csv(input_file)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        # Map columns according to configuration
        column_mapping = self.config['column_mapping']
        required_columns = ['question', 'answer', 'contexts', 'ground_truth']
        
        # Verify required columns exist
        for col in required_columns:
            mapped_col = column_mapping.get(col)
            if mapped_col not in df.columns:
                raise ValueError(
                    f"Required column '{mapped_col}' (mapped from '{col}') not found in data."
                )
        
        # Prepare data for RAGAS
        # Convert contexts to list of lists if it's a string
        if column_mapping['contexts'] in df.columns:
            df[column_mapping['contexts']] = df[column_mapping['contexts']].apply(
                lambda x: x if isinstance(x, list) else [x] if isinstance(x, str) else []
            )
        
        # Create dataset with mapped columns
        ragas_data = {
            'question': df[column_mapping['question']].tolist(),
            'answer': df[column_mapping['answer']].tolist(),
            'contexts': df[column_mapping['contexts']].tolist(),
            'ground_truth': df[column_mapping['ground_truth']].tolist(),
        }
        
        return Dataset.from_dict(ragas_data)
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Run RAGAS evaluation on the dataset.
        
        Returns:
            Dictionary containing evaluation results
        """
        print("Loading evaluation data...")
        dataset = self._load_data()
        
        print(f"Evaluating {len(dataset)} samples with {len(self.metrics)} metrics...")
        print(f"Metrics: {[m.name for m in self.metrics]}")
        
        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )
        
        # Convert results to dictionary
        results_dict = results.to_pandas().to_dict('records')
        
        # Calculate summary statistics
        # summary = self._calculate_summary(results)
        
        return {
            'detailed_results': results_dict,
            'summary': "summary",
            'results_df': results.to_pandas(),
        }
    
    # def _calculate_summary(self, results) -> Dict[str, Any]:
    #     """Calculate summary statistics from evaluation results."""
    #     df = results.to_pandas()
    #     summary = {}
        
    #     # Calculate mean and std for each metric
    #     metric_columns = [col for col in df.columns if col not in ['question', 'answer', 'contexts', 'ground_truth']]
        
    #     for metric in metric_columns:
    #         summary[metric] = {
    #             'mean': float(df[metric].mean()),
    #             'std': float(df[metric].std()),
    #             'min': float(df[metric].min()),
    #             'max': float(df[metric].max()),
    #         }
        
    #     # Overall average score
    #     summary['overall_average'] = float(df[metric_columns].mean().mean())
        
    #     return summary
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        output_config = self.config['output']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        if output_config.get('save_detailed_results', True):
            df = results['results_df']
            
            if 'csv' in output_config.get('export_formats', []):
                csv_path = self.output_dir / f"detailed_results_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Detailed results saved to: {csv_path}")
            
            if 'json' in output_config.get('export_formats', []):
                json_path = self.output_dir / f"detailed_results_{timestamp}.json"
                df.to_json(json_path, orient='records', indent=2)
                print(f"Detailed results saved to: {json_path}")


def main():
    """Main execution function."""
    try:
        # Initialize evaluator
        evaluator = RAGASEvaluator()
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Save results
        evaluator.save_results(results)
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
