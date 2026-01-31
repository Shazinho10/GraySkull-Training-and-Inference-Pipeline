"""
Qwen 1.5B Inference Script using Ollama
========================================
This script provides inference capabilities for Qwen 2.5 1.5B model via Ollama.

Features:
- Single and batch inference
- Streaming and non-streaming responses
- Execution time tracking
- Token usage tracking
- Configurable via YAML config file

Usage:
    python qwen-inference.py --prompt "Your question here"
    python qwen-inference.py --interactive
    python qwen-inference.py --batch prompts.txt

Prerequisites:
    1. Install Ollama: https://ollama.ai/download
    2. Pull the model: ollama pull qwen2.5:1.5b
    3. Install dependencies: pip install requests pyyaml

Author: GraySkull AI Pipeline
Date: 2026
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from contextlib import contextmanager

import yaml

# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class OllamaConfig:
    """Ollama server configuration."""
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:1.5b"
    timeout: int = 120


@dataclass
class GenerationConfig:
    """Model generation parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    seed: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """Track performance metrics for each inference."""
    prompt: str = ""
    response: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    execution_time_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    model: str = ""
    timestamp: str = ""


# =============================================================================
# Timer Context Manager
# =============================================================================

@contextmanager
def timer(description: str = "Operation"):
    """Context manager for timing code execution."""
    start = time.perf_counter()
    print(f"\n⏱️  {description} started...")
    yield
    elapsed = time.perf_counter() - start
    print(f"✅ {description} completed in {elapsed:.3f} seconds")


# =============================================================================
# Qwen Ollama Inference Client
# =============================================================================

class QwenOllamaClient:
    """
    Client for Qwen inference using Ollama.
    
    Provides methods for single inference, batch inference, and streaming.
    Tracks execution time and token usage for performance analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Qwen Ollama client.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        self.config = self._load_config(config_path)
        self.ollama_config = OllamaConfig(**self.config.get('ollama', {}))
        self.generation_config = GenerationConfig(**self.config.get('generation', {}))
        self.system_prompt = self.config.get('system_prompt', "You are a helpful AI assistant.")
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Setup logging directory
        self._setup_logging()
        
        # Verify Ollama connection
        self._verify_connection()
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try to find config in the same directory as the script
            script_dir = Path(__file__).parent
            config_path = script_dir / "qwen_inference_config.yaml"
        
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                print(f"📁 Loaded configuration from: {config_path}")
                return yaml.safe_load(f)
        else:
            print(f"⚠️  Config file not found at {config_path}, using defaults")
            return {}
    
    def _setup_logging(self):
        """Setup logging directory."""
        log_config = self.config.get('logging', {})
        if log_config.get('log_to_file', False):
            log_file = Path(log_config.get('log_file', 'logs/qwen_inference.log'))
            log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _verify_connection(self):
        """Verify connection to Ollama server."""
        try:
            response = requests.get(
                f"{self.ollama_config.base_url}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                print(f"✅ Connected to Ollama server at {self.ollama_config.base_url}")
                print(f"📦 Available models: {', '.join(model_names) if model_names else 'None'}")
                
                # Check if our model is available
                if not any(self.ollama_config.model_name in name for name in model_names):
                    print(f"\n⚠️  Model '{self.ollama_config.model_name}' not found!")
                    print(f"   Run: ollama pull {self.ollama_config.model_name}")
            else:
                print(f"⚠️  Ollama server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Cannot connect to Ollama server at {self.ollama_config.base_url}")
            print("   Make sure Ollama is running: ollama serve")
            sys.exit(1)
    
    def _build_request_payload(self, prompt: str) -> dict:
        """Build the request payload for Ollama API."""
        payload = {
            "model": self.ollama_config.model_name,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {
                "temperature": self.generation_config.temperature,
                "top_p": self.generation_config.top_p,
                "top_k": self.generation_config.top_k,
                "num_predict": self.generation_config.max_tokens,
                "repeat_penalty": self.generation_config.repeat_penalty,
            }
        }
        
        if self.generation_config.seed is not None:
            payload["options"]["seed"] = self.generation_config.seed
        
        return payload
    
    def generate(self, prompt: str) -> tuple[str, PerformanceMetrics]:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt
            
        Returns:
            Tuple of (response_text, performance_metrics)
        """
        metrics = PerformanceMetrics(
            prompt=prompt,
            model=self.ollama_config.model_name,
            timestamp=datetime.now().isoformat()
        )
        
        metrics.start_time = time.perf_counter()
        
        payload = self._build_request_payload(prompt)
        
        try:
            response_text = self._non_stream_response(payload, metrics)
            
            metrics.end_time = time.perf_counter()
            metrics.execution_time_seconds = metrics.end_time - metrics.start_time
            metrics.response = response_text
            
            # Calculate tokens per second
            if metrics.completion_tokens > 0 and metrics.execution_time_seconds > 0:
                metrics.tokens_per_second = metrics.completion_tokens / metrics.execution_time_seconds
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            return response_text, metrics
            
        except requests.exceptions.Timeout:
            print(f"\n❌ Request timed out after {self.ollama_config.timeout} seconds")
            raise
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Request failed: {e}")
            raise
    
    def _non_stream_response(self, payload: dict, metrics: PerformanceMetrics) -> str:
        """Handle non-streaming response."""
        response = requests.post(
            f"{self.ollama_config.base_url}/api/generate",
            json=payload,
            timeout=self.ollama_config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract token counts
        metrics.prompt_tokens = result.get('prompt_eval_count', 0)
        metrics.completion_tokens = result.get('eval_count', 0)
        metrics.total_tokens = metrics.prompt_tokens + metrics.completion_tokens
        
        return result.get('response', '')
    
    def batch_generate(self, prompts: List[str], show_progress: bool = True) -> List[tuple[str, PerformanceMetrics]]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            show_progress: Whether to show progress
            
        Returns:
            List of (response, metrics) tuples
        """
        results = []
        total = len(prompts)
        
        print(f"\n🔄 Processing {total} prompts...")
        batch_start = time.perf_counter()
        
        for i, prompt in enumerate(prompts, 1):
            if show_progress:
                print(f"\n[{i}/{total}] Processing prompt {i}...")
            
            try:
                response, metrics = self.generate(prompt, stream=False)
                results.append((response, metrics))
            except Exception as e:
                print(f"❌ Error processing prompt {i}: {e}")
                results.append(("", PerformanceMetrics(prompt=prompt)))
        
        batch_end = time.perf_counter()
        total_time = batch_end - batch_start
        
        print(f"\n✅ Batch processing completed!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Average time per prompt: {total_time/total:.2f} seconds")
        
        return results
    
    def print_metrics(self, metrics: PerformanceMetrics):
        """Print formatted performance metrics."""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE METRICS")
        print("=" * 60)
        print(f"   Model:              {metrics.model}")
        print(f"   Execution Time:     {metrics.execution_time_seconds:.3f} seconds")
        print(f"   Prompt Tokens:      {metrics.prompt_tokens}")
        print(f"   Completion Tokens:  {metrics.completion_tokens}")
        print(f"   Total Tokens:       {metrics.total_tokens}")
        print(f"   Tokens/Second:      {metrics.tokens_per_second:.2f}")
        print(f"   Timestamp:          {metrics.timestamp}")
        print("=" * 60)
    
    def save_metrics(self, filepath: Optional[str] = None):
        """Save all metrics to a JSON file."""
        if filepath is None:
            perf_config = self.config.get('performance', {})
            filepath = perf_config.get('metrics_file', 'logs/qwen_metrics.json')
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metrics to dictionaries
        metrics_data = []
        for m in self.metrics_history:
            metrics_data.append({
                'prompt': m.prompt[:100] + '...' if len(m.prompt) > 100 else m.prompt,
                'response_length': len(m.response),
                'execution_time_seconds': m.execution_time_seconds,
                'prompt_tokens': m.prompt_tokens,
                'completion_tokens': m.completion_tokens,
                'total_tokens': m.total_tokens,
                'tokens_per_second': m.tokens_per_second,
                'model': m.model,
                'timestamp': m.timestamp
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"📁 Metrics saved to: {filepath}")
    
    def interactive_chat(self):
        """Run an interactive chat session."""
        print("\n" + "=" * 60)
        print("🤖 QWEN INTERACTIVE CHAT")
        print("=" * 60)
        print(f"   Model: {self.ollama_config.model_name}")
        print("   Type 'quit' or 'exit' to end the session")
        print("   Type 'metrics' to see performance summary")
        print("   Type 'clear' to clear conversation history")
        print("=" * 60)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'metrics':
                    self._print_session_summary()
                    continue
                
                if user_input.lower() == 'clear':
                    conversation_history.clear()
                    print("🧹 Conversation cleared!")
                    continue
                
                # Build context-aware prompt
                context_prompt = self._build_context_prompt(conversation_history, user_input)
                
                response, metrics = self.generate(context_prompt)
                print(f"\n🤖 Qwen: {response}")
                
                # Store in conversation history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response})
                
                # Print metrics
                print(f"\n   ⏱️  Time: {metrics.execution_time_seconds:.2f}s | "
                      f"📊 Tokens: {metrics.total_tokens} | "
                      f"⚡ Speed: {metrics.tokens_per_second:.1f} tok/s")
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
    
    def _build_context_prompt(self, history: List[dict], current_input: str) -> str:
        """Build a context-aware prompt from conversation history."""
        if not history:
            return current_input
        
        # Keep last 5 exchanges for context
        recent_history = history[-10:]
        
        context_parts = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        
        context_parts.append(f"User: {current_input}")
        
        return "\n".join(context_parts)
    
    def _print_session_summary(self):
        """Print summary of session performance."""
        if not self.metrics_history:
            print("\n📊 No metrics recorded yet.")
            return
        
        total_requests = len(self.metrics_history)
        total_time = sum(m.execution_time_seconds for m in self.metrics_history)
        total_tokens = sum(m.total_tokens for m in self.metrics_history)
        avg_time = total_time / total_requests
        avg_tokens_per_sec = sum(m.tokens_per_second for m in self.metrics_history) / total_requests
        
        print("\n" + "=" * 60)
        print("📊 SESSION SUMMARY")
        print("=" * 60)
        print(f"   Total Requests:        {total_requests}")
        print(f"   Total Time:            {total_time:.2f} seconds")
        print(f"   Total Tokens:          {total_tokens}")
        print(f"   Avg Time per Request:  {avg_time:.2f} seconds")
        print(f"   Avg Tokens/Second:     {avg_tokens_per_sec:.1f}")
        print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Qwen 1.5B Inference using Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qwen-inference.py --prompt "What is machine learning?"
  python qwen-inference.py --interactive
  python qwen-inference.py --batch prompts.txt
Setup Instructions:
  1. Install Ollama: https://ollama.ai/download
  2. Start Ollama: ollama serve
  3. Pull model: ollama pull qwen2.5:1.5b
  4. Run inference: python qwen-inference.py --interactive
        """
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        help='Single prompt to process'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive chat mode'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Path to file with prompts (one per line)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--save-metrics',
        action='store_true',
        help='Save performance metrics to file'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("🚀 QWEN 1.5B INFERENCE (OLLAMA)")
    print("=" * 60)
    
    # Initialize client
    with timer("Initializing Qwen client"):
        client = QwenOllamaClient(config_path=args.config)
    
    try:
        if args.interactive:
            # Interactive mode
            client.interactive_chat()
            
        elif args.batch:
            # Batch mode
            batch_file = Path(args.batch)
            if not batch_file.exists():
                print(f"❌ Batch file not found: {args.batch}")
                sys.exit(1)
            
            with open(batch_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            with timer(f"Processing {len(prompts)} prompts"):
                results = client.batch_generate(prompts)
            
            for i, (response, metrics) in enumerate(results, 1):
                print(f"\n--- Response {i} ---")
                print(response[:500] + "..." if len(response) > 500 else response)
                client.print_metrics(metrics)
                
        elif args.prompt:
            # Single prompt mode
            with timer("Generating response"):
                response, metrics = client.generate(args.prompt)
            
            print(f"\n📝 Response:\n{response}")
            client.print_metrics(metrics)
            
        else:
            # Default: show help or run a demo
            print("\n📋 No prompt specified. Running demo...")
            
            demo_prompt = "Explain what is deep learning in 2-3 sentences."
            print(f"\n💬 Demo prompt: {demo_prompt}")
            
            with timer("Generating demo response"):
                response, metrics = client.generate(demo_prompt)
            
            print(f"\n📝 Response:\n{response}")
            client.print_metrics(metrics)
            
            print("\n💡 Tip: Use --interactive for chat mode or --prompt for single queries")
        
        # Save metrics if requested
        if args.save_metrics:
            client.save_metrics()
            
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
