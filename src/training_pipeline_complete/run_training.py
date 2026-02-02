"""run_training.py - Main wrapper to run training for any supported model"""
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training_pipeline_complete.base import (
    TrainingConfig,
    print_header,
    print_info,
    print_success,
    print_error
)
from src.training_pipeline_complete.pipelines import get_pipeline, PEFT_AVAILABLE
from src.utils.path import TRAINING_CONFIG_PATH_V2


def run_training(
    model_type: str = "qwen",
    config_path: str = None,
    mode: str = "full"
) -> dict:
    """
    Run the training pipeline for the specified model.
    
    Args:
        model_type: Type of model to train ('qwen' or 'llama')
        config_path: Path to YAML configuration file (optional)
        mode: Training mode - 'full' for complete training, 'test' for forward pass only
    
    Returns:
        Dictionary with training results
    """
    if config_path is None:
        config_path = TRAINING_CONFIG_PATH_V2
    
    print_header(f"UNIFIED TRAINING PIPELINE", "=")
    print(f"  Model Type: {model_type.upper()}")
    print(f"  Configuration: {config_path}")
    print(f"  Mode: {mode}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PEFT/LoRA Available: {PEFT_AVAILABLE}")
    
    try:
        # Create configuration from YAML
        config = TrainingConfig(config_path=config_path, model_type=model_type)
        
        # Print configuration summary
        config.print_summary()
        
        # Create appropriate pipeline
        pipeline = get_pipeline(model_type, config)
        
        # Run based on mode
        if mode == "full":
            results = pipeline.run_full_pipeline()
        elif mode == "test":
            # Just initialize and run forward pass
            pipeline.initialize()
            single_loss = pipeline.run_single_forward_pass()
            batch_loss = pipeline.run_batch_forward_pass()
            results = {
                'single_loss': single_loss,
                'batch_loss': batch_loss,
                'mode': 'test'
            }
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'full' or 'test'")
        
        print_success("Training completed successfully!")
        return results
        
    except Exception as e:
        print_error(f"Training failed: {e}")
        raise


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Unified Training Pipeline for LLM Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Qwen model (default)
  python -m src.training_pipeline_complete.run_training
  
  # Train Llama model
  python -m src.training_pipeline_complete.run_training --model llama
  
  # Test mode (forward pass only)
  python -m src.training_pipeline_complete.run_training --model qwen --mode test
  
  # Custom config file
  python -m src.training_pipeline_complete.run_training --config path/to/config.yaml
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="qwen",
        choices=["qwen", "llama"],
        help="Model type to train (default: qwen)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML configuration file (default: training_config.yaml)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "test"],
        help="Training mode: 'full' for complete training, 'test' for forward pass only (default: full)"
    )
    
    args = parser.parse_args()
    
    # Run training
    results = run_training(
        model_type=args.model,
        config_path=args.config,
        mode=args.mode
    )
    
    return results


if __name__ == "__main__":
    results = main()
