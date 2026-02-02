"""training_pipeline_llama.py - Dynamic YAML Configuration Version"""
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from datetime import datetime
import json
import yaml
from src.utils.path import TRAINING_CONFIG_PATH


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================
class TrainingConfig:
    """Load training configuration from YAML file."""
    
    def __init__(self, config_path: str = TRAINING_CONFIG_PATH, model_type: str = "llama"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            model_type: Type of model to configure ('qwen' or 'llama')
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load global settings
        global_config = config['global']
        self.SEED = global_config['seed']
        self.SAVE_CHECKPOINTS = global_config['save_checkpoints']
        
        # Load model-specific settings
        model_config = config['models'][model_type]
        self.MODEL_NAME = model_config['name']
        self.USE_FLOAT16 = model_config['use_float16']
        self.DETERMINISTIC = model_config['deterministic']
        self.CHECKPOINT_DIR = Path(model_config['checkpoint_dir'])
        self.SINGLE_CHECKPOINT_DIR = Path(model_config['single_checkpoint_dir'])
        self.BATCH_CHECKPOINT_DIR = Path(model_config['batch_checkpoint_dir'])
        
        # Load training data
        training_data = config['training_data']
        self.SINGLE_TEXTS = training_data['single_texts']
        self.BATCH_TEXTS = training_data['batch_texts']
        
        # Load pipeline settings
        pipeline_config = config['pipeline']
        self.TOTAL_STEPS = pipeline_config['total_steps']
        self.REPRODUCIBILITY_TOLERANCE = pipeline_config['reproducibility_tolerance']
        
        # Set tolerance based on float type
        if self.USE_FLOAT16:
            self.TOLERANCE = self.REPRODUCIBILITY_TOLERANCE['float16']
        else:
            self.TOLERANCE = self.REPRODUCIBILITY_TOLERANCE['float32']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def print_header(text: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def print_step(step_num: int, total_steps: int, description: str):
    """Print a step indicator with progress."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 50)


def print_info(key: str, value: str):
    """Print key-value information."""
    print(f"  → {key}: {value}")


def print_success(message: str):
    """Print success message."""
    print(f"  ✓ {message}")


def print_warning(message: str):
    """Print warning message."""
    print(f"  ⚠ {message}")


def print_error(message: str):
    """Print error message."""
    print(f"  ✗ {message}")


def print_gpu_memory():
    """Print detailed GPU memory information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"  → GPU Memory: Allocated={allocated:.2f}GB | Reserved={reserved:.2f}GB | Peak={max_allocated:.2f}GB")
    else:
        print("  → GPU: Not available")


def get_device_info():
    """Get detailed device information."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return device, {
            "type": "cuda",
            "name": device_name,
            "total_memory_gb": total_memory
        }
    else:
        return torch.device("cpu"), {"type": "cpu", "name": "CPU"}


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if deterministic:
        torch.use_deterministic_algorithms(True)
        print_info("Deterministic Mode", "ENABLED")
    
    print_info("Seed", str(seed))
    print_success("Random seeds set for reproducibility")


def save_training_log(log_path: Path, data: dict):
    """Save training log to JSON file."""
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# =============================================================================
# LLAMA TRAINING PIPELINE CLASS
# =============================================================================
class LlamaTrainingPipeline:
    """Training pipeline for Llama Dolphin model (8B parameters)."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.device_info = None
        self.training_log = {
            "model": config.MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "seed": config.SEED,
                "use_float16": config.USE_FLOAT16,
                "deterministic": config.DETERMINISTIC,
                "save_checkpoints": config.SAVE_CHECKPOINTS
            },
            "device": None,
            "runs": []
        }
        
    def initialize(self):
        """Initialize the training pipeline."""
        print_header("LLAMA DOLPHIN TRAINING PIPELINE - INITIALIZATION", "=")
        print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Model: {self.config.MODEL_NAME}")
        print(f"  Model Size: 8 Billion Parameters")
        
        # Step 1: Set seeds
        print_step(1, self.config.TOTAL_STEPS, "Setting up reproducibility")
        set_seed(self.config.SEED, self.config.DETERMINISTIC)
        
        # Step 2: Check device
        print_step(2, self.config.TOTAL_STEPS, "Checking compute device")
        self.device, self.device_info = get_device_info()
        self.training_log["device"] = self.device_info
        
        if self.device.type == "cuda":
            print_success(f"CUDA available: {self.device_info['name']}")
            print_info("Total GPU Memory", f"{self.device_info['total_memory_gb']:.2f} GB")
            print_gpu_memory()
        else:
            print_warning("CUDA not available - Running on CPU")
            print_warning("This will be VERY SLOW for an 8B model!")
        
        # Step 3: Load tokenizer
        print_step(3, self.config.TOTAL_STEPS, "Loading Llama tokenizer")
        print("  Downloading/loading tokenizer from HuggingFace...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        print_success("Tokenizer loaded")
        print_info("Vocabulary Size", str(len(self.tokenizer)))
        
        # Step 4: Load model
        print_step(4, self.config.TOTAL_STEPS, "Loading Llama model (8B parameters)")
        print("  ⏳ This may take several minutes...")
        print_info("Float16 Mode", str(self.config.USE_FLOAT16))
        print_info("Device Map", "auto")
        
        dtype = torch.float16 if self.config.USE_FLOAT16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            device_map="auto",
            torch_dtype=dtype
        )
        self.model.eval()
        
        # Print model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print_success("Model loaded successfully!")
        print_info("Total Parameters", f"{total_params:,}")
        print_info("Trainable Parameters", f"{trainable_params:,}")
        print_info("Model Dtype", str(next(self.model.parameters()).dtype))
        print_info("Model Device", str(next(self.model.parameters()).device))
        
        if torch.cuda.is_available():
            print_gpu_memory()
        
        # Create checkpoint directories
        if self.config.SAVE_CHECKPOINTS:
            self.config.SINGLE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            self.config.BATCH_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            print_success("Checkpoint directories created")
            print_info("Checkpoint Root", str(self.config.CHECKPOINT_DIR))
        else:
            print_warning("Checkpoint saving disabled in config")
        
        print_header("INITIALIZATION COMPLETE", "-")
        
    def tokenize_input(self, texts: list, padding: bool = True) -> dict:
        """Tokenize input texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=padding)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs
    
    def run_single_forward_pass(self) -> float:
        """Run a single forward pass with configured texts."""
        print_step(5, self.config.TOTAL_STEPS, "SINGLE FORWARD PASS (Dry-run)")
        print_info("Number of Texts", str(len(self.config.SINGLE_TEXTS)))
        
        for i, text in enumerate(self.config.SINGLE_TEXTS):
            preview = text[:40] + "..." if len(text) > 40 else text
            print(f"    Text {i+1}: {preview}")
        
        # Tokenize
        print("\n  Tokenizing inputs...")
        inputs = self.tokenize_input(self.config.SINGLE_TEXTS, padding=True)
        labels = inputs["input_ids"].clone()
        
        print_info("Input Shape", str(inputs["input_ids"].shape))
        print_info("Total Tokens", str(inputs["input_ids"].numel()))
        
        # Forward pass
        print("  Computing forward pass...")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
        
        loss = outputs.loss.item()
        print_success("Forward pass complete")
        print_info("Loss", f"{loss:.6f}")
        
        if torch.cuda.is_available():
            print_gpu_memory()
        
        # Log the run
        self.training_log["runs"].append({
            "type": "single_batch",
            "texts": self.config.SINGLE_TEXTS,
            "loss": loss,
            "num_tokens": inputs["input_ids"].numel()
        })
        
        return loss
    
    def save_checkpoint(self, checkpoint_dir: Path, checkpoint_name: str) -> Path:
        """Save model checkpoint."""
        if not self.config.SAVE_CHECKPOINTS:
            print_warning(f"Checkpoint saving disabled - skipping '{checkpoint_name}'")
            return None
            
        print(f"\n  Saving checkpoint '{checkpoint_name}'...")
        print_info("Location", str(checkpoint_dir))
        
        # Save model and tokenizer
        print("  Saving model weights...")
        # Use safe_serialization=False to avoid meta tensor issues with device_map="auto"
        try:
            self.model.save_pretrained(
                checkpoint_dir,
                safe_serialization=False,  # Use PyTorch .bin format instead of safetensors
                max_shard_size="2GB"
            )
            print_success("Model weights saved")
            
            # Verify save was successful
            has_index = (checkpoint_dir / "pytorch_model.bin.index.json").exists()
            has_sharded = any(checkpoint_dir.glob("pytorch_model-*.bin"))
            has_config = (checkpoint_dir / "config.json").exists()
            
            if not (has_index or has_sharded) and not has_config:
                print_warning("Model save may have failed - no checkpoint files found")
            else:
                if has_index:
                    print_info("Checkpoint Format", "Sharded PyTorch (with index)")
                elif has_sharded:
                    print_info("Checkpoint Format", "Sharded PyTorch")
                else:
                    print_info("Checkpoint Format", "Single file")
                    
        except Exception as e:
            print_error(f"Model save failed: {str(e)[:200]}")
            print_warning("Checkpoint may be incomplete")
            # Continue anyway to save tokenizer and log

        print("  Saving tokenizer...")
        try:
            self.tokenizer.save_pretrained(checkpoint_dir)
            print_success("Tokenizer saved")
        except Exception as e:
            print_error(f"Tokenizer save failed: {str(e)[:200]}")
        
        # Save training log
        log_path = checkpoint_dir / "training_log.json"
        try:
            save_training_log(log_path, self.training_log)
            print_success("Training log saved")
        except Exception as e:
            print_warning(f"Training log save failed: {str(e)[:100]}")
        
        print_success(f"Checkpoint '{checkpoint_name}' save process completed")
        
        # List saved files
        files = list(checkpoint_dir.glob("*"))
        if files:
            print_info("Files Saved", str(len(files)))
            for f in files[:5]:  # Show first 5 files
                print(f"    - {f.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more files")
        else:
            print_warning("No files found in checkpoint directory!")
        
        return checkpoint_dir
    
    def run_batch_forward_pass(self) -> float:
        """Run forward pass on configured batch of texts."""
        print_step(6, self.config.TOTAL_STEPS, "BATCH FORWARD PASS (Dry-run)")
        print_info("Batch Size", str(len(self.config.BATCH_TEXTS)))
        
        # Tokenize all at once
        print("\n  Tokenizing batch...")
        inputs = self.tokenize_input(self.config.BATCH_TEXTS, padding=True)
        labels = inputs["input_ids"].clone()
        
        print_info("Batch Shape", str(inputs["input_ids"].shape))
        
        # Forward pass
        print("  Computing batch forward pass...")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
        
        batch_loss = outputs.loss.item()
        print_success("Batch forward pass complete")
        print_info("Batch Loss", f"{batch_loss:.6f}")
        
        if torch.cuda.is_available():
            print_gpu_memory()
        
        # Log the batch run
        self.training_log["runs"].append({
            "type": "batch",
            "texts": self.config.BATCH_TEXTS,
            "loss": batch_loss,
            "batch_size": len(self.config.BATCH_TEXTS)
        })
        
        return batch_loss
    
    def verify_reproducibility(self, original_loss: float, checkpoint_dir: Path) -> bool:
        """Verify that checkpoint produces same results."""
        if not self.config.SAVE_CHECKPOINTS:
            print_warning("Checkpoint saving disabled - skipping reproducibility check")
            return None
            
        print_step(8, self.config.TOTAL_STEPS, "REPRODUCIBILITY VERIFICATION")
        print("  Reloading model from checkpoint...")
        
        # Check if checkpoint directory exists
        if not checkpoint_dir.exists():
            print_error(f"Checkpoint directory does not exist: {checkpoint_dir}")
            return False
        
        # Check for checkpoint files
        has_index = (checkpoint_dir / "pytorch_model.bin.index.json").exists()
        has_sharded = any(checkpoint_dir.glob("pytorch_model-*.bin"))
        has_single = (checkpoint_dir / "pytorch_model.bin").exists()
        
        if not (has_index or has_sharded or has_single):
            print_error("No checkpoint files found in directory")
            print_info("Available files", ", ".join([f.name for f in checkpoint_dir.glob("*")[:5]]))
            return False
        
        # Reload model
        dtype = torch.float16 if self.config.USE_FLOAT16 else torch.float32
        
        try:
            # Try loading directly from checkpoint (should work with index.json)
            print("  Loading from checkpoint directory...")
            # Use use_safetensors=False to force PyTorch format loading
            model_reload = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_dir),  # Convert Path to string
                device_map="auto",
                torch_dtype=dtype,
                use_safetensors=False  # Force PyTorch format since we saved with safe_serialization=False
            )
            print_success("Model loaded from checkpoint")
        except Exception as e:
            error_msg = str(e)
            print_warning(f"Direct load failed: {error_msg[:150]}")
            
            # If it's a safetensors error, try again with explicit format
            if "safetensors" in error_msg.lower():
                print("  Retrying with explicit PyTorch format...")
                try:
                    model_reload = AutoModelForCausalLM.from_pretrained(
                        str(checkpoint_dir),
                        device_map="auto",
                        torch_dtype=dtype,
                        use_safetensors=False,
                        ignore_mismatched_sizes=True
                    )
                    print_success("Model loaded from checkpoint (retry successful)")
                except Exception as e2:
                    print_error(f"Retry also failed: {str(e2)[:150]}")
                    print_warning("Skipping reproducibility check - checkpoint loading failed")
                    return False
            else:
                print_error(f"Could not load checkpoint: {error_msg[:150]}")
                print_warning("Skipping reproducibility check")
                return False
            
        model_reload.eval()
        
        # Re-run the same forward pass
        print("  Running forward pass on reloaded model...")
        inputs = self.tokenize_input(self.config.SINGLE_TEXTS, padding=True)
        labels = inputs["input_ids"].clone()
        
        with torch.no_grad():
            outputs = model_reload(**inputs, labels=labels)
        
        reload_loss = outputs.loss.item()
        
        # Compare losses
        print_info("Original Loss", f"{original_loss:.8f}")
        print_info("Reloaded Loss", f"{reload_loss:.8f}")
        
        diff = abs(original_loss - reload_loss)
        print_info("Absolute Difference", f"{diff:.10f}")
        
        is_reproducible = diff < self.config.TOLERANCE
        
        if is_reproducible:
            print_success(f"REPRODUCIBILITY VERIFIED (tolerance: {self.config.TOLERANCE})")
        else:
            print_error("REPRODUCIBILITY FAILED")
        
        # Clean up
        del model_reload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return is_reproducible
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print_header("LLAMA DOLPHIN FULL TRAINING PIPELINE", "=")
        print("  This pipeline will:")
        print("    1. Initialize model and tokenizer")
        print("    2. Run single forward pass")
        if self.config.SAVE_CHECKPOINTS:
            print("    3. Save single checkpoint")
        print("    4. Run batch forward pass")
        if self.config.SAVE_CHECKPOINTS:
            print("    5. Save batch checkpoint")
            print("    6. Verify reproducibility")
        
        # Initialize
        self.initialize()
        
        # Single forward pass
        single_loss = self.run_single_forward_pass()
        
        is_reproducible = None
        
        # Save single checkpoint (if enabled)
        if self.config.SAVE_CHECKPOINTS:
            print_step(5, self.config.TOTAL_STEPS, "SAVING SINGLE CHECKPOINT")
            self.save_checkpoint(self.config.SINGLE_CHECKPOINT_DIR, "single")
        else:
            print_step(5, self.config.TOTAL_STEPS, "SKIPPING CHECKPOINT SAVE")
            print_warning("Checkpoint saving disabled in config")
        
        # Batch forward pass
        batch_loss = self.run_batch_forward_pass()
        
        # Save batch checkpoint (if enabled)
        if self.config.SAVE_CHECKPOINTS:
            print_step(7, self.config.TOTAL_STEPS, "SAVING BATCH CHECKPOINT")
            self.save_checkpoint(self.config.BATCH_CHECKPOINT_DIR, "batch")
            
            # Verify reproducibility (only if checkpoints are saved)
            is_reproducible = self.verify_reproducibility(
                single_loss, 
                self.config.SINGLE_CHECKPOINT_DIR
            )
        else:
            print_step(7, self.config.TOTAL_STEPS, "SKIPPING BATCH CHECKPOINT")
            print_warning("Checkpoint saving disabled in config")
        
        # Final summary
        print_header("TRAINING PIPELINE SUMMARY", "=")
        print_info("Model", self.config.MODEL_NAME)
        print_info("Device", self.device_info.get("name", "Unknown"))
        print_info("Float16 Mode", str(self.config.USE_FLOAT16))
        print_info("Single Run Loss", f"{single_loss:.6f}")
        print_info("Batch Run Loss", f"{batch_loss:.6f}")
        if is_reproducible is not None:
            print_info("Reproducibility", "✓ PASSED" if is_reproducible else "✗ FAILED")
        else:
            print_info("Reproducibility", "SKIPPED (checkpoints disabled)")
        print_info("Save Checkpoints", str(self.config.SAVE_CHECKPOINTS))
        if self.config.SAVE_CHECKPOINTS:
            print_info("Single Checkpoint", str(self.config.SINGLE_CHECKPOINT_DIR))
            print_info("Batch Checkpoint", str(self.config.BATCH_CHECKPOINT_DIR))
        
        if torch.cuda.is_available():
            print_info("Peak GPU Memory", f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        print(f"\n  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_success("LLAMA TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        
        return {
            "single_loss": single_loss,
            "batch_loss": batch_loss,
            "reproducible": is_reproducible
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main(config_path: str = "training_config.yaml"):
    """
    Run the Llama training pipeline with configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    """
    print_header("LLAMA DOLPHIN TRAINING PIPELINE", "=")
    print(f"  Model: 8 Billion Parameters")
    print(f"  Configuration: {config_path}")
    print("  Purpose: Forward pass training with checkpointing")
    print("  ⚠ Note: This is a large model - GPU with 16GB+ VRAM recommended")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create configuration from YAML
    config = TrainingConfig(config_path=config_path, model_type="llama")
    
    # Create and run pipeline
    pipeline = LlamaTrainingPipeline(config)
    results = pipeline.run_full_pipeline()
    
    return results


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else TRAINING_CONFIG_PATH
    results = main(config_file)