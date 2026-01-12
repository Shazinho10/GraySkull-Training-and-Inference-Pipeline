"""training_pipeline_qwen.py - Dynamic YAML Configuration Version"""
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from datetime import datetime
import json
import yaml


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================
class TrainingConfig:
    """Load training configuration from YAML file."""
    
    def __init__(self, config_path: str = "training_config.yaml", model_type: str = "qwen"):
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


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print_info("Device", f"CUDA ({torch.cuda.get_device_name(0)})")
        print_info("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    else:
        print_warning("CUDA not available, using CPU")
        return torch.device("cpu")


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
        json.dump(data, f, indent=2)


# =============================================================================
# QWEN TRAINING PIPELINE CLASS
# =============================================================================
class QwenTrainingPipeline:
    """Training pipeline for Qwen Dolphin model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.training_log = {
            "model": config.MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "seed": config.SEED,
                "use_float16": config.USE_FLOAT16,
                "deterministic": config.DETERMINISTIC,
                "save_checkpoints": config.SAVE_CHECKPOINTS
            },
            "runs": []
        }
        
    def initialize(self):
        """Initialize the training pipeline."""
        print_header("QWEN DOLPHIN TRAINING PIPELINE", "=")
        print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Model: {self.config.MODEL_NAME}")
        
        # Step 1: Set seeds
        print_step(1, self.config.TOTAL_STEPS, "Setting up reproducibility")
        set_seed(self.config.SEED, self.config.DETERMINISTIC)
        
        # Step 2: Get device
        print_step(2, self.config.TOTAL_STEPS, "Checking compute device")
        self.device = get_device()
        
        # Step 3: Load tokenizer
        print_step(3, self.config.TOTAL_STEPS, "Loading tokenizer")
        print("  Loading Qwen tokenizer from HuggingFace...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        print_success("Tokenizer loaded")
        print_info("Vocabulary Size", str(len(self.tokenizer)))
        
        # Step 4: Load model
        print_step(4, self.config.TOTAL_STEPS, "Loading model")
        print("  Loading Qwen model weights...")
        
        dtype = torch.float16 if self.config.USE_FLOAT16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode for forward pass
        
        # Print model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print_success("Model loaded")
        print_info("Total Parameters", f"{total_params:,}")
        print_info("Trainable Parameters", f"{trainable_params:,}")
        print_info("Model Device", str(next(self.model.parameters()).device))
        print_info("Model Dtype", str(next(self.model.parameters()).dtype))
        
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            print_info("GPU Memory Used", f"{mem:.2f} GB")
        
        # Create checkpoint directories
        if self.config.SAVE_CHECKPOINTS:
            self.config.SINGLE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            self.config.BATCH_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            print_success("Checkpoint directories created")
        else:
            print_warning("Checkpoint saving disabled in config")
        
        print_header("INITIALIZATION COMPLETE", "-")
        
    def tokenize_chat(self, messages: list) -> torch.Tensor:
        """Tokenize chat messages using chat template."""
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)
    
    def run_single_forward_pass(self) -> float:
        """Run a single forward pass with configured texts."""
        print_step(5, self.config.TOTAL_STEPS, "SINGLE FORWARD PASS")
        
        losses = []
        for i, message in enumerate(self.config.SINGLE_TEXTS):
            print(f"\n  Text {i+1}/{len(self.config.SINGLE_TEXTS)}:")
            print_info("Input", message[:50] + "..." if len(message) > 50 else message)
            
            # Prepare input
            messages = [{"role": "user", "content": message}]
            inputs = self.tokenize_chat(messages)
            labels = inputs.clone()
            
            print_info("Input Tokens", str(inputs.shape[-1]))
            print("  Computing forward pass...")
            
            # Forward pass (no gradient computation)
            with torch.no_grad():
                outputs = self.model(input_ids=inputs, labels=labels)
            
            loss = outputs.loss.item()
            losses.append(loss)
            print_success("Forward pass complete")
            print_info("Loss", f"{loss:.6f}")
            
            # Log the run
            self.training_log["runs"].append({
                "type": "single",
                "message": message,
                "loss": loss,
                "tokens": inputs.shape[-1]
            })
        
        mean_loss = sum(losses) / len(losses)
        print_info("Mean Loss", f"{mean_loss:.6f}")
        
        return mean_loss
    
    def save_checkpoint(self, checkpoint_dir: Path, checkpoint_name: str) -> Path:
        """Save model checkpoint."""
        if not self.config.SAVE_CHECKPOINTS:
            print_warning(f"Checkpoint saving disabled - skipping '{checkpoint_name}'")
            return None
            
        print(f"\n  Saving checkpoint to {checkpoint_dir}...")
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training log
        log_path = checkpoint_dir / "training_log.json"
        save_training_log(log_path, self.training_log)
        
        print_success(f"Checkpoint '{checkpoint_name}' saved")
        print_info("Location", str(checkpoint_dir))
        
        # List saved files
        files = list(checkpoint_dir.glob("*"))
        print_info("Files Saved", str(len(files)))
        
        return checkpoint_dir
    
    def run_batch_forward_pass(self) -> tuple:
        """Run forward pass on configured batch of messages."""
        print_step(6, self.config.TOTAL_STEPS, "BATCH FORWARD PASS")
        print_info("Batch Size", str(len(self.config.BATCH_TEXTS)))
        
        losses = []
        
        print("\n  Processing batch:")
        with torch.no_grad():
            for i, msg in enumerate(self.config.BATCH_TEXTS):
                msg_preview = msg[:30] + "..." if len(msg) > 30 else msg
                print(f"    [{i+1}/{len(self.config.BATCH_TEXTS)}] {msg_preview}")
                
                inputs = self.tokenize_chat([{"role": "user", "content": msg}])
                labels = inputs.clone()
                
                outputs = self.model(input_ids=inputs, labels=labels)
                loss = outputs.loss.item()
                losses.append(loss)
                
                print(f"         Loss: {loss:.6f}")
                
                # Log each run
                self.training_log["runs"].append({
                    "type": "batch_item",
                    "message": msg,
                    "loss": loss,
                    "tokens": inputs.shape[-1]
                })
        
        mean_loss = sum(losses) / len(losses)
        print_success("Batch forward pass complete")
        print_info("Mean Loss", f"{mean_loss:.6f}")
        print_info("Min Loss", f"{min(losses):.6f}")
        print_info("Max Loss", f"{max(losses):.6f}")
        
        return losses, mean_loss
    
    def verify_reproducibility(self, original_loss: float, checkpoint_dir: Path) -> bool:
        """Verify that checkpoint produces same results."""
        if not self.config.SAVE_CHECKPOINTS:
            print_warning("Checkpoint saving disabled - skipping reproducibility check")
            return None
            
        print_step(8, self.config.TOTAL_STEPS, "REPRODUCIBILITY VERIFICATION")
        print("  Reloading model from checkpoint...")
        
        # Reload model from checkpoint
        dtype = torch.float16 if self.config.USE_FLOAT16 else torch.float32
        model_reload = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=dtype
        )
        model_reload.to(self.device)
        model_reload.eval()
        print_success("Model reloaded from checkpoint")
        
        # Re-run the same forward pass (use first single text)
        print("  Running forward pass on reloaded model...")
        test_message = self.config.SINGLE_TEXTS[0]
        messages = [{"role": "user", "content": test_message}]
        inputs = self.tokenize_chat(messages)
        labels = inputs.clone()
        
        with torch.no_grad():
            outputs = model_reload(input_ids=inputs, labels=labels)
        
        reload_loss = outputs.loss.item()
        
        # Compare losses
        print_info("Original Loss", f"{original_loss:.8f}")
        print_info("Reloaded Loss", f"{reload_loss:.8f}")
        print_info("Difference", f"{abs(original_loss - reload_loss):.10f}")
        
        is_reproducible = abs(original_loss - reload_loss) < self.config.TOLERANCE
        
        if is_reproducible:
            print_success(f"REPRODUCIBILITY VERIFIED (tolerance: {self.config.TOLERANCE})")
        else:
            print_error("REPRODUCIBILITY FAILED - Losses do not match")
        
        # Clean up
        del model_reload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return is_reproducible
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print_header("STARTING FULL TRAINING PIPELINE", "=")
        
        # Initialize
        self.initialize()
        
        # Single forward pass
        single_loss = self.run_single_forward_pass()
        
        # Save single checkpoint
        if self.config.SAVE_CHECKPOINTS:
            print_step(5, self.config.TOTAL_STEPS, "SAVING SINGLE CHECKPOINT")
            self.save_checkpoint(self.config.SINGLE_CHECKPOINT_DIR, "single")
        
        # Batch forward pass
        batch_losses, mean_batch_loss = self.run_batch_forward_pass()
        
        # Save batch checkpoint
        if self.config.SAVE_CHECKPOINTS:
            print_step(7, self.config.TOTAL_STEPS, "SAVING BATCH CHECKPOINT")
            self.save_checkpoint(self.config.BATCH_CHECKPOINT_DIR, "batch")
        
        # Verify reproducibility
        is_reproducible = self.verify_reproducibility(single_loss, self.config.SINGLE_CHECKPOINT_DIR)
        
        # Final summary
        print_header("TRAINING PIPELINE SUMMARY", "=")
        print_info("Model", self.config.MODEL_NAME)
        print_info("Single Run Loss", f"{single_loss:.6f}")
        print_info("Batch Mean Loss", f"{mean_batch_loss:.6f}")
        if is_reproducible is not None:
            print_info("Reproducibility", "✓ PASSED" if is_reproducible else "✗ FAILED")
        else:
            print_info("Reproducibility", "SKIPPED (checkpoints disabled)")
        
        if self.config.SAVE_CHECKPOINTS:
            print_info("Single Checkpoint", str(self.config.SINGLE_CHECKPOINT_DIR))
            print_info("Batch Checkpoint", str(self.config.BATCH_CHECKPOINT_DIR))
        
        if torch.cuda.is_available():
            print_info("Peak GPU Memory", f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        print(f"\n  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_success("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        
        return {
            "single_loss": single_loss,
            "batch_losses": batch_losses,
            "mean_batch_loss": mean_batch_loss,
            "reproducible": is_reproducible
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main(config_path: str = "training_config.yaml"):
    """
    Run the Qwen training pipeline with configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    """
    print_header("QWEN DOLPHIN TRAINING PIPELINE", "=")
    print(f"  Configuration: {config_path}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create configuration from YAML
    config = TrainingConfig(config_path=config_path, model_type="qwen")
    
    # Create and run pipeline
    pipeline = QwenTrainingPipeline(config)
    results = pipeline.run_full_pipeline()
    
    # Return results for external use
    return results


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "training_config.yaml"
    results = main(config_file)