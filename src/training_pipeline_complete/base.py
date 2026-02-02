"""base.py - Shared utilities, configuration, dataset, and metrics classes"""
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import yaml
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

# Try to import datasets for external dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not installed. External datasets will not be available.")


# =============================================================================
# PRINT UTILITIES
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


# =============================================================================
# DEVICE UTILITIES
# =============================================================================
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
# CONFIGURATION LOADER
# =============================================================================
class TrainingConfig:
    """Load training configuration from YAML file."""
    
    def __init__(self, config_path: str, model_type: str = "qwen"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            model_type: Type of model to configure ('qwen' or 'llama')
        """
        self.model_type = model_type
        
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
        self.FINAL_CHECKPOINT_DIR = Path(model_config.get('final_checkpoint_dir', f'checkpoints_{model_type}/final'))
        
        # Load training hyperparameters
        training_config = config.get('training', {})
        self.NUM_EPOCHS = training_config.get('num_epochs', 3)
        self.BATCH_SIZE = training_config.get('batch_size', 2)
        self.GRADIENT_ACCUMULATION_STEPS = training_config.get('gradient_accumulation_steps', 2)
        self.MAX_SEQ_LENGTH = training_config.get('max_seq_length', 512)
        self.LEARNING_RATE = training_config.get('learning_rate', 2e-5)
        self.WEIGHT_DECAY = training_config.get('weight_decay', 0.01)
        self.WARMUP_RATIO = training_config.get('warmup_ratio', 0.1)
        self.MAX_GRAD_NORM = training_config.get('max_grad_norm', 1.0)
        self.SCHEDULER_TYPE = training_config.get('scheduler_type', 'cosine')
        self.LOGGING_STEPS = training_config.get('logging_steps', 1)
        self.EVAL_STEPS = training_config.get('eval_steps', 5)
        self.SAVE_STEPS = training_config.get('save_steps', 10)
        self.SAVE_TOTAL_LIMIT = training_config.get('save_total_limit', 3)
        
        # LoRA settings
        self.USE_LORA = training_config.get('use_lora', True)
        lora_config = training_config.get('lora', {})
        self.LORA_R = lora_config.get('r', 16)
        self.LORA_ALPHA = lora_config.get('alpha', 32)
        self.LORA_DROPOUT = lora_config.get('dropout', 0.05)
        self.LORA_TARGET_MODULES = lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.LORA_BIAS = lora_config.get('bias', 'none')
        
        # Load training data
        training_data = config.get('training_data', {})
        
        # External dataset configuration
        self.USE_EXTERNAL_DATASET = training_data.get('use_external_dataset', False)
        if self.USE_EXTERNAL_DATASET:
            ext_config = training_data.get('external_dataset', {})
            self.EXTERNAL_DATASET_NAME = ext_config.get('name', '')
            self.EXTERNAL_DATASET_SPLIT = ext_config.get('split', 'train')
            self.EXTERNAL_DATASET_STREAMING = ext_config.get('streaming', True)
            self.EXTERNAL_DATASET_TEST_MODE = ext_config.get('test_mode', True)
            self.EXTERNAL_DATASET_TEST_SAMPLES = ext_config.get('test_samples', 100)
            self.EXTERNAL_DATASET_MAX_SAMPLES = ext_config.get('max_samples', None)
            self.EXTERNAL_DATASET_VAL_RATIO = ext_config.get('val_ratio', 0.1)
            self.EXTERNAL_DATASET_SHUFFLE_SEED = ext_config.get('shuffle_seed', 42)
            # Column mapping
            col_mapping = ext_config.get('column_mapping', {})
            self.INSTRUCTION_COLUMN = col_mapping.get('instruction', 'instruction')
            self.RESPONSE_COLUMN = col_mapping.get('response', 'response')
        
        # Fallback training data
        self.TRAIN_SAMPLES = training_data.get('train_samples', [])
        self.VAL_SAMPLES = training_data.get('val_samples', [])
        self.SINGLE_TEXTS = training_data.get('single_texts', [])
        self.BATCH_TEXTS = training_data.get('batch_texts', [])
        
        # Metrics configuration
        metrics_config = config.get('metrics', {})
        self.METRICS_OUTPUT_DIR = Path(metrics_config.get('output_dir', 'training_metrics'))
        self.SAVE_METRICS_EVERY_EPOCH = metrics_config.get('save_metrics_every_epoch', True)
        self.GENERATE_PLOTS = metrics_config.get('generate_plots', True)
        
        # Load pipeline settings
        pipeline_config = config.get('pipeline', {})
        self.TOTAL_STEPS = pipeline_config.get('total_steps', 8)
        self.REPRODUCIBILITY_TOLERANCE = pipeline_config.get('reproducibility_tolerance', {'float32': 1e-7, 'float16': 1e-5})
        
        # Set tolerance based on float type
        if self.USE_FLOAT16:
            self.TOLERANCE = self.REPRODUCIBILITY_TOLERANCE['float16']
        else:
            self.TOLERANCE = self.REPRODUCIBILITY_TOLERANCE['float32']
    
    def print_summary(self):
        """Print configuration summary."""
        print_header("CONFIGURATION SUMMARY", "-")
        print_info("Model Type", self.model_type.upper())
        print_info("Model", self.MODEL_NAME)
        print_info("Epochs", str(self.NUM_EPOCHS))
        print_info("Batch Size", str(self.BATCH_SIZE))
        print_info("Learning Rate", str(self.LEARNING_RATE))
        print_info("Use LoRA", str(self.USE_LORA))
        print_info("Use Float16", str(self.USE_FLOAT16))
        print_info("External Dataset", str(self.USE_EXTERNAL_DATASET))
        if not self.USE_EXTERNAL_DATASET:
            print_info("Training Samples", str(len(self.TRAIN_SAMPLES)))
            print_info("Validation Samples", str(len(self.VAL_SAMPLES)))


# =============================================================================
# CUSTOM DATASET CLASS
# =============================================================================
class InstructionDataset(Dataset):
    """Dataset for instruction-response pairs."""
    
    def __init__(
        self, 
        samples: List[Dict], 
        tokenizer, 
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of dicts with 'instruction' and 'response' keys
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Format as chat messages
        messages = [
            {"role": "user", "content": sample['instruction']},
            {"role": "assistant", "content": sample['response']}
        ]
        
        # Tokenize using chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        
        # Labels are same as input_ids for causal LM
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# =============================================================================
# EXTERNAL DATASET LOADER
# =============================================================================
def load_external_dataset(config: TrainingConfig) -> Tuple[List[Dict], List[Dict]]:
    """Load training data from external HuggingFace dataset."""
    print("\n  Loading external dataset...")
    print_info("Dataset", config.EXTERNAL_DATASET_NAME)
    print_info("Streaming", str(config.EXTERNAL_DATASET_STREAMING))
    
    # Determine number of samples to load
    if config.EXTERNAL_DATASET_TEST_MODE:
        num_samples = config.EXTERNAL_DATASET_TEST_SAMPLES
        print_info("Mode", f"TEST ({num_samples} samples)")
    elif config.EXTERNAL_DATASET_MAX_SAMPLES:
        num_samples = config.EXTERNAL_DATASET_MAX_SAMPLES
        print_info("Mode", f"LIMITED ({num_samples} samples)")
    else:
        num_samples = None  # Load all
        print_info("Mode", "FULL DATASET")
    
    try:
        # Load dataset
        ds = load_dataset(
            config.EXTERNAL_DATASET_NAME,
            split=config.EXTERNAL_DATASET_SPLIT,
            streaming=config.EXTERNAL_DATASET_STREAMING
        )
        
        # Collect samples
        samples = []
        instruction_col = config.INSTRUCTION_COLUMN
        response_col = config.RESPONSE_COLUMN
        
        print(f"  Collecting samples (instruction: '{instruction_col}', response: '{response_col}')...")
        
        for i, item in enumerate(ds):
            if num_samples and i >= num_samples:
                break
            
            instruction = item.get(instruction_col, '')
            response = item.get(response_col, '')
            
            if instruction and response:
                samples.append({
                    'instruction': instruction,
                    'response': response
                })
            
            if (i + 1) % 50 == 0:
                print(f"    Loaded {i + 1} samples...")
        
        print_success(f"Loaded {len(samples)} samples from external dataset")
        
        # Shuffle and split into train/val
        random.seed(config.EXTERNAL_DATASET_SHUFFLE_SEED)
        random.shuffle(samples)
        
        val_size = int(len(samples) * config.EXTERNAL_DATASET_VAL_RATIO)
        val_samples = samples[:val_size]
        train_samples = samples[val_size:]
        
        print_info("Training Samples", str(len(train_samples)))
        print_info("Validation Samples", str(len(val_samples)))
        
        return train_samples, val_samples
        
    except Exception as e:
        print_error(f"Failed to load external dataset: {e}")
        print_warning("Falling back to config samples")
        return config.TRAIN_SAMPLES, config.VAL_SAMPLES


# =============================================================================
# METRICS TRACKER CLASS
# =============================================================================
class MetricsTracker:
    """Track and store training metrics."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'perplexity': [],
            'val_perplexity': [],
            'learning_rate': [],
            'gradient_norm': [],
            'epoch': [],
            'step': [],
            'timestamp': []
        }
        
        self.epoch_metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_perplexity': [],
            'val_perplexity': [],
            'learning_rate': [],
            'duration_seconds': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def log_step(
        self,
        step: int,
        epoch: int,
        train_loss: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None
    ):
        """Log metrics for a training step."""
        self.metrics['step'].append(step)
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['perplexity'].append(math.exp(min(train_loss, 20)))  # Cap to avoid overflow
        self.metrics['timestamp'].append(datetime.now().isoformat())
        
        if gradient_norm is not None:
            self.metrics['gradient_norm'].append(gradient_norm)
    
    def log_validation(self, val_loss: float):
        """Log validation metrics."""
        self.metrics['val_loss'].append(val_loss)
        self.metrics['val_perplexity'].append(math.exp(min(val_loss, 20)))
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        learning_rate: float,
        duration: float
    ):
        """Log metrics for an epoch."""
        self.epoch_metrics['epoch'].append(epoch)
        self.epoch_metrics['train_loss'].append(train_loss)
        self.epoch_metrics['val_loss'].append(val_loss)
        self.epoch_metrics['train_perplexity'].append(math.exp(min(train_loss, 20)))
        self.epoch_metrics['val_perplexity'].append(math.exp(min(val_loss, 20)))
        self.epoch_metrics['learning_rate'].append(learning_rate)
        self.epoch_metrics['duration_seconds'].append(duration)
        
        # Track best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
    
    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save all metrics to JSON file."""
        filepath = self.output_dir / filename
        
        all_metrics = {
            'step_metrics': self.metrics,
            'epoch_metrics': self.epoch_metrics,
            'summary': {
                'best_val_loss': self.best_val_loss,
                'best_epoch': self.best_epoch,
                'total_steps': len(self.metrics['step']),
                'total_epochs': len(self.epoch_metrics['epoch'])
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print_success(f"Metrics saved to {filepath}")
        return filepath
    
    def generate_plots(self):
        """Generate training visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Metrics', fontsize=14, fontweight='bold')
        
        # Plot 1: Training Loss over Steps
        ax1 = axes[0, 0]
        ax1.plot(self.metrics['step'], self.metrics['train_loss'], 'b-', label='Train Loss', alpha=0.7)
        if self.metrics['val_loss']:
            val_steps = self.metrics['step'][-len(self.metrics['val_loss']):]
            ax1.plot(val_steps, self.metrics['val_loss'], 'r-', label='Val Loss', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss over Training Steps')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss per Epoch
        ax2 = axes[0, 1]
        if self.epoch_metrics['epoch']:
            ax2.plot(self.epoch_metrics['epoch'], self.epoch_metrics['train_loss'], 'b-o', label='Train Loss')
            ax2.plot(self.epoch_metrics['epoch'], self.epoch_metrics['val_loss'], 'r-o', label='Val Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Loss per Epoch')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Perplexity
        ax3 = axes[1, 0]
        ax3.plot(self.metrics['step'], self.metrics['perplexity'], 'g-', label='Train Perplexity', alpha=0.7)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Perplexity')
        ax3.set_title('Perplexity over Training Steps')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate Schedule
        ax4 = axes[1, 1]
        ax4.plot(self.metrics['step'], self.metrics['learning_rate'], 'm-', label='Learning Rate')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'training_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print_success(f"Training plots saved to {plot_path}")
        return plot_path
    
    def print_summary(self):
        """Print training summary."""
        print_header("TRAINING METRICS SUMMARY", "-")
        
        if self.epoch_metrics['epoch']:
            print_info("Total Epochs", str(len(self.epoch_metrics['epoch'])))
            print_info("Total Steps", str(len(self.metrics['step'])))
            print_info("Best Epoch", str(self.best_epoch))
            print_info("Best Val Loss", f"{self.best_val_loss:.6f}")
            print_info("Best Val Perplexity", f"{math.exp(min(self.best_val_loss, 20)):.4f}")
            
            # Final metrics
            print_info("Final Train Loss", f"{self.epoch_metrics['train_loss'][-1]:.6f}")
            print_info("Final Val Loss", f"{self.epoch_metrics['val_loss'][-1]:.6f}")
            
            # Improvement
            if len(self.epoch_metrics['train_loss']) > 1:
                initial_loss = self.epoch_metrics['train_loss'][0]
                final_loss = self.epoch_metrics['train_loss'][-1]
                improvement = ((initial_loss - final_loss) / initial_loss) * 100
                print_info("Training Loss Improvement", f"{improvement:.2f}%")
