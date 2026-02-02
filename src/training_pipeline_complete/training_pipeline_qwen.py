"""training_pipeline_qwen.py - Complete Training Pipeline with Metrics"""
import torch
import random
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_scheduler,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import yaml
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from src.utils.path import TRAINING_CONFIG_PATH_V2

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. LoRA will not be available.")

# Try to import datasets for external dataset loading
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not installed. External datasets will not be available.")


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================
class TrainingConfig:
    """Load training configuration from YAML file."""
    
    def __init__(self, config_path: str = TRAINING_CONFIG_PATH_V2, model_type: str = "qwen"):
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
        self.FINAL_CHECKPOINT_DIR = Path(model_config.get('final_checkpoint_dir', 'checkpoints/final'))
        
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
        training_data = config['training_data']
        
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


# =============================================================================
# QWEN TRAINING PIPELINE CLASS
# =============================================================================
class QwenTrainingPipeline:
    """Complete training pipeline for Qwen Dolphin model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.optimizer = None
        self.scheduler = None
        self.metrics_tracker = MetricsTracker(config.METRICS_OUTPUT_DIR)
        
        self.training_log = {
            "model": config.MODEL_NAME,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "seed": config.SEED,
                "use_float16": config.USE_FLOAT16,
                "deterministic": config.DETERMINISTIC,
                "save_checkpoints": config.SAVE_CHECKPOINTS,
                "num_epochs": config.NUM_EPOCHS,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "use_lora": config.USE_LORA
            },
            "runs": []
        }
        
    def initialize(self):
        """Initialize the training pipeline."""
        print_header("QWEN DOLPHIN TRAINING PIPELINE", "=")
        print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Model: {self.config.MODEL_NAME}")
        
        # Step 1: Set seeds
        print_step(1, 8, "Setting up reproducibility")
        set_seed(self.config.SEED, self.config.DETERMINISTIC)
        
        # Step 2: Get device
        print_step(2, 8, "Checking compute device")
        self.device = get_device()
        
        # Step 3: Load tokenizer
        print_step(3, 8, "Loading tokenizer")
        print("  Loading Qwen tokenizer from HuggingFace...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        print_success("Tokenizer loaded")
        print_info("Vocabulary Size", str(len(self.tokenizer)))
        print_info("Pad Token", str(self.tokenizer.pad_token))
        
        # Step 4: Load model
        print_step(4, 8, "Loading model")
        print("  Loading Qwen model weights...")
        
        dtype = torch.float16 if self.config.USE_FLOAT16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=dtype
        )
        self.model.to(self.device)
        
        # Print model statistics before LoRA
        total_params = sum(p.numel() for p in self.model.parameters())
        print_success("Base model loaded")
        print_info("Total Parameters", f"{total_params:,}")
        
        # Apply LoRA if enabled and available
        if self.config.USE_LORA and PEFT_AVAILABLE:
            print_step(5, 8, "Applying LoRA adapters")
            self._apply_lora()
        elif self.config.USE_LORA and not PEFT_AVAILABLE:
            print_warning("LoRA requested but PEFT not installed. Training full model.")
        
        # Print final model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print_info("Trainable Parameters", f"{trainable_params:,}")
        print_info("Training Ratio", f"{100 * trainable_params / total_params:.2f}%")
        print_info("Model Device", str(next(self.model.parameters()).device))
        print_info("Model Dtype", str(next(self.model.parameters()).dtype))
        
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1e9
            print_info("GPU Memory Used", f"{mem:.2f} GB")
        
        # Create checkpoint directories
        if self.config.SAVE_CHECKPOINTS:
            self.config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            self.config.FINAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            print_success("Checkpoint directories created")
        
        print_header("INITIALIZATION COMPLETE", "-")
    
    def _load_external_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Load training data from external HuggingFace dataset."""
        print("\n  Loading external dataset...")
        print_info("Dataset", self.config.EXTERNAL_DATASET_NAME)
        print_info("Streaming", str(self.config.EXTERNAL_DATASET_STREAMING))
        
        # Determine number of samples to load
        if self.config.EXTERNAL_DATASET_TEST_MODE:
            num_samples = self.config.EXTERNAL_DATASET_TEST_SAMPLES
            print_info("Mode", f"TEST ({num_samples} samples)")
        elif self.config.EXTERNAL_DATASET_MAX_SAMPLES:
            num_samples = self.config.EXTERNAL_DATASET_MAX_SAMPLES
            print_info("Mode", f"LIMITED ({num_samples} samples)")
        else:
            num_samples = None  # Load all
            print_info("Mode", "FULL DATASET")
        
        try:
            # Load dataset
            ds = load_dataset(
                self.config.EXTERNAL_DATASET_NAME,
                split=self.config.EXTERNAL_DATASET_SPLIT,
                streaming=self.config.EXTERNAL_DATASET_STREAMING
            )
            
            # Collect samples
            samples = []
            instruction_col = self.config.INSTRUCTION_COLUMN
            response_col = self.config.RESPONSE_COLUMN
            
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
            random.seed(self.config.EXTERNAL_DATASET_SHUFFLE_SEED)
            random.shuffle(samples)
            
            val_size = int(len(samples) * self.config.EXTERNAL_DATASET_VAL_RATIO)
            val_samples = samples[:val_size]
            train_samples = samples[val_size:]
            
            print_info("Training Samples", str(len(train_samples)))
            print_info("Validation Samples", str(len(val_samples)))
            
            return train_samples, val_samples
            
        except Exception as e:
            print_error(f"Failed to load external dataset: {e}")
            print_warning("Falling back to config samples")
            return self.config.TRAIN_SAMPLES, self.config.VAL_SAMPLES
    
    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.LORA_R,
            lora_alpha=self.config.LORA_ALPHA,
            lora_dropout=self.config.LORA_DROPOUT,
            target_modules=self.config.LORA_TARGET_MODULES,
            bias=self.config.LORA_BIAS
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print_success("LoRA adapters applied")
        print_info("LoRA Rank (r)", str(self.config.LORA_R))
        print_info("LoRA Alpha", str(self.config.LORA_ALPHA))
        print_info("LoRA Dropout", str(self.config.LORA_DROPOUT))
        print_info("Target Modules", ", ".join(self.config.LORA_TARGET_MODULES))
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        print_step(6, 8, "Creating datasets and dataloaders")
        
        # Load samples from external dataset or use config samples
        if self.config.USE_EXTERNAL_DATASET and DATASETS_AVAILABLE:
            train_samples, val_samples = self._load_external_dataset()
        else:
            train_samples = self.config.TRAIN_SAMPLES
            val_samples = self.config.VAL_SAMPLES
        
        # Create datasets
        train_dataset = InstructionDataset(
            train_samples,
            self.tokenizer,
            self.config.MAX_SEQ_LENGTH
        )
        
        val_dataset = InstructionDataset(
            val_samples,
            self.tokenizer,
            self.config.MAX_SEQ_LENGTH
        )
        
        print_info("Training Samples", str(len(train_dataset)))
        print_info("Validation Samples", str(len(val_dataset)))
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        print_info("Train Batches", str(len(train_loader)))
        print_info("Val Batches", str(len(val_loader)))
        print_success("Dataloaders created")
        
        return train_loader, val_loader
    
    def _setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        print_step(7, 8, "Setting up optimizer and scheduler")
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        print_info("Optimizer", "AdamW")
        print_info("Learning Rate", str(self.config.LEARNING_RATE))
        print_info("Weight Decay", str(self.config.WEIGHT_DECAY))
        
        # Calculate warmup steps
        num_warmup_steps = int(num_training_steps * self.config.WARMUP_RATIO)
        
        # Setup scheduler
        self.scheduler = get_scheduler(
            name=self.config.SCHEDULER_TYPE,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print_info("Scheduler", self.config.SCHEDULER_TYPE)
        print_info("Warmup Steps", str(num_warmup_steps))
        print_info("Total Training Steps", str(num_training_steps))
        print_success("Optimizer and scheduler configured")
    
    def _train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int,
        global_step: int
    ) -> Tuple[float, int]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS}",
            leave=True
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / self.config.GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.MAX_GRAD_NORM
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                global_step += 1
                
                # Get actual loss value
                actual_loss = loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
                total_loss += actual_loss
                num_batches += 1
                
                # Log metrics
                current_lr = self.scheduler.get_last_lr()[0]
                self.metrics_tracker.log_step(
                    step=global_step,
                    epoch=epoch + 1,
                    train_loss=actual_loss,
                    learning_rate=current_lr,
                    gradient_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                )
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{actual_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'ppl': f'{math.exp(min(actual_loss, 20)):.2f}'
                })
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, global_step
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        self.metrics_tracker.log_validation(avg_loss)
        
        return avg_loss
    
    def train(self):
        """Run the complete training loop."""
        print_header("STARTING TRAINING", "=")
        
        # Create dataloaders
        train_loader, val_loader = self._create_dataloaders()
        
        # Calculate total training steps
        steps_per_epoch = len(train_loader) // self.config.GRADIENT_ACCUMULATION_STEPS
        total_training_steps = steps_per_epoch * self.config.NUM_EPOCHS
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(total_training_steps)
        
        print_step(8, 8, "Training Loop")
        print_info("Epochs", str(self.config.NUM_EPOCHS))
        print_info("Steps per Epoch", str(steps_per_epoch))
        print_info("Total Steps", str(total_training_steps))
        print_info("Gradient Accumulation", str(self.config.GRADIENT_ACCUMULATION_STEPS))
        print_info("Effective Batch Size", str(self.config.BATCH_SIZE * self.config.GRADIENT_ACCUMULATION_STEPS))
        
        global_step = 0
        best_val_loss = float('inf')
        training_start_time = datetime.now()
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = datetime.now()
            
            print(f"\n{'='*60}")
            print(f" EPOCH {epoch + 1}/{self.config.NUM_EPOCHS}")
            print(f"{'='*60}")
            
            # Training
            train_loss, global_step = self._train_epoch(train_loader, epoch, global_step)
            
            # Validation
            print("\n  Running validation...")
            val_loss = self._validate(val_loader)
            
            # Calculate epoch duration
            epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
            
            # Log epoch metrics
            current_lr = self.scheduler.get_last_lr()[0]
            self.metrics_tracker.log_epoch(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                duration=epoch_duration
            )
            
            # Print epoch summary
            print(f"\n  Epoch {epoch + 1} Summary:")
            print_info("Train Loss", f"{train_loss:.6f}")
            print_info("Train Perplexity", f"{math.exp(min(train_loss, 20)):.4f}")
            print_info("Val Loss", f"{val_loss:.6f}")
            print_info("Val Perplexity", f"{math.exp(min(val_loss, 20)):.4f}")
            print_info("Learning Rate", f"{current_lr:.2e}")
            print_info("Duration", f"{epoch_duration:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print_success(f"New best model! Val Loss: {val_loss:.6f}")
                
                if self.config.SAVE_CHECKPOINTS:
                    self._save_checkpoint(
                        self.config.CHECKPOINT_DIR / "best_model",
                        epoch + 1,
                        train_loss,
                        val_loss
                    )
            
            # Save periodic checkpoint
            if self.config.SAVE_CHECKPOINTS and (epoch + 1) % max(1, self.config.NUM_EPOCHS // 3) == 0:
                self._save_checkpoint(
                    self.config.CHECKPOINT_DIR / f"epoch_{epoch + 1}",
                    epoch + 1,
                    train_loss,
                    val_loss
                )
        
        # Training complete
        total_training_time = (datetime.now() - training_start_time).total_seconds()
        
        print_header("TRAINING COMPLETE", "=")
        print_info("Total Training Time", f"{total_training_time:.2f}s ({total_training_time/60:.2f}m)")
        print_info("Final Train Loss", f"{train_loss:.6f}")
        print_info("Final Val Loss", f"{val_loss:.6f}")
        print_info("Best Val Loss", f"{best_val_loss:.6f}")
        
        # Save final model
        if self.config.SAVE_CHECKPOINTS:
            self._save_checkpoint(
                self.config.FINAL_CHECKPOINT_DIR,
                self.config.NUM_EPOCHS,
                train_loss,
                val_loss,
                is_final=True
            )
        
        return {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'total_steps': global_step,
            'training_time_seconds': total_training_time
        }
    
    def _save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        train_loss: float,
        val_loss: float,
        is_final: bool = False
    ):
        """Save model checkpoint with metadata."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_type = "final" if is_final else f"epoch_{epoch}"
        print(f"\n  Saving {checkpoint_type} checkpoint...")
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_name': self.config.MODEL_NAME,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'learning_rate': self.config.LEARNING_RATE,
                'batch_size': self.config.BATCH_SIZE,
                'num_epochs': self.config.NUM_EPOCHS,
                'use_lora': self.config.USE_LORA
            }
        }
        
        state_path = checkpoint_dir / "training_state.json"
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
        
        print_success(f"Checkpoint saved to {checkpoint_dir}")
    
    def tokenize_chat(self, messages: list) -> torch.Tensor:
        """Tokenize chat messages using chat template."""
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)
    
    def run_single_forward_pass(self) -> float:
        """Run a single forward pass with configured texts (for validation)."""
        print_header("VALIDATION FORWARD PASS", "-")
        
        self.model.eval()
        losses = []
        
        for i, message in enumerate(self.config.SINGLE_TEXTS[:2]):  # Use first 2 texts
            print(f"\n  Text {i+1}:")
            print_info("Input", message[:50] + "..." if len(message) > 50 else message)
            
            # Prepare input
            messages = [{"role": "user", "content": message}]
            inputs = self.tokenize_chat(messages)
            labels = inputs.clone()
            
            print_info("Input Tokens", str(inputs.shape[-1]))
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids=inputs, labels=labels)
            
            loss = outputs.loss.item()
            losses.append(loss)
            print_info("Loss", f"{loss:.6f}")
            print_info("Perplexity", f"{math.exp(min(loss, 20)):.4f}")
        
        mean_loss = sum(losses) / len(losses)
        print_info("Mean Loss", f"{mean_loss:.6f}")
        
        return mean_loss
    
    def save_metrics_and_plots(self):
        """Save all metrics and generate plots."""
        print_header("SAVING METRICS AND PLOTS", "-")
        
        # Save metrics to JSON
        metrics_path = self.metrics_tracker.save_metrics()
        
        # Generate plots if enabled
        if self.config.GENERATE_PLOTS:
            try:
                plot_path = self.metrics_tracker.generate_plots()
            except Exception as e:
                print_warning(f"Could not generate plots: {e}")
        
        # Print summary
        self.metrics_tracker.print_summary()
        
        return metrics_path
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print_header("STARTING FULL TRAINING PIPELINE", "=")
        
        # Initialize
        self.initialize()
        
        # Run training
        training_results = self.train()
        
        # Run validation forward pass to verify model
        val_loss = self.run_single_forward_pass()
        
        # Save metrics and generate plots
        metrics_path = self.save_metrics_and_plots()
        
        # Final summary
        print_header("TRAINING PIPELINE SUMMARY", "=")
        print_info("Model", self.config.MODEL_NAME)
        print_info("Final Train Loss", f"{training_results['final_train_loss']:.6f}")
        print_info("Final Val Loss", f"{training_results['final_val_loss']:.6f}")
        print_info("Best Val Loss", f"{training_results['best_val_loss']:.6f}")
        print_info("Total Steps", str(training_results['total_steps']))
        print_info("Training Time", f"{training_results['training_time_seconds']:.2f}s")
        
        if self.config.USE_LORA and PEFT_AVAILABLE:
            print_info("Fine-tuning Method", "LoRA")
        else:
            print_info("Fine-tuning Method", "Full Model")
        
        if self.config.SAVE_CHECKPOINTS:
            print_info("Final Checkpoint", str(self.config.FINAL_CHECKPOINT_DIR))
            print_info("Metrics File", str(metrics_path))
        
        if torch.cuda.is_available():
            print_info("Peak GPU Memory", f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        
        print(f"\n  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_success("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        
        return {
            **training_results,
            'validation_loss': val_loss,
            'metrics_path': str(metrics_path)
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main(config_path: str = TRAINING_CONFIG_PATH_V2):
    """
    Run the Qwen training pipeline with configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    """
    print_header("QWEN DOLPHIN COMPLETE TRAINING PIPELINE", "=")
    print(f"  Configuration: {config_path}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PEFT/LoRA Available: {PEFT_AVAILABLE}")
    
    # Create configuration from YAML
    config = TrainingConfig(config_path=config_path, model_type="qwen")
    
    # Print configuration summary
    print_header("CONFIGURATION SUMMARY", "-")
    print_info("Model", config.MODEL_NAME)
    print_info("Epochs", str(config.NUM_EPOCHS))
    print_info("Batch Size", str(config.BATCH_SIZE))
    print_info("Learning Rate", str(config.LEARNING_RATE))
    print_info("Use LoRA", str(config.USE_LORA))
    print_info("Training Samples", str(len(config.TRAIN_SAMPLES)))
    print_info("Validation Samples", str(len(config.VAL_SAMPLES)))
    
    # Create and run pipeline
    pipeline = QwenTrainingPipeline(config)
    results = pipeline.run_full_pipeline()
    
    # Return results for external use
    return results


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else TRAINING_CONFIG_PATH_V2
    results = main(config_file)