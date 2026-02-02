"""pipelines.py - Base and model-specific training pipeline implementations"""
import torch
import math
import json
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_scheduler
)

from .base import (
    TrainingConfig,
    MetricsTracker,
    InstructionDataset,
    load_external_dataset,
    print_header,
    print_step,
    print_info,
    print_success,
    print_warning,
    print_error,
    print_gpu_memory,
    get_device,
    get_device_info,
    set_seed,
    save_training_log,
    DATASETS_AVAILABLE
)

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. LoRA will not be available.")


# =============================================================================
# BASE TRAINING PIPELINE (Abstract)
# =============================================================================
class BaseTrainingPipeline(ABC):
    """Abstract base class for training pipelines."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.device_info = None
        self.optimizer = None
        self.scheduler = None
        self.metrics_tracker = MetricsTracker(config.METRICS_OUTPUT_DIR)
        
        self.training_log = {
            "model": config.MODEL_NAME,
            "model_type": config.model_type,
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
    
    @abstractmethod
    def _load_model(self):
        """Load the model - implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_pipeline_name(self) -> str:
        """Get the pipeline name for display."""
        pass
    
    def initialize(self):
        """Initialize the training pipeline."""
        print_header(f"{self._get_pipeline_name()} - INITIALIZATION", "=")
        print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Model: {self.config.MODEL_NAME}")
        
        # Step 1: Set seeds
        print_step(1, self.config.TOTAL_STEPS, "Setting up reproducibility")
        set_seed(self.config.SEED, self.config.DETERMINISTIC)
        
        # Step 2: Get device
        print_step(2, self.config.TOTAL_STEPS, "Checking compute device")
        self.device, self.device_info = get_device_info()
        self.training_log["device"] = self.device_info
        
        if self.device.type == "cuda":
            print_success(f"CUDA available: {self.device_info['name']}")
            print_info("Total GPU Memory", f"{self.device_info['total_memory_gb']:.2f} GB")
            print_gpu_memory()
        else:
            print_warning("CUDA not available - Running on CPU")
        
        # Step 3: Load tokenizer
        print_step(3, self.config.TOTAL_STEPS, "Loading tokenizer")
        print("  Loading tokenizer from HuggingFace...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        print_success("Tokenizer loaded")
        print_info("Vocabulary Size", str(len(self.tokenizer)))
        print_info("Pad Token", str(self.tokenizer.pad_token))
        
        # Step 4: Load model (implemented by subclass)
        print_step(4, self.config.TOTAL_STEPS, "Loading model")
        self._load_model()
        
        # Print model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print_success("Model loaded")
        print_info("Total Parameters", f"{total_params:,}")
        print_info("Trainable Parameters", f"{trainable_params:,}")
        print_info("Model Dtype", str(next(self.model.parameters()).dtype))
        print_info("Model Device", str(next(self.model.parameters()).device))
        
        if torch.cuda.is_available():
            print_gpu_memory()
        
        # Create checkpoint directories
        if self.config.SAVE_CHECKPOINTS:
            self.config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            self.config.SINGLE_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            self.config.BATCH_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            self.config.FINAL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            print_success("Checkpoint directories created")
        
        print_header("INITIALIZATION COMPLETE", "-")
    
    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        if not PEFT_AVAILABLE:
            print_warning("PEFT not installed. Cannot apply LoRA.")
            return
        
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
        print_step(6, self.config.TOTAL_STEPS, "Creating datasets and dataloaders")
        
        # Load samples from external dataset or use config samples
        if self.config.USE_EXTERNAL_DATASET and DATASETS_AVAILABLE:
            train_samples, val_samples = load_external_dataset(self.config)
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
        print_step(7, self.config.TOTAL_STEPS, "Setting up optimizer and scheduler")
        
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
            'model_type': self.config.model_type,
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
    
    def tokenize_input(self, texts: list, padding: bool = True) -> dict:
        """Tokenize input texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=padding)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs
    
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
        
        texts = self.config.SINGLE_TEXTS[:2] if self.config.SINGLE_TEXTS else []
        if not texts:
            print_warning("No single texts configured for forward pass")
            return 0.0
        
        for i, message in enumerate(texts):
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
        
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        print_info("Mean Loss", f"{mean_loss:.6f}")
        
        return mean_loss
    
    def run_batch_forward_pass(self) -> float:
        """Run forward pass on configured batch of texts."""
        print_step(6, self.config.TOTAL_STEPS, "BATCH FORWARD PASS (Dry-run)")
        
        if not self.config.BATCH_TEXTS:
            print_warning("No batch texts configured")
            return 0.0
        
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
        
        print_step(8, self.config.TOTAL_STEPS, "Training Loop")
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
    
    def run_full_pipeline(self):
        """Run the complete training pipeline."""
        print_header(f"STARTING {self._get_pipeline_name()} FULL PIPELINE", "=")
        
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
        print_info("Model Type", self.config.model_type.upper())
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
        print_success(f"{self._get_pipeline_name()} COMPLETED SUCCESSFULLY")
        
        return {
            **training_results,
            'validation_loss': val_loss,
            'metrics_path': str(metrics_path)
        }


# =============================================================================
# QWEN TRAINING PIPELINE
# =============================================================================
class QwenTrainingPipeline(BaseTrainingPipeline):
    """Training pipeline for Qwen Dolphin model (small, ~0.5B parameters)."""
    
    def _get_pipeline_name(self) -> str:
        return "QWEN DOLPHIN TRAINING PIPELINE"
    
    def _load_model(self):
        """Load Qwen model."""
        print("  Loading Qwen model weights...")
        
        dtype = torch.float16 if self.config.USE_FLOAT16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            torch_dtype=dtype
        )
        self.model.to(self.device)
        
        # Apply LoRA if enabled
        if self.config.USE_LORA and PEFT_AVAILABLE:
            print_step(5, self.config.TOTAL_STEPS, "Applying LoRA adapters")
            self._apply_lora()
        elif self.config.USE_LORA and not PEFT_AVAILABLE:
            print_warning("LoRA requested but PEFT not installed. Training full model.")


# =============================================================================
# LLAMA TRAINING PIPELINE
# =============================================================================
class LlamaTrainingPipeline(BaseTrainingPipeline):
    """Training pipeline for Llama Dolphin model (large, ~8B parameters)."""
    
    def _get_pipeline_name(self) -> str:
        return "LLAMA DOLPHIN TRAINING PIPELINE"
    
    def _load_model(self):
        """Load Llama model with device_map='auto' for large models."""
        print("  Loading Llama model weights...")
        print("  ⏳ This may take several minutes for 8B model...")
        print_info("Float16 Mode", str(self.config.USE_FLOAT16))
        print_info("Device Map", "auto")
        
        dtype = torch.float16 if self.config.USE_FLOAT16 else torch.float32
        
        # Use device_map="auto" for large models
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME,
            device_map="auto",
            torch_dtype=dtype
        )
        
        # Note: For Llama, we typically don't apply LoRA by default in inference mode
        # but can be enabled for training
        if self.config.USE_LORA and PEFT_AVAILABLE:
            print_step(5, self.config.TOTAL_STEPS, "Applying LoRA adapters")
            self._apply_lora()
        elif self.config.USE_LORA and not PEFT_AVAILABLE:
            print_warning("LoRA requested but PEFT not installed. Training full model.")
    
    def _save_checkpoint(
        self,
        checkpoint_dir: Path,
        epoch: int,
        train_loss: float,
        val_loss: float,
        is_final: bool = False
    ):
        """Save model checkpoint - special handling for large models."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_type = "final" if is_final else f"epoch_{epoch}"
        print(f"\n  Saving {checkpoint_type} checkpoint...")
        print_info("Location", str(checkpoint_dir))
        
        # Save model and tokenizer - use safe_serialization=False for device_map="auto"
        print("  Saving model weights...")
        try:
            self.model.save_pretrained(
                checkpoint_dir,
                safe_serialization=False,  # Use PyTorch .bin format
                max_shard_size="2GB"
            )
            print_success("Model weights saved")
        except Exception as e:
            print_error(f"Model save failed: {str(e)[:200]}")
            print_warning("Checkpoint may be incomplete")
        
        print("  Saving tokenizer...")
        try:
            self.tokenizer.save_pretrained(checkpoint_dir)
            print_success("Tokenizer saved")
        except Exception as e:
            print_error(f"Tokenizer save failed: {str(e)[:200]}")
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_name': self.config.MODEL_NAME,
            'model_type': self.config.model_type,
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


# =============================================================================
# PIPELINE FACTORY
# =============================================================================
def get_pipeline(model_type: str, config: TrainingConfig) -> BaseTrainingPipeline:
    """Factory function to get the appropriate pipeline for a model type."""
    pipelines = {
        'qwen': QwenTrainingPipeline,
        'llama': LlamaTrainingPipeline
    }
    
    if model_type.lower() not in pipelines:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(pipelines.keys())}")
    
    return pipelines[model_type.lower()](config)
