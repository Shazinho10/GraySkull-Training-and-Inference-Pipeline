"""inference_llama.py - Dynamic YAML Configuration Version"""

import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import yaml
from typing import Optional, Dict, Any, List


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================
class InferenceConfig:
    """Load inference configuration from YAML file."""
    
    def __init__(self, config_path: str = "inference_config.yaml", model_type: str = "llama"):
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
        
        # Load model-specific settings
        model_config = config['models'][model_type]
        self.MODEL_NAME = model_config['name']
        self.USE_FLOAT16 = model_config['use_float16']
        self.DEVICE_MAP = model_config['device_map']
        
        # Load generation parameters
        gen_config = config['generation']
        self.MAX_NEW_TOKENS = gen_config['max_new_tokens']
        self.TEMPERATURE = gen_config['temperature']
        self.TOP_P = gen_config['top_p']
        self.DO_SAMPLE = gen_config['do_sample']
        
        # Check for model-specific generation overrides
        if 'model_overrides' in gen_config and model_type in gen_config['model_overrides']:
            overrides = gen_config['model_overrides'][model_type]
            self.MAX_NEW_TOKENS = overrides.get('max_new_tokens', self.MAX_NEW_TOKENS)
            self.TEMPERATURE = overrides.get('temperature', self.TEMPERATURE)
            self.TOP_P = overrides.get('top_p', self.TOP_P)
            self.DO_SAMPLE = overrides.get('do_sample', self.DO_SAMPLE)
        
        # Load example prompts
        self.EXAMPLE_PROMPTS = config['example_prompts']
        
        # Load inference settings
        inference_config = config['inference']
        self.USE_CHAT_TEMPLATE = inference_config['use_chat_template']
        self.VERBOSE = inference_config['verbose']
        self.SHOW_MEMORY_STATS = inference_config['show_memory_stats']


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def print_header(text: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def print_step(step_num: int, description: str):
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 50)


def print_info(key: str, value: str):
    """Print key-value information."""
    print(f"  → {key}: {value}")


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"  → GPU Memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak: {max_allocated:.2f} GB")


def get_device_info():
    """Get and print device information."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return device, device_name, total_memory
    else:
        return torch.device("cpu"), "CPU", 0


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# =============================================================================
# LLAMA INFERENCE CLASS
# =============================================================================
class LlamaInference:
    """Inference handler for Llama Dolphin model."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def initialize(self):
        """Initialize the model and tokenizer."""
        print_header("LLAMA DOLPHIN MODEL - INITIALIZATION", "=")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Set seeds
        print_step(1, "Setting random seeds for reproducibility")
        set_seed(self.config.SEED)
        print_info("Seed", str(self.config.SEED))
        print("  ✓ Seeds set successfully")
        
        # Step 2: Check device
        print_step(2, "Checking available device")
        self.device, device_name, total_memory = get_device_info()
        print_info("Device", str(self.device))
        print_info("Device Name", device_name)
        if self.device.type == "cuda":
            print_info("Total GPU Memory", f"{total_memory:.2f} GB")
            if self.config.SHOW_MEMORY_STATS:
                print_gpu_memory()
            print("  ✓ CUDA is available - Using GPU")
        else:
            print("  ⚠ CUDA not available - Using CPU (this will be slow for 8B model)")
        
        # Step 3: Load tokenizer
        print_step(3, "Loading Llama tokenizer")
        print_info("Model", self.config.MODEL_NAME)
        print("  Loading tokenizer from HuggingFace Hub...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        print("  ✓ Tokenizer loaded successfully")
        print_info("Vocab Size", str(len(self.tokenizer)))
        
        # Step 4: Load model
        print_step(4, "Loading Llama model (8B parameters)")
        print("  ⏳ Loading model weights... (this may take several minutes)")
        print_info("Using Float16", str(self.config.USE_FLOAT16))
        print_info("Device Map", str(self.config.DEVICE_MAP))
        
        # Load with optimizations for large model
        dtype = torch.float16 if self.config.USE_FLOAT16 else torch.float32
        
        if self.config.DEVICE_MAP:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                device_map=self.config.DEVICE_MAP,
                torch_dtype=dtype
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.MODEL_NAME,
                torch_dtype=dtype
            )
            self.model.to(self.device)
        
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("  ✓ Model loaded successfully")
        print_info("Total Parameters", f"{total_params:,}")
        print_info("Trainable Parameters", f"{trainable_params:,}")
        print_info("Model Device", str(next(self.model.parameters()).device))
        print_info("Model Dtype", str(next(self.model.parameters()).dtype))
        
        if self.device.type == "cuda" and self.config.SHOW_MEMORY_STATS:
            print_gpu_memory()
        
        print_header("INITIALIZATION COMPLETE", "-")
        
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, 
                 temperature: Optional[float] = None, top_p: Optional[float] = None,
                 do_sample: Optional[bool] = None) -> str:
        """Generate text from a prompt."""
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        temperature = temperature or self.config.TEMPERATURE
        top_p = top_p or self.config.TOP_P
        do_sample = do_sample if do_sample is not None else self.config.DO_SAMPLE
        
        if self.config.VERBOSE:
            print_header("GENERATING RESPONSE", "-")
            print_info("Prompt", prompt[:50] + "..." if len(prompt) > 50 else prompt)
            print_info("Max New Tokens", str(max_new_tokens))
            print_info("Temperature", str(temperature))
            print_info("Top P", str(top_p))
            print_info("Do Sample", str(do_sample))
        
        # Tokenize input
        target_device = self.device if not self.config.DEVICE_MAP else self.model.device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        
        if self.config.VERBOSE:
            print_info("Input Tokens", str(input_length))
            print("  ⏳ Generating response...")
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_tokens = len(output_ids[0]) - input_length
        
        if self.config.VERBOSE:
            print("  ✓ Generation complete")
            print_info("Generated Tokens", str(generated_tokens))
            
            if torch.cuda.is_available() and self.config.SHOW_MEMORY_STATS:
                print_gpu_memory()
        
        return response
    
    def generate_chat(self, messages: List[Dict[str, str]], max_new_tokens: Optional[int] = None,
                      temperature: Optional[float] = None, top_p: Optional[float] = None,
                      do_sample: Optional[bool] = None) -> str:
        """Generate response from chat messages (if chat template available)."""
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.MAX_NEW_TOKENS
        temperature = temperature or self.config.TEMPERATURE
        top_p = top_p or self.config.TOP_P
        do_sample = do_sample if do_sample is not None else self.config.DO_SAMPLE
        
        if self.config.VERBOSE:
            print_header("CHAT GENERATION", "-")
            print_info("Number of Messages", str(len(messages)))
            print_info("Max New Tokens", str(max_new_tokens))
        
        target_device = self.device if not self.config.DEVICE_MAP else self.model.device
        
        # Check if chat template is available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            if self.config.VERBOSE:
                print("  Using chat template...")
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        else:
            # Fallback: concatenate messages
            if self.config.VERBOSE:
                print("  ⚠ No chat template available, using concatenation...")
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[-1]
        
        if self.config.VERBOSE:
            print_info("Input Tokens", str(input_length))
            print("  ⏳ Generating response...")
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        response = self.tokenizer.decode(
            output_ids[0][input_length:],
            skip_special_tokens=True
        )
        
        if self.config.VERBOSE:
            print("  ✓ Generation complete")
            print_info("Generated Tokens", str(len(output_ids[0]) - input_length))
            
            if torch.cuda.is_available() and self.config.SHOW_MEMORY_STATS:
                print_gpu_memory()
        
        return response
    
    def batch_generate(self, prompts: list, **kwargs) -> list:
        """Generate responses for multiple prompts."""
        print_header(f"BATCH GENERATION ({len(prompts)} prompts)", "=")
        
        responses = []
        for i, prompt in enumerate(prompts):
            if self.config.VERBOSE:
                print(f"\n  Processing prompt {i+1}/{len(prompts)}...")
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        
        print_header("BATCH GENERATION COMPLETE", "-")
        return responses


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main(config_path: str = "inference_config.yaml"):
    """
    Run the Llama inference with configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    """
    print_header("LLAMA DOLPHIN INFERENCE SCRIPT", "=")
    print(f"  Model: 8 Billion Parameters")
    print(f"  Configuration: {config_path}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize configuration from YAML
    config = InferenceConfig(config_path=config_path, model_type="llama")
    
    # Create inference handler
    inference = LlamaInference(config)
    inference.initialize()
    
    # =========================
    # Example 1: Simple prompt
    # =========================
    print_header("EXAMPLE 1: SIMPLE PROMPT", "=")
    
    prompt = config.EXAMPLE_PROMPTS['single'][0]
    response = inference.generate(prompt, max_new_tokens=20)
    
    print("\n" + "=" * 70)
    print("PROMPT:")
    print(prompt)
    print("\nRESPONSE:")
    print(response)
    print("=" * 70)
    
    # =========================
    # Example 2: Chat-style
    # =========================
    print_header("EXAMPLE 2: CHAT-STYLE GENERATION", "=")
    
    # Use first chat example from config
    messages = [config.EXAMPLE_PROMPTS['chat'][0]]
    
    response = inference.generate_chat(messages, max_new_tokens=20)
    
    print("\n" + "=" * 70)
    print("CHAT MESSAGES:")
    for msg in messages:
        print(f"  [{msg['role']}]: {msg['content']}")
    print("\nRESPONSE:")
    print(response)
    print("=" * 70)
    
    # =========================
    # Example 3: Multiple prompts
    # =========================
    print_header("EXAMPLE 3: MULTIPLE PROMPTS", "=")
    
    prompts = config.EXAMPLE_PROMPTS['batch']
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1} ---")
        response = inference.generate(prompt, max_new_tokens=20)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
    
    # =========================
    # Summary
    # =========================
    print_header("INFERENCE COMPLETE", "=")
    print(f"  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if torch.cuda.is_available() and config.SHOW_MEMORY_STATS:
        print_info("Peak GPU Memory", f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("\n  ✓ All inference tasks completed successfully!")


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "inference_config.yaml"
    main(config_file)