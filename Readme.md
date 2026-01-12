# Example Configurations

This directory contains pre-configured YAML files for common use cases. Each configuration is optimized for specific types of tasks.

## 📁 Available Configs

### 1. Creative Writing (`creative_writing_config.yaml`)

**Optimized for:** Stories, poems, creative content
**Settings:**
- High temperature (0.95-1.0) for maximum creativity
- Long responses (500 tokens)
- Diverse sampling

**Use for:**
- Writing short stories
- Generating poetry
- Creating fictional narratives
- Brainstorming creative ideas

**Usage:**
```bash
cd inference
python inference_qwen_dynamic.py ../example_configs/creative_writing_config.yaml
```

**Example prompts:**
- "Write a short story about..."
- "Create a poem about..."
- "Describe a character who..."

---

### 2. Technical Q&A (`technical_qa_config.yaml`)

**Optimized for:** Factual answers, educational content, technical explanations
**Settings:**
- Low temperature (0.2-0.3) for focused responses
- Deterministic generation
- Moderate length (200 tokens)

**Use for:**
- Answering technical questions
- Explaining concepts
- Educational content
- Documentation generation

**Usage:**
```bash
cd inference
python inference_llama_dynamic.py ../example_configs/technical_qa_config.yaml
```

**Example prompts:**
- "Explain how X works..."
- "What is the difference between..."
- "How does Y algorithm work?"

---

### 3. Fast Testing (`fast_test_config.yaml`)

**Optimized for:** Quick iteration, debugging, experimentation
**Settings:**
- Short responses (30 tokens)
- Minimal verbosity
- Fast execution

**Use for:**
- Testing model loading
- Quick checks
- Debugging configurations
- Performance benchmarking

**Usage:**
```bash
cd inference
python inference_qwen_dynamic.py ../example_configs/fast_test_config.yaml
```

**Example prompts:**
- "Test"
- "Hello"
- Quick one-line questions

---

### 4. Code Generation (`code_generation_config.yaml`)

**Optimized for:** Programming assistance, code generation
**Settings:**
- Moderate temperature (0.3-0.5)
- Balanced creativity and structure
- Sufficient length for code blocks (300 tokens)

**Use for:**
- Writing functions
- Creating code snippets
- Implementing algorithms
- Programming examples

**Usage:**
```bash
cd inference
python inference_llama_dynamic.py ../example_configs/code_generation_config.yaml
```

**Example prompts:**
- "Write a function to..."
- "Implement a class for..."
- "Create an API endpoint for..."

---

## 🎯 Quick Reference

| Config | Temperature | Max Tokens | Best For |
|--------|------------|------------|----------|
| Creative Writing | 0.95-1.0 | 500 | Stories, poems |
| Technical Q&A | 0.2-0.3 | 200 | Explanations, facts |
| Fast Testing | 0.7 | 30 | Quick tests |
| Code Generation | 0.3-0.5 | 300 | Programming |

## 🔧 Customizing Configs

### Create Your Own Config

1. Copy an existing config:
```bash
cp creative_writing_config.yaml my_custom_config.yaml
```

2. Edit the settings:
```yaml
generation:
  max_new_tokens: 400        # Your preferred length
  temperature: 0.8           # Your preferred creativity
```

3. Add your prompts:
```yaml
example_prompts:
  single:
    - "Your custom prompt 1"
    - "Your custom prompt 2"
```

4. Use it:
```bash
python inference_qwen_dynamic.py my_custom_config.yaml
```

### Mix and Match Settings

Combine settings from different configs:

```yaml
# My hybrid config
generation:
  max_new_tokens: 300        # From code_generation
  temperature: 0.8           # From creative_writing
  top_p: 0.9                 # From technical_qa
  do_sample: true
```

## 📊 Parameter Impact

### Temperature

**Effect on output:**
```
Temperature 0.1: "The sky is blue."
Temperature 0.5: "The sky appears blue due to Rayleigh scattering."
Temperature 0.9: "The sky dances in shades of azure and cerulean."
Temperature 1.2: "Sky blue purple dreams floating whispers."
```

### Max New Tokens

**Effect on response length:**
- **30 tokens**: 1-2 sentences
- **100 tokens**: 1 paragraph
- **300 tokens**: 2-3 paragraphs
- **500 tokens**: Multiple paragraphs / long-form content

### Top P (Nucleus Sampling)

**Effect on diversity:**
- **0.9**: Balanced diversity
- **0.95**: More diverse vocabulary
- **0.98**: Maximum diversity
- **1.0**: No restriction

### Do Sample

**Effect on determinism:**
- **False (Greedy)**: Always picks most likely token (deterministic)
- **True (Sampling)**: Randomly samples from distribution (non-deterministic)

## 🎨 Use Case Matrix

| Task | Recommended Config | Suggested Model |
|------|-------------------|-----------------|
| Story writing | Creative Writing | Llama |
| Poetry | Creative Writing | Qwen |
| Technical docs | Technical Q&A | Llama |
| Quick answers | Technical Q&A | Qwen |
| Code snippets | Code Generation | Llama |
| Function writing | Code Generation | Qwen |
| Testing setup | Fast Testing | Qwen |
| Brainstorming | Creative Writing | Either |

## 💡 Pro Tips

### 1. Start Conservative
Begin with lower temperatures and increase if output is too boring.

### 2. Model-Specific Overrides
Use `model_overrides` when models behave differently:
```yaml
model_overrides:
  qwen:
    temperature: 0.8
  llama:
    temperature: 0.6  # Llama needs less temperature
```

### 3. Iterative Refinement
1. Start with base config
2. Run a few tests
3. Adjust one parameter at a time
4. Save successful configs

### 4. Domain-Specific Configs
Create configs for your specific domains:
- `medical_qa_config.yaml`
- `legal_writing_config.yaml`
- `marketing_copy_config.yaml`

### 5. Version Control
Git track your configs:
```bash
git add example_configs/my_domain_config.yaml
git commit -m "Add custom config for domain X"
```

## 🧪 Experimentation Guide

### Test Different Settings

Create variations:
```bash
# High creativity
cp creative_writing_config.yaml creative_high_temp.yaml
# Edit: temperature: 1.2

# Medium creativity  
cp creative_writing_config.yaml creative_med_temp.yaml
# Edit: temperature: 0.8

# Low creativity
cp creative_writing_config.yaml creative_low_temp.yaml
# Edit: temperature: 0.5
```

Run all three and compare outputs!

### A/B Testing

```python
# test_configs.py
configs = [
    "creative_high_temp.yaml",
    "creative_med_temp.yaml",
    "creative_low_temp.yaml"
]

prompt = "Write a story about a robot"

for config in configs:
    print(f"\n=== Testing {config} ===")
    # Run inference with config
    # Compare outputs
```

## 📝 Config Template

```yaml
# my_custom_config.yaml

global:
  seed: 42

models:
  qwen:
    name: "dphn/dolphin-2.9.3-qwen2-0.5b"
    use_float16: false
    device_map: null
    
  llama:
    name: "dphn/dolphin-2.9-llama3-8b"
    use_float16: true
    device_map: "auto"

generation:
  max_new_tokens: ___        # Fill in: 30-500
  temperature: ___           # Fill in: 0.0-2.0
  top_p: ___                 # Fill in: 0.5-1.0
  do_sample: ___             # Fill in: true/false
  
  # Optional: model-specific settings
  # model_overrides:
  #   qwen:
  #     temperature: ___
  #   llama:
  #     temperature: ___

example_prompts:
  single:
    - "Your prompt here"
  
  batch:
    - "Batch prompt 1"
    - "Batch prompt 2"
  
  chat:
    - role: "user"
      content: "Chat message"

inference:
  use_chat_template: true
  verbose: true              # true for debugging, false for production
  show_memory_stats: true    # true to monitor GPU usage
```