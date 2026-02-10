# Code Generation Inference

A Streamlit-based web application that uses Ollama's `qwen3-coder` model to generate and preview web code in real-time.

## Features

- **Code Generation**: Generate HTML, CSS, and JavaScript code from natural language descriptions
- **Live Preview**: Render generated code directly in the browser
- **Multiple Code Types**: Support for different code formats:
  - Complete HTML Pages
  - HTML Components
  - HTML + CSS
  - HTML + CSS + JS
  - Tailwind CSS
  - Bootstrap
- **Code Download**: Download generated code as separate files or combined HTML
- **Customizable Settings**: Adjust temperature and preview settings

## Prerequisites

1. **Ollama** installed and running
2. **qwen3-coder** model pulled:
   ```bash
   ollama pull qwen3-coder:latest
   ```

## Installation

```bash
# Navigate to the directory
cd code-generation-inference

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Select Code Type**: Choose the type of code you want to generate from the sidebar
2. **Enter Your Prompt**: Describe the web component/page you want to create
3. **Generate**: Click "Generate Code" to create the code
4. **View Code**: See the generated HTML, CSS, and JavaScript in separate tabs
5. **Preview**: Toggle the live preview to see the rendered result
6. **Download**: Download the code files for use in your projects

## Example Prompts

- "Create a modern login form with email and password"
- "Build a responsive navigation bar with dropdown menu"
- "Create a pricing table with 3 tiers"
- "Make an animated loading spinner"
- "Create a card component with image and text"
- "Build a contact form with validation"

## Configuration

Edit `config.yaml` to customize:
- Ollama endpoint
- Model name
- Generation parameters
- UI settings
