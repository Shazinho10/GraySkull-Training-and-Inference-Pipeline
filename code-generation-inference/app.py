"""
Code Generation Inference - Streamlit UI
Uses Ollama's qwen3-coder model to generate web code and render it in the browser.
"""

import streamlit as st
import requests
import json
import re
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="Web Code Generator",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextArea textarea {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    }
    .code-container {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
    }
    .render-frame {
        border: 2px solid #4a4a4a;
        border-radius: 8px;
        background: white;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3-coder:latest"


def check_ollama_status() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(MODEL_NAME.replace(":latest", "") in name for name in model_names)
        return False
    except requests.exceptions.RequestException:
        return False


def extract_code_from_response(response: str) -> dict:
    """Extract HTML, CSS, and JavaScript code from the LLM response."""
    result = {
        "html": "",
        "css": "",
        "js": "",
        "full_response": response
    }
    
    # Try to extract code blocks with language tags
    html_pattern = r"```html\s*([\s\S]*?)```"
    css_pattern = r"```css\s*([\s\S]*?)```"
    js_pattern = r"```(?:javascript|js)\s*([\s\S]*?)```"
    
    html_matches = re.findall(html_pattern, response, re.IGNORECASE)
    css_matches = re.findall(css_pattern, response, re.IGNORECASE)
    js_matches = re.findall(js_pattern, response, re.IGNORECASE)
    
    if html_matches:
        result["html"] = "\n".join(html_matches)
    if css_matches:
        result["css"] = "\n".join(css_matches)
    if js_matches:
        result["js"] = "\n".join(js_matches)
    
    # If no specific blocks found, try to find a single code block
    if not any([result["html"], result["css"], result["js"]]):
        generic_pattern = r"```\s*([\s\S]*?)```"
        generic_matches = re.findall(generic_pattern, response)
        if generic_matches:
            combined = "\n".join(generic_matches)
            # Check if it looks like HTML
            if "<html" in combined.lower() or "<div" in combined.lower() or "<body" in combined.lower():
                result["html"] = combined
    
    return result


def create_renderable_html(html: str, css: str, js: str) -> str:
    """Combine HTML, CSS, and JS into a single renderable HTML document."""
    # Check if HTML already has complete structure
    if "<html" in html.lower() and "</html>" in html.lower():
        # Insert CSS and JS if not already present
        if css and "<style>" not in html.lower():
            html = html.replace("</head>", f"<style>{css}</style></head>")
        if js and "<script>" not in html.lower():
            html = html.replace("</body>", f"<script>{js}</script></body>")
        return html
    
    # Build complete HTML document
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Code Preview</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            padding: 20px;
        }}
        {css}
    </style>
</head>
<body>
    {html}
    <script>
        {js}
    </script>
</body>
</html>"""


def generate_code(prompt: str, system_prompt: str) -> Optional[str]:
    """Generate code using Ollama's qwen3-coder model."""
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": f"{system_prompt}\n\nUser Request: {prompt}\n\nGenerate the code:",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 4096
            }
        }
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. The model might be generating a complex response.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def main():
    """Main application entry point."""
    
    # Header
    st.markdown("<h1 class='main-header'>🖥️ Web Code Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Generate and preview web code using qwen3-coder</p>", unsafe_allow_html=True)
    
    # Check Ollama status
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ollama_status = check_ollama_status()
        if ollama_status:
            st.success(f"✅ Model: {MODEL_NAME}")
        else:
            st.error(f"❌ Model not available")
            st.info("Make sure Ollama is running and the model is pulled:\n```\nollama pull qwen3-coder:latest\n```")
        
        st.divider()
        
        # Code type selection
        code_type = st.selectbox(
            "Code Type",
            ["Complete HTML Page", "HTML Component", "HTML + CSS", "HTML + CSS + JS", "Tailwind CSS", "Bootstrap"],
            help="Select the type of code to generate"
        )
        
        # Temperature slider
        temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7, 0.1)
        
        st.divider()
        
        # Example prompts
        st.subheader("📝 Example Prompts")
        example_prompts = [
            "Create a modern login form with email and password",
            "Build a responsive navigation bar with dropdown menu",
            "Create a pricing table with 3 tiers",
            "Make an animated loading spinner",
            "Create a card component with image and text",
            "Build a contact form with validation"
        ]
        
        for prompt in example_prompts:
            if st.button(prompt, key=prompt, use_container_width=True):
                st.session_state.user_prompt = prompt
    
    # System prompt based on code type
    system_prompts = {
        "Complete HTML Page": """You are a web development expert. Generate a complete, functional HTML page with embedded CSS and JavaScript.
Always provide clean, well-structured code with proper indentation. Include responsive design considerations.
Format your response with code blocks: ```html, ```css, ```javascript""",
        
        "HTML Component": """You are a web development expert. Generate a reusable HTML component.
Provide clean, semantic HTML code. Focus on accessibility and best practices.
Format your response with code blocks: ```html""",
        
        "HTML + CSS": """You are a web development expert. Generate HTML with accompanying CSS.
Provide modern, responsive CSS using flexbox or grid where appropriate.
Format your response with separate code blocks: ```html and ```css""",
        
        "HTML + CSS + JS": """You are a web development expert. Generate HTML, CSS, and JavaScript.
Provide interactive, functional code with modern JavaScript practices.
Format your response with separate code blocks: ```html, ```css, and ```javascript""",
        
        "Tailwind CSS": """You are a web development expert specializing in Tailwind CSS.
Generate HTML using Tailwind CSS utility classes for styling.
Include the Tailwind CDN link in the HTML. Format your response with: ```html""",
        
        "Bootstrap": """You are a web development expert specializing in Bootstrap.
Generate HTML using Bootstrap 5 classes for styling and components.
Include the Bootstrap CDN links in the HTML. Format your response with: ```html"""
    }
    
    # Initialize session state
    if "user_prompt" not in st.session_state:
        st.session_state.user_prompt = ""
    if "generated_code" not in st.session_state:
        st.session_state.generated_code = None
    if "show_preview" not in st.session_state:
        st.session_state.show_preview = True
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Enter Your Request")
        
        user_prompt = st.text_area(
            "Describe the web code you want to generate:",
            value=st.session_state.user_prompt,
            height=150,
            placeholder="E.g., Create a modern card component with a profile picture, name, title, and social media links"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn1:
            generate_btn = st.button("🚀 Generate Code", type="primary", use_container_width=True)
        
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        with col_btn3:
            st.session_state.show_preview = st.toggle("Show Preview", value=st.session_state.show_preview)
        
        if clear_btn:
            st.session_state.generated_code = None
            st.session_state.user_prompt = ""
            st.rerun()
        
        if generate_btn and user_prompt:
            if not ollama_status:
                st.error("Please ensure Ollama is running with the qwen3-coder model.")
            else:
                with st.spinner("🔄 Generating code..."):
                    response = generate_code(user_prompt, system_prompts[code_type])
                    if response:
                        st.session_state.generated_code = extract_code_from_response(response)
                        st.session_state.user_prompt = user_prompt
    
    with col2:
        st.subheader("💻 Generated Code")
        
        if st.session_state.generated_code:
            code_data = st.session_state.generated_code
            
            # Tabs for different code sections
            tabs = st.tabs(["📄 HTML", "🎨 CSS", "⚡ JavaScript", "📜 Full Response"])
            
            with tabs[0]:
                if code_data["html"]:
                    st.code(code_data["html"], language="html")
                    st.download_button(
                        "Download HTML",
                        code_data["html"],
                        "generated.html",
                        "text/html"
                    )
                else:
                    st.info("No HTML code extracted")
            
            with tabs[1]:
                if code_data["css"]:
                    st.code(code_data["css"], language="css")
                    st.download_button(
                        "Download CSS", 
                        code_data["css"],
                        "styles.css",
                        "text/css"
                    )
                else:
                    st.info("No CSS code extracted")
            
            with tabs[2]:
                if code_data["js"]:
                    st.code(code_data["js"], language="javascript")
                    st.download_button(
                        "Download JS",
                        code_data["js"],
                        "script.js",
                        "text/javascript"
                    )
                else:
                    st.info("No JavaScript code extracted")
            
            with tabs[3]:
                st.text_area("Full Response", code_data["full_response"], height=300)
        else:
            st.info("Enter a prompt and click 'Generate Code' to get started")
    
    # Preview section
    if st.session_state.show_preview and st.session_state.generated_code:
        st.divider()
        st.subheader("🖼️ Live Preview")
        
        code_data = st.session_state.generated_code
        combined_html = create_renderable_html(
            code_data["html"],
            code_data["css"],
            code_data["js"]
        )
        
        # Preview options
        preview_col1, preview_col2 = st.columns([3, 1])
        
        with preview_col2:
            preview_height = st.slider("Preview Height", 200, 800, 500, 50)
            
            # Download combined file
            st.download_button(
                "📥 Download Combined HTML",
                combined_html,
                "complete_page.html",
                "text/html",
                use_container_width=True
            )
        
        with preview_col1:
            # Render in iframe using st.components
            import streamlit.components.v1 as components
            components.html(combined_html, height=preview_height, scrolling=True)


if __name__ == "__main__":
    main()
