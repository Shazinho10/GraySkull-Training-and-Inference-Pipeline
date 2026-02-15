"""ui-interference.py - Streamlit UI for Qwen & Llama Inference"""

import streamlit as st
import torch
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.inference_qwen import (
    QwenInference,
    InferenceConfig as QwenConfig,
)
from src.inference.inference_llama import (
    LlamaInference,
    InferenceConfig as LlamaConfig,
)
from src.utils.path import INFERENCE_CONFIG_PATH

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraySkull Inference",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 GraySkull Inference")
st.markdown("Generate text using **Qwen 0.5B** or **Llama 8B** Dolphin models.")

# ── Sidebar – model selection & generation params ────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        options=["Qwen (0.5B)", "Llama (8B)"],
        index=0,
    )

    st.subheader("Generation Parameters")
    max_new_tokens = st.slider("Max New Tokens", 16, 512, 100, step=16)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    do_sample = st.checkbox("Sampling", value=True)

    if model_choice.startswith("Qwen"):
        use_chat_template = st.checkbox("Use Chat Template", value=True)
    else:
        use_chat_template = st.checkbox("Use Chat Template", value=True)

    st.divider()
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.memory_allocated() / 1e9
        st.caption(f"GPU: {gpu_name}")
        st.caption(f"VRAM in use: {mem:.2f} GB")
    else:
        st.caption("Running on CPU")


# ── Helper: load model (cached so it only runs once per selection) ───────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(model_key: str):
    """Load and initialise the selected model. Cached across reruns."""
    if model_key == "qwen":
        config = QwenConfig(config_path=str(INFERENCE_CONFIG_PATH), model_type="qwen")
        engine = QwenInference(config)
    else:
        config = LlamaConfig(config_path=str(INFERENCE_CONFIG_PATH), model_type="llama")
        engine = LlamaInference(config)

    engine.initialize()
    return engine


# ── Chat history ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User input & generation ──────────────────────────────────────────────────
user_query = st.chat_input("Ask anything…")

if user_query:
    # Show the user message immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Determine model key
    model_key = "qwen" if model_choice.startswith("Qwen") else "llama"

    # Load / retrieve cached model
    engine = load_model(model_key)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating…"):
            if model_key == "qwen":
                response = engine.generate(
                    prompt=user_query,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    use_chat_template=use_chat_template,
                )
            else:
                # Llama: use chat template path when requested
                if use_chat_template:
                    messages = [{"role": "user", "content": user_query}]
                    response = engine.generate_chat(
                        messages=messages,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
                else:
                    response = engine.generate(
                        prompt=user_query,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
