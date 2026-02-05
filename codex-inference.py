import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-coder:latest"


def generate(
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            # ❌ do NOT set num_predict
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()["response"]


if __name__ == "__main__":
    prompt = """
Make a website using React for a simple todo web app.
Include components, state management, and basic styling.
"""

    output = generate(prompt)
    print("\n=== MODEL OUTPUT ===\n")
    print(output)
