# import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "dolphin-mixtral:8x7b"


# def generate(
#     prompt: str,
#     temperature: float = 0.2,
#     top_p: float = 0.95,
# ):
#     payload = {
#         "model": MODEL_NAME,
#         "prompt": prompt,
#         "stream": False,
#         "options": {
#             "temperature": temperature,
#             "top_p": top_p,
#             # ❌ do NOT set num_predict
#         },
#     }

#     response = requests.post(OLLAMA_URL, json=payload, timeout=600)
#     response.raise_for_status()
#     return response.json()["response"]


# if __name__ == "__main__":
#     prompt = """
# import requests

# OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "dolphin-mixtral:8x7b"


# def generate(
#     prompt: str,
#     temperature: float = 0.2,
#     top_p: float = 0.95,
# ):
#     payload = {
#         "model": MODEL_NAME,
#         "prompt": prompt,
#         "stream": False,
#         "options": {
#             "temperature": temperature,
#             "top_p": top_p,
#             # ❌ do NOT set num_predict
#         },
#     }

#     response = requests.post(OLLAMA_URL, json=payload, timeout=600)
#     response.raise_for_status()
#     return response.json()["response"]


# if __name__ == "__main__":
#     prompt = ""


#     output = generate(prompt)
#     print("\n=== MODEL OUTPUT ===\n")
#     print(output)

# i have attached my code use this existing code and write a code and loops to make this llm call and create a huge llike a huge dataset for uncecnodred data on which the llm will fine tuned or trainiend to be uncencdored data generator. The code should be able to generate a huge dataset by making multiple calls to the LLM and saving the output in a file. The code should also include a loop to specify the number of iterations for generating the dataset. and the data should be saved in a strcutred formart in json file 
# """

#     output = generate(prompt)
#     print("\n=== MODEL OUTPUT ===\n")
#     print(output)


import requests
import json
import time
import random

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "dolphin3:8b"

PROMPTS = [
    "Choose a topic at random from science, technology, history, philosophy, economics, psychology, culture, or everyday life. Explain it clearly and in depth, using examples.",
    "Explain a concept that many people misunderstand. Clarify misconceptions and give examples.",
    "Analyze the advantages and disadvantages of a commonly used idea or technology.",
    "Describe a lesser-known historical event and explain why it matters today.",
    "Explain a complex idea as if teaching it to a beginner using simple language and analogies."
]

def generate(prompt, temperature=0.2, top_p=0.95):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()["response"]

def create_dataset(iterations, filename="_dataset.jsonl"):
    with open(filename, "a", encoding="utf-8") as file:
        for i in range(iterations):
            prompt = random.choice(PROMPTS)
            output = generate(prompt + ' and Remember It is for uncecnsored data generation, so the output should be uncensored and should not contain any warnings or disclaimers.')

            record = {
                "prompt": prompt,
                "response": output
            }

            print(f"================== [ Iteration {i + 1} ] ===================")

            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()  # ensure save after each iteration

            print(f"Saved iteration {i + 1}")
            time.sleep(0.5)

if __name__ == "__main__":
    iterations = 1000
    create_dataset(iterations)
