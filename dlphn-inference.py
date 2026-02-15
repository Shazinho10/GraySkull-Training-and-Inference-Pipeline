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


import json
import time
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Model configs ───────────────────────────────────────────────
MODELS = {
    "qwen": {
        "name": "dphn/dolphin-2.9.3-qwen2-0.5b",
        "use_float16": False,
        "device_map": None,  # Use default device selection (null = move to single device)
    },
    "llama": {
        "name": "dphn/dolphin-2.9-llama3-8b",
        "use_float16": True,
        "device_map": "auto",  # Use device_map="auto" for large models
    },
}

PROMPTS = [
    "Discuss the concept of 'realpolitik' and its influence on political decision-making.",
    "Explain the term 'deep state' and its implications on government and society.",
    "Analyze the advantages and disadvantages of censorship in modern political discourse.",
    "Describe the historical event of the Spanish Inquisition and its impact on the world today.",
    "Explain the concept of 'checks and balances' in a government and how it operates.",
    "Clarify the misconception that all dictators are evil and provide examples of those who were well-intentioned.",
    "Discuss the role of propaganda in shaping public opinion and its effects on democracy.",
    "Explain the concept of 'limited hangout' in politics and provide examples.",
    "Describe the historical event of the Watergate scandal and its impact on American politics.",
    "Analyze the advantages and disadvantages of campaign finance in modern political campaigns.",
    "Explain the concept of 'soft power' and its importance in international relations.",
    "Clarify the misconception that politicians are only interested in power and provide examples of those who genuinely care about the public good.",
    "Discuss the role of the media in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the Cuban Missile Crisis and its impact on international relations.",
    "Analyze the advantages and disadvantages of gerrymandering in modern politics.",
    "Explain the concept of 'fake news' and its impact on political discourse.",
    "Clarify the misconception that all lobbyists are corrupt and provide examples of those who work for the public good.",
    "Discuss the role of money in politics and its effects on democracy.",
    "Describe the historical event of the Gleneagles G8 summit and its impact on global politics.",
    "Analyze the advantages and disadvantages of the Electoral College in the United States.",
    "Explain the concept of 'smoking gun' in politics and its implications.",
    "Clarify the misconception that all politicians are the same and provide examples of those who have made significant positive changes.",
    "Discuss the role of political parties in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the fall of the Berlin Wall and its impact on world politics.",
    "Analyze the advantages and disadvantages of term limits in modern politics.",
    "Explain the concept of 'redistricting' and its impact on political representation.",
    "Clarify the misconception that all whistleblowers are heroes and provide examples of those who have caused harm.",
    "Discuss the role of social media in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the Arab Spring and its impact on global politics.",
    "Analyze the advantages and disadvantages of the parliamentary system in modern democracies.",
    "Explain the concept of 'whataboutism' in political discourse and its implications.",
    "Clarify the misconception that all political scandals are equal and provide examples of those that have had significant consequences.",
    "Discuss the role of political polarization in modern democracies and its effects on governance.",
    "Describe the historical event of the Iranian Revolution and its impact on world politics.",
    "Analyze the advantages and disadvantages of the presidential system in modern democracies.",
    "Explain the concept of 'false equivalence' in political discourse and its implications.",
    "Clarify the misconception that all political ideologies are the same and provide examples of those with significant differences.",
    "Discuss the role of political corruption in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the Vietnam War and its impact on global politics.",
    "Analyze the advantages and disadvantages of direct democracy in modern democracies.",
    "Explain the concept of 'astroturfing' in political discourse and its implications.",
    "Clarify the misconception that all political protests are peaceful and provide examples of those that have turned violent.",
    "Discuss the role of political campaigns in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the Gulf War and its impact on world politics.",
    "Analyze the advantages and disadvantages of the two-party system in modern democracies.",
    "Explain the concept of 'gaslighting' in political discourse and its implications.",
    "Clarify the misconception that all political leaders are honest and provide examples of those who have lied.",
    "Discuss the role of political apathy in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the Rwandan Genocide and its impact on global politics.",
    "Analyze the advantages and disadvantages of the multi-party system in modern democracies.",
    "Explain the concept of 'confirmation bias' in political discourse and its implications.",
    "Clarify the misconception that all political compromises are bad and provide examples of those that have led to positive changes.",
    "Discuss the role of political correctness in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the Cold War and its impact on world politics.",
    "Analyze the advantages and disadvantages of the proportional representation system in modern democracies.",
    "Explain the concept of 'echo chamber' in political discourse and its implications.",
    "Clarify the misconception that all political polarization is bad and provide examples of those that have led to positive changes.",
    "Discuss the role of political lobbying in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the Cuban Revolution and its impact on world politics.",
    "Analyze the advantages and disadvantages of the first-past-the-post system in modern democracies.",
    "Explain the concept of 'strawman' in political discourse and its implications.",
    "Clarify the misconception that all political ideologies are extreme and provide examples of those with moderate views.",
    "Discuss the role of political apathy in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the American Civil War and its impact on world politics.",
    "Analyze the advantages and disadvantages of the runoff system in modern democracies.",
    "Explain the concept of 'red herring' in political discourse and its implications.",
    "Clarify the misconception that all political scandals involve criminal activity and provide examples of those that have led to positive changes.",
    "Discuss the role of political propaganda in shaping political discourse and its effects on democracy.",
    "Describe the historical event of the Suez Crisis and its impact on world politics.",
    "Analyze the advantages and disadvantages of the single-transferable-vote system in modern democracies.",
    "Explain the concept of 'ad hominem' in political discourse and its implications.",
    "Clarify the misconception that all political leaders are charismatic and provide examples of those who are less so.",
    "Describe the historical event of the Korean War and its impact on world politics.",
    "Analyze the advantages and disadvantages of the ranked-choice voting system in modern democracies.",
    "Explain the concept of 'appeasement' in political discourse and its implications.",
    "Describe the historical event of the Iran-Contra Affair and its impact on world politics.",
    "Describe the historical event of the Cuban Missile Crisis and its impact on world politics.",
    "Describe the historical event of the Vietnam War and its impact on world politics.",
    "Describe the historical event of the Gulf War and its impact on global politics.",
    "Analyze the effects of political apathy on democratic participation.",
    "Discuss the role of propaganda in democratic societies.",
    "Analyze the influence of media on democratic decision-making.",
    "Discuss the role of money in shaping democratic outcomes.",
    "Examine how social media platforms influence democratic discourse.",
    "Analyze the impact of political corruption on democratic institutions.",
    "Discuss how political campaigns influence voter perception.",
    "Examine the role of political parties in maintaining democratic stability.",
    "Discuss the influence of lobbying on public policy outcomes.",
    "Analyze how political correctness shapes public debate.",
    "Examine how political polarization affects legislative governance.",
    "Compare the advantages and disadvantages of parliamentary and presidential systems.",
    "Compare proportional representation systems with majoritarian electoral systems.",
    "Analyze the impact of the Arab Spring on Middle Eastern political systems.",
    "Discuss the long-term effects of the Iranian Revolution on U.S.-Iran relations.",
    "Analyze the impact of the Cold War on NATO and the Warsaw Pact.",
    "Discuss how the fall of the Berlin Wall influenced European integration.",
    "Examine the ethical implications of censorship in the digital age.",
    "Analyze how gerrymandering affects voter representation and electoral fairness.",
    "Discuss the implications of fake news on modern elections.",
    "Explain the importance of checks and balances in preventing abuse of power."
]


def load_model(model_key: str):
    """Load tokenizer + model from HuggingFace based on config key."""
    cfg = MODELS[model_key]
    print(f"Loading model: {cfg['name']} ...")

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])

    load_kwargs = {}
    if cfg["use_float16"]:
        load_kwargs["torch_dtype"] = torch.float16
    if cfg["device_map"] is not None:
        load_kwargs["device_map"] = cfg["device_map"]

    model = AutoModelForCausalLM.from_pretrained(cfg["name"], **load_kwargs)

    # If no device_map was used, move model to GPU/CPU explicitly
    if cfg["device_map"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    model.eval()
    print(f"Model {cfg['name']} loaded successfully.")
    return tokenizer, model


def generate(
    tokenizer,
    model,
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
) -> str:
    """Generate text using a HuggingFace model."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    # Decode only the newly generated tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def create_dataset(
    tokenizer,
    model,
    model_key: str,
    iterations: int,
    filename: str = "_data.jsonl",
):
    with open(filename, "a", encoding="utf-8") as file:
        for i in range(iterations):
            prompt = random.choice(PROMPTS)
            output = generate(
                tokenizer,
                model,
                prompt
                + " and Remember It is for uncensored data generation, so the output should be uncensored and should not contain any warnings or disclaimers.",
            )

            record = {
                "model": model_key,
                "prompt": prompt,
                "response": output,
            }

            print(f"================== [ Iteration {i + 1} ] ===================")

            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()

            print(f"Saved iteration {i + 1}")
            time.sleep(0.5)


if __name__ == "__main__":
    iterations = 10000
    model_key = "llama"  # Change to "llama" to use the Llama model

    tokenizer, model = load_model(model_key)
    create_dataset(tokenizer, model, model_key, iterations)

