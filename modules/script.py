import os
import requests
import json
from dotenv import load_dotenv
from utils.prompt import get_ad_script_prompt

load_dotenv()

# ——— CONFIG ———
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ——— SCRIPT GENERATOR ———
def generate_ad_script(prompt: str) -> list:
    """
    Sends the user prompt to Groq's LLM model and returns
    a list of scene dicts:
    [
      {
        "scene": 1,
        "duration": "5s",
        "visual_description": "...",
        "dialogue": "...",
        "on_screen_text": "..."
      },
      ...
    ]
    """
    # Get prompts from the prompt module
    prompts = get_ad_script_prompt(prompt)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        # "model": "llama-3.3-70b-versatile",
        "model": "mistral-saba-24b",
        "messages": [
            {"role": "system", "content": prompts["system_prompt"]},
            {"role": "user", "content": prompts["user_prompt"]},
        ],
        "temperature": 0.7,
        "max_tokens": 800,
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # Parse out the JSON array
    try:
        script = json.loads(content)
    except json.JSONDecodeError:
        # If the model returns extra text, try to extract the JSON substring
        start = content.find("[")
        end = content.rfind("]") + 1
        script = json.loads(content[start:end])

    return script