import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Available models for your API key:")
for model in client.models.list():
    if "generateContent" in (model.supported_actions or []):
        print(f" - {model.name}")