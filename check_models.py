import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Your API key has access to these models:")
for model in client.models.list().data:
    print(f"- {model.id}")