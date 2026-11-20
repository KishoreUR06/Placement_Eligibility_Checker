
import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "allam-2-7b",
    "messages": [
        {
            "role": "user",
            "content": 'Return this JSON: {"status": "success", "message": "AI is working"}'
        }
    ],
    "response_format": {
        "type": "json_object"
    },
    "temperature": 0
}

print("Testing AI response...")

try:
    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        print("\nAI response received successfully!")
        print(answer)

    else:
        print("\nAI request failed!")
        print("Status code:", response.status_code)
        print("Error:", response.text)

except requests.exceptions.RequestException as error:
    print("Network error:", error)

