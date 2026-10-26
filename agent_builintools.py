
import anthropic
from dotenv import load_dotenv


# Load the API key from the .env file next to this script
load_dotenv()

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
response = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "what is the latest version of Claude?",}],
    tools=[{"type": "web_search_20260318", "name": "web_search"}], # built in tool
)

for block in response.content:
    if block.type == "text":
        print(block.text)