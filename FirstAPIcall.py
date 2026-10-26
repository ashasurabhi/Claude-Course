from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

msg = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ],
)

print(msg.content[0].text)