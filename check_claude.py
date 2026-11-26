"""Smoke test: can we reach Claude with our key?"""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # reads .env into environment variables

client = Anthropic()  # picks up ANTHROPIC_API_KEY automatically
model = os.environ["CLAUDE_MODEL"]

message = client.messages.create(
    model=model,
    max_tokens=100,
    messages=[{"role": "user", "content": "In one sentence: what is RAG?"}],
)

print("Model:      ", message.model)
print("Answer:     ", message.content[0].text)
print("Stop reason:", message.stop_reason)
print("Tokens in/out:", message.usage.input_tokens, "/", message.usage.output_tokens)