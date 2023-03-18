import openai
import json
import tiktoken

with open("./config.json", "r") as f:
    CONFIG = json.loads(f.read())

openai.api_key = CONFIG['openai_apikey']

persona = """
chris is a talented and creative tattoo artist who runs a tattoo studio in the city. he's known for his ability to create unique and functional tattoo designs for his clients.
"""

name = persona.split(" ",1)[0]

player_input = "Tell me about yourself"

messages = [
    {"role": "system", "content": "You are a helpful assistant that generates dialogue from a given Stardew Valley persona."},
    {"role": "user", "content": f'Create a conversation between a Stardew Valley character and a player. Each should speak three times. The persona is "{persona}". The first line from the player is "{player_input}"' }
]
model = "gpt-3.5-turbo"

encoding = tiktoken.encoding_for_model(model)
def num_tokens_from_string(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

tokens = 0
for message in messages:
    tokens += num_tokens_from_string(message['content'], encoding)
print(tokens)

response = openai.ChatCompletion.create(model = model,
                                        messages = messages,
                                        max_tokens = 300)

print(response)

