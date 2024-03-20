"""
Code implementation to access Claude 3 family models from Anthropic AI.
The most capable model is `claude-3-opus-20240229`.
"""

import base64
import httpx
from anthropic import Anthropic
from typing_extensions import Literal

#store the API key in .txt file
api_key = open("your-api-key/anthropic.txt").read()
client = Anthropic(api_key=api_key)

#For chat completion
def chat_completion(question: str,
                    model: Literal[
                    'claude-3-haiku-20240307',
                    'claude-3-sonnet-20240229',
                    'claude-3-opus-20240229'] = None):
    '''
    Fortunately, system prompt is set straight as the parameter, not in the `message`.
    '''
  
    with client.messages.stream(
        max_tokens=1024,
        system="You are Claude 3 model, from Anthropic. You will answer and response courteously and truthful.",
        messages=[
            {"role": "user","content": question}
            ],
        model=model,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

#For vision capabilities
def vision(question: str,
           image_url: str,
           model: Literal[
           'claude-3-haiku-20240307',
           'claude-3-sonnet-20240229', 
           'claude-3-opus-20240229'] = None):
  
    image_url = image_url #example: https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg
    image_media_type = "image/jpeg"
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

    client = Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data,
                        },
                    },
                    {
                    "type": "text",
                    "text": question
                    }
                ],
            }
        ],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

if __name__ == '__main__':
  #select one of the two functions we have created earlier.
