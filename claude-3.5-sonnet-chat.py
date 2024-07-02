from anthropic import Anthropic

import asyncio
import base64
import httpx

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.style import Style
from rich.syntax import Syntax

api_key = open("./path_to/anthropic_api_key.txt").read()
client = Anthropic(api_key=api_key)

conversation_history = []
collected_assistant_input = []
last_used_image_file = []

class StreamingChatCompletion:
    r"""
    This allows markdown renderable from streamed message and codeblocks (if exists) beautifully
    from streaming mode Anthropic API calls.
    Each chunk of token outputs will be automatically rendered in `Markdown`.
    
    So far my implementation is going well, till further modification.
    """
    def __init__(self):
        self.console = Console()
        self.full_text = ""
        self.code_block = ""
        self.in_code_block = False

    async def simulate_api_call(self, stream):
        for text in stream.text_stream:
            collected_assistant_input.append(text)
            yield text
            
    def update_content(self, chunk):
        if self.in_code_block:
            if "```" in chunk:
                self.code_block += chunk.split("```")[0]
                self.full_text += f"```{self.code_block}```"
                self.in_code_block = False
                self.code_block = ""
            else:
                self.code_block += chunk
        else:
            if chunk.startswith("```"):
                self.in_code_block = True
                self.code_block = chunk[3:]
            else:
                self.full_text += chunk

    def render_content(self):
        md = Markdown(self.full_text,
                    style=Style(color='#d78700'))
        if self.in_code_block:
            md = Markdown(self.full_text + f"```{self.code_block}```",
                        style=Style(color="#d78700"))
        return md

    async def stream_chat_completion(self, stream):
        with Live(self.render_content(), refresh_per_second=10) as live:
            async for chunk in self.simulate_api_call(stream):
                self.update_content(chunk)
                live.update(self.render_content())

def chat_completion_engine(user_input, prompt):

    with client.messages.stream(
        max_tokens=1024,
        system=f"You are a helpful assistant.\nThis is conversation history between you and user:\n{prompt}",
        messages=[
            {"role": "user","content": user_input}
            ],
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        top_p=1
        ) as stream:

        # Call streamer
        streamer = StreamingChatCompletion()
        asyncio.run(streamer.stream_chat_completion(stream))
    output = "".join(collected_assistant_input)
    return output

def vision_chat_engine(question: str, image_file: None, prompt: str):
    if image_file != None:
        image_url = image_file
    else:
        image_url = last_used_image_file[0].strip()

    image_media_type = "image/png"
    with open(image_url, 'rb') as file:
        image_data = base64.b64encode(file.read()).decode("utf-8")

    with client.messages.stream(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2048,
        temperature=0,
        system=f"You are a helpful assistant.\nThis is conversation history between you and user:\n{prompt}",
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
      
        # call streamer
        streamer = StreamingChatCompletion()
        asyncio.run(streamer.stream_chat_completion(stream))
    output = "".join(collected_assistant_input)
    return output

if __name__ == "__main__":

    # Conversation loop begins.
    while True:
        question = input("> ").strip()
        
        conversation_history.append(f"user: {question}")

        prompt = "\n".join(message for message in conversation_history)

        # Uncomment to choose one of the engines
        #output = vision_chat_engine(question=question, image_file="./path_to_your_image/image.png",prompt=prompt)
        output = chat_completion_engine(user_input=question, prompt=prompt)

        conversation_history.append(f"Assistant: {output}")
