import os

from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_llm() -> OpenAI:
    return OpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
