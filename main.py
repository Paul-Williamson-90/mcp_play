import asyncio
import logging

from llama_index.core.memory import ChatMemoryBuffer

from client import MCPAgent, get_llm


logging.basicConfig(level=logging.INFO, filename="debug.log")


LLM = get_llm()
CHAT_MEMORY = ChatMemoryBuffer(token_limit=40_000)

async def run_agent(input: str, chat_history: ChatMemoryBuffer) -> ChatMemoryBuffer:
    AGENT = MCPAgent(
        llm=LLM,
        chat_history=chat_history,
    )
    try:
        await AGENT.run(input=input)
        return AGENT.chat_history
    finally:
        await AGENT.cleanup()
    
user_input = "Hello, what is the weather in New Jersey?" #input("Enter your input: ")

chat_history = asyncio.run(run_agent(user_input, CHAT_MEMORY))

chat_history = chat_history.get_all()

for message in chat_history:
    print(f"{message.role}: {message.content}")