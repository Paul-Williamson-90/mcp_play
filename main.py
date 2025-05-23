import asyncio
import logging
import os

from llama_index.core.memory import ChatMemoryBuffer

from client import MCPAgent, get_llm, MCP


if os.path.exists("debug.log"):
    os.remove("debug.log")
    
logging.basicConfig(level=logging.INFO, filename="debug.log")


def main():
    LLM = get_llm()
    CHAT_MEMORY = ChatMemoryBuffer(token_limit=40_000)

    async def run_agent(input: str, chat_history: ChatMemoryBuffer) -> ChatMemoryBuffer:
        async with MCP() as mcp:
            agent = MCPAgent(llm=LLM, mcp=mcp, chat_history=chat_history)
            await agent.run(input=input)
            return agent.chat_history
        
    user_input = "Hello, fetch any weather alerts for New Jersey."

    chat_history = asyncio.run(run_agent(user_input, CHAT_MEMORY))

    chat_history = chat_history.get_all()

    for message in chat_history:
        print(f"{message.role}: {message.content}")
        

if __name__ == "__main__":
    main()
