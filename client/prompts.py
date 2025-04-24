from llama_index.core import PromptTemplate


ReasoningPrompt = PromptTemplate(
    "System: "
    "You are a helpful chatbot assistant that supports users with their requests. "
    "You have a set of tools that are defined via the Model Context Protocol (MCP) standard. "
    "You can use these tools to gather information whenever it is appropriate to facilitate a user's request. "
    "Once you have gathered all the information you need, you can choose to continue to the response step by setting the tool_call to None. "
    "Specific instructions on how your output should be formatted are provided below, you msut follow these exactly and provide no additional information or preamble. "
    "\n\nHere are the tools you can use: "
    "\n{available_tools}"
    "\n\nChat History: "
    "{chat_history}"
    "\n\nYour thoughts and context gathered so far:"
    "\n{thoughts}"
)

ResponsePrompt = PromptTemplate(
    "System: "
    "You are a helpful chatbot assistant that supports users with their requests. "
    "You have received a request from a user and have already gathered all the information you need to respond. "
    "Base on the information below, you should provide a response to the user. "
    "\n\nChat History: "
    "\n{chat_history}"
    "\n\nYour thoughts and context gathered so far:"
    "\n{thoughts}"
    "\n\nNow, write your response to the user below:"
    "\nAssistant: "
)