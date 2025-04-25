import logging
from typing import Any, Optional

from mcp.types import TextContent, ImageContent, EmbeddedResource
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.core import PromptTemplate

from client.events import AgentReasoningStep, ToolExecutionStep, AgentResponseStep
from client.prompts import ReasoningPrompt, ResponsePrompt
from client.pydantics import ReasoningStep, ToolSelection
from client.invocations import invocation_validator
from client.mcp import MCP


logger = logging.getLogger(__name__)


DEFAULT_TOKEN_LIMIT = 40_000


class MCPAgent(Workflow):
    
    _reasoning_prompt: PromptTemplate = ReasoningPrompt
    _response_prompt: PromptTemplate = ResponsePrompt
    _reasoning_llm_kwargs: dict[str, Any] = {"max_tokens": 3000}
    _response_llm_kwargs: dict[str, Any] = {"max_tokens": 3000}
    
    def __init__(
        self,
        llm: LLM,
        mcp: MCP,
        chat_history: Optional[ChatMemoryBuffer] = None,
    ):
        super().__init__(timeout=None)
        self.mcp = mcp
        self.llm = llm
        self.chat_history = (
            chat_history or ChatMemoryBuffer(token_limit=DEFAULT_TOKEN_LIMIT)
        )
        self.internal_memory = ChatMemoryBuffer(token_limit=DEFAULT_TOKEN_LIMIT)
        
    async def get_chat_history(self) -> str:
        chat_history = self.chat_history.get_all()
        if not chat_history:
            return "No chat history available."
        return "\n".join([str(msg) for msg in chat_history])
    
    async def get_thoughts(self) -> str:
        internal_memory = self.internal_memory
        if not isinstance(internal_memory, ChatMemoryBuffer):
            raise ValueError("No internal memory available.")
        else:
            internal_memory_list = internal_memory.get_all()
            thoughts_str = "\n".join([str(memory) for memory in internal_memory_list])
        return thoughts_str
        
    @step
    async def agent_preparation_step(self, ev: StartEvent) -> AgentReasoningStep:
        user_input = ev.input
        user_msg = ChatMessage(role=MessageRole.USER, content=user_input)
        self.chat_history.put(user_msg)
        self.internal_memory = ChatMemoryBuffer(token_limit=DEFAULT_TOKEN_LIMIT)
        logger.info(f"User input: {user_input}")
        return AgentReasoningStep()
    
    @step
    async def agent_reasoning_step(self, ev: AgentReasoningStep) -> ToolExecutionStep | AgentResponseStep:
        chat_history = await self.get_chat_history()
        available_tools = await self.mcp.get_available_tools()
        thoughts_str = await self.get_thoughts()
        
        prompt = self._reasoning_prompt.format(
            chat_history=chat_history,
            available_tools=available_tools,
            thoughts=thoughts_str,
        )
        
        output = await invocation_validator.structured_invocation(
            llm=self.llm,
            context=prompt,
            pydantic_object=ReasoningStep,
            llm_kwargs=self._reasoning_llm_kwargs,
        )
        
        logger.info(f"Agent Reasoning: {output}")
        
        if not isinstance(output, ReasoningStep):
            raise ValueError("Output is not a valid ReasoningStep")
        
        thoughts = "\n".join([str(thought) for thought in output.thoughts])
        self.internal_memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=thoughts))
        
        tool_call = output.tool_call
        if isinstance(tool_call, ToolSelection):
            return ToolExecutionStep(
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments
            )
        
        return AgentResponseStep()
    
    @step
    async def tool_execution_step(self, ev: ToolExecutionStep) -> AgentReasoningStep:
        tool_name = ev.tool_name
        arguments = ev.arguments
        
        try:
            tool_response = await self.mcp.call_tool(
                tool_name=tool_name, arguments=arguments
            )
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            msg = f"Error calling tool {tool_name}: {e}"
            self.internal_memory.put(
                ChatMessage(
                    role=MessageRole.TOOL, 
                    content=msg, 
                    additional_kwargs={
                        "tool_call_id": tool_name
                    }
                )
            )
            logger.info(f"Tool Response: {msg}")
            return AgentReasoningStep()
        
        content_list = tool_response.content
        
        texts: list[str] = []
        for content in content_list:
            if isinstance(content, TextContent):
                text = content.text
                self.internal_memory.put(
                    ChatMessage(
                        role=MessageRole.TOOL, 
                        content=text, 
                        additional_kwargs={
                            "tool_call_id": tool_name
                        }
                    )
                )
                texts.append(text)
            
            elif isinstance(content, ImageContent):
                logger.info("Tool Response: Image content is not supported yet.")
                raise NotImplementedError("Image content is not supported yet.")
            elif isinstance(content, EmbeddedResource):
                logger.info("Tool Response: Embedded resource is not supported yet.")
                raise NotImplementedError("Embedded resource is not supported yet.")
            else:
                logger.error("Unknown content type returned from tool.")
                raise ValueError("Unknown content type returned from tool.")
            
        logger.info(f"Tool Response: {"\n".join(texts)}")
        return AgentReasoningStep()
        
        
    @step
    async def agent_response_step(self, ev: AgentResponseStep) -> StopEvent:
        chat_history = await self.get_chat_history()
        thoughts_str = await self.get_thoughts()
        
        prompt = self._response_prompt.format(
            chat_history=chat_history,
            thoughts=thoughts_str,
        )
        
        output = await invocation_validator.non_structured_invocation(
            llm=self.llm,
            prompt=prompt,
            inference_kwargs=self._response_llm_kwargs,
        )
        
        self.chat_history.put(
            ChatMessage(role=MessageRole.ASSISTANT, content=output)
        )
        
        logger.info(f"Agent Response: {output}")
        return StopEvent(result=output)
