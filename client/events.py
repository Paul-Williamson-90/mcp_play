from typing import Any

from llama_index.core.workflow import Event


class AgentReasoningStep(Event):
    pass


class ToolExecutionStep(Event):
    tool_name: str
    arguments: dict[str, Any]
    
    
class AgentResponseStep(Event):
    pass