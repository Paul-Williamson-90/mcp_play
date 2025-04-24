from typing import Any, Optional

from pydantic import BaseModel


class Step(BaseModel):
    """Use this schema to think through the user's request and the context so far.

    Parameters
    ----------
    thought : str
        A thought you have as you think through the user's request and the context received so far.
    conclusion : str
        Your conclusion based on the thought.
    """
    thought: str
    conclusion: str
    
    def __str__(self) -> str:
        return f"Thought: {self.thought}\nConclusion: {self.conclusion}"
    
    
class ToolSelection(BaseModel):
    """Use this schema to select a tool and provide the arguments it requires.

    Parameters
    ----------
    tool_name : str
        The name of the tool to be called.
    arguments : dict[str, Any]
        The arguments required by the tool.
    """
    tool_name: str
    arguments: dict[str, Any]
    

class ReasoningStep(BaseModel):
    """Use this schema to think through the user's request and the context so far.
    Based on your thoughts, you can either decide to call a tool, or choose to move to the next step of responding to the user.
    Selecting a tool will require you to provide the name of the tool and any arguments it requires.
    If you decide not to use a tool and wish to instead respond to the user, you can simply return None for the tool_call.

    Parameters
    ----------
    thoughts : list[Step]
        A list of thoughts as you think though the user's request and the context received so far.
    tool_call : Optional[ToolSelection] (default=None)
        If you decide to call a tool, provide the name of the tool and any arguments it requires.
        If you decide not to use a tool, this can be None.
    """
    thoughts: list[Step]
    tool_call: Optional[ToolSelection] = None