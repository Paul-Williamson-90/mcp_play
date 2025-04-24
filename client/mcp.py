from typing import Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCP:
    def __init__(
        self,
        server_script_path: Optional[str] = "servers/weather.py"
    ):
        self.server_script_path = server_script_path
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self):
        await self.exit_stack.__aenter__()  # Enter the exit stack
        await self._connect_to_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()  # Ensure cleanup is called
        await self.exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.exit_stack.aclose()
        except RuntimeError as e:
            logger.error(f"Error during cleanup: {e}")
            raise
        
    async def get_available_tools(self) -> list[dict[str, Any]]:
        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        return available_tools
        
    async def _connect_to_server(self):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        if not self.server_script_path:
            logger.warning("No server script path provided. Skipping server connection.")
            return
        
        is_python = self.server_script_path.endswith('.py')
        is_js = self.server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[self.server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"Connected to server with tools: {", ".join([str(tool.name) for tool in tools])}")
   
   async def call_tool(self, tool_name: str, arguments: dict[str, Any]):
      try:
       tool_response = await self.session.call_tool(
         name=tool_name, arguments=arguments
       )
       return tool_response
      except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}")
