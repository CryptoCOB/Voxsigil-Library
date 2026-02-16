"""
Tool Registry Module

This module provides a registry for tools and functors that can be used by agents.
"""
import logging
import datetime
from typing import Dict, Any, List, Callable, Optional

logger = logging.getLogger("metaconsciousness.tools.registry")

class ToolRegistry:
    """Registry for tools that can be called by agents."""

    def __init__(self, tools: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Initialize the tool registry.

        Args:
            tools: Initial tools dictionary
        """
        self.tools = tools or {}

    def register_tool(self, name: str, function: Callable, schema: Dict[str, Any]) -> None:
        """
        Register a tool with the registry.

        Args:
            name: Tool name
            function: Tool function
            schema: Tool schema
        """
        self.tools[name] = {
            "function": function,
            "schema": schema
        }
        logger.info(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool function or None if not found
        """
        tool_entry = self.tools.get(name)
        if tool_entry:
            return tool_entry["function"]
        return None

    def get_schema(self, name: str) -> Dict[str, Any]:
        """
        Get a tool's schema by name.

        Args:
            name: Tool name

        Returns:
            Tool schema or empty dict if not found
        """
        tool_entry = self.tools.get(name)
        if tool_entry:
            return tool_entry.get("schema", {})
        return {}

    def list_tools(self) -> List[str]:
        """
        List all registered tools.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

# Sample tool implementations

def search_web(query: str) -> str:
    """
    Simulate searching the web for information.

    Args:
        query: Search query

    Returns:
        Search results
    """
    return f"Simulated search results for: '{query}'"

def get_time() -> str:
    """
    Get the current UTC time.

    Returns:
        Current UTC time as ISO format string
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def get_weather(location: str) -> str:
    """
    Get weather information for a location.

    Args:
        location: Location to get weather for

    Returns:
        Weather information
    """
    return f"Simulated weather for {location}: Sunny, 22°C"

def calculate(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.

    Args:
        expression: Mathematical expression

    Returns:
        Calculation result
    """
    try:
        # Simple and safe way to evaluate expressions
        import ast
        import operator

        # Define allowed operators
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg
        }

        def eval_expr(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_expr(node.operand))
            else:
                raise TypeError(f"Unsupported operation: {node}")

        result = eval_expr(ast.parse(expression, mode='eval').body)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

# Create registry with sample tools
tool_registry = {
    "search_web": {
        "function": search_web,
        "schema": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            },
            "args": ["query"]
        }
    },
    "get_time": {
        "function": get_time,
        "schema": {
            "name": "get_time",
            "description": "Get the current UTC time",
            "parameters": {
                "type": "object",
                "properties": {}
            },
            "args": []
        }
    },
    "get_weather": {
        "function": get_weather,
        "schema": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for"
                    }
                },
                "required": ["location"]
            },
            "args": ["location"]
        }
    },
    "calculate": {
        "function": calculate,
        "schema": {
            "name": "calculate",
            "description": "Calculate the result of a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate"
                    }
                },
                "required": ["expression"]
            },
            "args": ["expression"]
        }
    }
}

# Create the default registry
default_registry = ToolRegistry(tool_registry)

# For backward compatibility
def execute_tool(name: str, *args, **kwargs) -> Any:
    """
    Execute a tool by name with arguments.

    Args:
        name: Tool name
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Tool execution result
    """
    tool = default_registry.get_tool(name)
    if tool:
        try:
            return tool(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool {name}: {str(e)}"
    else:
        return f"Unknown tool: {name}"
