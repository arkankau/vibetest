"""Base classes and utilities for tools."""

from collections.abc import Callable

from inspect_ai.tool import Tool, tool


class ToolRegistry:
    """Registry for managing tools available to the agent."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool_fn: Callable) -> Tool:
        """Register a tool function.

        Args:
            tool_fn: Function decorated with @tool

        Returns:
            The registered tool
        """
        # If not already a Tool, wrap it
        if not isinstance(tool_fn, Tool):
            tool_fn = tool(tool_fn)
        self._tools[tool_fn.name] = tool_fn
        return tool_fn

    def get_tools(self) -> list[Tool]:
        """Get all registered tools.

        Returns:
            List of all tools
        """
        return list(self._tools.values())

    def get_tool(self, name: str) -> Tool | None:
        """Get a specific tool by name.

        Args:
            name: Tool name

        Returns:
            Tool or None if not found
        """
        return self._tools.get(name)


# Global registry instance
default_registry = ToolRegistry()
