"""File system tools for the agent."""

import os
from pathlib import Path
from typing import Annotated
from inspect_ai.tool import tool


@tool
def read_file(path: Annotated[str, "Path to file to read"]) -> str:
    """Read contents of a file.

    Args:
        path: Path to the file

    Returns:
        File contents as string
    """
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def list_directory(
    path: Annotated[str, "Directory path to list"] = "."
) -> str:
    """List contents of a directory.

    Args:
        path: Directory path (default: current directory)

    Returns:
        Formatted directory listing
    """
    try:
        items = os.listdir(path)
        result = []
        for item in sorted(items):
            full_path = os.path.join(path, item)
            item_type = "DIR" if os.path.isdir(full_path) else "FILE"
            result.append(f"[{item_type}] {item}")
        return "\n".join(result) if result else "Directory is empty"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def find_files(
    pattern: Annotated[str, "File pattern to search for (e.g., '*.py')"],
    directory: Annotated[str, "Directory to search in"] = ".",
) -> str:
    """Find files matching a pattern.

    Args:
        pattern: Glob pattern to match
        directory: Starting directory for search

    Returns:
        List of matching file paths
    """
    try:
        from pathlib import Path

        base_path = Path(directory)
        matches = list(base_path.rglob(pattern))
        if matches:
            return "\n".join(str(p) for p in matches)
        else:
            return f"No files matching '{pattern}' found in {directory}"
    except Exception as e:
        return f"Error finding files: {str(e)}"
