"""Execution tools for running code."""

import subprocess
import sys
from typing import Annotated
from inspect_ai.tool import tool


@tool
def run_python(
    code: Annotated[str, "Python code to execute"],
    working_dir: Annotated[str, "Working directory for execution"] = ".",
) -> str:
    """Execute Python code and return the output.

    Args:
        code: Python code to run
        working_dir: Directory to run code in

    Returns:
        Combined stdout and stderr from execution
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        output += f"\nReturn code: {result.returncode}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out after 60 seconds"
    except Exception as e:
        return f"Error executing Python code: {str(e)}"


@tool
def run_command(
    command: Annotated[str, "Shell command to execute"],
    working_dir: Annotated[str, "Working directory for execution"] = ".",
) -> str:
    """Execute a shell command and return the output.

    Args:
        command: Command to run
        working_dir: Directory to run command in

    Returns:
        Combined stdout and stderr from execution
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        output += f"\nReturn code: {result.returncode}"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"
