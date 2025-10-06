"""Analysis and visualization tools."""

import json
import re
from pathlib import Path
from typing import Annotated
from inspect_ai.tool import tool


@tool
def create_plot(
    data: Annotated[str, "JSON string of data to plot (e.g., {'x': [...], 'y': [...]})"],
    output_path: Annotated[str, "Path to save the plot"],
    title: Annotated[str, "Plot title"] = "Plot",
    xlabel: Annotated[str, "X-axis label"] = "X",
    ylabel: Annotated[str, "Y-axis label"] = "Y",
) -> str:
    """Create a plot from data and save it.

    Args:
        data: JSON string containing plot data
        output_path: Where to save the plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        Status message
    """
    try:
        import matplotlib.pyplot as plt
        import json

        plot_data = json.loads(data)

        plt.figure(figsize=(10, 6))
        if "x" in plot_data and "y" in plot_data:
            plt.plot(plot_data["x"], plot_data["y"])
        elif "y" in plot_data:
            plt.plot(plot_data["y"])
        else:
            return "Error: Data must contain at least 'y' values"

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        return f"Plot saved to {output_path}"
    except Exception as e:
        return f"Error creating plot: {str(e)}"


@tool
def parse_logs(
    log_text: Annotated[str, "Log text to parse"],
    pattern: Annotated[str, "Regex pattern to search for"],
) -> str:
    """Parse log text using a regex pattern.

    Args:
        log_text: Text to parse
        pattern: Regular expression pattern

    Returns:
        JSON string of matches
    """
    try:
        matches = re.findall(pattern, log_text)
        return json.dumps({"matches": matches, "count": len(matches)}, indent=2)
    except Exception as e:
        return f"Error parsing logs: {str(e)}"
