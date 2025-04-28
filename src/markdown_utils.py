"""
Utility functions for Markdown elements.
"""

import numpy as np
from IPython.core.display_functions import display
from IPython.display import Markdown


def show_comparison_table(
        metric_names: list[str],
        first_metrics: list[float],
        second_metrics: list[float],
        first_column_name: str = "Model Metrics 1",
        second_column_name: str = "Model Metrics 2",
) -> None:
    """
    Creates and show a Markdown table comparing default and optimized model metrics.

    Args:
        metric_names (list[str]): List of metric names.
        first_metrics (list[float]): List of metric values for the default model.
        second_metrics (list[float]): List of metric values for the optimized model.
        first_column_name (str): Name of the first column in the table.
        second_column_name (str): Name of the second column in the table.

    Returns:
        str: A Markdown table as a string.

    Raise:
        Exception: the metrics length is different from the default or optimized metrics.
    """

    if len(metric_names) != len(first_metrics) or len(metric_names) != len(second_metrics):
        raise Exception("Error: Metric lists must have the same length.")

    markdown_table = f"| Metric | {first_column_name} | {second_column_name} |\n"
    markdown_table += "|---|---|---|\n"

    for i in range(len(metric_names)):
        markdown_table += f"| {metric_names[i]} | {np.round(first_metrics[i], 3)} | {np.round(second_metrics[i], 3)} |\n"

    # display the table
    display(Markdown(markdown_table))

    return None
