"""Simple vizualization utils.
"""

from typing import List


def table_print(headings: List[str], table: List[List[str]]) -> str:
    """Print a table in markdown format."""
    head_line = " & ".join(headings) + r"\\ \hline" + "\n"
    table_lines = []
    for row in table:
        table_lines.append(" & ".join(row))
    table = head_line + "\\\\ \n".join(table_lines)
    print(table)
