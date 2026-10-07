"""ARC Prompt Engineering & Formatting

Formats ARC tasks (input/output grids) for code generation LLMs
(Qwen2.5-Coder, DeepSeek-Coder, Claude, Gemini) with spatial reasoning guidelines.
"""

from typing import Dict, Any, List
import numpy as np

COLOR_NAMES = {
    0: "black (0)",
    1: "blue (1)",
    2: "red (2)",
    3: "green (3)",
    4: "yellow (4)",
    5: "gray (5)",
    6: "magenta (6)",
    7: "orange (7)",
    8: "teal (8)",
    9: "brown (9)",
}

def grid_to_string(grid: List[List[int]]) -> str:
    """Format a 2D list into a clean row-by-row string with dimensions."""
    arr = np.array(grid)
    h, w = arr.shape
    rows_str = "\n".join("  [" + ", ".join(f"{val}" for val in row) + "]" for row in arr)
    return f"Shape: {h}x{w}\n[\n{rows_str}\n]"

def build_coder_prompt(task: Dict[str, Any], include_system: bool = True) -> str:
    """Build a comprehensive coding prompt for programmatic ARC transformation."""
    system_prompt = (
        "You are an expert mathematician and computer scientist specializing in spatial reasoning, "
        "cellular automata, and the Abstraction and Reasoning Corpus (ARC).\n"
        "Your task is to identify the core rule that maps input grids to output grids, and write a "
        "concise, robust Python function `def transform(grid: list[list[int]]) -> list[list[int]]:` "
        "that implements this transformation."
    )
    
    body = ["Here are the demonstration input/output pairs for this ARC challenge:\n"]
    
    for i, pair in enumerate(task.get("train", [])):
        in_grid = pair["input"]
        out_grid = pair["output"]
        body.append(f"### Example {i+1}:")
        body.append(f"Input Grid:\n{grid_to_string(in_grid)}")
        body.append(f"Output Grid:\n{grid_to_string(out_grid)}\n")
        
    for i, pair in enumerate(task.get("test", [])):
        in_grid = pair["input"]
        body.append(f"### Test Query {i+1}:")
        body.append(f"Input Grid:\n{grid_to_string(in_grid)}")
        body.append("(Generate the transformation function that produces the matching output)\n")
        
    instructions = (
        "### Instructions:\n"
        "1. Identify the invariant features, colors, shapes, boundaries, symmetries, or arithmetic patterns.\n"
        "2. Formulate the transformation hypothesis.\n"
        "3. Implement `def transform(grid: list[list[int]]) -> list[list[int]]:`.\n"
        "   - You may use standard libraries like `numpy` (as `np`), `math`, `collections`, `itertools`.\n"
        "   - The function must take a list of lists of ints and return a list of lists of ints.\n"
        "   - Wrap your complete, self-contained Python code in ```python ... ```."
    )
    body.append(instructions)
    
    full_prompt = "\n".join(body)
    if include_system:
        return f"{system_prompt}\n\n{full_prompt}"
    return full_prompt
