"""ARC Prompt Engineering & Formatting

Formats ARC tasks (input/output grids) compactly for code generation LLMs
(Qwen2.5-Coder, DeepSeek-Coder) with spatial reasoning guidelines.
"""

from typing import Dict, Any, List
import numpy as np

def grid_to_string(grid: List[List[int]]) -> str:
    """Format a 2D list compactly to minimize token consumption while preserving spatial layout."""
    arr = np.array(grid)
    h, w = arr.shape
    
    # For small grids (<= 12x12), output python list representation
    if h * w <= 144:
        rows = [f"  [{','.join(str(v) for v in row)}]" for row in arr]
        return f"Shape {h}x{w}:\n[\n" + ",\n".join(rows) + "\n]"
    else:
        # Compact single-character representation per cell for large grids
        # ARC colors are strictly 0..9, so each cell is a single digit
        rows = ["  " + "".join(str(v) for v in row) for row in arr]
        return f"Shape {h}x{w} (compact digits):\n" + "\n".join(rows)

def build_coder_prompt(task: Dict[str, Any], include_system: bool = True) -> str:
    """Build a concise, high-signal coding prompt for programmatic ARC transformation."""
    system_prompt = (
        "You are an expert ARC (Abstraction and Reasoning Corpus) programmatic solver.\n"
        "Analyze the demonstration input/output grids and implement a Python function:\n"
        "`def transform(grid: list[list[int]]) -> list[list[int]]:`\n"
        "Requirements:\n"
        "- Return the correct output grid as a list of list of ints.\n"
        "- Standard libraries `numpy` (as `np`), `math`, `collections` are available.\n"
        "- Enclose your code in ```python ... ```."
    )
    
    body = ["Below are the task demonstration pairs:\n"]
    
    for i, pair in enumerate(task.get("train", [])):
        in_grid = pair["input"]
        out_grid = pair["output"]
        body.append(f"### Example {i+1}:")
        body.append(f"Input ({grid_to_string(in_grid)})")
        body.append(f"Output ({grid_to_string(out_grid)})\n")
        
    for i, pair in enumerate(task.get("test", [])):
        in_grid = pair["input"]
        body.append(f"### Test Query {i+1}:")
        body.append(f"Input ({grid_to_string(in_grid)})")
        body.append("(Implement the transformation function to produce this output)\n")
        
    full_prompt = "\n".join(body)
    if include_system:
        return f"{system_prompt}\n\n{full_prompt}"
    return full_prompt
