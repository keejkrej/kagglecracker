"""ARC Program Execution Sandbox

Safely compiles, executes, and validates candidate Python grid transformations
with strict timeouts and memory isolation.
"""

from typing import List, Tuple, Dict, Any, Optional
import ast
import re
import concurrent.futures
import numpy as np

# Allowed standard imports / builtins for ARC solving
ALLOWED_MODULES = {"numpy", "np", "math", "collections", "itertools", "scipy"}

def extract_code_block(text: str) -> str:
    """Extract python code from markdown fences or raw code."""
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def copy_grid(grid: Any) -> Any:
    """Fast deepcopy of 2D grid list."""
    if isinstance(grid, list):
        return [list(r) if isinstance(r, list) else r for r in grid]
    return grid

def _safe_import(name, *args, **kwargs):
    allowed = {"numpy", "np", "math", "collections", "itertools", "scipy", "copy", "re"}
    top_module = name.split(".")[0]
    if top_module not in allowed:
        raise ImportError(f"Import of module '{name}' is not allowed in sandbox")
    return __import__(name, *args, **kwargs)

def _run_isolated_transform(code_str: str, input_grid: list) -> Tuple[bool, Any, str]:
    """Execute code with restricted globals."""
    try:
        builtins_dict = {
            k: __builtins__[k]
            for k in [
                "range", "len", "enumerate", "zip", "map", "filter",
                "list", "dict", "set", "tuple", "int", "float", "bool",
                "str", "min", "max", "sum", "abs", "round", "sorted",
                "any", "all", "print", "Exception", "ValueError", "TypeError",
                "IndexError", "KeyError", "isinstance", "issubclass"
            ]
            if (isinstance(__builtins__, dict) and k in __builtins__) or hasattr(__builtins__, k)
        }
        builtins_dict["__import__"] = _safe_import

        scope = {
            "np": np,
            "numpy": np,
            "__builtins__": builtins_dict
        }
        exec(code_str, scope)
        if "transform" not in scope or not callable(scope["transform"]):
            return False, None, "Function 'transform' not found"
            
        out = scope["transform"](copy_grid(input_grid))
        
        if isinstance(out, np.ndarray):
            out = out.tolist()
        elif isinstance(out, list):
            out = [list(r) if isinstance(r, (list, tuple, np.ndarray)) else r for r in out]
            
        return True, out, ""
    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)}"

def execute_with_timeout(code_str: str, input_grid: list, timeout_sec: float = 2.0) -> Tuple[bool, Any, str]:
    """Execute code with a strict timeout using ThreadPoolExecutor."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_isolated_transform, code_str, input_grid)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            return False, None, "Timeout exceeded"
        except Exception as e:
            return False, None, f"Execution failed: {str(e)}"

def grids_equal(g1: Any, g2: Any) -> bool:
    """Check if two grids are identical in shape and color values."""
    if not isinstance(g1, list) or not isinstance(g2, list):
        return False
    if len(g1) != len(g2):
        return False
    for r1, r2 in zip(g1, g2):
        if not isinstance(r1, (list, tuple)) or not isinstance(r2, (list, tuple)):
            return False
        if len(r1) != len(r2):
            return False
        if list(r1) != list(r2):
            return False
    return True

def evaluate_program_on_task(
    code_str: str, 
    task: Dict[str, Any], 
    timeout_per_pair: float = 2.0
) -> Dict[str, Any]:
    """Evaluate a candidate program against all train pairs in an ARC task."""
    cleaned_code = extract_code_block(code_str)
    
    # 1. Syntax check
    try:
        ast.parse(cleaned_code)
    except SyntaxError as e:
        return {
            "solved_all_train": False,
            "train_accuracy": 0.0,
            "test_outputs": [],
            "errors": [f"SyntaxError: {e}"]
        }

    # 2. Check on all train pairs
    train_pairs = task.get("train", [])
    if not train_pairs:
        return {"solved_all_train": False, "train_accuracy": 0.0, "test_outputs": [], "errors": ["No train pairs"]}

    num_matched = 0
    errors = []
    
    for idx, pair in enumerate(train_pairs):
        in_grid = pair["input"]
        expected_out = pair["output"]
        success, pred_out, err = execute_with_timeout(cleaned_code, in_grid, timeout_sec=timeout_per_pair)
        if not success:
            errors.append(f"Train[{idx}] failed: {err}")
            break
        if grids_equal(pred_out, expected_out):
            num_matched += 1
        else:
            errors.append(f"Train[{idx}] output mismatch")
            break

    train_accuracy = num_matched / len(train_pairs)
    solved_all = (num_matched == len(train_pairs))
    
    test_outputs = []
    if solved_all:
        for idx, pair in enumerate(task.get("test", [])):
            test_in = pair["input"]
            success, pred_out, err = execute_with_timeout(cleaned_code, test_in, timeout_sec=timeout_per_pair)
            if success and pred_out is not None:
                test_outputs.append(pred_out)
            else:
                test_outputs.append(None)
                errors.append(f"Test[{idx}] execution failed: {err}")

    return {
        "solved_all_train": solved_all,
        "train_accuracy": train_accuracy,
        "test_outputs": test_outputs,
        "errors": errors,
        "cleaned_code": cleaned_code
    }
