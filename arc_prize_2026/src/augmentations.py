"""ARC Data Augmentation Module

Implements Dihedral D4 group symmetries, color space permutations,
and inverse transformations for test-time augmentation (TTA).
"""

from typing import List, Tuple, Dict, Any, Callable
import copy
import random
import numpy as np

# Dihedral D4 operations
D4_OPERATIONS = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
]

def apply_d4(grid: np.ndarray, op: str) -> np.ndarray:
    """Apply D4 dihedral symmetry transformation to a 2D numpy grid."""
    if op == "identity":
        return grid.copy()
    elif op == "rot90":
        return np.rot90(grid, -1).copy()  # Clockwise 90
    elif op == "rot180":
        return np.rot90(grid, 2).copy()
    elif op == "rot270":
        return np.rot90(grid, 1).copy()   # Clockwise 270 / Counter-clockwise 90
    elif op == "flip_h":
        return np.fliplr(grid).copy()
    elif op == "flip_v":
        return np.flipud(grid).copy()
    elif op == "transpose":
        return grid.T.copy()
    elif op == "anti_transpose":
        return np.rot90(grid.T, 2).copy()
    else:
        raise ValueError(f"Unknown D4 operation: {op}")

def invert_d4(grid: np.ndarray, op: str) -> np.ndarray:
    """Apply the inverse of the D4 transformation."""
    if op == "identity":
        return grid.copy()
    elif op == "rot90":
        return np.rot90(grid, 1).copy()   # Inverse of clockwise 90 is CCW 90
    elif op == "rot180":
        return np.rot90(grid, 2).copy()
    elif op == "rot270":
        return np.rot90(grid, -1).copy()  # Inverse of CW 270 is CW 90
    elif op == "flip_h":
        return np.fliplr(grid).copy()
    elif op == "flip_v":
        return np.flipud(grid).copy()
    elif op == "transpose":
        return grid.T.copy()
    elif op == "anti_transpose":
        return np.rot90(grid.T, 2).copy()
    else:
        raise ValueError(f"Unknown D4 operation: {op}")

def apply_color_perm(grid: np.ndarray, perm_dict: Dict[int, int]) -> np.ndarray:
    """Apply a color bijection to the grid."""
    result = grid.copy()
    for src, dst in perm_dict.items():
        result[grid == src] = dst
    return result

def invert_color_perm(grid: np.ndarray, perm_dict: Dict[int, int]) -> np.ndarray:
    """Invert the color permutation mapping."""
    inv_dict = {v: k for k, v in perm_dict.items()}
    return apply_color_perm(grid, inv_dict)

def generate_color_permutation(keep_zero: bool = True, seed: int = None) -> Dict[int, int]:
    """Generate a random 1-to-1 color permutation among 0..9."""
    rng = random.Random(seed)
    if keep_zero:
        colors = list(range(1, 10))
        shuffled = colors.copy()
        rng.shuffle(shuffled)
        mapping = {0: 0}
        for src, dst in zip(colors, shuffled):
            mapping[src] = dst
        return mapping
    else:
        colors = list(range(10))
        shuffled = colors.copy()
        rng.shuffle(shuffled)
        return dict(zip(colors, shuffled))

def augment_task_d4(task: Dict[str, Any], op: str) -> Dict[str, Any]:
    """Augment an entire ARC task (train pairs + test input) using a D4 symmetry."""
    new_task = copy.deepcopy(task)
    for pair in new_task.get("train", []):
        pair["input"] = apply_d4(np.array(pair["input"]), op).tolist()
        pair["output"] = apply_d4(np.array(pair["output"]), op).tolist()
    
    for pair in new_task.get("test", []):
        pair["input"] = apply_d4(np.array(pair["input"]), op).tolist()
        if "output" in pair:
            pair["output"] = apply_d4(np.array(pair["output"]), op).tolist()
            
    return new_task

def augment_task_color(task: Dict[str, Any], perm: Dict[int, int]) -> Dict[str, Any]:
    """Augment an entire ARC task using a color permutation."""
    new_task = copy.deepcopy(task)
    for pair in new_task.get("train", []):
        pair["input"] = apply_color_perm(np.array(pair["input"]), perm).tolist()
        pair["output"] = apply_color_perm(np.array(pair["output"]), perm).tolist()
        
    for pair in new_task.get("test", []):
        pair["input"] = apply_color_perm(np.array(pair["input"]), perm).tolist()
        if "output" in pair:
            pair["output"] = apply_color_perm(np.array(pair["output"]), perm).tolist()
            
    return new_task
