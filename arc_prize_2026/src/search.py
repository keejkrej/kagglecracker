"""ARC Test-Time Search and Ensembling Engine

Implements Test-Time Augmentation (TTA), multi-candidate sampling,
sandbox program verification, and Kaggle Top-2 ensembling.
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
from collections import Counter
import json
import numpy as np

from .augmentations import (
    D4_OPERATIONS,
    apply_d4,
    invert_d4,
    augment_task_d4,
    generate_color_permutation,
    augment_task_color,
    invert_color_perm
)
from .sandbox import evaluate_program_on_task, copy_grid

def grid_to_key(grid: List[List[int]]) -> str:
    """Hashable representation of a 2D grid."""
    if grid is None:
        return "None"
    return json.dumps(grid)

def key_to_grid(key: str) -> Optional[List[List[int]]]:
    """Parse JSON key back to 2D grid."""
    if key == "None":
        return None
    return json.loads(key)

class ARCSearchEngine:
    def __init__(
        self,
        generator_fn: Optional[Callable[[str, int, float], List[str]]] = None,
        use_d4_tta: bool = True,
        samples_per_variant: int = 4,
        temperature: float = 0.7,
        timeout_per_pair: float = 2.0
    ):
        """
        Args:
            generator_fn: Function (prompt, num_samples, temperature) -> list of code strings
            use_d4_tta: Whether to augment with D4 rotations/reflections
            samples_per_variant: Number of code samples generated per augmentation
            temperature: Sampling temperature
            timeout_per_pair: Execution timeout in seconds per grid pair
        """
        self.generator_fn = generator_fn
        self.use_d4_tta = use_d4_tta
        self.samples_per_variant = samples_per_variant
        self.temperature = temperature
        self.timeout_per_pair = timeout_per_pair

    def solve_task(
        self, 
        task: Dict[str, Any], 
        from_prompt_builder: Callable[[Dict[str, Any]], str]
    ) -> Dict[str, Any]:
        """
        Run test-time search over task variants.
        Returns:
            {
                "top_predictions": list of list of [attempt_1, attempt_2] per test query,
                "verified_programs": list of verified code strings,
                "all_candidates_count": int
            }
        """
        num_test = len(task.get("test", []))
        # Frequency counts of predictions per test pair: list of Counter[grid_key]
        pred_counters = [Counter() for _ in range(num_test)]
        verified_programs = []
        
        # Determine task variants: canonical + D4 operations
        variants: List[Tuple[str, Dict[str, Any]]] = [("identity", task)]
        if self.use_d4_tta:
            for op in D4_OPERATIONS:
                if op != "identity":
                    variants.append((op, augment_task_d4(task, op)))

        for op, variant_task in variants:
            if self.generator_fn is None:
                continue

            prompt = from_prompt_builder(variant_task)
            candidates = self.generator_fn(prompt, self.samples_per_variant, self.temperature)
            
            for code in candidates:
                eval_res = evaluate_program_on_task(
                    code, 
                    variant_task, 
                    timeout_per_pair=self.timeout_per_pair
                )
                
                if eval_res["solved_all_train"]:
                    verified_programs.append(eval_res["cleaned_code"])
                    test_outs = eval_res["test_outputs"]
                    
                    for t_idx, test_out in enumerate(test_outs):
                        if test_out is not None:
                            # Invert D4 transformation back to canonical orientation
                            if op != "identity":
                                restored_out = invert_d4(np.array(test_out), op).tolist()
                            else:
                                restored_out = test_out
                                
                            pred_counters[t_idx][grid_to_key(restored_out)] += 1

        # Build top-2 submissions for each test query
        final_test_predictions = []
        for t_idx in range(num_test):
            counter = pred_counters[t_idx]
            most_common = counter.most_common(2)
            
            p1 = None
            p2 = None
            if len(most_common) > 0:
                p1 = key_to_grid(most_common[0][0])
            if len(most_common) > 1:
                p2 = key_to_grid(most_common[1][0])
                
            # Fallback if no verified candidate found: copy input grid
            default_grid = copy_grid(task["test"][t_idx]["input"])
            if p1 is None:
                p1 = default_grid
            if p2 is None:
                p2 = copy_grid(p1)

            final_test_predictions.append({"attempt_1": p1, "attempt_2": p2})

        return {
            "predictions": final_test_predictions,
            "verified_count": len(verified_programs),
            "verified_programs": verified_programs
        }
