"""ARC Evaluation and Benchmark Runner

Evaluates candidate solutions or search engines across task benchmarks,
computes top-1 and top-2 exact match accuracy, and produces performance metrics.
"""

from typing import Dict, Any, List, Optional
import os
import glob
import json
import time

from .sandbox import grids_equal
from .submission import format_submission, save_submission

def load_tasks_from_dir(task_dir: str, limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """Load JSON task files from a directory."""
    files = glob.glob(os.path.join(task_dir, "*.json"))
    tasks = {}
    for fpath in files:
        task_id = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, "r", encoding="utf-8") as f:
            tasks[task_id] = json.load(f)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks

def evaluate_predictions_against_solutions(
    predictions: Dict[str, List[Dict[str, List[List[int]]]]],
    solutions: Dict[str, List[List[List[int]]]]
) -> Dict[str, Any]:
    """
    Evaluate top-1 and top-2 accuracy.
    In ARC competition rules, a test pair is solved if either attempt_1 OR attempt_2 matches ground truth.
    A multi-query task is solved only if ALL test queries are correct.
    """
    total_tasks = 0
    solved_tasks = 0
    top1_tasks = 0
    total_test_pairs = 0
    solved_test_pairs = 0
    top1_test_pairs = 0
    per_task_results = {}

    for task_id, ground_truths in solutions.items():
        if task_id not in predictions:
            continue
        total_tasks += 1
        preds = predictions[task_id]
        
        task_all_solved = True
        task_top1_all_solved = True
        pair_matches = []

        for p_idx, gt in enumerate(ground_truths):
            total_test_pairs += 1
            if p_idx < len(preds):
                att1 = preds[p_idx].get("attempt_1")
                att2 = preds[p_idx].get("attempt_2")
                
                m1 = grids_equal(att1, gt)
                m2 = grids_equal(att2, gt)
                
                if m1:
                    top1_test_pairs += 1
                if m1 or m2:
                    solved_test_pairs += 1
                else:
                    task_all_solved = False
                    
                if not m1:
                    task_top1_all_solved = False
                    
                pair_matches.append({"pair_idx": p_idx, "m1": m1, "m2": m2})
            else:
                task_all_solved = False
                task_top1_all_solved = False
                pair_matches.append({"pair_idx": p_idx, "m1": False, "m2": False})

        if task_all_solved:
            solved_tasks += 1
        if task_top1_all_solved:
            top1_tasks += 1
            
        per_task_results[task_id] = {
            "solved": task_all_solved,
            "top1_solved": task_top1_all_solved,
            "pairs": pair_matches
        }

    return {
        "total_tasks": total_tasks,
        "solved_tasks": solved_tasks,
        "task_accuracy_pct": (solved_tasks / total_tasks * 100) if total_tasks > 0 else 0.0,
        "top1_task_accuracy_pct": (top1_tasks / total_tasks * 100) if total_tasks > 0 else 0.0,
        "total_test_pairs": total_test_pairs,
        "solved_test_pairs": solved_test_pairs,
        "pair_accuracy_pct": (solved_test_pairs / total_test_pairs * 100) if total_test_pairs > 0 else 0.0,
        "details": per_task_results
    }
