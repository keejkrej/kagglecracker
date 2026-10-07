"""ARC Submission Formatter and Kaggle Submitter

Formats predictions into the official Kaggle ARC JSON structure,
validates integrity (grid ranges 1-30, color values 0-9), and submits via CLI.
"""

from typing import Dict, Any, List
import json
import os
import subprocess

def validate_submission_grid(grid: Any) -> bool:
    """Validate that a predicted grid matches ARC requirements."""
    if not isinstance(grid, list) or len(grid) == 0:
        return False
    h = len(grid)
    if h > 30:
        return False
    w = len(grid[0])
    if w == 0 or w > 30:
        return False
    for row in grid:
        if not isinstance(row, list) or len(row) != w:
            return False
        for val in row:
            if not isinstance(val, int) or val < 0 or val > 9:
                return False
    return True

def format_submission(results: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, Any]:
    """Validate and format dictionary into standard submission structure."""
    clean_submission = {}
    for task_id, attempts_list in results.items():
        clean_task_preds = []
        for pair_attempts in attempts_list:
            att1 = pair_attempts.get("attempt_1", [[0]])
            att2 = pair_attempts.get("attempt_2", [[0]])
            
            # Fallback invalid grids to 1x1 [0]
            if not validate_submission_grid(att1):
                att1 = [[0]]
            if not validate_submission_grid(att2):
                att2 = att1
                
            clean_task_preds.append({
                "attempt_1": att1,
                "attempt_2": att2
            })
        clean_submission[task_id] = clean_task_preds
    return clean_submission

def save_submission(
    submission_dict: Dict[str, Any], 
    filepath: str
) -> str:
    """Save submission JSON to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(submission_dict, f)
    return filepath

def submit_to_kaggle(
    submission_path: str, 
    competition: str = "arc-prize-2026-arc-agi-2", 
    message: str = "ARC SOTA programmatic synthesis submission",
    kaggle_bin: str = "kaggle"
) -> Dict[str, Any]:
    """Submit the predictions file to Kaggle competition via CLI."""
    cmd = [
        kaggle_bin,
        "competitions",
        "submit",
        "-c", competition,
        "-f", submission_path,
        "-m", message
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"success": True, "output": proc.stdout}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.stderr or e.stdout}
