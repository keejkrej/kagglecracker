"""Unit tests for ARC pipeline components."""

import unittest
import numpy as np
import json
import glob
import os

from arc_prize_2026.src.augmentations import apply_d4, invert_d4, D4_OPERATIONS, generate_color_permutation, apply_color_perm, invert_color_perm
from arc_prize_2026.src.sandbox import evaluate_program_on_task, execute_with_timeout, grids_equal
from arc_prize_2026.src.prompts import build_coder_prompt
from arc_prize_2026.src.submission import validate_submission_grid, format_submission

class TestARCPipeline(unittest.TestCase):
    def test_d4_inverses(self):
        rng = np.random.RandomState(42)
        grid = rng.randint(0, 10, size=(5, 7))
        for op in D4_OPERATIONS:
            aug = apply_d4(grid, op)
            inv = invert_d4(aug, op)
            np.testing.assert_array_equal(inv, grid, err_msg=f"Failed D4 inversion for {op}")

    def test_color_perm_inverses(self):
        rng = np.random.RandomState(42)
        grid = rng.randint(0, 10, size=(6, 6))
        perm = generate_color_permutation(keep_zero=True, seed=42)
        permuted = apply_color_perm(grid, perm)
        restored = invert_color_perm(permuted, perm)
        np.testing.assert_array_equal(restored, grid)

    def test_sandbox_execution(self):
        code = """
import numpy as np
def transform(grid):
    # Invert non-zero to 1
    arr = np.array(grid)
    arr[arr > 0] = 1
    return arr.tolist()
"""
        task = {
            "train": [
                {"input": [[0, 2], [3, 0]], "output": [[0, 1], [1, 0]]},
                {"input": [[5, 5], [0, 0]], "output": [[1, 1], [0, 0]]}
            ],
            "test": [
                {"input": [[4, 0], [0, 6]]}
            ]
        }
        res = evaluate_program_on_task(code, task)
        self.assertTrue(res["solved_all_train"])
        self.assertEqual(res["train_accuracy"], 1.0)
        self.assertEqual(res["test_outputs"][0], [[1, 0], [0, 1]])

    def test_submission_validation(self):
        valid_grid = [[1, 2], [3, 4]]
        invalid_grid = [[1, 12], [3, 4]] # color 12 > 9
        self.assertTrue(validate_submission_grid(valid_grid))
        self.assertFalse(validate_submission_grid(invalid_grid))

if __name__ == "__main__":
    unittest.main()
