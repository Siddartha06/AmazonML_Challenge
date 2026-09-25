"""
Unit tests for src/validation.py — Business Entity Resolution Challenge
"""

import os
import sys
import unittest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation import (
    calculate_entity_f05,
    evaluate_predictions,
    evaluate_thresholds,
    create_entity_level_split,
    get_match_count_category
)


class TestValidationFramework(unittest.TestCase):

    def test_singleton_evaluations(self):
        """Test singleton behavior (Ground Truth has 0 matches)."""
        gt_singleton = set()
        
        # 1. Correctly predicting no matches -> 1.0
        score_correct = calculate_entity_f05(gt_singleton, set())
        self.assertEqual(score_correct, 1.0)
        
        # 2. Incorrectly predicting 1 match -> 0.0
        score_fp1 = calculate_entity_f05(gt_singleton, {"S2-100"})
        self.assertEqual(score_fp1, 0.0)

        # 3. Incorrectly predicting multiple matches -> 0.0
        score_fp2 = calculate_entity_f05(gt_singleton, {"S2-100", "S3-200"})
        self.assertEqual(score_fp2, 0.0)

        # 4. Input as string / empty string
        self.assertEqual(calculate_entity_f05("", ""), 1.0)
        self.assertEqual(calculate_entity_f05("", "S2-100"), 0.0)

    def test_single_match_evaluations(self):
        """Test non-singleton behavior with 1 true match."""
        gt = {"S2-100"}

        # 1. Missing match (empty prediction) -> 0.0
        self.assertEqual(calculate_entity_f05(gt, set()), 0.0)

        # 2. Exact match -> 1.0
        self.assertEqual(calculate_entity_f05(gt, {"S2-100"}), 1.0)

        # 3. 1 True Positive + 1 False Positive -> P=0.5, R=1.0 -> F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.555556
        f05_fp = calculate_entity_f05(gt, {"S2-100", "S2-999"})
        self.assertAlmostEqual(f05_fp, 5.0 / 9.0, places=5)

        # 4. Completely wrong match -> 0.0
        self.assertEqual(calculate_entity_f05(gt, {"S2-999"}), 0.0)

    def test_multi_match_evaluations(self):
        """Test non-singleton behavior with multiple true matches."""
        gt = {"S2-100", "S3-200", "S3-300"}

        # 1. Exact match (3/3) -> 1.0
        self.assertEqual(calculate_entity_f05(gt, {"S2-100", "S3-200", "S3-300"}), 1.0)

        # 2. Partial match (2/3 TP, 0 FP) -> P=1.0, R=2/3 -> F0.5 = (1.25 * 1.0 * (2/3)) / (0.25 * 1.0 + 2/3) = (5/6) / (11/12) = 10/11 ~ 0.909091
        f05_partial = calculate_entity_f05(gt, {"S2-100", "S3-200"})
        self.assertAlmostEqual(f05_partial, 10.0 / 11.0, places=5)

    def test_duplicate_prediction_handling(self):
        """Test that duplicate predictions in lists are automatically deduplicated."""
        gt = {"S2-100"}
        pred_list = ["S2-100", "S2-100", "S2-100"]
        score = calculate_entity_f05(gt, pred_list)
        self.assertEqual(score, 1.0)

    def test_macro_averaging(self):
        """Test macro-averaging across multiple Source-1 entities."""
        gt_dict = {
            "S1-001": set(),                  # Singleton
            "S1-002": {"S2-10"},              # 1 match
            "S1-003": {"S2-20", "S3-30"},      # 2 matches
            "S1-004": {"S2-40", "S3-50", "S3-60"} # 3 matches
        }

        # Perfect predictions across all
        pred_dict_perfect = {
            "S1-001": set(),
            "S1-002": {"S2-10"},
            "S1-003": {"S2-20", "S3-30"},
            "S1-004": {"S2-40", "S3-50", "S3-60"}
        }
        res_perfect = evaluate_predictions(gt_dict, pred_dict_perfect)
        self.assertEqual(res_perfect["macro_f05"], 1.0)
        self.assertEqual(res_perfect["singleton_f05"], 1.0)
        self.assertEqual(res_perfect["1_match_f05"], 1.0)

        # Mixed predictions
        pred_dict_mixed = {
            "S1-001": {"S2-99"},              # FP for singleton -> 0.0
            "S1-002": {"S2-10"},              # Perfect -> 1.0
            "S1-003": {"S2-20"},              # Partial TP=1, P=1.0, R=0.5 -> F0.5 = 5/6 ~ 0.833333
            "S1-004": set()                   # Empty for non-singleton -> 0.0
        }
        res_mixed = evaluate_predictions(gt_dict, pred_dict_mixed)
        expected_macro = (0.0 + 1.0 + (5.0 / 6.0) + 0.0) / 4.0
        self.assertAlmostEqual(res_mixed["macro_f05"], round(expected_macro, 6), places=5)
        self.assertEqual(res_mixed["singleton_f05"], 0.0)
        self.assertEqual(res_mixed["1_match_f05"], 1.0)

    def test_entity_level_split(self):
        """Test entity-level train/validation split reproducibility and zero overlap."""
        gt_dict = {}
        # Create synthetic ground truth with 100 entities across all categories
        for i in range(100):
            s1_id = f"S1-{i:04d}"
            if i < 20:
                gt_dict[s1_id] = set()
            elif i < 50:
                gt_dict[s1_id] = {f"S2-{i}"}
            elif i < 80:
                gt_dict[s1_id] = {f"S2-{i}", f"S3-{i}"}
            else:
                gt_dict[s1_id] = {f"S2-{i}", f"S3-{i}", f"S3-{i+1000}"}

        train_ids, val_ids = create_entity_level_split(gt_dict, val_size=0.2, random_state=42)

        # 1. Check total count
        self.assertEqual(len(train_ids) + len(val_ids), 100)

        # 2. Check ZERO overlap
        overlap = set(train_ids).intersection(set(val_ids))
        self.assertEqual(len(overlap), 0)

        # 3. Check reproducibility with same seed
        train_ids_2, val_ids_2 = create_entity_level_split(gt_dict, val_size=0.2, random_state=42)
        self.assertEqual(train_ids, train_ids_2)
        self.assertEqual(val_ids, val_ids_2)

        # 4. Check stratification proportions
        val_singletons = sum(1 for sid in val_ids if len(gt_dict[sid]) == 0)
        self.assertTrue(3 <= val_singletons <= 5) # ~20% of 20 singletons = 4

    def test_evaluate_thresholds(self):
        """Test threshold optimization on synthetic candidate probabilities."""
        gt_dict = {
            "S1-001": {"S2-100"},
            "S1-002": {"S3-200"}
        }
        candidate_probs = [
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-100", "probability": 0.95},
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-999", "probability": 0.40},
            {"source1_entity_id": "S1-002", "candidate_entity_id": "S3-200", "probability": 0.85},
            {"source1_entity_id": "S1-002", "candidate_entity_id": "S3-888", "probability": 0.30},
        ]

        res = evaluate_thresholds(candidate_probs, gt_dict, thresholds=[0.20, 0.50, 0.80])
        # At thresh=0.50 or 0.80, only true matches >= thresh are picked -> F0.5 = 1.0
        # At thresh=0.20, low probability false positives are also picked -> lower F0.5
        self.assertIn(res["best_threshold"], [0.50, 0.80])
        self.assertEqual(res["best_macro_f05"], 1.0)


if __name__ == "__main__":
    unittest.main()
