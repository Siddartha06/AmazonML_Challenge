"""
Unit tests for src/blocking.py — Multi-Stage Blocking System
"""

import os
import sys
import unittest

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalization import normalize_record
from src.blocking import (
    generate_candidates,
    evaluate_blocking_recall,
    calculate_reduction_ratio,
    analyze_candidate_distribution,
    MultiBlocker
)


class TestBlockingSystem(unittest.TestCase):

    def setUp(self):
        """Create synthetic records for testing blocking strategies."""
        self.s1_records = [
            normalize_record("S1-001", "Acme Corporation", "123 Main Street", "US"),
            normalize_record("S1-002", "Bistro Parisien", "15 Rue de la Paix, Paris 75002", "France"),
            normalize_record("S1-003", "Tata Consultancy Services Pvt Ltd", "Plot 45 MG Road Bengaluru 560034", "India"),
            normalize_record("S1-004", "Singleton Enterprise", "999 Nowhere Lane", "US"), # Singleton
        ]

        self.s2_records = [
            normalize_record("S2-100", "Acme Corp.", "123 Main St.", "US"),                 # Match to S1-001
            normalize_record("S2-200", "Bistro Parisien SARL", "15 Rue de la Paix 75002", "France"), # Match to S1-002
            normalize_record("S2-300", "TCS Pvt Ltd", "Plot 45 MG Road 560034", "India"),     # Match to S1-003
            normalize_record("S2-999", "Unrelated US Business", "100 Broadway Ave", "US"),  # Distractor
        ]

        self.s3_records = [
            normalize_record("S3-100", "Acme Co", "123 Main Street Suite 4", "US"),        # Match to S1-001
            normalize_record("S3-300", "Tata Consultancy Services", "560034 Bengaluru", "India"), # Match to S1-003
            normalize_record("S3-888", "Unrelated Indian Shop", "50 FC Road Pune", "India"), # Distractor
        ]

        self.gt_dict = {
            "S1-001": {"S2-100", "S3-100"},
            "S1-002": {"S2-200"},
            "S1-003": {"S2-300", "S3-300"},
            "S1-004": set(), # Singleton
        }

    def test_single_block_exact_name(self):
        """Test exact name blocking strategy (BLOCK1)."""
        config = {"block_types": ["BLOCK1_exact_name"]}
        cands = generate_candidates(self.s1_records, self.s2_records, self.s3_records, config)
        
        self.assertIn("S2-100", cands["S1-001"]) # Acme Corp matches Acme Corporation
        self.assertIn("S2-200", cands["S1-002"]) # Bistro Parisien SARL matches Bistro Parisien

    def test_single_block_postal_code(self):
        """Test postal code blocking strategy (BLOCK6)."""
        config = {"block_types": ["BLOCK6_address_postal"]}
        cands = generate_candidates(self.s1_records, self.s2_records, self.s3_records, config)

        self.assertIn("S2-200", cands["S1-002"]) # 75002 Paris match
        self.assertIn("S2-300", cands["S1-003"]) # 560034 Bengaluru match
        self.assertIn("S3-300", cands["S1-003"]) # 560034 Bengaluru match

    def test_union_blocking_recall(self):
        """Test UNION of all blocking strategies achieves 100% recall on synthetic data."""
        cands = generate_candidates(self.s1_records, self.s2_records, self.s3_records)
        eval_res = evaluate_blocking_recall(cands, self.gt_dict)

        self.assertEqual(eval_res["total_gt_matches"], 5)
        self.assertEqual(eval_res["recovered_gt_matches"], 5)
        self.assertEqual(eval_res["blocking_recall"], 1.0)
        self.assertEqual(eval_res["s1_with_full_recall"], 3) # 3 non-singleton S1 entities fully recovered

    def test_no_country_cross_matching(self):
        """Test that country filtering prevents matching across different countries."""
        cands = generate_candidates(self.s1_records, self.s2_records, self.s3_records)

        # S1-002 is France -> must not contain US or India candidates
        for cand in cands["S1-002"]:
            self.assertEqual(cand, "S2-200") # S2-200 is France

    def test_reduction_ratio_and_distribution(self):
        """Test Reduction Ratio calculation and candidate distribution analysis."""
        cands = generate_candidates(self.s1_records, self.s2_records, self.s3_records)
        dist = analyze_candidate_distribution(cands)
        rr = calculate_reduction_ratio(dist["total_candidate_pairs"], 4, 4, 3)

        self.assertGreater(rr, 0.5)
        self.assertIn("mean_candidates", dist)
        self.assertIn("median_candidates", dist)
        self.assertIn("p95_candidates", dist)
        self.assertIn("max_candidates", dist)


if __name__ == "__main__":
    unittest.main()
