"""
ML Challenge 2026 — Multi-Stage High-Recall Blocking & Candidate Generation System

This module implements multiple independent blocking strategies for Business Entity Resolution
and combines them via UNION to maximize true-match candidate recall while constraining
the candidate pair volume.

Key Principles:
1. Preserve Source-2 and Source-3 candidate IDs.
2. Never produce Source-1 self-matches.
3. Strictly enforce Country Matching (same country only).
4. Support singletons (0 matches), 1-match, and multi-matches.
5. Work seamlessly across open-set countries (e.g. France, US, India).
6. Frequency-prune high-cardinality blocking keys to prevent pair explosions.
"""

import os
import csv
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union, Optional

from src.normalization import normalize_record, NormalizedRecord


# =============================================================================
# Inverted Index Builder & Key Generators
# =============================================================================

class MultiBlocker:
    """Engine for generating candidate pairs across multiple independent blocking strategies."""

    def __init__(self, max_key_frequency: int = 500):
        """
        Args:
            max_key_frequency: Maximum allowed candidate records per blocking key to prevent pair explosion.
        """
        self.max_key_frequency = max_key_frequency

    @staticmethod
    def get_block_keys_for_record(rec: NormalizedRecord, block_type: str) -> List[str]:
        """Generate blocking keys for a single normalized record based on strategy block type."""
        country = rec.country_clean if rec.country_clean else "UNKNOWN"
        keys = []

        if block_type == "BLOCK1_exact_name":
            # Exact clean name or exact lower name
            if rec.name_no_legal_suffix:
                keys.append(f"{country}||b1||{rec.name_no_legal_suffix}")
            elif rec.name_lower:
                keys.append(f"{country}||b1||{rec.name_lower}")

        elif block_type == "BLOCK2_exact_compact":
            # Exact compact name (no spaces/punct)
            if rec.name_compact:
                keys.append(f"{country}||b2||{rec.name_compact}")

        elif block_type == "BLOCK3_name_prefix":
            # Name compact prefix (4-char & 6-char)
            if len(rec.name_compact) >= 4:
                keys.append(f"{country}||b3_4||{rec.name_compact[:4]}")
            if len(rec.name_compact) >= 6:
                keys.append(f"{country}||b3_6||{rec.name_compact[:6]}")

        elif block_type == "BLOCK4_important_tokens":
            # Name tokens with length >= 4 (excluding legal suffix terms)
            no_legal_tokens = rec.name_no_legal_suffix.split()
            for tok in no_legal_tokens:
                if len(tok) >= 4 and not tok.isdigit():
                    keys.append(f"{country}||b4||{tok}")

        elif block_type == "BLOCK5_sorted_tokens":
            # Sorted name tokens
            if rec.name_sorted_tokens:
                keys.append(f"{country}||b5||{rec.name_sorted_tokens}")

        elif block_type == "BLOCK6_address_postal":
            # Postal/PIN code blocking
            if rec.address_postal_code:
                keys.append(f"{country}||b6||{rec.address_postal_code}")

        elif block_type == "BLOCK7_address_number_token":
            # House number + address token
            if rec.address_numeric_tokens and rec.address_tokens:
                num = rec.address_numeric_tokens[0]
                for tok in rec.address_tokens:
                    if len(tok) >= 4 and not tok.isdigit():
                        keys.append(f"{country}||b7||{num}||{tok}")
                        break  # take first qualifying street token

        elif block_type == "BLOCK8_name_address_token":
            # Name token + Address numeric token
            no_legal_tokens = rec.name_no_legal_suffix.split()
            if no_legal_tokens and rec.address_numeric_tokens:
                name_tok = no_legal_tokens[0]
                num_tok = rec.address_numeric_tokens[0]
                if len(name_tok) >= 3:
                    keys.append(f"{country}||b8||{name_tok}||{num_tok}")

        elif block_type == "BLOCK9_char_trigrams":
            # Character 3-grams of compact name
            comp = rec.name_compact
            if len(comp) >= 5:
                # generate character 3-grams
                grams = set(comp[i:i+3] for i in range(len(comp) - 2))
                for g in list(grams)[:5]:  # limit to top 5 grams per name
                    keys.append(f"{country}||b9||{g}")

        return keys

    def build_index(
        self,
        candidate_records: List[NormalizedRecord],
        block_types: List[str]
    ) -> Dict[str, List[str]]:
        """Build an inverted index mapping blocking_key -> List of candidate entity_ids."""
        index = defaultdict(list)
        for rec in candidate_records:
            for btype in block_types:
                keys = self.get_block_keys_for_record(rec, btype)
                for k in keys:
                    index[k].append(rec.entity_id)
        return index

    def generate_candidates_for_s1(
        self,
        s1_records: List[NormalizedRecord],
        candidate_index: Dict[str, List[str]],
        block_types: List[str]
    ) -> Dict[str, Set[str]]:
        """Query inverted index for each S1 entity to generate candidate sets."""
        candidates = {}
        for rec in s1_records:
            cand_set = set()
            for btype in block_types:
                keys = self.get_block_keys_for_record(rec, btype)
                for k in keys:
                    matched_ids = candidate_index.get(k, [])
                    # Prune over-frequent keys to avoid candidate pair explosion
                    if 0 < len(matched_ids) <= self.max_key_frequency:
                        cand_set.update(matched_ids)
                    elif len(matched_ids) > self.max_key_frequency:
                        # For high-frequency keys, take a deterministic prefix subset if necessary
                        cand_set.update(matched_ids[:100])
            candidates[rec.entity_id] = cand_set
        return candidates


# =============================================================================
# High-Level Blocking Interface
# =============================================================================

def generate_candidates(
    source1_records: List[Union[NormalizedRecord, Dict]],
    source2_records: List[Union[NormalizedRecord, Dict]],
    source3_records: List[Union[NormalizedRecord, Dict]],
    config: Optional[Dict] = None
) -> Dict[str, Set[str]]:
    """Generate candidate pairs for Source-1 entities using multi-stage blocking.
    
    Args:
        source1_records: List of S1 records (NormalizedRecord objects or raw dicts).
        source2_records: List of S2 records.
        source3_records: List of S3 records.
        config: Dict specifying blocking configuration:
            - 'block_types': List of block strategy names to include.
            - 'max_key_frequency': Max candidate cardinality per blocking key (default 500).

    Returns:
        Dict mapping source1_entity_id -> Set of candidate S2/S3 entity IDs.
    """
    if config is None:
        config = {}

    block_types = config.get("block_types", [
        "BLOCK1_exact_name",
        "BLOCK2_exact_compact",
        "BLOCK3_name_prefix",
        "BLOCK4_important_tokens",
        "BLOCK5_sorted_tokens",
        "BLOCK6_address_postal",
        "BLOCK7_address_number_token",
        "BLOCK8_name_address_token",
    ])
    max_freq = config.get("max_key_frequency", 500)

    # Convert dicts to NormalizedRecord objects if necessary
    def ensure_normalized(recs):
        normed = []
        for r in recs:
            if isinstance(r, NormalizedRecord):
                normed.append(r)
            else:
                eid = r.get("entity_id", r.get("source1_entity_id", ""))
                normed.append(normalize_record(
                    eid,
                    r.get("business_name"),
                    r.get("business_address"),
                    r.get("country")
                ))
        return normed

    s1_norm = ensure_normalized(source1_records)
    s2_norm = ensure_normalized(source2_records)
    s3_norm = ensure_normalized(source3_records)

    all_candidates = s2_norm + s3_norm

    blocker = MultiBlocker(max_key_frequency=max_freq)
    index = blocker.build_index(all_candidates, block_types)
    candidate_dict = blocker.generate_candidates_for_s1(s1_norm, index, block_types)

    return candidate_dict


# =============================================================================
# Evaluation & Analysis Metrics
# =============================================================================

def evaluate_blocking_recall(
    candidate_dict: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]]
) -> Dict[str, Union[float, int]]:
    """Evaluate blocking recall across all evaluated Source-1 entities against ground truth.
    
    Recall = (Total True Matches Recovered in Candidates) / (Total True Matches in GT)
    """
    total_gt_matches = 0
    recovered_gt_matches = 0

    s1_with_full_recall = 0
    s1_with_partial_recall = 0
    s1_with_zero_recall = 0
    total_s1 = 0

    for s1_id, gt_set in gt_dict.items():
        total_s1 += 1
        num_gt = len(gt_set)
        if num_gt == 0:
            continue  # singletons have 0 matches in GT

        total_gt_matches += num_gt
        cands = candidate_dict.get(s1_id, set())
        recovered = len(gt_set.intersection(cands))
        recovered_gt_matches += recovered

        if recovered == num_gt:
            s1_with_full_recall += 1
        elif recovered > 0:
            s1_with_partial_recall += 1
        else:
            s1_with_zero_recall += 1

    recall = (recovered_gt_matches / total_gt_matches) if total_gt_matches > 0 else 1.0

    return {
        "total_gt_matches": total_gt_matches,
        "recovered_gt_matches": recovered_gt_matches,
        "blocking_recall": round(float(recall), 6),
        "total_s1_entities": total_s1,
        "s1_with_full_recall": s1_with_full_recall,
        "s1_with_partial_recall": s1_with_partial_recall,
        "s1_with_zero_recall": s1_with_zero_recall,
    }


def calculate_reduction_ratio(
    total_candidate_pairs: int,
    num_s1: int,
    num_s2: int,
    num_s3: int
) -> float:
    """Calculate the Reduction Ratio (RR) metric.
    
    RR = 1 - (Total Candidate Pairs) / (Num S1 * (Num S2 + Num S3))
    """
    max_possible_pairs = num_s1 * (num_s2 + num_s3)
    if max_possible_pairs == 0:
        return 1.0
    rr = 1.0 - (total_candidate_pairs / max_possible_pairs)
    return round(float(rr), 8)


def analyze_candidate_distribution(
    candidate_dict: Dict[str, Set[str]]
) -> Dict[str, Union[int, float]]:
    """Analyze the distribution of candidate set sizes per Source-1 entity."""
    counts = [len(cands) for cands in candidate_dict.values()]
    if not counts:
        return {
            "total_candidate_pairs": 0,
            "mean_candidates": 0.0,
            "median_candidates": 0,
            "p95_candidates": 0,
            "p99_candidates": 0,
            "max_candidates": 0,
            "min_candidates": 0,
            "empty_s1_count": 0,
        }

    counts.sort()
    n = len(counts)
    total_pairs = sum(counts)
    mean_c = total_pairs / n
    median_c = counts[n // 2]
    p95_c = counts[int(0.95 * n)]
    p99_c = counts[int(0.99 * n)]
    max_c = counts[-1]
    min_c = counts[0]
    empty_cnt = sum(1 for c in counts if c == 0)

    return {
        "total_candidate_pairs": total_pairs,
        "mean_candidates": round(float(mean_c), 2),
        "median_candidates": median_c,
        "p95_candidates": p95_c,
        "p99_candidates": p99_c,
        "max_candidates": max_c,
        "min_candidates": min_c,
        "empty_s1_count": empty_cnt,
    }
