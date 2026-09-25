"""
ML Challenge 2026 — Business Entity Resolution Validation Framework

This module provides a leak-free entity-level validation framework and macro F0.5 metric evaluation.

Key Requirements:
1. Macro-averaged F0.5 evaluation over Source-1 entities.
2. Correct handling of singletons (0 matches GT -> 1.0 if 0 pred, 0.0 if >0 pred).
3. Correct handling of non-singletons (F0.5 = (1.25 * P * R) / (0.25 * P + R)).
4. Entity-level train/validation split (no pair-level split, no candidate leakage).
5. Stratified splitting by match-count categories (0, 1, 2, 3+).
"""

import os
import csv
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union, Optional


def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Load ground truth TSV into a mapping from source1_entity_id to a set of matched entity IDs.
    
    Args:
        gt_path: Path to train_ground_truth.tsv

    Returns:
        Dict mapping source1_entity_id -> set of matched S2/S3 entity IDs.
    """
    gt_dict = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        
        # Determine column indices
        if header is None:
            raise ValueError(f"Ground truth file {gt_path} is empty.")
        
        s1_col = 0
        match_col = 1
        for idx, col in enumerate(header):
            col_lower = col.strip().lower()
            if col_lower == "source1_entity_id":
                s1_col = idx
            elif col_lower in ("matched_entity_ids", "matched_ids"):
                match_col = idx

        for row in reader:
            if not row or len(row) <= s1_col:
                continue
            s1_id = row[s1_col].strip()
            if not s1_id:
                continue
                
            raw_matches = row[match_col].strip() if len(row) > match_col else ""
            if raw_matches:
                matches = set(x.strip() for x in raw_matches.split(",") if x.strip())
            else:
                matches = set()
                
            gt_dict[s1_id] = matches

    return gt_dict


def _to_set(val: Union[Set[str], List[str], Tuple[str, ...], str]) -> Set[str]:
    """Helper function to normalize predicted or GT matches into a set of non-empty strings."""
    if isinstance(val, set):
        return set(x.strip() for x in val if x and x.strip())
    if isinstance(val, (list, tuple)):
        return set(x.strip() for x in val if x and x.strip())
    if isinstance(val, str):
        if not val.strip():
            return set()
        return set(x.strip() for x in val.split(",") if x.strip())
    return set()


def calculate_entity_f05(
    gt_matches: Union[Set[str], List[str], str],
    pred_matches: Union[Set[str], List[str], str]
) -> float:
    """Calculate F0.5 score for a single Source-1 entity according to challenge rules.
    
    Metric Definition:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Singleton rules (Ground Truth has 0 matches):
    - If prediction has 0 matches: F0.5 = 1.0 (perfect prediction of singleton)
    - If prediction has >0 matches: F0.5 = 0.0 (false positive predictions)
    
    Non-singleton rules (Ground Truth has >0 matches):
    - If prediction has 0 matches: F0.5 = 0.0 (recall is 0.0)
    - If Precision + Recall == 0: F0.5 = 0.0
    - Otherwise: F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    gt_set = _to_set(gt_matches)
    pred_set = _to_set(pred_matches)

    # 1. Singleton case: Ground truth has 0 matches
    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    # 2. Non-singleton case: Ground truth has >0 matches, but prediction has 0
    if len(pred_set) == 0:
        return 0.0

    # 3. Both GT and prediction have >0 matches
    true_positives = len(gt_set.intersection(pred_set))
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(pred_set)
    recall = true_positives / len(gt_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    f05 = (1.25 * precision * recall) / denom
    return float(f05)


def evaluate_predictions(
    gt_dict: Dict[str, Set[str]],
    pred_dict: Dict[str, Union[Set[str], List[str], str]],
    return_per_entity: bool = False
) -> Dict[str, Union[float, Dict]]:
    """Evaluate predictions against ground truth using macro-averaged F0.5 score across all Source-1 entities.
    
    Args:
        gt_dict: Dict mapping source1_entity_id -> set of ground truth matched IDs.
        pred_dict: Dict mapping source1_entity_id -> set/list/string of predicted matched IDs.
        return_per_entity: If True, include per-entity scores in the returned dict.

    Returns:
        Dict containing:
            - macro_f05: Overall macro-averaged F0.5 score
            - macro_precision: Overall macro-averaged precision
            - macro_recall: Overall macro-averaged recall
            - total_entities: Number of evaluated S1 entities
            - singleton_f05: Macro F0.5 on singletons (0-match entities)
            - 1_match_f05: Macro F0.5 on 1-match entities
            - 2_match_f05: Macro F0.5 on 2-match entities
            - 3plus_match_f05: Macro F0.5 on 3+ match entities
            - (optional) per_entity_scores: Dict[str, float] if return_per_entity=True
    """
    total_entities = len(gt_dict)
    if total_entities == 0:
        raise ValueError("Ground truth dictionary is empty.")

    f05_scores = []
    precisions = []
    recalls = []

    group_f05 = defaultdict(list)
    per_entity_scores = {}

    for s1_id, gt_set in gt_dict.items():
        pred_val = pred_dict.get(s1_id, set())
        gt_s = _to_set(gt_set)
        pred_s = _to_set(pred_val)

        # F0.5 score
        score = calculate_entity_f05(gt_s, pred_s)
        f05_scores.append(score)
        if return_per_entity:
            per_entity_scores[s1_id] = score

        # Precision & Recall tracking for macro summary
        if len(gt_s) == 0:
            p = 1.0 if len(pred_s) == 0 else 0.0
            r = 1.0 if len(pred_s) == 0 else 0.0
            cat = "singleton"
        else:
            cat = "1_match" if len(gt_s) == 1 else ("2_match" if len(gt_s) == 2 else "3plus_match")
            if len(pred_s) == 0:
                p = 0.0
                r = 0.0
            else:
                tp = len(gt_s.intersection(pred_s))
                p = tp / len(pred_s)
                r = tp / len(gt_s)

        precisions.append(p)
        recalls.append(r)
        group_f05[cat].append(score)

    macro_f05 = float(sum(f05_scores) / total_entities)
    macro_p = float(sum(precisions) / total_entities)
    macro_r = float(sum(recalls) / total_entities)

    summary = {
        "macro_f05": round(macro_f05, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "total_entities": total_entities,
        "singleton_f05": round(sum(group_f05["singleton"]) / len(group_f05["singleton"]), 6) if group_f05["singleton"] else 0.0,
        "singleton_count": len(group_f05["singleton"]),
        "1_match_f05": round(sum(group_f05["1_match"]) / len(group_f05["1_match"]), 6) if group_f05["1_match"] else 0.0,
        "1_match_count": len(group_f05["1_match"]),
        "2_match_f05": round(sum(group_f05["2_match"]) / len(group_f05["2_match"]), 6) if group_f05["2_match"] else 0.0,
        "2_match_count": len(group_f05["2_match"]),
        "3plus_match_f05": round(sum(group_f05["3plus_match"]) / len(group_f05["3plus_match"]), 6) if group_f05["3plus_match"] else 0.0,
        "3plus_match_count": len(group_f05["3plus_match"]),
    }

    if return_per_entity:
        summary["per_entity_scores"] = per_entity_scores

    return summary


def evaluate_thresholds(
    candidate_probabilities: List[Dict[str, Union[str, float]]],
    gt_dict: Dict[str, Set[str]],
    thresholds: Optional[List[float]] = None
) -> Dict[str, Union[float, Dict[float, Dict]]]:
    """Evaluate candidate match probabilities across multiple probability thresholds to find the optimal F0.5 threshold.
    
    Args:
        candidate_probabilities: List of dicts, each containing:
            - 'source1_entity_id': S1 ID
            - 'candidate_entity_id': S2 or S3 ID
            - 'probability': float score in [0.0, 1.0]
        gt_dict: Dict mapping source1_entity_id -> set of ground truth matched IDs.
        thresholds: List of probability thresholds to evaluate (default: 0.1 to 0.9 in steps of 0.05).

    Returns:
        Dict containing:
            - best_threshold: Threshold yielding maximum macro_f05
            - best_macro_f05: Highest macro F0.5 score achieved
            - threshold_results: Mapping of threshold -> evaluation summary dict
    """
    if thresholds is None:
        thresholds = [round(x * 0.05, 2) for x in range(1, 20)]

    # Group candidate probabilities by source1_entity_id
    s1_candidates = defaultdict(list)
    for row in candidate_probabilities:
        s1_id = str(row["source1_entity_id"]).strip()
        cand_id = str(row["candidate_entity_id"]).strip()
        prob = float(row["probability"])
        s1_candidates[s1_id].append((cand_id, prob))

    results = {}
    best_thresh = thresholds[0]
    best_score = -1.0

    for thresh in thresholds:
        pred_dict = {}
        for s1_id in gt_dict.keys():
            preds = set()
            for cand_id, prob in s1_candidates.get(s1_id, []):
                if prob >= thresh:
                    preds.add(cand_id)
            pred_dict[s1_id] = preds

        eval_res = evaluate_predictions(gt_dict, pred_dict)
        results[thresh] = eval_res
        
        score = float(eval_res["macro_f05"])
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return {
        "best_threshold": best_thresh,
        "best_macro_f05": best_score,
        "threshold_results": results
    }


def get_match_count_category(num_matches: int) -> str:
    """Classify match count into stratification buckets: '0', '1', '2', or '3+'."""
    if num_matches == 0:
        return "0"
    elif num_matches == 1:
        return "1"
    elif num_matches == 2:
        return "2"
    else:
        return "3+"


def create_entity_level_split(
    gt_input: Union[Dict[str, Set[str]], str],
    val_size: float = 0.2,
    random_state: int = 42,
    stratify_by_match_count: bool = True
) -> Tuple[List[str], List[str]]:
    """Split Source-1 entities into training and validation groups at the entity level.
    
    Guarantees:
    1. The split occurs strictly at the Source-1 entity level (no candidate-pair splitting).
    2. Zero overlap between train S1 IDs and validation S1 IDs.
    3. Reproducible using a fixed random seed.
    4. Stratified by match-count categories (0, 1, 2, 3+).

    Args:
        gt_input: Path to train_ground_truth.tsv or pre-loaded ground truth dict.
        val_size: Fraction of entities to assign to validation set (0.0 < val_size < 1.0).
        random_state: Fixed random seed for reproducibility.
        stratify_by_match_count: Whether to stratify split by match count categories.

    Returns:
        Tuple of (train_s1_ids, val_s1_ids) as sorted lists of strings.
    """
    if isinstance(gt_input, str):
        gt_dict = load_ground_truth(gt_input)
    else:
        gt_dict = gt_input

    if not gt_dict:
        raise ValueError("Ground truth dictionary is empty.")

    s1_ids = sorted(gt_dict.keys())
    
    if stratify_by_match_count:
        # Group S1 IDs by match category
        buckets = defaultdict(list)
        for s1_id in s1_ids:
            num_m = len(gt_dict[s1_id])
            cat = get_match_count_category(num_m)
            buckets[cat].append(s1_id)

        train_ids = []
        val_ids = []

        # Stratified split per bucket
        for cat in sorted(buckets.keys()):
            cat_ids = buckets[cat]
            # Shuffle deterministically per bucket
            rng = random.Random(random_state + hash(cat) % 10000)
            shuffled = list(cat_ids)
            rng.shuffle(shuffled)

            val_count = max(1, int(round(len(shuffled) * val_size))) if val_size > 0 else 0
            val_ids.extend(shuffled[:val_count])
            train_ids.extend(shuffled[val_count:])
    else:
        rng = random.Random(random_state)
        shuffled = list(s1_ids)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_size))
        val_ids = shuffled[:val_count]
        train_ids = shuffled[val_count:]

    train_ids.sort()
    val_ids.sort()

    # Integrity verification
    train_set = set(train_ids)
    val_set = set(val_ids)
    overlap = train_set.intersection(val_set)
    if overlap:
        raise RuntimeError(f"Data leakage detected! {len(overlap)} S1 entities exist in both train and validation splits.")

    return train_ids, val_ids
