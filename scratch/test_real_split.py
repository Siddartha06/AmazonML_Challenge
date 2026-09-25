import os
import sys
import time

sys.path.insert(0, r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource")

from src.validation import load_ground_truth, create_entity_level_split, evaluate_predictions

def test_real_gt_split():
    gt_path = r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource\dataset\train\train_ground_truth.tsv"
    
    print("Loading Ground Truth...", flush=True)
    t0 = time.time()
    gt_dict = load_ground_truth(gt_path)
    print(f"Loaded {len(gt_dict)} S1 entities in {time.time() - t0:.2f} seconds.", flush=True)

    print("Creating Stratified Entity-Level Split (80% Train, 20% Val)...", flush=True)
    t0 = time.time()
    train_ids, val_ids = create_entity_level_split(gt_dict, val_size=0.2, random_state=42, stratify_by_match_count=True)
    print(f"Split created in {time.time() - t0:.2f} seconds.", flush=True)

    print(f"Train S1 Count: {len(train_ids)} ({len(train_ids)/len(gt_dict)*100:.2f}%)")
    print(f"Val S1 Count:   {len(val_ids)} ({len(val_ids)/len(gt_dict)*100:.2f}%)")

    # Check zero overlap
    train_set = set(train_ids)
    val_set = set(val_ids)
    overlap = train_set.intersection(val_set)
    print(f"Train & Val ID Overlap: {len(overlap)} (Must be 0)")

    # Stratification stats
    from collections import Counter
    def get_cat(n):
        return "0" if n == 0 else ("1" if n == 1 else ("2" if n == 2 else "3+"))

    train_cats = Counter(get_cat(len(gt_dict[sid])) for sid in train_ids)
    val_cats = Counter(get_cat(len(gt_dict[sid])) for sid in val_ids)

    print("\n--- Train Category Breakdown ---")
    for cat in ["0", "1", "2", "3+"]:
        cnt = train_cats[cat]
        pct = cnt / len(train_ids) * 100
        print(f"  Category {cat:4s}: {cnt:10d} ({pct:.2f}%)")

    print("\n--- Val Category Breakdown ---")
    for cat in ["0", "1", "2", "3+"]:
        cnt = val_cats[cat]
        pct = cnt / len(val_ids) * 100
        print(f"  Category {cat:4s}: {cnt:10d} ({pct:.2f}%)")

    # Test baseline predictions (all empty predictions / zero matches)
    val_gt = {sid: gt_dict[sid] for sid in val_ids}
    empty_preds = {sid: set() for sid in val_ids}
    
    print("\n--- Baseline All-Empty Predictions Evaluation on Validation Set ---", flush=True)
    val_eval = evaluate_predictions(val_gt, empty_preds)
    for k, v in val_eval.items():
        print(f"  {k:20s}: {v}")

if __name__ == "__main__":
    test_real_gt_split()
