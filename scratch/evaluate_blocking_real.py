import os
import sys
import csv
import json
import time

sys.path.insert(0, r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource")

from src.normalization import normalize_record
from src.validation import load_ground_truth, create_entity_level_split
from src.blocking import (
    generate_candidates,
    evaluate_blocking_recall,
    calculate_reduction_ratio,
    analyze_candidate_distribution,
    MultiBlocker
)

def run_blocking_benchmark():
    dataset_dir = r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource\dataset"
    gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train", "train_source3.tsv")

    print("1. Loading Ground Truth and Creating Validation Split...", flush=True)
    t0 = time.time()
    gt_dict = load_ground_truth(gt_path)
    train_ids, val_ids = create_entity_level_split(gt_dict, val_size=0.2, random_state=42)
    val_id_set = set(val_ids)
    val_gt_dict = {sid: gt_dict[sid] for sid in val_ids}
    print(f"Validation S1 count: {len(val_ids)} (created in {time.time()-t0:.2f}s)", flush=True)

    # For benchmark efficiency, sample 20,000 validation S1 entities and associated S2/S3 candidates
    sample_val_ids = val_ids[:20000]
    sample_val_set = set(sample_val_ids)
    sample_val_gt = {sid: val_gt_dict[sid] for sid in sample_val_ids}
    
    # Collect all ground truth matched candidate IDs for sample
    sample_matched_cands = set(cid for sid in sample_val_ids for cid in sample_val_gt[sid])
    print(f"Sample Validation S1 count: {len(sample_val_ids)} with {len(sample_matched_cands)} GT matched candidate IDs.", flush=True)

    print("2. Reading and Normalizing Source 1 Validation Sample...", flush=True)
    s1_records = []
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader) # header
        for row in reader:
            if row[0] in sample_val_set:
                s1_records.append(normalize_record(row[0], row[1], row[2], row[3]))

    print(f"Loaded {len(s1_records)} normalized S1 validation records.", flush=True)

    print("3. Reading and Normalizing Source 2 and Source 3 Candidate Pool...", flush=True)
    # Read S2 and S3 candidates (sampling relevant candidates + distractors)
    s2_records = []
    s3_records = []

    t0 = time.time()
    with open(s2_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for idx, row in enumerate(reader):
            # Include all GT matched S2 candidates + sample distractors
            if row[0] in sample_matched_cands or (idx % 10 == 0):
                s2_records.append(normalize_record(row[0], row[1], row[2], row[3]))

    with open(s3_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for idx, row in enumerate(reader):
            if row[0] in sample_matched_cands or (idx % 10 == 0):
                s3_records.append(normalize_record(row[0], row[1], row[2], row[3]))

    print(f"Loaded {len(s2_records)} S2 candidates and {len(s3_records)} S3 candidates in {time.time()-t0:.2f}s.", flush=True)

    all_candidates = s2_records + s3_records

    # List of blocking strategies to evaluate individually
    strategies = [
        ("BLOCK 1: Exact Normalized Name", ["BLOCK1_exact_name"]),
        ("BLOCK 2: Exact Compact Name", ["BLOCK2_exact_compact"]),
        ("BLOCK 3: Name Prefix", ["BLOCK3_name_prefix"]),
        ("BLOCK 4: Important Name Tokens", ["BLOCK4_important_tokens"]),
        ("BLOCK 5: Sorted Name Tokens", ["BLOCK5_sorted_tokens"]),
        ("BLOCK 6: Address Postal Code", ["BLOCK6_address_postal"]),
        ("BLOCK 7: Address Number + Token", ["BLOCK7_address_number_token"]),
        ("BLOCK 8: Name Token + Address Number", ["BLOCK8_name_address_token"]),
        ("BLOCK 9: Character Trigrams", ["BLOCK9_char_trigrams"]),
    ]

    all_block_types = [s[1][0] for s in strategies]
    strategies.append(("UNION OF ALL BLOCKS (1-9)", all_block_types))

    blocker = MultiBlocker(max_key_frequency=500)

    results_report = []

    print("\n=== EVALUATING BLOCKING STRATEGIES ===", flush=True)

    for label, btypes in strategies:
        t0 = time.time()
        index = blocker.build_index(all_candidates, btypes)
        cands = blocker.generate_candidates_for_s1(s1_records, index, btypes)
        t_el = time.time() - t0

        rec_eval = evaluate_blocking_recall(cands, sample_val_gt)
        dist_eval = analyze_candidate_distribution(cands)
        rr = calculate_reduction_ratio(dist_eval["total_candidate_pairs"], len(s1_records), len(s2_records), len(s3_records))

        res_dict = {
            "strategy": label,
            "true_matches_in_gt": rec_eval["total_gt_matches"],
            "recovered_true_matches": rec_eval["recovered_gt_matches"],
            "blocking_recall": rec_eval["blocking_recall"],
            "total_candidate_pairs": dist_eval["total_candidate_pairs"],
            "mean_candidates_per_s1": dist_eval["mean_candidates"],
            "median_candidates": dist_eval["median_candidates"],
            "p95_candidates": dist_eval["p95_candidates"],
            "max_candidates": dist_eval["max_candidates"],
            "reduction_ratio": rr,
            "execution_time_sec": round(t_el, 2)
        }
        results_report.append(res_dict)

        print(f"\n{label}:")
        print(f"  Recall: {rec_eval['blocking_recall']*100:.2f}% ({rec_eval['recovered_gt_matches']}/{rec_eval['total_gt_matches']})")
        print(f"  Total Candidate Pairs: {dist_eval['total_candidate_pairs']:,}")
        print(f"  Mean Candidates / S1: {dist_eval['mean_candidates']}")
        print(f"  Median: {dist_eval['median_candidates']}, P95: {dist_eval['p95_candidates']}, Max: {dist_eval['max_candidates']}")
        print(f"  Reduction Ratio: {rr:.8f}")

    # Save benchmark to JSON for report generation
    out_json = r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource\scratch\blocking_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(results_report, f, indent=2)

    print(f"\nSaved benchmark results to {out_json}", flush=True)

if __name__ == "__main__":
    run_blocking_benchmark()
