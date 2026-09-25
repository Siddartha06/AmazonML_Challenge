import os
import sys
import csv
import json
import time
from collections import defaultdict

sys.path.insert(0, r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource")

from src.normalization import normalize_record
from src.validation import load_ground_truth, create_entity_level_split
from src.blocking import (
    evaluate_blocking_recall,
    calculate_reduction_ratio,
    analyze_candidate_distribution,
    MultiBlocker
)

def run_ultra_fast_benchmark():
    dataset_dir = r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource\dataset"
    gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(dataset_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(dataset_dir, "train", "train_source3.tsv")

    print("1. Loading Ground Truth and Creating Validation Split...", flush=True)
    t0 = time.time()
    gt_dict = load_ground_truth(gt_path)
    train_ids, val_ids = create_entity_level_split(gt_dict, val_size=0.2, random_state=42)
    val_gt_dict = {sid: gt_dict[sid] for sid in val_ids}
    print(f"Validation S1 count: {len(val_ids)} (created in {time.time()-t0:.2f}s)", flush=True)

    # Sample 10,000 validation S1 entities for rapid blocking evaluation
    sample_val_ids = val_ids[:10000]
    sample_val_set = set(sample_val_ids)
    sample_val_gt = {sid: val_gt_dict[sid] for sid in sample_val_ids}
    
    # Target ground truth matched candidate IDs
    gt_target_cands = set(cid for sid in sample_val_ids for cid in sample_val_gt[sid])
    print(f"Sample Validation S1 count: {len(sample_val_ids)} with {len(gt_target_cands)} GT matched candidate IDs.", flush=True)

    print("2. Reading and Normalizing Source 1 Validation Sample...", flush=True)
    s1_records = []
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[0] in sample_val_set:
                s1_records.append(normalize_record(row[0], row[1], row[2], row[3]))

    print(f"Loaded {len(s1_records)} normalized S1 validation records.", flush=True)

    # List of blocking strategies to evaluate individually
    strategies = [
        ("BLOCK 1: Exact Normalized Name", "BLOCK1_exact_name"),
        ("BLOCK 2: Exact Compact Name", "BLOCK2_exact_compact"),
        ("BLOCK 3: Name Prefix", "BLOCK3_name_prefix"),
        ("BLOCK 4: Important Name Tokens", "BLOCK4_important_tokens"),
        ("BLOCK 5: Sorted Name Tokens", "BLOCK5_sorted_tokens"),
        ("BLOCK 6: Address Postal Code", "BLOCK6_address_postal"),
        ("BLOCK 7: Address Number + Token", "BLOCK7_address_number_token"),
        ("BLOCK 8: Name Token + Address Number", "BLOCK8_name_address_token"),
        ("BLOCK 9: Character Trigrams", "BLOCK9_char_trigrams"),
    ]

    blocker = MultiBlocker(max_key_frequency=500)
    
    # Store candidates per block: block_label -> {s1_id -> set of candidate_ids}
    block_candidate_dicts = {label: {rec.entity_id: set() for rec in s1_records} for label, _ in strategies}
    union_label = "UNION OF ALL BLOCKS (1-9)"
    block_candidate_dicts[union_label] = {rec.entity_id: set() for rec in s1_records}

    # Pre-index S1 blocking keys: btype -> {key -> list of s1_ids}
    s1_key_map = {}
    active_s1_keys = set()
    for label, btype in strategies:
        kmap = defaultdict(list)
        for rec in s1_records:
            keys = blocker.get_block_keys_for_record(rec, btype)
            for k in keys:
                kmap[k].append(rec.entity_id)
                active_s1_keys.add(k)
        s1_key_map[btype] = kmap

    print(f"Pre-indexed {len(active_s1_keys)} active target blocking keys.", flush=True)

    print("3. Streaming Candidate Pools (S2 & S3)...", flush=True)
    t0 = time.time()
    total_s2_cnt = 0
    total_s3_cnt = 0
    processed_cnt = 0

    for path, is_s2 in [(s2_path, True), (s3_path, False)]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                processed_cnt += 1
                if is_s2:
                    total_s2_cnt += 1
                else:
                    total_s3_cnt += 1

                cid, name, addr, ctry = row[0], row[1], row[2], row[3]
                
                # Fast check: normalize record
                rec = normalize_record(cid, name, addr, ctry)

                for label, btype in strategies:
                    keys = blocker.get_block_keys_for_record(rec, btype)
                    kmap = s1_key_map[btype]
                    for k in keys:
                        if k in kmap:
                            matched_s1_list = kmap[k]
                            if 0 < len(matched_s1_list) <= 500:
                                for s1_id in matched_s1_list:
                                    block_candidate_dicts[label][s1_id].add(cid)
                                    block_candidate_dicts[union_label][s1_id].add(cid)

                if processed_cnt % 2000000 == 0:
                    print(f"  Processed {processed_cnt:,} candidate records ({time.time()-t0:.1f}s)...", flush=True)

    print(f"Completed streaming {processed_cnt:,} candidate records in {time.time()-t0:.2f}s.", flush=True)

    results_report = []

    print("\n=== BENCHMARK RESULTS PER BLOCKING STRATEGY ===", flush=True)

    all_strategies_eval = strategies + [(union_label, "UNION")]

    for label, _ in all_strategies_eval:
        cands = block_candidate_dicts[label]
        rec_eval = evaluate_blocking_recall(cands, sample_val_gt)
        dist_eval = analyze_candidate_distribution(cands)
        rr = calculate_reduction_ratio(dist_eval["total_candidate_pairs"], len(s1_records), total_s2_cnt, total_s3_cnt)

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
        }
        results_report.append(res_dict)

        print(f"\n{label}:")
        print(f"  Recall: {rec_eval['blocking_recall']*100:.2f}% ({rec_eval['recovered_gt_matches']}/{rec_eval['total_gt_matches']})")
        print(f"  Total Candidate Pairs: {dist_eval['total_candidate_pairs']:,}")
        print(f"  Mean Candidates / S1: {dist_eval['mean_candidates']}")
        print(f"  Median: {dist_eval['median_candidates']}, P95: {dist_eval['p95_candidates']}, Max: {dist_eval['max_candidates']}")
        print(f"  Reduction Ratio: {rr:.8f}")

    out_json = r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource\scratch\blocking_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(results_report, f, indent=2)

    print(f"\nSaved benchmark results to {out_json}", flush=True)

if __name__ == "__main__":
    run_ultra_fast_benchmark()
