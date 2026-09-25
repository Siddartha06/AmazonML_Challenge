import os
import sys
import json
import csv
import re
from collections import Counter

def run_fast_eda():
    base_dir = r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource"
    dataset_dir = os.path.join(base_dir, "dataset")

    file_paths = {
        "train_source1": os.path.join(dataset_dir, "train", "train_source1.tsv"),
        "train_source2": os.path.join(dataset_dir, "train", "train_source2.tsv"),
        "train_source3": os.path.join(dataset_dir, "train", "train_source3.tsv"),
        "train_ground_truth": os.path.join(dataset_dir, "train", "train_ground_truth.tsv"),
        "test_source1": os.path.join(dataset_dir, "test", "test_source1.tsv"),
        "test_source2": os.path.join(dataset_dir, "test", "test_source2.tsv"),
        "test_source3": os.path.join(dataset_dir, "test", "test_source3.tsv"),
    }

    results = {}
    row_counts = {}
    columns_map = {}
    missing_stats = {}
    unique_entities = {}
    country_dist = {}
    business_name_stats = {}
    address_stats = {}
    address_patterns = {}
    duplicate_checks = {}
    entity_id_sets = {}

    in_pin_regex = re.compile(r'\b\d{6}\b')
    us_zip_regex = re.compile(r'\b\d{5}(-\d{4})?\b')
    digit_regex = re.compile(r'\d')

    for name, path in file_paths.items():
        print(f"Reading {name}...", flush=True)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            headers = next(reader)
            columns_map[name] = headers
            
            rows = list(reader)
            total_rows = len(rows)
            row_counts[name] = total_rows

            id_idx = headers.index("entity_id") if "entity_id" in headers else headers.index("source1_entity_id")
            name_idx = headers.index("business_name") if "business_name" in headers else -1
            addr_idx = headers.index("business_address") if "business_address" in headers else -1
            ctry_idx = headers.index("country") if "country" in headers else -1
            matched_idx = headers.index("matched_entity_ids") if "matched_entity_ids" in headers else -1

            # Missing stats
            missing_counts = {h: 0 for h in headers}
            ids = set()
            dup_ids = 0

            # Text stats
            name_char_lens = []
            name_word_cnts = []
            addr_char_lens = []
            addr_word_cnts = []
            has_digit_cnt = 0
            has_in_pin_cnt = 0
            has_us_zip_cnt = 0
            ctry_counter = Counter()

            row_tuples = []

            for row in rows:
                for idx, col_val in enumerate(row):
                    h = headers[idx]
                    val_str = col_val.strip()
                    if val_str == "":
                        missing_counts[h] += 1

                # entity ID
                eid = row[id_idx].strip()
                if eid in ids:
                    dup_ids += 1
                else:
                    ids.add(eid)

                if ctry_idx != -1:
                    ctry_counter[row[ctry_idx].strip()] += 1

                if name_idx != -1:
                    b_name = row[name_idx]
                    name_char_lens.append(len(b_name))
                    name_word_cnts.append(len(b_name.split()))

                if addr_idx != -1:
                    b_addr = row[addr_idx]
                    addr_char_lens.append(len(b_addr))
                    addr_word_cnts.append(len(b_addr.split()))
                    if digit_regex.search(b_addr):
                        has_digit_cnt += 1
                    if in_pin_regex.search(b_addr):
                        has_in_pin_cnt += 1
                    if us_zip_regex.search(b_addr):
                        has_us_zip_cnt += 1

                if name_idx != -1 and addr_idx != -1 and ctry_idx != -1:
                    row_tuples.append((row[name_idx].strip(), row[addr_idx].strip(), row[ctry_idx].strip()))

            entity_id_sets[name] = ids

            missing_stats[name] = {
                h: {
                    "null_count": missing_counts[h],
                    "null_pct": round(missing_counts[h] / total_rows * 100, 4) if total_rows > 0 else 0
                } for h in headers
            }

            unique_entities[name] = {
                "unique_ids": len(ids),
                "duplicate_ids": dup_ids
            }

            if ctry_idx != -1:
                country_dist[name] = {
                    k: {"count": v, "pct": round(v / total_rows * 100, 2)}
                    for k, v in ctry_counter.items()
                }

            if name_char_lens:
                name_char_lens.sort()
                name_word_cnts.sort()
                business_name_stats[name] = {
                    "char_length": {
                        "min": name_char_lens[0],
                        "max": name_char_lens[-1],
                        "mean": round(sum(name_char_lens) / len(name_char_lens), 2),
                        "median": name_char_lens[len(name_char_lens)//2],
                        "p5": name_char_lens[int(0.05 * len(name_char_lens))],
                        "p95": name_char_lens[int(0.95 * len(name_char_lens))],
                    },
                    "word_count": {
                        "min": name_word_cnts[0],
                        "max": name_word_cnts[-1],
                        "mean": round(sum(name_word_cnts) / len(name_word_cnts), 2),
                        "median": name_word_cnts[len(name_word_cnts)//2],
                    }
                }

            if addr_char_lens:
                addr_char_lens.sort()
                addr_word_cnts.sort()
                address_stats[name] = {
                    "char_length": {
                        "min": addr_char_lens[0],
                        "max": addr_char_lens[-1],
                        "mean": round(sum(addr_char_lens) / len(addr_char_lens), 2),
                        "median": addr_char_lens[len(addr_char_lens)//2],
                        "p5": addr_char_lens[int(0.05 * len(addr_char_lens))],
                        "p95": addr_char_lens[int(0.95 * len(addr_char_lens))],
                    },
                    "word_count": {
                        "min": addr_word_cnts[0],
                        "max": addr_word_cnts[-1],
                        "mean": round(sum(addr_word_cnts) / len(addr_word_cnts), 2),
                        "median": addr_word_cnts[len(addr_word_cnts)//2],
                    }
                }
                address_patterns[name] = {
                    "has_digits": has_digit_cnt,
                    "has_6digit_in_pin_pattern": has_in_pin_cnt,
                    "has_5digit_us_zip_pattern": has_us_zip_cnt,
                }

            if row_tuples:
                dup_content_cnt = total_rows - len(set(row_tuples))
                duplicate_checks[name] = {
                    "exact_full_duplicate_rows": 0, # checked separately if needed
                    "duplicate_records_excluding_id": dup_content_cnt
                }

    results["row_counts"] = row_counts
    results["columns"] = columns_map
    results["missing_stats"] = missing_stats
    results["unique_entities"] = unique_entities
    results["country_dist"] = country_dist
    results["business_name_stats"] = business_name_stats
    results["address_stats"] = address_stats
    results["address_patterns"] = address_patterns
    results["duplicate_checks"] = duplicate_checks

    results["id_overlap"] = {
        "s1_overlap": len(entity_id_sets["train_source1"].intersection(entity_id_sets["test_source1"])),
        "s2_overlap": len(entity_id_sets["train_source2"].intersection(entity_id_sets["test_source2"])),
        "s3_overlap": len(entity_id_sets["train_source3"].intersection(entity_id_sets["test_source3"])),
    }

    # Ground Truth detailed analysis
    print("Parsing Ground Truth...", flush=True)
    gt_path = file_paths["train_ground_truth"]
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader)
        gt_rows = list(reader)

    gt_s1_cnt = len(gt_rows)
    matches_list = []
    for r in gt_rows:
        val = r[1].strip()
        if val:
            matches_list.append([x.strip() for x in val.split(",") if x.strip()])
        else:
            matches_list.append([])

    match_counts = [len(m) for m in matches_list]
    match_counts_sorted = sorted(match_counts)

    results["gt_match_stats"] = {
        "total_s1_in_gt": gt_s1_cnt,
        "min_matches": match_counts_sorted[0],
        "max_matches": match_counts_sorted[-1],
        "mean_matches": round(sum(match_counts_sorted) / gt_s1_cnt, 4),
        "median_matches": match_counts_sorted[gt_s1_cnt // 2],
        "quantile_25": match_counts_sorted[int(0.25 * gt_s1_cnt)],
        "quantile_75": match_counts_sorted[int(0.75 * gt_s1_cnt)],
        "quantile_90": match_counts_sorted[int(0.90 * gt_s1_cnt)],
        "quantile_99": match_counts_sorted[int(0.99 * gt_s1_cnt)],
    }

    freq = Counter(match_counts)
    results["gt_match_count_frequency"] = {k: v for k, v in sorted(freq.items())}

    zero_m = freq.get(0, 0)
    one_m = freq.get(1, 0)
    two_m = freq.get(2, 0)
    multi_m = sum(v for k, v in freq.items() if k >= 3)

    results["gt_match_distribution_pct"] = {
        "zero_matches_singletons": {"count": zero_m, "pct": round(zero_m / gt_s1_cnt * 100, 2)},
        "one_match": {"count": one_m, "pct": round(one_m / gt_s1_cnt * 100, 2)},
        "two_matches": {"count": two_m, "pct": round(two_m / gt_s1_cnt * 100, 2)},
        "three_or_more_matches": {"count": multi_m, "pct": round(multi_m / gt_s1_cnt * 100, 2)},
    }

    all_matched_ids = [item for sublist in matches_list for item in sublist]
    s2_matched_ids = [m for m in all_matched_ids if m.startswith("S2-")]
    s3_matched_ids = [m for m in all_matched_ids if m.startswith("S3-")]

    unique_s2_matched = set(s2_matched_ids)
    unique_s3_matched = set(s3_matched_ids)

    total_s2_in_tr = row_counts["train_source2"]
    total_s3_in_tr = row_counts["train_source3"]

    results["s2_s3_ground_truth_coverage"] = {
        "total_s2_matches_in_gt": len(s2_matched_ids),
        "unique_s2_matched_entities": len(unique_s2_matched),
        "total_s2_records_in_train": total_s2_in_tr,
        "s2_matched_pct": round(len(unique_s2_matched) / total_s2_in_tr * 100, 2),
        "s2_distractors_unmatched": total_s2_in_tr - len(unique_s2_matched),

        "total_s3_matches_in_gt": len(s3_matched_ids),
        "unique_s3_matched_entities": len(unique_s3_matched),
        "total_s3_records_in_train": total_s3_in_tr,
        "s3_matched_pct": round(len(unique_s3_matched) / total_s3_in_tr * 100, 2),
        "s3_distractors_unmatched": total_s3_in_tr - len(unique_s3_matched),
    }

    s2_counter = Counter(s2_matched_ids)
    s3_counter = Counter(s3_matched_ids)
    s2_multi_matched = {k: v for k, v in s2_counter.items() if v > 1}
    s3_multi_matched = {k: v for k, v in s3_counter.items() if v > 1}

    results["multi_matched_candidates"] = {
        "s2_entities_matched_to_multiple_s1": len(s2_multi_matched),
        "s3_entities_matched_to_multiple_s1": len(s3_multi_matched),
    }

    out_json = os.path.join(base_dir, "scratch", "eda_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"--- Fast EDA Completed Successfully! Saved to {out_json} ---", flush=True)

if __name__ == "__main__":
    run_fast_eda()
