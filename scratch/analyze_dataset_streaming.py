import os
import sys
import json
import gc
import re
from collections import Counter
import pandas as pd
import numpy as np

def analyze():
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

    for name, path in file_paths.items():
        print(f"Processing {name}...")
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        row_counts[name] = len(df)
        columns_map[name] = list(df.columns)

        # Missing stats
        missing_stats[name] = {}
        for col in df.columns:
            if name == "train_ground_truth" and col == "matched_entity_ids":
                null_cnt = int((df[col] == "").sum())
            else:
                null_cnt = int((df[col].isna() | (df[col] == "") | df[col].str.strip().eq("")).sum())
            missing_stats[name][col] = {
                "null_count": null_cnt,
                "null_pct": round(float(null_cnt) / len(df) * 100, 4)
            }

        # Unique entities
        id_col = "entity_id" if "entity_id" in df.columns else "source1_entity_id"
        id_series = df[id_col]
        u_cnt = int(id_series.nunique())
        unique_entities[name] = {
            "unique_ids": u_cnt,
            "duplicate_ids": len(df) - u_cnt
        }
        entity_id_sets[name] = set(id_series)

        # Country distribution
        if "country" in df.columns:
            vc = df["country"].value_counts().to_dict()
            total = len(df)
            country_dist[name] = {str(k): {"count": int(v), "pct": round(v / total * 100, 2)} for k, v in vc.items()}

        # Business name stats
        if "business_name" in df.columns:
            names = df["business_name"]
            char_lens = names.str.len()
            # Fast word count
            word_cnts = names.str.split().str.len()

            business_name_stats[name] = {
                "char_length": {
                    "min": int(char_lens.min()),
                    "max": int(char_lens.max()),
                    "mean": round(float(char_lens.mean()), 2),
                    "median": float(char_lens.median()),
                    "p5": float(np.percentile(char_lens, 5)),
                    "p95": float(np.percentile(char_lens, 95)),
                },
                "word_count": {
                    "min": int(word_cnts.min()),
                    "max": int(word_cnts.max()),
                    "mean": round(float(word_cnts.mean()), 2),
                    "median": float(word_cnts.median()),
                }
            }

        # Address stats
        if "business_address" in df.columns:
            addrs = df["business_address"]
            char_lens = addrs.str.len()
            word_cnts = addrs.str.split().str.len()

            address_stats[name] = {
                "char_length": {
                    "min": int(char_lens.min()),
                    "max": int(char_lens.max()),
                    "mean": round(float(char_lens.mean()), 2),
                    "median": float(char_lens.median()),
                    "p5": float(np.percentile(char_lens, 5)),
                    "p95": float(np.percentile(char_lens, 95)),
                },
                "word_count": {
                    "min": int(word_cnts.min()),
                    "max": int(word_cnts.max()),
                    "mean": round(float(word_cnts.mean()), 2),
                    "median": float(word_cnts.median()),
                }
            }

            has_digit = int(addrs.str.contains(r"\d", regex=True).sum())
            has_in_pin = int(addrs.str.contains(r"\b\d{6}\b", regex=True).sum())
            has_us_zip = int(addrs.str.contains(r"\b\d{5}\b", regex=True).sum())
            address_patterns[name] = {
                "has_digits": has_digit,
                "has_6digit_in_pin_pattern": has_in_pin,
                "has_5digit_us_zip_pattern": has_us_zip,
            }

        # Duplicate checks
        if "entity_id" in df.columns:
            exact_dup_full = int(df.duplicated().sum())
            exact_dup_no_id = int(df.duplicated(subset=["business_name", "business_address", "country"]).sum())
            duplicate_checks[name] = {
                "exact_full_duplicate_rows": exact_dup_full,
                "duplicate_records_excluding_id": exact_dup_no_id
            }

        # Free memory
        del df
        gc.collect()

    results["row_counts"] = row_counts
    results["columns"] = columns_map
    results["missing_stats"] = missing_stats
    results["unique_entities"] = unique_entities
    results["country_dist"] = country_dist
    results["business_name_stats"] = business_name_stats
    results["address_stats"] = address_stats
    results["address_patterns"] = address_patterns
    results["duplicate_checks"] = duplicate_checks

    # ID Overlaps
    results["id_overlap"] = {
        "s1_overlap": len(entity_id_sets["train_source1"].intersection(entity_id_sets["test_source1"])),
        "s2_overlap": len(entity_id_sets["train_source2"].intersection(entity_id_sets["test_source2"])),
        "s3_overlap": len(entity_id_sets["train_source3"].intersection(entity_id_sets["test_source3"])),
    }

    # Ground Truth detailed analysis
    print("Processing Ground Truth details...")
    df_gt = pd.read_csv(file_paths["train_ground_truth"], sep="\t", dtype=str, keep_default_na=False)
    matches_list = [val.split(",") if val else [] for val in df_gt["matched_entity_ids"]]
    match_counts = [len(m) for m in matches_list]
    mc_series = pd.Series(match_counts)

    gt_s1_cnt = len(df_gt)
    results["gt_match_stats"] = {
        "total_s1_in_gt": gt_s1_cnt,
        "min_matches": int(mc_series.min()),
        "max_matches": int(mc_series.max()),
        "mean_matches": round(float(mc_series.mean()), 4),
        "median_matches": float(mc_series.median()),
        "quantile_25": float(mc_series.quantile(0.25)),
        "quantile_75": float(mc_series.quantile(0.75)),
        "quantile_90": float(mc_series.quantile(0.90)),
        "quantile_99": float(mc_series.quantile(0.99)),
    }

    freq = Counter(match_counts)
    results["gt_match_count_frequency"] = {int(k): int(v) for k, v in sorted(freq.items())}

    zero_m = freq.get(0, 0)
    one_m = freq.get(1, 0)
    two_m = freq.get(2, 0)
    multi_m = sum(v for k, v in freq.items() if k >= 3)

    results["gt_match_distribution_pct"] = {
        "zero_matches_singletons": {"count": int(zero_m), "pct": round(zero_m / gt_s1_cnt * 100, 2)},
        "one_match": {"count": int(one_m), "pct": round(one_m / gt_s1_cnt * 100, 2)},
        "two_matches": {"count": int(two_m), "pct": round(two_m / gt_s1_cnt * 100, 2)},
        "three_or_more_matches": {"count": int(multi_m), "pct": round(multi_m / gt_s1_cnt * 100, 2)},
    }

    all_matched_ids = [item.strip() for sublist in matches_list for item in sublist if item.strip()]
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

    print(f"--- Streaming Analysis Completed Successfully! Saved to {out_json} ---")

if __name__ == "__main__":
    analyze()
