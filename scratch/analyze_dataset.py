import os
import sys
import json
import re
from collections import Counter

def analyze():
    import pandas as pd
    import numpy as np

    base_dir = r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource"
    dataset_dir = os.path.join(base_dir, "dataset")

    train_s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
    train_s2_path = os.path.join(dataset_dir, "train", "train_source2.tsv")
    train_s3_path = os.path.join(dataset_dir, "train", "train_source3.tsv")
    train_gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")

    test_s1_path = os.path.join(dataset_dir, "test", "test_source1.tsv")
    test_s2_path = os.path.join(dataset_dir, "test", "test_source2.tsv")
    test_s3_path = os.path.join(dataset_dir, "test", "test_source3.tsv")

    print("--- Reading Datasets ---")
    df_tr_s1 = pd.read_csv(train_s1_path, sep="\t", dtype=str)
    df_tr_s2 = pd.read_csv(train_s2_path, sep="\t", dtype=str)
    df_tr_s3 = pd.read_csv(train_s3_path, sep="\t", dtype=str)
    df_tr_gt = pd.read_csv(train_gt_path, sep="\t", dtype=str, keep_default_na=False)

    df_te_s1 = pd.read_csv(test_s1_path, sep="\t", dtype=str)
    df_te_s2 = pd.read_csv(test_s2_path, sep="\t", dtype=str)
    df_te_s3 = pd.read_csv(test_s3_path, sep="\t", dtype=str)

    results = {}

    # 1. Row counts & Columns & Dtypes
    datasets = {
        "train_source1": df_tr_s1,
        "train_source2": df_tr_s2,
        "train_source3": df_tr_s3,
        "train_ground_truth": df_tr_gt,
        "test_source1": df_te_s1,
        "test_source2": df_te_s2,
        "test_source3": df_te_s3,
    }

    results["row_counts"] = {k: len(v) for k, v in datasets.items()}
    results["columns"] = {k: list(v.columns) for k, v in datasets.items()}

    # 2. Missing-value statistics
    missing_stats = {}
    for name, df in datasets.items():
        missing_stats[name] = {}
        for col in df.columns:
            if name == "train_ground_truth" and col == "matched_entity_ids":
                # For GT, empty string means zero matches
                null_cnt = (df[col].isna() | (df[col] == "")).sum()
            else:
                null_cnt = (df[col].isna() | (df[col] == "") | df[col].str.strip().eq("")).sum()
            missing_stats[name][col] = {
                "null_count": int(null_cnt),
                "null_pct": round(float(null_cnt) / len(df) * 100, 4)
            }
    results["missing_stats"] = missing_stats

    # 3. Unique entities & Duplicate entity_ids
    unique_entities = {}
    for name, df in datasets.items():
        if "entity_id" in df.columns:
            u_ids = df["entity_id"].nunique()
            dup_ids = len(df) - u_ids
            unique_entities[name] = {"unique_ids": int(u_ids), "duplicate_ids": int(dup_ids)}
        elif "source1_entity_id" in df.columns:
            u_ids = df["source1_entity_id"].nunique()
            dup_ids = len(df) - u_ids
            unique_entities[name] = {"unique_ids": int(u_ids), "duplicate_ids": int(dup_ids)}
    results["unique_entities"] = unique_entities

    # Train vs Test entity ID overlap
    tr_s1_ids = set(df_tr_s1["entity_id"])
    te_s1_ids = set(df_te_s1["entity_id"])
    tr_s2_ids = set(df_tr_s2["entity_id"])
    te_s2_ids = set(df_te_s2["entity_id"])
    tr_s3_ids = set(df_tr_s3["entity_id"])
    te_s3_ids = set(df_te_s3["entity_id"])

    results["id_overlap"] = {
        "s1_overlap": len(tr_s1_ids.intersection(te_s1_ids)),
        "s2_overlap": len(tr_s2_ids.intersection(te_s2_ids)),
        "s3_overlap": len(tr_s3_ids.intersection(te_s3_ids))
    }

    # 4. Country distribution
    country_dist = {}
    for name, df in datasets.items():
        if "country" in df.columns:
            vc = df["country"].value_counts(dropna=False).to_dict()
            total = len(df)
            country_dist[name] = {str(k): {"count": int(v), "pct": round(v / total * 100, 2)} for k, v in vc.items()}
    results["country_dist"] = country_dist

    # 5. Business Name Statistics
    def get_text_stats(df_dict):
        stats = {}
        for name, df in df_dict.items():
            if "business_name" not in df.columns:
                continue
            names = df["business_name"].fillna("").astype(str)
            char_lens = names.apply(len)
            word_cnts = names.apply(lambda x: len(x.split()))
            has_non_ascii = names.apply(lambda x: any(ord(c) > 127 for c in x)).sum()
            leading_trailing_ws = names.apply(lambda x: x != x.strip()).sum()
            
            stats[name] = {
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
                },
                "has_non_ascii": int(has_non_ascii),
                "has_whitespace_issues": int(leading_trailing_ws)
            }
        return stats

    results["business_name_stats"] = get_text_stats(datasets)

    # 6. Address Statistics
    def get_address_stats(df_dict):
        stats = {}
        for name, df in df_dict.items():
            if "business_address" not in df.columns:
                continue
            addrs = df["business_address"].fillna("").astype(str)
            char_lens = addrs.apply(len)
            word_cnts = addrs.apply(lambda x: len(x.split()))
            has_pincode_in = addrs.apply(lambda x: bool(re.search(r'\b\d{6}\b', x))).sum() # India PIN pattern
            has_zip_us = addrs.apply(lambda x: bool(re.search(r'\b\d{5}(-\d{4})?\b', x))).sum() # US ZIP pattern
            has_digits = addrs.apply(lambda x: bool(re.search(r'\d', x))).sum()
            has_newlines = addrs.apply(lambda x: '\n' in x or '\r' in x).sum()
            leading_trailing_ws = addrs.apply(lambda x: x != x.strip()).sum()

            stats[name] = {
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
                },
                "has_digits": int(has_digits),
                "has_in_pincode_pattern": int(has_pincode_in),
                "has_us_zip_pattern": int(has_zip_us),
                "has_newlines": int(has_newlines),
                "has_whitespace_issues": int(leading_trailing_ws)
            }
        return stats

    results["address_stats"] = get_address_stats(datasets)

    # 7. Ground Truth Match Analysis
    # train_ground_truth.tsv has source1_entity_id, matched_entity_ids
    # Parse matched_entity_ids
    gt_s1_cnt = len(df_tr_gt)
    
    # Process matches
    def parse_matches(val):
        if not val or val == "" or pd.isna(val):
            return []
        return [x.strip() for x in val.split(",") if x.strip()]

    df_tr_gt["matches_list"] = df_tr_gt["matched_entity_ids"].apply(parse_matches)
    df_tr_gt["match_count"] = df_tr_gt["matches_list"].apply(len)
    
    # Match count stats
    match_counts = df_tr_gt["match_count"]
    results["gt_match_stats"] = {
        "total_s1_in_gt": gt_s1_cnt,
        "min_matches": int(match_counts.min()),
        "max_matches": int(match_counts.max()),
        "mean_matches": round(float(match_counts.mean()), 4),
        "median_matches": float(match_counts.median()),
        "quantile_25": float(np.percentile(match_counts, 25)),
        "quantile_75": float(np.percentile(match_counts, 75)),
        "quantile_90": float(np.percentile(match_counts, 90)),
        "quantile_99": float(np.percentile(match_counts, 99)),
    }

    # Match count breakdown
    mc_counts = match_counts.value_counts().sort_index().to_dict()
    results["gt_match_count_frequency"] = {int(k): int(v) for k, v in mc_counts.items()}

    # Categorical breakdown: 0 (singleton), 1, 2, 3+
    zero_m = (match_counts == 0).sum()
    one_m = (match_counts == 1).sum()
    two_m = (match_counts == 2).sum()
    multi_m = (match_counts >= 3).sum()

    results["gt_match_distribution_pct"] = {
        "zero_matches_singletons": {"count": int(zero_m), "pct": round(zero_m / gt_s1_cnt * 100, 2)},
        "one_match": {"count": int(one_m), "pct": round(one_m / gt_s1_cnt * 100, 2)},
        "two_matches": {"count": int(two_m), "pct": round(two_m / gt_s1_cnt * 100, 2)},
        "three_or_more_matches": {"count": int(multi_m), "pct": round(multi_m / gt_s1_cnt * 100, 2)},
    }

    # Matches per source analysis (S2 vs S3)
    def count_s2_s3(matches):
        s2_c = sum(1 for m in matches if m.startswith("S2-"))
        s3_c = sum(1 for m in matches if m.startswith("S3-"))
        return s2_c, s3_c

    s2_s3_counts = df_tr_gt["matches_list"].apply(count_s2_s3)
    df_tr_gt["s2_match_count"] = [x[0] for x in s2_s3_counts]
    df_tr_gt["s3_match_count"] = [x[1] for x in s2_s3_counts]

    results["s2_s3_breakdown"] = {
        "s2_only_matches": int(((df_tr_gt["s2_match_count"] > 0) & (df_tr_gt["s3_match_count"] == 0)).sum()),
        "s3_only_matches": int(((df_tr_gt["s2_match_count"] == 0) & (df_tr_gt["s3_match_count"] > 0)).sum()),
        "both_s2_and_s3_matches": int(((df_tr_gt["s2_match_count"] > 0) & (df_tr_gt["s3_match_count"] > 0)).sum()),
        "neither_matches": int(zero_m),
    }

    # 8. S2 and S3 records in Ground Truth
    all_matched_ids = [m for sublist in df_tr_gt["matches_list"] for m in sublist]
    s2_matched_ids = [m for m in all_matched_ids if m.startswith("S2-")]
    s3_matched_ids = [m for m in all_matched_ids if m.startswith("S3-")]

    unique_s2_matched = set(s2_matched_ids)
    unique_s3_matched = set(s3_matched_ids)

    total_s2_in_tr = len(df_tr_s2)
    total_s3_in_tr = len(df_tr_s3)

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

    # Check if any S2/S3 entity is matched to MULTIPLE S1 entities
    s2_counter = Counter(s2_matched_ids)
    s3_counter = Counter(s3_matched_ids)
    s2_multi_matched = {k: v for k, v in s2_counter.items() if v > 1}
    s3_multi_matched = {k: v for k, v in s3_counter.items() if v > 1}

    results["multi_matched_candidates"] = {
        "s2_entities_matched_to_multiple_s1": len(s2_multi_matched),
        "s3_entities_matched_to_multiple_s1": len(s3_multi_matched),
    }

    # 9. Duplicates & Suspicious Records
    duplicate_checks = {}
    for name, df in datasets.items():
        if "entity_id" in df.columns:
            cols_no_id = [c for c in df.columns if c != "entity_id"]
            exact_dup_full = df.duplicated().sum()
            exact_dup_no_id = df.duplicated(subset=cols_no_id).sum()
            duplicate_checks[name] = {
                "exact_full_duplicate_rows": int(exact_dup_full),
                "duplicate_records_excluding_id": int(exact_dup_no_id)
            }
    results["duplicate_checks"] = duplicate_checks

    # Check exact match business_name + business_address across sources (Train)
    # Are there identical business name + address pairs across S1, S2, S3?
    df_tr_s1_pair = df_tr_s1.set_index("entity_id")
    df_tr_s2_pair = df_tr_s2.set_index("entity_id")
    df_tr_s3_pair = df_tr_s3.set_index("entity_id")

    # Name normalization check for exact overlap
    tr_s1_norm = set(df_tr_s1["business_name"].str.lower().str.strip() + "||" + df_tr_s1["business_address"].str.lower().str.strip())
    tr_s2_norm = set(df_tr_s2["business_name"].str.lower().str.strip() + "||" + df_tr_s2["business_address"].str.lower().str.strip())
    tr_s3_norm = set(df_tr_s3["business_name"].str.lower().str.strip() + "||" + df_tr_s3["business_address"].str.lower().str.strip())

    results["exact_text_matches_across_sources_train"] = {
        "s1_s2_exact_text_overlap": len(tr_s1_norm.intersection(tr_s2_norm)),
        "s1_s3_exact_text_overlap": len(tr_s1_norm.intersection(tr_s3_norm)),
        "s2_s3_exact_text_overlap": len(tr_s2_norm.intersection(tr_s3_norm)),
        "all_three_exact_text_overlap": len(tr_s1_norm.intersection(tr_s2_norm).intersection(tr_s3_norm))
    }

    # Cross train-test exact business_name + address overlap
    te_s1_norm = set(df_te_s1["business_name"].str.lower().str.strip() + "||" + df_te_s1["business_address"].str.lower().str.strip())
    te_s2_norm = set(df_te_s2["business_name"].str.lower().str.strip() + "||" + df_te_s2["business_address"].str.lower().str.strip())
    te_s3_norm = set(df_te_s3["business_name"].str.lower().str.strip() + "||" + df_te_s3["business_address"].str.lower().str.strip())

    results["exact_text_overlap_train_vs_test"] = {
        "s1_train_vs_test_text_overlap": len(tr_s1_norm.intersection(te_s1_norm)),
        "s2_train_vs_test_text_overlap": len(tr_s2_norm.intersection(te_s2_norm)),
        "s3_train_vs_test_text_overlap": len(tr_s3_norm.intersection(te_s3_norm)),
    }

    # Save output to JSON
    out_json = os.path.join(base_dir, "scratch", "eda_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"--- Analysis Completed. Saved to {out_json} ---")

if __name__ == "__main__":
    analyze()
