import os
import sys
import json
import re
from collections import Counter
import polars as pl

def analyze():
    base_dir = r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource"
    dataset_dir = os.path.join(base_dir, "dataset")

    train_s1_path = os.path.join(dataset_dir, "train", "train_source1.tsv")
    train_s2_path = os.path.join(dataset_dir, "train", "train_source2.tsv")
    train_s3_path = os.path.join(dataset_dir, "train", "train_source3.tsv")
    train_gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")

    test_s1_path = os.path.join(dataset_dir, "test", "test_source1.tsv")
    test_s2_path = os.path.join(dataset_dir, "test", "test_source2.tsv")
    test_s3_path = os.path.join(dataset_dir, "test", "test_source3.tsv")

    print("--- Reading Datasets with Polars ---")
    df_tr_s1 = pl.read_csv(train_s1_path, separator="\t", schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8})
    df_tr_s2 = pl.read_csv(train_s2_path, separator="\t", schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8})
    df_tr_s3 = pl.read_csv(train_s3_path, separator="\t", schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8})
    df_tr_gt = pl.read_csv(train_gt_path, separator="\t", schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})

    df_te_s1 = pl.read_csv(test_s1_path, separator="\t", schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8})
    df_te_s2 = pl.read_csv(test_s2_path, separator="\t", schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8})
    df_te_s3 = pl.read_csv(test_s3_path, separator="\t", schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8})

    results = {}

    datasets = {
        "train_source1": df_tr_s1,
        "train_source2": df_tr_s2,
        "train_source3": df_tr_s3,
        "train_ground_truth": df_tr_gt,
        "test_source1": df_te_s1,
        "test_source2": df_te_s2,
        "test_source3": df_te_s3,
    }

    # 1. Row counts & Columns
    results["row_counts"] = {k: len(v) for k, v in datasets.items()}
    results["columns"] = {k: v.columns for k, v in datasets.items()}

    # 2. Missing-value statistics
    missing_stats = {}
    for name, df in datasets.items():
        missing_stats[name] = {}
        for col in df.columns:
            if name == "train_ground_truth" and col == "matched_entity_ids":
                null_cnt = df[col].is_null().sum() + (df[col] == "").sum()
            else:
                s = df[col]
                null_cnt = s.is_null().sum() + (s.str.strip_chars() == "").sum()
            missing_stats[name][col] = {
                "null_count": int(null_cnt),
                "null_pct": round(float(null_cnt) / len(df) * 100, 4)
            }
    results["missing_stats"] = missing_stats

    # 3. Unique entities & Duplicate entity_ids
    unique_entities = {}
    for name, df in datasets.items():
        col_id = "entity_id" if "entity_id" in df.columns else "source1_entity_id"
        u_ids = df[col_id].n_unique()
        dup_ids = len(df) - u_ids
        unique_entities[name] = {"unique_ids": int(u_ids), "duplicate_ids": int(dup_ids)}
    results["unique_entities"] = unique_entities

    # Train vs Test entity ID overlap
    tr_s1_ids = set(df_tr_s1["entity_id"].to_list())
    te_s1_ids = set(df_te_s1["entity_id"].to_list())
    tr_s2_ids = set(df_tr_s2["entity_id"].to_list())
    te_s2_ids = set(df_te_s2["entity_id"].to_list())
    tr_s3_ids = set(df_tr_s3["entity_id"].to_list())
    te_s3_ids = set(df_te_s3["entity_id"].to_list())

    results["id_overlap"] = {
        "s1_overlap": len(tr_s1_ids.intersection(te_s1_ids)),
        "s2_overlap": len(tr_s2_ids.intersection(te_s2_ids)),
        "s3_overlap": len(tr_s3_ids.intersection(te_s3_ids))
    }

    # 4. Country distribution
    country_dist = {}
    for name, df in datasets.items():
        if "country" in df.columns:
            vc = df["country"].value_counts().to_dict(as_series=False)
            total = len(df)
            counts = dict(zip(vc["country"], vc["count"]))
            country_dist[name] = {str(k): {"count": int(v), "pct": round(v / total * 100, 2)} for k, v in counts.items()}
    results["country_dist"] = country_dist

    # 5. Business Name Statistics
    def get_text_stats(df_dict, col_name):
        stats = {}
        for name, df in df_dict.items():
            if col_name not in df.columns:
                continue
            s = df[col_name].fill_null("")
            char_lens = s.str.len_bytes()
            word_cnts = s.str.split(" ").list.len()
            
            # quantiles
            char_array = char_lens.to_numpy()
            word_array = word_cnts.to_numpy()

            stats[name] = {
                "char_length": {
                    "min": int(char_array.min()),
                    "max": int(char_array.max()),
                    "mean": round(float(char_array.mean()), 2),
                    "median": float(pl.Series(char_array).median()),
                    "p5": float(pl.Series(char_array).quantile(0.05)),
                    "p95": float(pl.Series(char_array).quantile(0.95)),
                },
                "word_count": {
                    "min": int(word_array.min()),
                    "max": int(word_array.max()),
                    "mean": round(float(word_array.mean()), 2),
                    "median": float(pl.Series(word_array).median()),
                }
            }
        return stats

    results["business_name_stats"] = get_text_stats(datasets, "business_name")
    results["address_stats"] = get_text_stats(datasets, "business_address")

    # 6. Additional Address & Name Patterns
    # PIN code/ZIP code presence in addresses
    addr_patterns = {}
    for name, df in datasets.items():
        if "business_address" not in df.columns:
            continue
        s = df["business_address"].fill_null("")
        has_digit = s.str.contains(r"\d").sum()
        has_in_pin = s.str.contains(r"\b\d{6}\b").sum()
        has_us_zip = s.str.contains(r"\b\d{5}\b").sum()
        addr_patterns[name] = {
            "has_digits": int(has_digit),
            "has_6digit_in_pin_pattern": int(has_in_pin),
            "has_5digit_us_zip_pattern": int(has_us_zip),
        }
    results["address_patterns"] = addr_patterns

    # 7. Ground Truth Match Analysis
    gt_s1_cnt = len(df_tr_gt)
    # Parse matches
    gt_matched_str = df_tr_gt["matched_entity_ids"].fill_null("")
    matches_list = [m.split(",") if m else [] for m in gt_matched_str.to_list()]
    match_counts = [len(m) for m in matches_list]

    mc_series = pl.Series("match_count", match_counts)
    
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

    # Frequency
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

    # Breakdown by S2 vs S3
    all_matched_ids = [item.strip() for sublist in matches_list for item in sublist if item.strip()]
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

    s2_counter = Counter(s2_matched_ids)
    s3_counter = Counter(s3_matched_ids)
    s2_multi_matched = {k: v for k, v in s2_counter.items() if v > 1}
    s3_multi_matched = {k: v for k, v in s3_counter.items() if v > 1}

    results["multi_matched_candidates"] = {
        "s2_entities_matched_to_multiple_s1": len(s2_multi_matched),
        "s3_entities_matched_to_multiple_s1": len(s3_multi_matched),
    }

    # 8. Duplicate Checks
    duplicate_checks = {}
    for name, df in datasets.items():
        if "entity_id" in df.columns:
            exact_dup_full = df.is_duplicated().sum()
            exact_dup_no_id = df.select(["business_name", "business_address", "country"]).is_duplicated().sum()
            duplicate_checks[name] = {
                "exact_full_duplicate_rows": int(exact_dup_full),
                "duplicate_records_excluding_id": int(exact_dup_no_id)
            }
    results["duplicate_checks"] = duplicate_checks

    # Save output to JSON
    out_json = os.path.join(base_dir, "scratch", "eda_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"--- Analysis Completed. Saved to {out_json} ---")

if __name__ == "__main__":
    analyze()
