import os
import sys
import csv
import json

sys.path.insert(0, r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource")
sys.stdout.reconfigure(encoding='utf-8')

from src.normalization import normalize_record

def test_real_records():
    dataset_dir = r"c:\Users\SIDDARTHA\Downloads\Amazon_Zip file\student_resource\dataset"
    
    samples = [
        ("train_source1.tsv", os.path.join(dataset_dir, "train", "train_source1.tsv")),
        ("train_source2.tsv", os.path.join(dataset_dir, "train", "train_source2.tsv")),
        ("test_source1.tsv", os.path.join(dataset_dir, "test", "test_source1.tsv")),
    ]

    print("=== Multi-View Normalization Demonstration on Real Records ===\n")

    for file_label, path in samples:
        print(f"--- Sampling 2 records from {file_label} ---")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            
            for i in range(2):
                row = next(reader)
                eid, name, addr, ctry = row[0], row[1], row[2], row[3]
                norm_rec = normalize_record(eid, name, addr, ctry)
                
                print(f"\n[Record {i+1} from {file_label}]")
                print(f"  Entity ID: {norm_rec.entity_id}")
                print(f"  Original Name: '{norm_rec.original_name}'")
                print(f"    - name_lower: '{norm_rec.name_lower}'")
                print(f"    - name_unicode: '{norm_rec.name_unicode}'")
                print(f"    - name_no_punct: '{norm_rec.name_no_punct}'")
                print(f"    - name_abbrev_expanded: '{norm_rec.name_abbrev_expanded}'")
                print(f"    - name_legal_suffix_norm: '{norm_rec.name_legal_suffix_norm}'")
                print(f"    - name_no_legal_suffix: '{norm_rec.name_no_legal_suffix}'")
                print(f"    - name_compact: '{norm_rec.name_compact}'")
                print(f"    - name_sorted_tokens: '{norm_rec.name_sorted_tokens}'")

                print(f"  Original Address: '{norm_rec.original_address}'")
                print(f"    - address_abbrev_expanded: '{norm_rec.address_abbrev_expanded}'")
                print(f"    - address_numeric_tokens: {norm_rec.address_numeric_tokens}")
                print(f"    - address_postal_code: {norm_rec.address_postal_code}")
                print(f"    - address_region_tokens: {norm_rec.address_region_tokens}")

                print(f"  Original Country: '{norm_rec.original_country}' -> clean: '{norm_rec.country_clean}'")

if __name__ == "__main__":
    test_real_records()
