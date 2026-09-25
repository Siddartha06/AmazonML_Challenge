# Business Entity Resolution Challenge — Reports Directory

This directory contains technical reports, exploratory data analysis (EDA), validation findings, and strategic project plans for the **Business Entity Resolution Challenge 2026**.

---

## Technical Reports Index

| Report File | Title & Description | Key Focus Areas | Status |
| :--- | :--- | :--- | :---: |
| [01_dataset_analysis.md](file:///c:/Users/SIDDARTHA/Downloads/Amazon_Zip%20file/student_resource/reports/01_dataset_analysis.md) | **Dataset Analysis & Project Plan** | Data volume, missingness, entity uniqueness, country distribution shift (France in test), ground truth match topology, distractor analysis, data quality hazards, leakage risks, and pipeline roadmap. | **Completed** |
| [02_validation.md](file:///c:/Users/SIDDARTHA/Downloads/Amazon_Zip%20file/student_resource/reports/02_validation.md) | **Validation Framework & Evaluator** | Leakage-safe entity-level stratified split, competition Macro F0.5 metric definition, singleton logic, threshold optimization, and unit test suite verification. | **Completed** |
| [03_normalization.md](file:///c:/Users/SIDDARTHA/Downloads/Amazon_Zip%20file/student_resource/reports/03_normalization.md) | **Multi-View Normalization System** | 10 derived name representations, 10 derived address representations, open-set country handling, Unicode accent stripping, abbreviation expansion, legal suffix standardization, and unit test suite verification. | **Completed** |

---

## Executive Summary of Core Findings

1. **Massive Scale:** 26.4 Million total rows across 7 TSV files (~2.52 GB). Train set contains 2.2M $S_1$ entities, 5.0M $S_2$ candidates, and 5.3M $S_3$ candidates. Test set contains 1.7M $S_1$ entities, 4.9M $S_2$ candidates, and 5.1M $S_3$ candidates.
2. **Domain Shift Risk:** Training set contains only **US (60%)** and **India (40%)**, while the test set includes **France (14.5%)**. Models must avoid hardcoded country assumptions.
3. **Strict Partition Topology:** Ground-truth analysis confirms that every $S_2$ and $S_3$ candidate matches **at most ONE** $S_1$ entity. No candidate matches multiple $S_1$ entities.
4. **Match Distribution:** Average of 3.46 matches per $S_1$ entity. 5.58% singletons (0 matches), 5.40% 1-match, 17.00% 2-matches, and 72.01% multi-matches (3+ matches).
5. **Evaluation Target:** Submission is evaluated on **Macro $F_{0.5}$ Score** (prioritizing precision over recall). Final submission requires `output/matching_results.tsv` and `output/candidate_pairs.tsv` verified via `utils/validate_submission.py`.

---

## Challenge Rules & Compliance

- **No External Data or APIs:** Strictly prohibited from using external business databases, Google Maps, web search, geocoding APIs, or external ER services.
- **Model Constraints:** Open-source models licensed under MIT/Apache 2.0 with parameter count $\le$ **8 Billion**.
- **No ID Leakage:** Disjoint entity IDs between train and test sets.
