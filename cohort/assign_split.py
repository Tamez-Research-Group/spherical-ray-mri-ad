"""
assign_split.py — documents the stratified train/validation/test split
assignment (Methods, Section 2.1).

The `split` column already present in cohort.csv was produced by exactly
this logic (extracted verbatim from the research repository's
eda_pca_datasets.ipynb, section "Stratified Train / Validation / Test
split"). This script is provided for transparency and is not meant to be
re-run against cohort.csv -- doing so would reproduce the same assignment
(same data, same seed) but cohort.csv already carries its result.

Strategy
--------
- The single MCI subject in the source cohort was merged into the majority
  class (CN) for stratification purposes only; the Diagnosis column itself
  retains the subject's original label.
- Stratified by Diagnosis, to preserve class proportions in every split.
- Ratios: 70% train / 15% validation / 15% test.
- Computed once on subject IDs (not on image rows), seed = 42.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


def assign_split(subjects: pd.DataFrame) -> pd.Series:
    """
    subjects : DataFrame with at least columns ['PTID', 'Diagnosis']
               (one row per subject).

    Returns a pandas Series of {'train','validation','test'} indexed like
    `subjects`, assigned by stratified random split on Diagnosis (with MCI
    folded into CN for stratification only).
    """
    strat_label = subjects["Diagnosis"].replace({"MCI": "CN"})

    ptids  = subjects["PTID"].values
    labels = strat_label.values

    # Step 1: split off test set (15%)
    ptids_trainval, ptids_test, labels_trainval, _ = train_test_split(
        ptids, labels,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # Step 2: split remaining into train / val (70/15 of total)
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    ptids_train, ptids_val, _, _ = train_test_split(
        ptids_trainval, labels_trainval,
        test_size=val_size_adjusted,
        stratify=labels_trainval,
        random_state=RANDOM_SEED,
    )

    split_map = (
        {p: "train"      for p in ptids_train} |
        {p: "validation" for p in ptids_val}   |
        {p: "test"       for p in ptids_test}
    )
    return subjects["PTID"].map(split_map)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    cohort_path = Path(__file__).parent / "cohort.csv"
    df = pd.read_csv(cohort_path)
    recomputed = assign_split(df[["PTID", "Diagnosis"]])
    mismatches = (recomputed != df["split"]).sum()
    print(f"Recomputed split differs from cohort.csv's stored split for "
          f"{mismatches}/{len(df)} subjects (expected: 0).")
    sys.exit(0 if mismatches == 0 else 1)
