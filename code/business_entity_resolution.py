import pandas as pd

source1 = pd.read_csv(
    "dataset/train/train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "dataset/train/train_source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "dataset/train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)

print("SOURCE 1")
print(source1.head())

print("\nSOURCE 2")
print(source2.head())

print("\nSOURCE 3")
print(source3.head())

print("\nGROUND TRUTH")
print(ground_truth.head())

print("\nDATASET SIZES")
print("Source 1:", source1.shape)
print("Source 2:", source2.shape)
print("Source 3:", source3.shape)
print("Ground Truth:", ground_truth.shape)