"""
data_prep.py
Run by the GitHub Actions data-prep job.
Loads the raw dataset from Hugging Face, cleans it, splits it, and
uploads train/test CSVs back to Hugging Face.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from huggingface_hub import HfApi, login

HF_TOKEN      = os.environ["HF_TOKEN"]
HF_DATASET    = "DineshDSV/tourism-wellness-dataset"

login(token=HF_TOKEN)
api = HfApi()

# Load
ds = load_dataset(HF_DATASET, data_files="tourism.csv", split="train")
df = ds.to_pandas()

# Clean
df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns], inplace=True)
df["Gender"] = df["Gender"].replace("Fe Male", "Female")
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# Split
X, y = df.drop(columns=["ProdTaken"]), df["ProdTaken"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_df = X_train.copy(); train_df["ProdTaken"] = y_train.values
test_df  = X_test.copy();  test_df["ProdTaken"]  = y_test.values

os.makedirs("tourism_project/data", exist_ok=True)
train_df.to_csv("tourism_project/data/train.csv", index=False)
test_df.to_csv("tourism_project/data/test.csv", index=False)

# Upload
for fname in ["train.csv", "test.csv"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/data/{fname}",
        path_in_repo=fname,
        repo_id=HF_DATASET,
        repo_type="dataset",
        token=HF_TOKEN
    )
    print(f"{fname} uploaded.")

print("Data preparation complete.")
