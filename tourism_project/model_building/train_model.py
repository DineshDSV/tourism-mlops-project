"""
train_model.py
Run by the GitHub Actions model-training job.
Loads train/test from Hugging Face, trains XGBoost with GridSearchCV,
logs to MLflow, saves model, and registers it on Hugging Face Model Hub.
"""
import os, json
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from datasets import load_dataset
from huggingface_hub import HfApi, login
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_DATASET  = "DineshDSV/tourism-wellness-dataset"
HF_MODEL    = "DineshDSV/tourism-wellness-model"

login(token=HF_TOKEN)
api = HfApi()

# Load data
train_ds = load_dataset(HF_DATASET, data_files="train.csv", split="train")
test_ds  = load_dataset(HF_DATASET, data_files="test.csv",  split="train")
train_df, test_df = train_ds.to_pandas(), test_ds.to_pandas()

X_train, y_train = train_df.drop(columns=["ProdTaken"]), train_df["ProdTaken"]
X_test,  y_test  = test_df.drop(columns=["ProdTaken"]),  test_df["ProdTaken"]

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", StandardScaler(), num_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", XGBClassifier(
        random_state=42, eval_metric="logloss", use_label_encoder=False,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    ))
])

param_grid = {
    "clf__n_estimators":  [100, 200],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth":     [4, 6]
}

mlflow.set_experiment("tourism_wellness_prediction")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with mlflow.start_run(run_name="XGBoost_CI"):
    gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_prob), 4)
    }
    mlflow.log_params(gs.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(best, "model")

    print(f"Best params: {gs.best_params_}")
    print(f"Metrics: {metrics}")

os.makedirs("tourism_project/model_building", exist_ok=True)
model_path = "tourism_project/model_building/best_model.joblib"
joblib.dump(best, model_path)

meta = {"model_name": "XGBoost", "best_params": gs.best_params_, "metrics": metrics,
        "feature_columns": cat_cols + num_cols, "target": "ProdTaken"}
meta_path = "tourism_project/model_building/model_metadata.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

api.create_repo(repo_id=HF_MODEL, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=model_path,  path_in_repo="best_model.joblib",
                repo_id=HF_MODEL, repo_type="model", token=HF_TOKEN)
api.upload_file(path_or_fileobj=meta_path, path_in_repo="model_metadata.json",
                repo_id=HF_MODEL, repo_type="model", token=HF_TOKEN)
print(f"Model registered at: https://huggingface.co/{HF_MODEL}")
