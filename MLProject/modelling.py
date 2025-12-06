import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import dagshub
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Initialize Dagshub Credentials
username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

if not username or not token:
    raise ValueError("DAGSHUB_USERNAME / DAGSHUB_TOKEN missing")

# 2. Configure MLflow
mlflow.set_tracking_uri(f"https://dagshub.com/{username}/SMSML_Alya.mlflow")
mlflow.set_experiment("student_performance-ci")

os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# Enable Auto logging
mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

# 3. Load Data
data = pd.read_csv("MLProject/StudentsPerformance_preprocessed.csv")

X = data.drop([
    "math score", "reading score", "writing score", "average_score", "performance_level"
], axis=1)

y = data["performance_level"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Start MLflow Run
with mlflow.start_run(run_name="RandomForest-Baseline-Auto"):

    n_estimators = 500
    max_depth = 20

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
    mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})

    # ARTIFACT FOLDER
    os.makedirs("artifacts", exist_ok=True)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)
    cm_path = "artifacts/confusion_matrix.png"
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="metrics")

    # CLASSIFICATION REPORT
    report = classification_report(y_test, y_pred)
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path, artifact_path="metrics")

    # ROC Curve
    classes = np.unique(y)
    y_bin = label_binarize(y_test, classes=classes)
    fpr, tpr, _ = roc_curve(y_bin[:, 0], y_proba[:, 0])
    roc_auc = auc(fpr, tpr)

    roc_path = "artifacts/roc_curve.png"
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve - AUC={roc_auc:.3f}")
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path, artifact_path="analysis")

    # SAVE MODEL (BEST PRACTICE)
    mlflow.sklearn.log_model(model, "model")
    
    example_path = "example_input.csv"
    X_train.iloc[:2].to_csv(example_path, index=False)
    mlflow.log_artifact(example_path, artifact_path="examples")


    print("\n=== Training selesai dan seluruh artifacts telah diupload ke Dagshub ===")
