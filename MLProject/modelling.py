import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import seaborn as sns
import matplotlib.pyplot as plt

# 1. SETTING DAGSHUB
username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")

if not username or not token:
    raise ValueError("Environment variable DAGSHUB_USERNAME atau DAGSHUB_TOKEN tidak ditemukan!")

mlflow.set_tracking_uri(f"https://dagshub.com/{username}/SMSML_Alya.mlflow")
mlflow.set_experiment("student_performance-ci")

os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# 2. LOAD DATA
data = pd.read_csv("MLProject/StudentsPerformance_preprocessed.csv")

X = data.drop(
    ['math score', 'reading score', 'writing score', 'average_score', 'performance_level'],
    axis=1
)
y = data['performance_level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

input_example = X_train.iloc[:2]

# 3. MLflow RUN 
with mlflow.start_run(run_name="Baseline RandomForest") as run:

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

    mlflow.log_params({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X_train.shape[1],
    })

    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    # Create artifacts folder 
    os.makedirs("artifacts", exist_ok=True)

    # Confusion Matrix PNG
    cm = confusion_matrix(y_test, y_pred)
    cm_path = "artifacts/confusion_matrix.png"
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="metrics")

    # Classification Report TXT
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact(report_path, artifact_path="metrics")

    # Feature Importance PNG
    fi = model.feature_importances_
    fi_path = "artifacts/feature_importance.png"
    plt.figure(figsize=(8,5))
    sns.barplot(x=fi, y=X_train.columns)
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(fi_path)
    plt.close()
    mlflow.log_artifact(fi_path, artifact_path="analysis")

    # ROC Curve Macro avg
    classes = np.unique(y)
    y_bin = label_binarize(y_test, classes=classes)

    fpr, tpr = {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)
    roc_auc_macro = auc(all_fpr, mean_tpr)

    mlflow.log_metric("auc_macro", float(roc_auc_macro))

    roc_path = "artifacts/roc_curve.png"
    plt.figure(figsize=(6,5))
    plt.plot(all_fpr, mean_tpr)
    plt.title(f"ROC Curve (Macro Average) - AUC={roc_auc_macro:.3f}")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    mlflow.log_artifact(roc_path, artifact_path="analysis")

    # Save + Log Model
    model_dir = "artifacts/random_forest_model"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=None
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("Model & artifacts berhasil diupload ke MLflow DagsHub!")
