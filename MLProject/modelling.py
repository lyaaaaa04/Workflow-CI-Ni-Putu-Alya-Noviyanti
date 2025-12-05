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

token = os.getenv("DAGSHUB_TOKEN")

if not token:
    print("WARNING: DAGSHUB_TOKEN is not found in environment!")
else:
    os.environ["DAGSHUB_API_TOKEN"] = token 

# 1. Init Dagshub
dagshub.init(
    repo_owner='lyaaaaa04',
    repo_name='SMSML_Alya',
    mlflow=True,
    host="https://dagshub.com"
)


mlflow.set_tracking_uri("https://dagshub.com/lyaaaaa04/SMSML_Alya.mlflow")
mlflow.set_experiment("Experiment Student Performance")

# 2. Load Data
data = pd.read_csv("StudentsPerformance_preprocessed.csv")

X = data.drop([
    'math score', 'reading score', 'writing score',
    'average_score', 'performance_level'
], axis=1)

y = data['performance_level']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

input_example = X_train.iloc[:2]

# 4. MLflow Training Run
with mlflow.start_run(run_name="Baseline RandomForest"):

    # Hyperparameters
    n_estimators = 500
    max_depth = 20

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 5. Log Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_params({
        "n_estimators": n_estimators,
        "max_depth": max_depth
    })

    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    # Create artifact folder
    os.makedirs("artifacts", exist_ok=True)

    # ARTIFACT 1 — Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_path = "artifacts/confusion_matrix.png"

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path, artifact_path="metrics")

    # ARTIFACT 2 — Classification Report
    report_path = "artifacts/classification_report.txt"

    report = classification_report(y_test, y_pred)
    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(report_path, artifact_path="metrics")

    # ARTIFACT 3 — Feature Importance
    fi_path = "artifacts/feature_importance.png"

    fi = model.feature_importances_
    plt.figure(figsize=(8,5))
    sns.barplot(x=fi, y=X_train.columns)
    plt.title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(fi_path)
    plt.close()

    mlflow.log_artifact(fi_path, artifact_path="analysis")

    # ARTIFACT 4 — ROC Curve (Macro Average)
    classes = np.unique(y)
    y_bin = label_binarize(y_test, classes=classes)

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(
        np.concatenate([fpr[i] for i in range(len(classes))])
    )
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= len(classes)
    roc_auc_macro = auc(all_fpr, mean_tpr)

    roc_path = "artifacts/roc_curve.png"
    plt.figure(figsize=(6,5))
    plt.plot(all_fpr, mean_tpr)
    plt.title(f"ROC Curve (Macro Average) - AUC = {roc_auc_macro:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    mlflow.log_artifact(roc_path, artifact_path="analysis")

    # 8. SAVE MODEL – Dagshub Compatible
    model_dir = "artifacts/random_forest_model"

    mlflow.sklearn.save_model(
        sk_model=model,
        path=model_dir,
        input_example=input_example
    )

    # Upload full folder as artifact
    mlflow.log_artifacts(model_dir, artifact_path="model")

    print("\n=== Model Training Selesai ===")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("Model + seluruh artefak berhasil diupload ke MLflow Dagshub!")
