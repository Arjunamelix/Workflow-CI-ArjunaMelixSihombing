"""
modelling.py (versi MLflow Project — Kriteria 3)
=================================================
Versi ini menerima argumen CLI agar bisa dijalankan oleh MLflow Project
dan GitHub Actions CI.

Cara penggunaan manual:
    python modelling.py --n_estimators 100 --max_depth 10

Cara via MLflow Project:
    mlflow run . -P n_estimators=100 -P max_depth=10
"""

import os
import argparse
import warnings
import json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay,
)

#  Parser argumen 
def parse_args():
    parser = argparse.ArgumentParser(description="Heart Disease — MLflow Training")
    parser.add_argument("--n_estimators",     type=int,   default=100)
    parser.add_argument("--max_depth",        type=int,   default=10)
    parser.add_argument("--min_samples_split",type=int,   default=5)
    parser.add_argument("--min_samples_leaf", type=int,   default=2)
    parser.add_argument("--data_dir",         type=str,   default="heart_disease_preprocessing")
    parser.add_argument("--random_state",     type=int,   default=42)
    return parser.parse_args()


def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return (
        train.drop("target", axis=1), test.drop("target", axis=1),
        train["target"], test["target"],
    )


def save_confusion_matrix(y_test, y_pred, path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Sehat", "Sakit"]).plot(cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_roc_curve(y_test, y_proba, path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1], [0,1], "--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve"); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    args = parse_args()

    # Baris ini dinonaktifkan agar tidak bentrok dengan GitHub Actions
    # mlflow.set_experiment("Heart-Disease-CI") 
    mlflow.sklearn.autolog(disable=True)

    X_train, X_test, y_train, y_test = load_data(args.data_dir)
    os.makedirs("artifacts", exist_ok=True)

    with mlflow.start_run():
        # Log parameter
        params = {
            "n_estimators"    : args.n_estimators,
            "max_depth"       : args.max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "random_state"    : args.random_state,
        }
        mlflow.log_params(params)

        # Training
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy" : accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall"   : recall_score(y_test, y_pred),
            "f1_score" : f1_score(y_test, y_pred),
            "roc_auc"  : roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        print(" Metrik:")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")

        # Artefak
        cm_path  = "artifacts/confusion_matrix.png"
        roc_path = "artifacts/roc_curve.png"
        save_confusion_matrix(y_test, y_pred, cm_path)
        save_roc_curve(y_test, y_proba, roc_path)

        # Log model dan artefak
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(cm_path,  artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # Tag disesuaikan
        mlflow.set_tags({
            "dataset"   : "UCI Heart Disease",
            "author"    : "Arjuna Melix Sihombing",
            "model_type": "RandomForestClassifier",
        })

        print(f"\n Training selesai! Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
