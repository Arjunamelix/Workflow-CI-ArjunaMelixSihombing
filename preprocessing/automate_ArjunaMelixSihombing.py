"""
automate_ArjunSinaga.py
=======================
Script otomatis untuk preprocessing data Heart Disease UCI.
Mengembalikan data yang sudah siap dilatih (train.csv dan test.csv).

Cara penggunaan:
    python automate_ArjunSinaga.py

Output:
    heart_disease_preprocessing/train.csv
    heart_disease_preprocessing/test.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ─── Konfigurasi ──────────────────────────────────────────────────
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
OUTPUT_DIR = "heart_disease_preprocessing"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ─── Fungsi-fungsi ────────────────────────────────────────────────

def load_data(url: str, column_names: list) -> pd.DataFrame:
    """Muat data dari URL, tandai '?' sebagai NaN."""
    print(f"[1/5] Memuat data dari {url} ...")
    df = pd.read_csv(url, names=column_names, na_values="?")
    print(f"      ✅ Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi target multi-kelas (0-4) menjadi biner (0/1)."""
    print("[2/5] Encoding target menjadi biner ...")
    df = df.copy()
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    dist = df["target"].value_counts().to_dict()
    print(f"      ✅ Distribusi target: Sehat={dist.get(0,0)}, Sakit={dist.get(1,0)}")
    return df


def handle_missing_values(df: pd.DataFrame, cols_to_impute: list) -> pd.DataFrame:
    """Isi missing value dengan nilai median (hanya untuk kolom yang ditentukan)."""
    print("[3/5] Menangani missing values ...")
    df = df.copy()
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        imputer = SimpleImputer(strategy="median")
        df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
    print(f"      ✅ Missing values: {missing_before} → {df.isnull().sum().sum()}")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Pisahkan data menjadi train dan test dengan stratifikasi."""
    print("[4/5] Membagi data menjadi train dan test ...")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"      ✅ Train: {X_train.shape[0]} baris | Test: {X_test.shape[0]} baris")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numerical_cols: list,
) -> tuple:
    """
    Terapkan StandardScaler hanya pada fitur numerik kontinu.
    Scaler di-fit hanya pada data train untuk mencegah data leakage.
    """
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_test


def save_preprocessed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """Simpan data train dan test ke folder output."""
    print(f"[5/5] Menyimpan hasil ke folder '{output_dir}/' ...")
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df["target"] = y_train.values
    train_path = os.path.join(output_dir, "train.csv")
    train_df.to_csv(train_path, index=False)

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_path = os.path.join(output_dir, "test.csv")
    test_df.to_csv(test_path, index=False)

    print(f"      ✅ Tersimpan: {train_path} ({len(train_df)} baris)")
    print(f"      ✅ Tersimpan: {test_path} ({len(test_df)} baris)")


def run_preprocessing() -> dict:
    """
    Fungsi utama yang menjalankan seluruh pipeline preprocessing
    dan mengembalikan dict berisi train/test DataFrame.
    """
    print("=" * 55)
    print("  AUTOMATE PREPROCESSING — Heart Disease Dataset")
    print("  Nama   : Arjun Sinaga")
    print("  Dataset: UCI Heart Disease (Cleveland)")
    print("=" * 55)

    df = load_data(DATA_URL, COLUMN_NAMES)
    df = encode_target(df)
    df = handle_missing_values(df, cols_to_impute=["ca", "thal"])
    X_train, X_test, y_train, y_test = split_data(
        df, TEST_SIZE, RANDOM_STATE
    )
    X_train, X_test = scale_features(X_train, X_test, NUMERICAL_FEATURES)
    save_preprocessed_data(X_train, X_test, y_train, y_test, OUTPUT_DIR)

    print("\n🎉 Preprocessing selesai!")
    print(f"   Output tersimpan di folder: {OUTPUT_DIR}/")

    # Kembalikan data untuk dipakai langsung jika di-import
    train_df = X_train.copy()
    train_df["target"] = y_train.values
    test_df = X_test.copy()
    test_df["target"] = y_test.values

    return {"train": train_df, "test": test_df}


# ─── Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_preprocessing()
    print(f"\nContoh data train (5 baris pertama):")
    print(result["train"].head())
