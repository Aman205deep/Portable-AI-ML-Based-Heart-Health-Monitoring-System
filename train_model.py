"""
train_model.py
---------------
Trains a Normal vs Abnormal ECG beat classifier using two CSV datasets:
  1. captured_beats.csv  -> your own recorded beats
  2. ecg_dataset.csv     -> MIT-BIH (or similar) labeled beats

Expected CSV format (same for both files):
  Columns "0","1",...,"204"  -> 205 numeric samples of one heartbeat
  Column  "Label"            -> "Normal" or "Abnormal"

What this script does:
  1. Loads both CSVs and merges them into one dataset
  2. Encodes labels (Normal -> 0, Abnormal -> 1)
  3. Scales features and does a stratified train/test split
  4. Trains TWO models so you can compare:
       a) RandomForest  (fast baseline, always available)
       b) 1D CNN        (better for raw waveform shape, needs tensorflow)
  5. Prints accuracy / precision / recall / F1 / confusion matrix for each
  6. Saves the best model + the scaler + label encoder to disk so you can
     reuse them later for real-time prediction on new beats

Usage:
    python train_model.py --recorded captured_beats.csv --mit ecg_dataset.csv

Outputs (saved to ./model_output/):
    scaler.pkl
    label_encoder.pkl
    random_forest_model.pkl
    cnn_model.keras          (only if tensorflow is installed)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

FEATURE_COLS = [str(i) for i in range(205)]  # columns "0" ... "204"
LABEL_COL = "Label"


def load_and_merge(recorded_path, mit_path):
    """Load both CSVs, validate columns, and merge them into one DataFrame."""
    print(f"Loading recorded data from: {recorded_path}")
    df_recorded = pd.read_csv(recorded_path)

    print(f"Loading MIT data from: {mit_path}")
    df_mit = pd.read_csv(mit_path)

    for name, df in [("recorded", df_recorded), ("mit", df_mit)]:
        missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
        if missing:
            sys.exit(
                f"ERROR: {name} dataset is missing expected columns: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}\n"
                f"Found columns: {list(df.columns)[:10]}..."
            )

    df_recorded = df_recorded[FEATURE_COLS + [LABEL_COL]].copy()
    df_mit = df_mit[FEATURE_COLS + [LABEL_COL]].copy()

    df_recorded["source"] = "recorded"
    df_mit["source"] = "mit"

    merged = pd.concat([df_recorded, df_mit], ignore_index=True)

    # Drop rows with any missing values in features/label
    before = len(merged)
    merged = merged.dropna(subset=FEATURE_COLS + [LABEL_COL])
    dropped = before - len(merged)
    if dropped:
        print(f"Dropped {dropped} rows with missing values.")

    # Normalize label text just in case of casing/whitespace issues
    merged[LABEL_COL] = merged[LABEL_COL].astype(str).str.strip().str.capitalize()

    print("\nMerged dataset summary:")
    print(f"  Total rows: {len(merged)}")
    print(f"  From recorded: {(merged['source'] == 'recorded').sum()}")
    print(f"  From MIT:      {(merged['source'] == 'mit').sum()}")
    print("  Label counts:")
    print(merged[LABEL_COL].value_counts().to_string())

    return merged


def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n--- Training RandomForest ---")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report_metrics("RandomForest", y_test, y_pred)
    return clf


def train_cnn(X_train, y_train, X_test, y_test):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except ImportError:
        print("\n[skip] tensorflow not installed — skipping CNN training.")
        print("       Install with: pip install tensorflow --break-system-packages")
        return None

    print("\n--- Training 1D CNN ---")
    X_train_cnn = X_train[..., np.newaxis]  # (samples, 205, 1)
    X_test_cnn = X_test[..., np.newaxis]

    model = models.Sequential([
        layers.Input(shape=(205, 1)),
        layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    class_weight = {
        0: len(y_train) / (2 * n_neg) if n_neg > 0 else 1.0,
        1: len(y_train) / (2 * n_pos) if n_pos > 0 else 1.0,
    }

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    model.fit(
        X_train_cnn,
        y_train,
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    y_pred_prob = model.predict(X_test_cnn, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    report_metrics("1D CNN", y_test, y_pred)
    return model


def report_metrics(name, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n[{name}] Results:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print("  Confusion matrix ([[TN, FP], [FN, TP]]):")
    print(f"  {cm}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))


def main():
    parser = argparse.ArgumentParser(description="Train ECG Normal/Abnormal classifier.")
    parser.add_argument("--recorded", default="captured_beats.csv", help="Path to your recorded CSV")
    parser.add_argument("--mit", default="ecg_dataset.csv", help="Path to the MIT dataset CSV")
    parser.add_argument("--output_dir", default="model_output", help="Where to save trained models")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for testing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    merged = load_and_merge(args.recorded, args.mit)

    X = merged[FEATURE_COLS].values.astype(np.float32)
    y_raw = merged[LABEL_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)  # Normal -> 0, Abnormal -> 1 (alphabetical, verify below)
    print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)
    cnn_model = train_cnn(X_train_scaled, y_train, X_test_scaled, y_test)

    # Save artifacts
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.pkl"))
    joblib.dump(rf_model, os.path.join(args.output_dir, "random_forest_model.pkl"))
    if cnn_model is not None:
        cnn_model.save(os.path.join(args.output_dir, "cnn_model.keras"))

    print(f"\nAll done. Artifacts saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
