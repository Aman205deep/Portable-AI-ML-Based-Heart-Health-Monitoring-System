import wfdb
import numpy as np
import pandas as pd
import os

DATASET_PATH = "dataset/mitbih"

records = ["100", "101", "102", "103", "104"]

X = []
y = []

for record_name in records:

    print("Processing:", record_name)

    record = wfdb.rdrecord(
        os.path.join(DATASET_PATH, record_name)
    )

    annotation = wfdb.rdann(
        os.path.join(DATASET_PATH, record_name),
        "atr"
    )

    signal = record.p_signal[:, 0]

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol
    ):

        if sample < 100:
            continue

        if sample >= len(signal) - 100:
            continue

        beat = signal[sample - 100:sample + 100]

        # Remove baseline
        beat = beat - np.mean(beat)

        # Normalize
        std = np.std(beat)

        if std != 0:
            beat = beat / std

        features = [
            np.mean(beat),
            np.std(beat),
            np.min(beat),
            np.max(beat),
            np.ptp(beat)
        ]

        # Add waveform samples
        features.extend(beat.tolist())

        X.append(features)

        # Beginner binary classification
        if symbol == "N":
            y.append("Normal")
        else:
            y.append("Abnormal")

data = pd.DataFrame(X)

data["Label"] = y

data.to_csv(
    "ecg_dataset.csv",
    index=False
)

print()
print("Dataset created!")
print("Total samples:", len(data))
print()
print(data["Label"].value_counts())