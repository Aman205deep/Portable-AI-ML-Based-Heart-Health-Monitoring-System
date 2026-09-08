import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

data = pd.read_csv("ecg_dataset.csv")

X = data.drop("Label", axis=1)
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print()
print("Training model...")

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print()
print("Accuracy:", accuracy)

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

print()
print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

joblib.dump(
    model,
    "model/ecg_model.joblib"
)

print()
print("MODEL SAVED!")