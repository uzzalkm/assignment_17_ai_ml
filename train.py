"""Train a heart-disease classifier and save it with joblib.

Usage:
    python train.py                       # uses data/heart.csv
    python train.py path/to/heart.csv     # e.g. the Kaggle johnsmith88 file
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"

data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/heart.csv")
df = pd.read_csv(data_path, encoding="utf-8-sig")  # strips a BOM if present

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Rows: {len(df)} | Test accuracy: {acc:.3f}")

out = Path("model/heart_model.joblib")
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({"model": model, "features": FEATURES, "test_accuracy": acc}, out)
print(f"Saved model to {out}")
