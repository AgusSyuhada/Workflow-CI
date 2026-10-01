"""Baseline modelling — Agus Syuhada (Basic K2).

- Latih di dataset PREPROCESSING (bukan raw).
- MLflow autolog, TANPA hyperparameter tuning (tuning hanya di modelling_tuning.py).
- Tracking lokal: mlruns/ di folder ini.

Jalankan:  python modelling.py
Lihat UI:   mlflow ui --port 5000   (buka http://127.0.0.1:5000)
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

TARGET = "HeartDisease"
RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent


def load_data(path: Path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in (TARGET, "split")]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    return train[feature_cols], test[feature_cols], train[TARGET], test[TARGET]


def main() -> None:
    # Dijalankan via `mlflow run` (MLflow Project): tracking URI, eksperimen,
    # dan run aktif diwarisi dari environment — jangan set ulang di sini.
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data(HERE / "heart_disease_preprocessing.csv")
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    with mlflow.start_run(run_name="randomforest_baseline"):
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Test accuracy (baseline): {acc:.4f}")


if __name__ == "__main__":
    main()
