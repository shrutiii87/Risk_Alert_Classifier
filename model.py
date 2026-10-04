from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "risk_status"

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "Risk_Alert_Classifier_Dataset_4600 - Risk_Alert_Classifier_Dataset_4600.csv.csv"
)

MODEL_PATH = Path(__file__).resolve().parent / "risk_alert_model.joblib"


def load_dataset(path=DATASET_PATH):
    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column: {TARGET}")

    X = df.drop(columns=[TARGET])
    numeric_features = X.select_dtypes(include="number").columns.tolist()

    if not numeric_features:
        raise ValueError("No numeric input features found.")

    y = df[TARGET].astype(int)

    return X[numeric_features], y, numeric_features


def train_model(path=DATASET_PATH):
    X, y, features = load_dataset(path)

    pipeline = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=42,
            max_iter=1000
        )),
    ])

    pipeline.fit(X[features], y)

    return pipeline, features


def save_model(path=DATASET_PATH, model_path=MODEL_PATH):
    model, features = train_model(path)

    joblib.dump(
        {
            "model": model,
            "features": features
        },
        model_path
    )

    return model, features


def load_or_train_model():
    if MODEL_PATH.exists():
        saved = joblib.load(MODEL_PATH)

        return saved["model"], saved["features"]

    return train_model()