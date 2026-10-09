"""Training, evaluation, and inference utilities for Risk Alert Classifier."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "risk_status"
IDENTIFIER_COLUMNS = ["customer_id"]
EXCLUDED_COLUMNS = ["last_transaction_date"]
CATEGORICAL_FEATURES = ["gender", "region", "employment_type"]
MODEL_VERSION = 2

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / (
    "Risk_Alert_Classifier_Dataset_4600 - "
    "Risk_Alert_Classifier_Dataset_4600.csv.csv"
)
MODEL_PATH = BASE_DIR / "risk_alert_model.joblib"

INTEGER_FEATURES = {
    "age",
    "credit_score",
    "missed_payments_12m",
    "monthly_transaction_count",
    "cash_advance_count_6m",
    "complaints_last_6m",
    "failed_login_attempts_3m",
    "account_tenure_months",
    "debt_balance_inr",
}
FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "region": "Region",
    "employment_type": "Employment Type",
    "annual_income_inr": "Annual Income (INR)",
    "credit_score": "Credit Score",
    "credit_utilization_ratio": "Credit Utilization Ratio",
    "missed_payments_12m": "Missed Payments (last 12 months)",
    "avg_late_payment_days": "Average Late Payment Days",
    "monthly_transaction_count": "Monthly Transaction Count",
    "monthly_spend_inr": "Monthly Spend (INR)",
    "cash_advance_count_6m": "Cash Advances (last 6 months)",
    "complaints_last_6m": "Complaints (last 6 months)",
    "failed_login_attempts_3m": "Failed Login Attempts (last 3 months)",
    "account_tenure_months": "Account Tenure (months)",
    "debt_balance_inr": "Debt Balance (INR)",
}


def load_dataset(path=DATASET_PATH):
    """Load the dataset and return features, target, and feature metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}. Keep the CSV beside app.py and model.py."
        )

    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain the target column '{TARGET}'.")

    y = pd.to_numeric(df[TARGET], errors="coerce")
    valid_target = y.isin([0, 1])
    df = df.loc[valid_target].copy()
    y = y.loc[valid_target].astype(int)

    if y.nunique() != 2:
        raise ValueError("risk_status must contain both classes 0 and 1.")

    excluded = {TARGET, *IDENTIFIER_COLUMNS, *EXCLUDED_COLUMNS}
    numeric_features = [
        c for c in df.select_dtypes(include="number").columns if c not in excluded
    ]
    categorical_features = [
        c for c in CATEGORICAL_FEATURES if c in df.columns and c not in excluded
    ]
    features = numeric_features + categorical_features

    if not features:
        raise ValueError("No usable input features were found in the dataset.")

    X = df[features].copy()
    for column in numeric_features:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in categorical_features:
        X[column] = X[column].where(X[column].notna(), np.nan).astype(object)

    return X, y, features, numeric_features, categorical_features, df


def build_pipeline(numeric_features, categorical_features):
    """Build preprocessing + classifier in one pipeline to prevent leakage."""
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(path=DATASET_PATH, test_size=0.2, random_state=42):
    """Evaluate on a held-out stratified test set; preprocessing fits only on train."""
    X, y, features, numeric_features, categorical_features, _ = load_dataset(path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    evaluation_pipeline = build_pipeline(numeric_features, categorical_features)
    evaluation_pipeline.fit(X_train[features], y_train)

    predictions = evaluation_pipeline.predict(X_test[features])
    probabilities = evaluation_pipeline.predict_proba(X_test[features])
    classes = list(evaluation_pipeline.classes_)
    if 1 not in classes:
        raise ValueError("The trained model does not contain the High Risk class.")
    risky_probability = probabilities[:, classes.index(1)]

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, risky_probability)),
        "average_precision": float(average_precision_score(y_test, risky_probability)),
        "confusion_matrix": cm,
        "test_records": int(len(y_test)),
        "train_records": int(len(y_train)),
        "features": features,
        "class_counts": {
            "low_risk": int((y_test == 0).sum()),
            "high_risk": int((y_test == 1).sum()),
        },
    }
    return metrics


def train_model(path=DATASET_PATH):
    """Train the final deployment model on all available labelled records."""
    X, y, features, numeric_features, categorical_features, _ = load_dataset(path)
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X[features], y)
    return pipeline, features


def save_model(path=DATASET_PATH, model_path=MODEL_PATH):
    """Train and save a versioned model artifact with its expected input schema."""
    model, features = train_model(path)
    artifact = {
        "version": MODEL_VERSION,
        "model": model,
        "features": features,
    }
    joblib.dump(artifact, model_path)
    return model, features


def load_or_train_model():
    """Load a compatible saved artifact, otherwise retrain and replace it."""
    if MODEL_PATH.exists():
        try:
            saved = joblib.load(MODEL_PATH)
            _, _, expected_features, _, _, _ = load_dataset(DATASET_PATH)
            if (
                isinstance(saved, dict)
                and saved.get("version") == MODEL_VERSION
                and saved.get("features") == expected_features
                and "model" in saved
            ):
                return saved["model"], saved["features"]
        except Exception:
            # A stale or unreadable artifact is rebuilt below.
            pass

    return save_model()
