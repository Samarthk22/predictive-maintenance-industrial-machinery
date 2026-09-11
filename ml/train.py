import os, json, joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/predictive_maintenance.csv"
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/predictive_maintenance_pipeline.joblib"

NUMERIC = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
CATEGORICAL = ["Type"]
TARGET = "Failure Type"

def build_features(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["Temperature Difference [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Mechanical Load Index"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] / 1000.0
    df["Wear Load Index"] = df["Tool wear [min]"] * df["Torque [Nm]"] / 100.0
    return df

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the Kaggle dataset first."
        )

    df = build_features(pd.read_csv(DATA_PATH))
    features = NUMERIC + [
        "Temperature Difference [K]",
        "Mechanical Load Index",
        "Wear Load Index"
    ] + CATEGORICAL

    X = df[features]
    y = df[TARGET].astype(str).fillna("No Failure")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = [c for c in features if c not in CATEGORICAL]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])

    model = RandomForestClassifier(
        n_estimators=350, random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2, n_jobs=-1
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "classification_report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classes": list(pipeline.named_steps["model"].classes_),
        "features": features
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
