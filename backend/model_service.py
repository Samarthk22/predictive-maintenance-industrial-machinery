import json, joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/predictive_maintenance_pipeline.joblib")
METRICS_PATH = Path("models/metrics.json")

def risk_category(score):
    if score >= 80: return "CRITICAL"
    if score >= 60: return "HIGH"
    if score >= 35: return "MEDIUM"
    return "LOW"

def priority(score):
    if score >= 80: return "Immediate"
    if score >= 60: return "Within 24 hours"
    if score >= 35: return "Within 7 days"
    return "Routine monitoring"

class LocalModelService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model missing. Run: python ml/train.py")
        self.pipeline = joblib.load(MODEL_PATH)
        self.metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

    @staticmethod
    def feature_engineer(df):
        df = df.copy()
        df["Temperature Difference [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
        df["Mechanical Load Index"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] / 1000.0
        df["Wear Load Index"] = df["Tool wear [min]"] * df["Torque [Nm]"] / 100.0
        return df

    def predict_one(self, data):
        df = self.feature_engineer(pd.DataFrame([data]))
        pred = str(self.pipeline.predict(df)[0])
        probs = self.pipeline.predict_proba(df)[0]
        classes = list(self.pipeline.named_steps["model"].classes_)
        confidence = float(np.max(probs))
        no_idx = classes.index("No Failure") if "No Failure" in classes else None
        failure_prob = float(1 - probs[no_idx]) if no_idx is not None else confidence
        return pred, confidence, failure_prob

    def predict_df(self, df):
        df2 = self.feature_engineer(df)
        pred = self.pipeline.predict(df2)
        probs = self.pipeline.predict_proba(df2)
        classes = list(self.pipeline.named_steps["model"].classes_)
        no_idx = classes.index("No Failure") if "No Failure" in classes else None
        failure_prob = 1 - probs[:, no_idx] if no_idx is not None else np.max(probs, axis=1)
        out = df.copy()
        out["Predicted Failure Type"] = pred
        out["Failure Probability"] = failure_prob
        out["Risk Score"] = np.clip(failure_prob * 100, 0, 100).round(1)
        out["Risk Category"] = out["Risk Score"].apply(risk_category)
        out["Maintenance Priority"] = out["Risk Score"].apply(priority)
        return out
