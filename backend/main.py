import os, io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from backend.model_service import LocalModelService, risk_category, priority
from backend.ibm_client import IBMScoringClient

load_dotenv()
app = FastAPI(title="AI Predictive Maintenance API", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

USE_IBM = os.getenv("USE_IBM", "false").lower() == "true"
local_service = None
FIELDS = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]

class MachineInput(BaseModel):
    Type: str = "M"
    air_temperature: float = Field(alias="Air temperature [K]")
    process_temperature: float = Field(alias="Process temperature [K]")
    rotational_speed: float = Field(alias="Rotational speed [rpm]")
    torque: float = Field(alias="Torque [Nm]")
    tool_wear: float = Field(alias="Tool wear [min]")
    model_config = {"populate_by_name": True}

def local():
    global local_service
    if local_service is None:
        local_service = LocalModelService()
    return local_service

def recommendation(pred, risk):
    if risk >= 80:
        return "Stop/isolate the machine and perform immediate inspection."
    if risk >= 60:
        return "Schedule maintenance within 24 hours and inspect sensors, torque, speed and wear."
    if risk >= 35:
        return "Increase monitoring frequency and plan preventive maintenance."
    return "Continue operation with routine condition monitoring."

@app.get("/health")
def health():
    return {"status": "ok", "mode": "IBM watsonx.ai Runtime" if USE_IBM else "Local model"}

@app.get("/model-info")
def model_info():
    if USE_IBM:
        return {"mode": "IBM watsonx.ai Runtime", "deployment": "online"}
    s = local()
    return {
        "mode": "Local", "accuracy": s.metrics.get("accuracy"),
        "classes": s.metrics.get("classes", []),
        "features": s.metrics.get("features", [])
    }

@app.post("/predict")
def predict(m: MachineInput):
    data = {
        "Type": m.Type,
        "Air temperature [K]": m.air_temperature,
        "Process temperature [K]": m.process_temperature,
        "Rotational speed [rpm]": m.rotational_speed,
        "Torque [Nm]": m.torque,
        "Tool wear [min]": m.tool_wear
    }

    if USE_IBM:
        try:
            return {"source": "IBM", "raw_response": IBMScoringClient().predict(FIELDS, [[data[x] for x in FIELDS]])}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"IBM scoring failed: {e}")

    try:
        pred, confidence, failure_prob = local().predict_one(data)
        risk = round(failure_prob * 100, 1)
        return {
            "source": "local",
            "predicted_failure_type": pred,
            "confidence": round(confidence, 4),
            "failure_probability": round(failure_prob, 4),
            "risk_score": risk,
            "risk_category": risk_category(risk),
            "maintenance_priority": priority(risk),
            "recommendation": recommendation(pred, risk)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file required.")
    df = pd.read_csv(io.BytesIO(await file.read()))
    try:
        out = local().predict_df(df)
        return {"rows": len(out), "data": out.fillna("").to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
