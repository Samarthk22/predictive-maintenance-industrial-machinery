# AI Predictive Maintenance System — IBM Cloud + Streamlit + FastAPI

Problem Statement No. 39: Predictive Maintenance of Industrial Machinery.

Dataset:
https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

Save the downloaded CSV as:
data/predictive_maintenance.csv

Expected columns:
UDI, Product ID, Type, Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min], Target, Failure Type

## Architecture

Kaggle CSV → preprocessing → Random Forest classification → IBM watsonx.ai Runtime
→ FastAPI REST backend → Streamlit frontend

## Added features
- Failure-type prediction
- Failure probability
- 0–100 risk score
- Risk category and maintenance priority
- Engineering load/wear indices
- Fleet/batch prediction
- Downloadable prediction report
- Interactive analytics
- Maintenance advisor
- IBM Cloud online scoring integration
- Local fallback model for development
- Swagger API
- Docker support
- Future-ready for OpenScale, IoT, SMS/email and RUL

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
python ml/train.py
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

Open:
- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

## IBM Cloud

Use IBM watsonx.ai Studio/Project, IBM watsonx.ai Runtime and Cloud Object Storage.
Train/import the model, promote it to a deployment space, create an Online deployment,
copy the scoring endpoint, and configure `.env`.

IBM docs:
https://cloud.ibm.com/docs/solution-tutorials?topic=solution-tutorials-create-deploy-retrain-machine-learning-model
