import os, requests
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
load_dotenv()

API = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="AI Predictive Maintenance", page_icon="⚙️", layout="wide")

st.title("⚙️ AI Predictive Maintenance System")
st.caption("Industrial machinery failure prediction • IBM Cloud • FastAPI • Streamlit")

try:
    h = requests.get(f"{API}/health", timeout=5).json()
    st.success(f"Backend connected — {h['mode']}")
except:
    st.error("Backend not reachable. Start FastAPI on port 8000.")

tabs = st.tabs(["🏠 Dashboard", "🔮 Prediction", "📁 Batch", "📊 Analytics", "🧠 Advisor", "☁️ IBM Cloud"])

with tabs[0]:
    st.header("Fleet Health Dashboard")
    a,b,c,d = st.columns(4)
    a.metric("ML Engine", "Random Forest")
    b.metric("Backend", "FastAPI")
    c.metric("Frontend", "Streamlit")
    d.metric("Cloud", "IBM Cloud")
    st.markdown("""
    **System capabilities**
    - Failure-type classification
    - Failure probability and 0–100 risk score
    - Maintenance priority and recommendation
    - Batch fleet screening
    - Interactive risk analytics
    - IBM watsonx.ai Runtime online scoring
    - Future IoT/SMS/email/CMMS integration
    """)

with tabs[1]:
    st.header("Single Machine Prediction")
    with st.form("p"):
        c1,c2,c3 = st.columns(3)
        typ = c1.selectbox("Machine Type", ["L","M","H"], index=1)
        air = c1.number_input("Air Temperature [K]", 295.,305.,300.,.1)
        process = c2.number_input("Process Temperature [K]",300.,315.,310.,.1)
        speed = c2.number_input("Rotational Speed [rpm]",1000.,3000.,1500.,10.)
        torque = c3.number_input("Torque [Nm]",0.,100.,40.,.5)
        wear = c3.number_input("Tool Wear [min]",0.,300.,80.,1.)
        go = st.form_submit_button("Predict Machine Condition", use_container_width=True)
    if go:
        payload = {
            "Type":typ, "Air temperature [K]":air, "Process temperature [K]":process,
            "Rotational speed [rpm]":speed, "Torque [Nm]":torque, "Tool wear [min]":wear
        }
        try:
            r = requests.post(f"{API}/predict", json=payload, timeout=30).json()
            if "detail" in r: st.error(r["detail"])
            elif "raw_response" in r: st.json(r["raw_response"])
            else:
                x,y,z = st.columns(3)
                x.metric("Failure Type", r["predicted_failure_type"])
                y.metric("Risk Score", f"{r['risk_score']}/100")
                z.metric("Risk", r["risk_category"])
                st.progress(min(100, int(r["risk_score"])))
                st.write("**Maintenance Priority:**", r["maintenance_priority"])
                st.warning(r["recommendation"])
                st.write({
                    "Temperature Difference [K]": round(process-air,2),
                    "Mechanical Load Index": round(torque*speed/1000,2),
                    "Wear Load Index": round(wear*torque/100,2)
                })
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tabs[2]:
    st.header("Batch Fleet Prediction")
    up = st.file_uploader("Upload the Kaggle CSV", type="csv")
    if up:
        df = pd.read_csv(up)
        st.write(f"Loaded **{len(df):,}** rows")
        st.dataframe(df.head(20), use_container_width=True)
        if st.button("Run Fleet Prediction", use_container_width=True):
            try:
                r = requests.post(
                    f"{API}/batch-predict",
                    files={"file": (up.name, up.getvalue(), "text/csv")},
                    timeout=120
                )
                r.raise_for_status()
                out = pd.DataFrame(r.json()["data"])
                st.session_state["out"] = out
                st.success(f"Predicted {len(out):,} machines")
                st.dataframe(out.head(100), use_container_width=True)
                st.download_button(
                    "⬇️ Download Prediction Report",
                    out.to_csv(index=False).encode(),
                    "predictive_maintenance_report.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

with tabs[3]:
    st.header("Fleet Analytics")
    if "out" not in st.session_state:
        st.info("Run a batch prediction first.")
    else:
        df = st.session_state["out"]
        risk = pd.to_numeric(df["Risk Score"], errors="coerce")
        a,b,c,d = st.columns(4)
        a.metric("Machines", len(df))
        b.metric("Average Risk", f"{risk.mean():.1f}")
        c.metric("High/Critical", int((risk>=60).sum()))
        d.metric("Critical", int((risk>=80).sum()))
        st.plotly_chart(px.histogram(df, x="Risk Score", nbins=20, title="Risk Distribution"), use_container_width=True)
        st.plotly_chart(px.pie(df, names="Predicted Failure Type", title="Failure Type Distribution"), use_container_width=True)
        if "Tool wear [min]" in df.columns:
            st.plotly_chart(
                px.scatter(df, x="Tool wear [min]", y="Risk Score", color="Risk Category",
                           title="Tool Wear vs Risk"),
                use_container_width=True
            )

with tabs[4]:
    st.header("🧠 Maintenance Advisor")
    st.markdown("""
    | Risk Score | Recommended Action |
    |---|---|
    | 0–34 | Routine monitoring |
    | 35–59 | Plan preventive maintenance |
    | 60–79 | Maintenance within 24 hours |
    | 80–100 | Immediate inspection / isolation |
    """)
    st.info("This rule layer can later connect to CMMS, email/SMS alerts, IoT sensors and an LLM maintenance assistant.")

with tabs[5]:
    st.header("☁️ IBM Cloud Integration")
    st.markdown("""
    **Production path**

    1. Upload the dataset/model to the IBM project.
    2. Train/import the model into watsonx.ai Runtime.
    3. Promote the model to a deployment space.
    4. Create an **Online** deployment.
    5. Copy the scoring endpoint.
    6. Set IBM credentials in `.env`.
    7. Set `USE_IBM=true`.
    8. FastAPI calls the IBM scoring endpoint.
    9. Streamlit displays the result.

    Never put an IBM API key directly in GitHub/source code.
    """)
    st.code("IBM_API_KEY=YOUR_KEY\nIBM_SCORING_URL=YOUR_ENDPOINT\nUSE_IBM=true\nBACKEND_URL=http://localhost:8000", language="bash")
