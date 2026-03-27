import streamlit as st
import requests
import pandas as pd

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Banking Risk Platform",
    layout="wide",
)

API_BASE_URL = "https://banking-risk-app-mukosi.onrender.com"

API_PREDICT_URL = f"{API_BASE_URL}/predict"
API_HEALTH_URL = f"{API_BASE_URL}/health"
API_BATCH_URL = f"{API_BASE_URL}/predict/batch"

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🏦 Banking Risk Platform")
st.write("Fraud + Credit Risk Decision System")

# -------------------------------------------------
# API HEALTH CHECK
# -------------------------------------------------
try:
    health = requests.get(API_HEALTH_URL, timeout=5)
    if health.status_code == 200:
        st.success("✅ API Connected")
    else:
        st.warning("⚠️ API running but not healthy")
except:
    st.error("❌ Cannot connect to API")

# -------------------------------------------------
# INPUT FORM
# -------------------------------------------------
st.subheader("📥 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💳 Fraud Inputs")
    transaction_amt = st.number_input("Transaction Amount", value=5000.0)
    card1 = st.number_input("Card1", value=1000.0)
    card2 = st.number_input("Card2", value=200.0)
    card3 = st.number_input("Card3", value=150.0)
    card5 = st.number_input("Card5", value=300.0)
    addr1 = st.number_input("Address 1", value=100.0)
    addr2 = st.number_input("Address 2", value=50.0)

with col2:
    st.markdown("### 📊 Credit Inputs")
    utilization = st.number_input("Utilization", value=0.2)
    age = st.number_input("Age", value=35.0)
    late_30_59 = st.number_input("Late 30-59 Days", value=0.0)
    debt_ratio = st.number_input("Debt Ratio", value=0.3)
    income = st.number_input("Monthly Income", value=5000.0)
    expenses = st.number_input("Monthly Expenses", value=2000.0)
    open_credit = st.number_input("Open Credit Lines", value=5.0)
    late_90 = st.number_input("Late 90 Days", value=0.0)
    real_estate = st.number_input("Real Estate Loans", value=1.0)
    late_60_89 = st.number_input("Late 60-89 Days", value=0.0)
    dependents = st.number_input("Dependents", value=0.0)

# -------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------
if st.button("🚀 Run Prediction", use_container_width=True):

    payload = {
        "transaction_amt": transaction_amt,
        "card1": card1,
        "card2": card2,
        "card3": card3,
        "card5": card5,
        "addr1": addr1,
        "addr2": addr2,
        "utilization": utilization,
        "age": age,
        "late_30_59": late_30_59,
        "debt_ratio": debt_ratio,
        "income": income,
        "monthly_expenses": expenses,
        "open_credit": open_credit,
        "late_90": late_90,
        "real_estate": real_estate,
        "late_60_89": late_60_89,
        "dependents": dependents,
    }

    try:
        response = requests.post(API_PREDICT_URL, json=payload, timeout=20)

        if response.status_code == 200:
            result = response.json()

            fraud_prob = result.get("fraud_probability", 0)
            credit_prob = result.get("credit_probability", 0)
            decision = result.get("decision", "UNKNOWN")

            st.success("✅ Prediction Complete")

            col1, col2, col3 = st.columns(3)

            col1.metric("Fraud Risk", f"{fraud_prob:.4f}")
            col2.metric("Credit Score", f"{credit_prob:.4f}")

            if "APPROVE" in decision.upper():
                col3.success(decision)
            else:
                col3.error(decision)

        else:
            st.error(f"API Error: {response.status_code}")
            st.code(response.text)

    except Exception as e:
        st.error("❌ Failed to connect to API")
        st.code(str(e))

# -------------------------------------------------
# BATCH UPLOAD
# -------------------------------------------------
st.subheader("📂 Batch Prediction")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    if st.button("Run Batch Prediction"):
        try:
            payload = df.to_dict(orient="records")
            response = requests.post(API_BATCH_URL, json=payload, timeout=60)

            if response.status_code == 200:
                results = pd.DataFrame(response.json()["results"])
                final = pd.concat([df, results], axis=1)

                st.success("Batch prediction complete")
                st.dataframe(final)

                st.download_button(
                    "Download Results",
                    final.to_csv(index=False),
                    "results.csv"
                )

            else:
                st.error("Batch API error")

        except Exception as e:
            st.error("Batch failed")
            st.code(str(e))