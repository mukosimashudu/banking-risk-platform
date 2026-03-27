import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Banking Risk Platform",
    layout="wide",
)

API_BASE_URL = "https://banking-risk-app-mukosi.onrender.com"
API_PREDICT_URL = f"{API_BASE_URL}/predict"
API_HEALTH_URL = f"{API_BASE_URL}/health"
API_BATCH_URL = f"{API_BASE_URL}/predict/batch"

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-text {
        color: #9aa0a6;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🏦 Banking Risk Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Fraud risk, credit risk, and decision support</div>',
    unsafe_allow_html=True,
)

try:
    health = requests.get(API_HEALTH_URL, timeout=10)
    if health.status_code == 200:
        st.success("API connected")
    else:
        st.warning("API is reachable, but health check did not return 200")
except Exception as e:
    st.error("Could not connect to the deployed API")
    st.code(str(e))

st.subheader("Single Application Scoring")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Fraud Inputs")
    transaction_amt = st.number_input("Transaction Amount", min_value=0.0, value=5000.0)
    card1 = st.number_input("Card1", min_value=0.0, value=1000.0)
    card2 = st.number_input("Card2", min_value=0.0, value=200.0)
    card3 = st.number_input("Card3", min_value=0.0, value=150.0)
    card5 = st.number_input("Card5", min_value=0.0, value=300.0)
    addr1 = st.number_input("Address 1", min_value=0.0, value=100.0)
    addr2 = st.number_input("Address 2", min_value=0.0, value=50.0)

with col2:
    st.markdown("### Credit Inputs")
    utilization = st.number_input("Utilization", min_value=0.0, value=0.20)
    age = st.number_input("Age", min_value=18.0, value=35.0)
    late_30_59 = st.number_input("Late 30-59 Days", min_value=0.0, value=0.0)
    debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.30)
    income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
    monthly_expenses = st.number_input("Monthly Expenses", min_value=0.0, value=2000.0)
    open_credit = st.number_input("Open Credit Lines", min_value=0.0, value=5.0)
    late_90 = st.number_input("Late 90 Days", min_value=0.0, value=0.0)
    real_estate = st.number_input("Real Estate Loans", min_value=0.0, value=1.0)
    late_60_89 = st.number_input("Late 60-89 Days", min_value=0.0, value=0.0)
    dependents = st.number_input("Dependents", min_value=0.0, value=0.0)
    marital_status = st.selectbox(
        "Marital Status",
        ["single", "married", "divorced", "widowed"],
        index=0,
    )

if st.button("🚀 Predict", use_container_width=True):
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
        "open_credit": open_credit,
        "late_90": late_90,
        "real_estate": real_estate,
        "late_60_89": late_60_89,
        "dependents": dependents,
        "monthly_expenses": monthly_expenses,
        "marital_status": marital_status,
    }

    try:
        response = requests.post(API_PREDICT_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            fraud_probability = float(result.get("fraud_probability", 0))
            credit_probability = float(result.get("credit_probability", 0))
            decision = result.get("decision", "UNKNOWN")
            top_features = result.get("top_features", [])

            st.success("Prediction completed")

            m1, m2, m3 = st.columns(3)
            m1.metric("Fraud Probability", f"{fraud_probability:.4f}")
            m2.metric("Credit Probability", f"{credit_probability:.4f}")
            if "APPROVE" in decision.upper():
                m3.success(decision)
            else:
                m3.error(decision)

            st.subheader("Application Summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Income", f"{income:,.2f}")
            s2.metric("Expenses", f"{monthly_expenses:,.2f}")
            s3.metric("Debt Ratio", f"{debt_ratio:.2f}")
            s4.metric("Age", f"{age:.0f}")

            if top_features:
                st.subheader("Top Risk Drivers")
                for item in top_features:
                    feature = item.get("feature", "unknown")
                    impact = float(item.get("impact", 0))
                    direction = "Increase risk" if impact > 0 else "Reduce risk"
                    st.write(f"**{feature}**: {direction} ({impact:.4f})")

        else:
            st.error(f"API returned status {response.status_code}")
            st.code(response.text)

    except Exception as e:
        st.error("Prediction request failed")
        st.code(str(e))

st.subheader("Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.dataframe(batch_df.head(), use_container_width=True)

    if st.button("Run Batch Prediction", use_container_width=True):
        try:
            payload = batch_df.to_dict(orient="records")
            response = requests.post(API_BATCH_URL, json=payload, timeout=120)

            if response.status_code == 200:
                results = pd.DataFrame(response.json().get("results", []))
                final_df = pd.concat(
                    [batch_df.reset_index(drop=True), results.reset_index(drop=True)],
                    axis=1,
                )

                st.success(f"Batch prediction completed for {len(final_df)} rows")
                st.dataframe(final_df, use_container_width=True)

                st.download_button(
                    label="Download Results CSV",
                    data=final_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_prediction_results.csv",
                    mime="text/csv",
                )
            else:
                st.error(f"Batch API returned status {response.status_code}")
                st.code(response.text)

        except Exception as e:
            st.error("Batch prediction failed")
            st.code(str(e))