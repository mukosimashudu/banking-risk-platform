import sys
from pathlib import Path
import os   # ✅ ADD THIS LINE

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


from src.scoring.explain import explain_credit

# -------------------------------------------------
# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Banking Risk Decision System",
    layout="wide",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

API_PREDICT_URL = f"{API_BASE_URL}/predict"
API_BATCH_PREDICT_URL = f"{API_BASE_URL}/predict/batch"
API_HEALTH_URL = f"{API_BASE_URL}/health"
API_KPI_URL = f"{API_BASE_URL}/kpi/summary"
API_MONITOR_URL = f"{API_BASE_URL}/monitoring/summary"
API_CHAT_URL = f"{API_BASE_URL}/chat/query"
# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "payload" not in st.session_state:
    st.session_state.payload = None

if "shap_df" not in st.session_state:
    st.session_state.shap_df = None

if "shap_single_exp" not in st.session_state:
    st.session_state.shap_single_exp = None

if "shap_error" not in st.session_state:
    st.session_state.shap_error = None

# -------------------------------------------------
# STYLE
# -------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.kpi-card {
    background-color: #111827;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 3px 14px rgba(0,0,0,0.20);
}

.kpi-icon {
    font-size: 28px;
    margin-bottom: 4px;
}

.kpi-title {
    color: #9ca3af;
    font-size: 14px;
    margin-bottom: 6px;
}

.kpi-value {
    color: #f9fafb;
    font-size: 28px;
    font-weight: 700;
}

.panel {
    background-color: #0f172a;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 12px;
}

.summary-approve {
    background: linear-gradient(135deg, #0b1720 0%, #10291d 100%);
    border-left: 6px solid #22c55e;
    border-radius: 14px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.06);
}

.summary-reject {
    background: linear-gradient(135deg, #0b1720 0%, #2a1212 100%);
    border-left: 6px solid #ef4444;
    border-radius: 14px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.06);
}

.reason-chip {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px 6px 4px 0;
    border-radius: 999px;
    background-color: #1f2937;
    color: #e5e7eb;
    border: 1px solid rgba(255,255,255,0.06);
    font-size: 13px;
}

.small-text {
    color: #cbd5e1;
    line-height: 1.5;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def kpi_card(title, value, icon):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def get_credit_band(credit_prob: float) -> str:
    if credit_prob >= 0.75:
        return "Excellent"
    if credit_prob >= 0.50:
        return "Moderate"
    if credit_prob >= 0.25:
        return "Weak"
    return "Very Weak"


def get_fraud_band(fraud_prob: float) -> str:
    if fraud_prob >= 0.70:
        return "High"
    if fraud_prob >= 0.30:
        return "Medium"
    return "Low"


def build_reason_summary(result: dict, payload: dict):
    fraud_prob = float(result.get("fraud_probability", 0))
    credit_prob = float(result.get("credit_probability", 0))

    reasons = []

    if fraud_prob >= 0.70:
        reasons.append("Fraud probability is high")
    elif fraud_prob >= 0.30:
        reasons.append("Fraud probability is moderate")
    else:
        reasons.append("Fraud probability is low")

    if credit_prob > 0.50:
        reasons.append("Default risk is elevated")
    elif credit_prob > 0.25:
        reasons.append("Credit quality is below average")
    else:
        reasons.append("Credit quality is acceptable")

    disposable_income = float(payload["income"]) - float(payload["monthly_expenses"])
    if disposable_income < 1000:
        reasons.append("Disposable income is limited")
    else:
        reasons.append("Disposable income supports affordability")

    if float(payload["debt_ratio"]) > 0.60:
        reasons.append("Debt ratio is high")

    if float(payload["utilization"]) > 0.75:
        reasons.append("Credit utilization is high")

    if (
        float(payload["late_30_59"]) > 0
        or float(payload["late_60_89"]) > 0
        or float(payload["late_90"]) > 0
    ):
        reasons.append("Delinquency indicators are present")

    return reasons


def extract_single_shap_explanation(shap_output, credit_df: pd.DataFrame):
    import shap

    exp = shap_output["explanation"]

    if hasattr(exp, "values") and hasattr(exp, "base_values"):
        values = exp.values
        base_values = exp.base_values

        # regression / single output
        if len(values.shape) == 2:
            return shap.Explanation(
                values=values[0],
                base_values=base_values[0] if hasattr(base_values, "__len__") else base_values,
                data=credit_df.iloc[0].values,
                feature_names=credit_df.columns.tolist(),
            )

        # binary / multi-class
        if len(values.shape) == 3:
            class_idx = 1 if values.shape[2] > 1 else 0
            base_val = base_values[0]
            if hasattr(base_val, "__len__"):
                base_val = base_val[class_idx]

            return shap.Explanation(
                values=values[0, :, class_idx],
                base_values=base_val,
                data=credit_df.iloc[0].values,
                feature_names=credit_df.columns.tolist(),
            )

    if isinstance(exp, list):
        arr = exp[1] if len(exp) > 1 else exp[0]
        return shap.Explanation(
            values=arr[0],
            base_values=0.0,
            data=credit_df.iloc[0].values,
            feature_names=credit_df.columns.tolist(),
        )

    raise ValueError("Unsupported SHAP explanation format.")


def build_shap_table(single_exp):
    df = pd.DataFrame({
        "feature": single_exp.feature_names,
        "input_value": single_exp.data,
        "shap_impact": single_exp.values,
    })
    df["abs_impact"] = df["shap_impact"].abs()
    df["direction"] = df["shap_impact"].apply(
        lambda x: "Toward approval" if x > 0 else "Toward rejection"
    )
    df = df.sort_values("abs_impact", ascending=False).reset_index(drop=True)
    return df


def make_risk_chart(fraud_prob: float, credit_prob: float):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Fraud Probability", "Credit Probability"], [fraud_prob, credit_prob])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Risk Snapshot")
    plt.tight_layout()
    return fig


def make_shap_bar_chart(shap_df: pd.DataFrame):
    plot_df = shap_df.head(5).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(plot_df["feature"], plot_df["shap_impact"])
    ax.set_title("Top SHAP Drivers")
    ax.set_xlabel("SHAP Impact")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    return fig


# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🏦 Banking Risk Decision System")
st.write("Fraud risk + credit risk + decision engine + explainability + monitoring + batch scoring")

# -------------------------------------------------
# HEALTH
# -------------------------------------------------
try:
    health = requests.get(API_HEALTH_URL, timeout=8)
    if health.status_code == 200:
        st.success("✅ FastAPI connected")
    else:
        st.warning("⚠️ FastAPI is running but health endpoint did not return 200")
except Exception:
    st.error("❌ FastAPI is not reachable. Start the API first with: uvicorn api.main:app --reload")

# -------------------------------------------------
# KPI SUMMARY (UNCHANGED LOGIC)
# -------------------------------------------------
st.subheader("📊 KPI Summary")
try:
    kpi_res = requests.get(API_KPI_URL, timeout=10)
    if kpi_res.status_code == 200:
        kpi = kpi_res.json()

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            kpi_card("Total Apps", int(kpi.get("total_applications") or 0), "📄")
        with c2:
            kpi_card("Approved", int(kpi.get("total_approved") or 0), "✅")
        with c3:
            kpi_card("Rejected", int(kpi.get("total_rejected") or 0), "❌")
        with c4:
            kpi_card("Avg Credit", round(float(kpi.get("avg_credit_probability") or 0), 4), "📈")
        with c5:
            kpi_card("Avg Fraud", round(float(kpi.get("avg_fraud_probability") or 0), 4), "⚠️")
except Exception as e:
    st.info(f"KPI summary not available yet: {e}")

# -------------------------------------------------
# INPUT FORM
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("💳 Fraud Inputs")
    transaction_amt = st.number_input("Transaction Amount", min_value=0.0, value=5000.0)
    card1 = st.number_input("Card1", min_value=0.0, value=1000.0)
    card2 = st.number_input("Card2", min_value=0.0, value=200.0)
    card3 = st.number_input("Card3", min_value=0.0, value=150.0)
    card5 = st.number_input("Card5", min_value=0.0, value=300.0)
    addr1 = st.number_input("Address 1", min_value=0.0, value=100.0)
    addr2 = st.number_input("Address 2", min_value=0.0, value=50.0)

with col2:
    st.subheader("📈 Credit Inputs")
    utilization = st.number_input("Utilization", min_value=0.0, value=0.20)
    age = st.number_input("Age", min_value=18.0, value=35.0)
    late_30_59 = st.number_input("Late 30–59 Days", min_value=0.0, value=0.0)
    debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.30)
    income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
    monthly_expenses = st.number_input("Monthly Expenses", min_value=0.0, value=2000.0)
    open_credit = st.number_input("Open Credit Lines", min_value=0.0, value=5.0)
    late_90 = st.number_input("Late 90 Days", min_value=0.0, value=0.0)
    real_estate = st.number_input("Real Estate Loans", min_value=0.0, value=1.0)
    late_60_89 = st.number_input("Late 60–89 Days", min_value=0.0, value=0.0)
    dependents = st.number_input("Dependents", min_value=0.0, value=0.0)
    marital_status = st.selectbox("Marital Status", ["single", "married", "divorced", "widowed"])

# -------------------------------------------------
# PREDICT
# -------------------------------------------------
if st.button("🚀 Predict Decision", use_container_width=True):
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
        "marital_status": marital_status
    }

    try:
        response = requests.post(API_PREDICT_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            st.session_state.result = result
            st.session_state.payload = payload

            credit_df = pd.DataFrame([{
                "utilization": utilization,
                "age": age,
                "late_30_59": late_30_59,
                "debt_ratio": debt_ratio,
                "income": income,
                "open_loans": open_credit,
                "late_90": late_90,
                "real_estate_loans": real_estate,
                "late_60_89": late_60_89,
                "dependents": dependents,
            }])

            shap_result = explain_credit(credit_df)

            if shap_result["ok"]:
                try:
                    single_exp = extract_single_shap_explanation(shap_result, credit_df)
                    shap_df = build_shap_table(single_exp)

                    st.session_state.shap_single_exp = single_exp
                    st.session_state.shap_df = shap_df
                    st.session_state.shap_error = None
                except Exception as shap_parse_error:
                    st.session_state.shap_single_exp = None
                    st.session_state.shap_df = None
                    st.session_state.shap_error = f"SHAP parsing error: {shap_parse_error}"
            else:
                st.session_state.shap_single_exp = None
                st.session_state.shap_df = None
                st.session_state.shap_error = shap_result["error"]

        else:
            st.error(f"API Error: {response.status_code}")
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to FastAPI. Start the API first.")
    except Exception as e:
        st.error("Unexpected Streamlit error")
        st.code(str(e))

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
if st.session_state.result:
    result = st.session_state.result
    payload = st.session_state.payload
    shap_df = st.session_state.shap_df
    shap_single_exp = st.session_state.shap_single_exp
    shap_error = st.session_state.shap_error

    fraud_prob = float(result["fraud_probability"])
    credit_prob = float(result["credit_probability"])
    decision = result["decision"]

    st.success("✅ Prediction Complete")

    a, b, c = st.columns(3)
    a.metric("Fraud Risk", f"{fraud_prob:.4f}")
    b.metric("Credit Score", f"{credit_prob:.4f}")

    if "APPROVE" in decision.upper():
        c.success(decision)
        summary_class = "summary-approve"
    else:
        c.error(decision)
        summary_class = "summary-reject"

    reasons = build_reason_summary(result, payload)
    reasons_html = "".join([f"<span class='reason-chip'>{r}</span>" for r in reasons])

    st.subheader("📋 Decision Summary")
    st.markdown(f"""
    <div class="{summary_class}">
        <div style="font-size:24px; font-weight:700; color:#f8fafc;">{decision}</div>
        <div class="small-text" style="margin-top:10px;">
            <b>Income:</b> {payload['income']:,.2f}<br>
            <b>Monthly Expenses:</b> {payload['monthly_expenses']:,.2f}<br>
            <b>Age:</b> {payload['age']:.0f}<br>
            <b>Marital Status:</b> {payload['marital_status'].title()}<br>
            <b>Fraud Risk Band:</b> {get_fraud_band(fraud_prob)}<br>
            <b>Credit Quality Band:</b> {get_credit_band(credit_prob)}
        </div>
        <div style="margin-top:12px;">
            {reasons_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("### 👤 Customer Input Summary")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Income", f"{payload['income']:,.2f}")
        p2.metric("Expenses", f"{payload['monthly_expenses']:,.2f}")
        p3.metric("Age", f"{payload['age']:.0f}")
        p4.metric("Dependents", f"{payload['dependents']:.0f}")

        p5, p6, p7, p8 = st.columns(4)
        p5.metric("Debt Ratio", f"{payload['debt_ratio']:.2f}")
        p6.metric("Utilization", f"{payload['utilization']:.2f}")
        p7.metric("Open Credit", f"{payload['open_credit']:.0f}")
        p8.metric("Marital Status", payload['marital_status'].title())

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("### 📊 Risk Snapshot")
        fig_risk = make_risk_chart(fraud_prob, credit_prob)
        st.pyplot(fig_risk, clear_figure=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("🔎 Explainability (SHAP)")
    ex1, ex2 = st.columns([1.15, 1])

    with ex1:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("### Top Model Drivers")

        if shap_df is not None and not shap_df.empty:
            st.dataframe(
                shap_df[["feature", "input_value", "shap_impact", "direction"]],
                use_container_width=True,
                hide_index=True
            )

            fig_bar = make_shap_bar_chart(shap_df)
            st.pyplot(fig_bar, clear_figure=True)
        else:
            st.warning(f"SHAP explanation unavailable. {shap_error if shap_error else ''}")

        st.markdown("</div>", unsafe_allow_html=True)

    with ex2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("### Interpretation")

        if shap_df is not None and not shap_df.empty:
            top3 = shap_df.head(3)
            notes = []
            for _, row in top3.iterrows():
                direction_text = "supported approval" if row["shap_impact"] > 0 else "pulled toward rejection"
                notes.append(f"- **{row['feature']}** with input value **{row['input_value']}** {direction_text}.")

            st.markdown(
                "The strongest drivers behind this decision were:\n\n" +
                "\n".join(notes) +
                "\n\nThis combines model scoring with feature-level explainability so the decision is interpretable for business and technical stakeholders."
            )
        else:
            st.info("Top SHAP reasons will appear here after a successful explanation build.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("### SHAP Waterfall Plot")

    if shap_single_exp is not None:
        try:
            import shap
            fig = plt.figure(figsize=(10, 4.8))
            shap.plots.waterfall(shap_single_exp, show=False)
            st.pyplot(fig, clear_figure=True)
        except Exception as e:
            st.warning(f"Could not render SHAP waterfall plot: {e}")
    else:
        st.info("Waterfall plot not available for this prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# BATCH SCORING
# -------------------------------------------------
st.subheader("📂 Batch Scoring")
uploaded_file = st.file_uploader("Upload CSV for batch scoring", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.dataframe(batch_df.head(), use_container_width=True)

    if st.button("Run Batch Prediction"):
        try:
            payload = batch_df.to_dict(orient="records")
            res = requests.post(API_BATCH_PREDICT_URL, json=payload, timeout=120)

            if res.status_code == 200:
                batch_results = pd.DataFrame(res.json()["results"])
                combined = pd.concat([batch_df.reset_index(drop=True), batch_results.reset_index(drop=True)], axis=1)

                st.success(f"Batch scoring completed for {len(combined)} records")
                st.dataframe(combined, use_container_width=True)

                csv_bytes = combined.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download Batch Results",
                    data=csv_bytes,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Batch API Error: {res.status_code}")
                st.code(res.text)
        except Exception as e:
            st.error(f"Batch scoring failed: {e}")

# -------------------------------------------------
# MONITORING
# -------------------------------------------------
st.subheader("📈 Model Monitoring")
try:
    mon_res = requests.get(API_MONITOR_URL, timeout=10)
    if mon_res.status_code == 200:
        mon = mon_res.json()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Predictions", int(mon.get("total_predictions") or 0))
        m2.metric("Avg Fraud", round(float(mon.get("avg_fraud") or 0), 4))
        m3.metric("Avg Credit", round(float(mon.get("avg_credit") or 0), 4))

        st.write({
            "first_prediction_at": mon.get("first_prediction_at"),
            "last_prediction_at": mon.get("last_prediction_at"),
        })
    else:
        st.warning("Monitoring endpoint not available.")
except Exception as e:
    st.warning(f"Monitoring not available: {e}")

# -------------------------------------------------
# CHATBOT
# -------------------------------------------------
st.subheader("🤖 Banking Assistant")
question = st.text_input(
    "Ask something like: How many approved? How many rejected? How many qualify? Total applications?"
)

if st.button("Ask Assistant"):
    try:
        res = requests.post(API_CHAT_URL, json={"question": question}, timeout=20)
        if res.status_code == 200:
            st.write(res.json())
        else:
            st.error(f"Chat API Error: {res.status_code}")
            st.code(res.text)
    except Exception as e:
        st.error(f"Assistant failed: {e}")