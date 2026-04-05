from __future__ import annotations

import os
import time
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://banking-risk-app-mukosi.onrender.com"
).rstrip("/")

AUTO_REFRESH_SECONDS = int(os.getenv("AUTO_REFRESH_SECONDS", "30"))


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Full Fintech Banking Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #030b1a;
        color: white;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(8,18,39,0.95), rgba(5,12,27,0.95));
        border: 1px solid rgba(90,110,160,0.20);
        border-radius: 16px;
        padding: 18px 18px 14px 18px;
        min-height: 120px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 8px 24px rgba(0,0,0,0.22);
    }

    .metric-title {
        font-size: 13px;
        color: #cbd5e1;
        margin-bottom: 10px;
        font-weight: 500;
    }

    .metric-value {
        font-size: 22px;
        color: #f8fafc;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-pill {
        display: inline-block;
        background: rgba(34,197,94,0.18);
        color: #4ade80;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
    }

    .metric-pill-live {
        display: inline-block;
        background: rgba(59,130,246,0.18);
        color: #60a5fa;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
    }

    .alert-banner {
        background: rgba(239,68,68,0.18);
        border: 1px solid rgba(239,68,68,0.22);
        color: #f87171;
        border-radius: 12px;
        padding: 14px 16px;
        font-weight: 600;
        margin-bottom: 16px;
        animation: pulse 1.5s infinite;
    }

    .info-card {
        background: rgba(30,64,175,0.22);
        border: 1px solid rgba(59,130,246,0.18);
        color: #dbeafe;
        border-radius: 12px;
        padding: 14px 16px;
    }

    .success-card {
        background: rgba(22,163,74,0.18);
        border: 1px solid rgba(34,197,94,0.18);
        color: #dcfce7;
        border-radius: 12px;
        padding: 14px 16px;
    }

    .section-title {
        font-size: 17px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 10px;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 12px;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.25); }
        70% { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
        100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def api_get(endpoint: str, timeout: int = 60) -> Any:
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(endpoint: str, payload: Dict[str, Any], timeout: int = 60) -> Any:
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def fmt_currency(value: Any) -> str:
    try:
        return f"R {float(value):,.2f}"
    except Exception:
        return "R 0.00"


def fmt_percent(value: Any) -> str:
    try:
        number = float(value)
        if number <= 1:
            number *= 100
        return f"{number:.2f}%"
    except Exception:
        return "0.00%"


def metric_card(title: str, value: str, pill: str = "", pill_live: bool = False) -> None:
    pill_html = ""
    if pill:
        cls = "metric-pill-live" if pill_live else "metric-pill"
        pill_html = f'<span class="{cls}">{pill}</span>'

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def to_dataframe(data: Any) -> pd.DataFrame:
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame()


def clear_all_cache() -> None:
    st.cache_data.clear()


# =========================================================
# AUTO REFRESH
# =========================================================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > AUTO_REFRESH_SECONDS:
    st.session_state.last_refresh = time.time()
    st.rerun()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Configuration")
    st.write(f"**API Base URL:** {API_BASE_URL}")
    st.write(f"**Auto refresh target:** {AUTO_REFRESH_SECONDS} sec")

    if st.button("🔄 Refresh now", use_container_width=True):
        st.session_state.last_refresh = time.time()
        clear_all_cache()
        st.rerun()

    st.markdown("---")
    st.markdown("## SQL Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_area(
        "Ask portfolio questions",
        placeholder="Example: how many rejected",
        height=90,
        key="chat_question_input",
    )

    if st.button("Ask SQL Assistant", use_container_width=True):
        if user_question.strip():
            chat_response = api_post("/api/chat/query", {"question": user_question.strip()})
            answer = chat_response.get("answer", "No answer returned.")
            st.session_state.chat_history.append(
                {"question": user_question.strip(), "answer": answer}
            )

    if st.session_state.chat_history:
        for item in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            st.markdown("---")


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(ttl=20)
def load_summary():
    return api_get("/api/portfolio/summary")


@st.cache_data(ttl=20)
def load_recent_loans():
    return api_get("/api/portfolio/recent-loans")


@st.cache_data(ttl=20)
def load_recent_credit():
    return api_get("/api/portfolio/recent-credit")


@st.cache_data(ttl=20)
def load_product_distribution():
    return api_get("/api/portfolio/product-distribution")


@st.cache_data(ttl=20)
def load_decision_distribution():
    return api_get("/api/portfolio/decision-distribution")


@st.cache_data(ttl=20)
def load_fraud_distribution():
    return api_get("/api/portfolio/fraud-distribution")


@st.cache_data(ttl=20)
def load_fraud_live():
    return api_get("/api/fraud/live")


summary = load_summary()
recent_loans = load_recent_loans()
recent_credit = load_recent_credit()
product_distribution = load_product_distribution()
decision_distribution = load_decision_distribution()
fraud_distribution = load_fraud_distribution()
fraud_live = load_fraud_live()

summary = summary if isinstance(summary, dict) else {}
loans_df = to_dataframe(recent_loans)
credit_df = to_dataframe(recent_credit)
product_df = to_dataframe(product_distribution)
decision_df = to_dataframe(decision_distribution)
fraud_df = to_dataframe(fraud_distribution)
fraud_live_df = to_dataframe(fraud_live)


# =========================================================
# HEADER
# =========================================================
st.markdown("# 🏦 Full Fintech Banking Platform")
st.caption("Loans • Credit Cards • Vehicle Finance • Home Loans • IFRS 9 • Explainable AI • Fraud Monitoring")

tabs = st.tabs(
    [
        "Executive Dashboard",
        "Loan Application",
        "Credit Application",
        "Fraud Monitoring",
        "Recent Applications",
    ]
)


# =========================================================
# TAB 1: EXECUTIVE DASHBOARD
# =========================================================
with tabs[0]:
    st.subheader("Executive Dashboard")

    critical_alerts = int(summary.get("critical_alerts", 0) or 0)
    high_alerts = int(summary.get("high_alerts", 0) or 0)

    if critical_alerts > 0:
        st.markdown(
            f'<div class="alert-banner">🚨 Live Alert: {critical_alerts} critical fraud event(s) detected. Immediate review recommended.</div>',
            unsafe_allow_html=True,
        )

    row1 = st.columns(4)
    with row1[0]:
        metric_card("Total Applications", f"{int(summary.get('total_applications', 0) or 0):,}", "recent")
    with row1[1]:
        metric_card("Approved Cases", f"{int(summary.get('approved_cases', 0) or 0):,}", "recent")
    with row1[2]:
        metric_card("Approval Rate", fmt_percent(summary.get("approval_rate", 0)), "live", pill_live=True)
    with row1[3]:
        metric_card("Lifetime ECL", fmt_currency(summary.get("lifetime_ecl", 0)))

    row2 = st.columns(4)
    with row2[0]:
        metric_card("Loan Exposure", fmt_currency(summary.get("loan_exposure", 0)))
    with row2[1]:
        metric_card("Credit Limits", fmt_currency(summary.get("credit_limits", 0)))
    with row2[2]:
        metric_card("Average PD", fmt_percent(summary.get("average_pd", 0)))
    with row2[3]:
        metric_card("Average Fraud Score", fmt_percent(summary.get("average_fraud_score", 0)))

    row3 = st.columns(4)
    with row3[0]:
        metric_card("Critical Alerts", str(critical_alerts))
    with row3[1]:
        metric_card("High Alerts", str(high_alerts))
    with row3[2]:
        metric_card("Average SHAP Risk", fmt_percent(summary.get("average_shap_risk", 0)))
    with row3[3]:
        metric_card("Data Refresh", time.strftime("%H:%M:%S"))

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Product Distribution")
        if not product_df.empty and "product" in product_df.columns and "count" in product_df.columns:
            fig = px.bar(
                product_df,
                x="product",
                y="count",
                text="count",
            )
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product distribution yet.")

        st.markdown("### Decision Distribution")
        if not decision_df.empty:
            decision_col = "decision" if "decision" in decision_df.columns else decision_df.columns[0]
            count_col = "count" if "count" in decision_df.columns else decision_df.columns[-1]
            fig = px.pie(decision_df, names=decision_col, values=count_col, hole=0.45)
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No decision distribution yet.")

    with col2:
        st.markdown("### Fraud Alert Distribution")
        if not fraud_df.empty:
            alert_col = "alert_level" if "alert_level" in fraud_df.columns else fraud_df.columns[0]
            count_col = "count" if "count" in fraud_df.columns else fraud_df.columns[-1]
            fig = px.bar(
                fraud_df,
                x=alert_col,
                y=count_col,
                text=count_col,
            )
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fraud distribution yet.")


# =========================================================
# TAB 2: LOAN APPLICATION
# =========================================================
with tabs[1]:
    st.subheader("Loan Application")

    c1, c2, c3 = st.columns(3)

    with c1:
        loan_customer_name = st.text_input("Customer Name", value="John Doe", key="loan_customer_name")
        loan_product_type = st.selectbox(
            "Product Type",
            ["personal_loan", "vehicle_finance", "home_loan", "business_loan"],
            key="loan_product_type",
        )
        loan_requested_amount = st.number_input("Requested Amount (R)", min_value=0.0, value=1500000.0, step=1000.0)
        loan_interest = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=15.5, step=0.1)
        loan_term_months = st.number_input("Term (Months)", min_value=1, value=60, step=1)
        loan_affordability = st.slider("Affordability Factor", 0.10, 1.00, 0.70, 0.01)

    with c2:
        loan_income = st.number_input("Net Monthly Income (R)", min_value=0.0, value=85000.0, step=500.0)
        loan_expenses = st.number_input("Monthly Expenses (R)", min_value=0.0, value=12000.0, step=500.0)
        loan_existing_debt = st.number_input("Existing Debt Payments (R)", min_value=0.0, value=3500.0, step=100.0)
        loan_credit_score = st.slider("Credit Score", 300, 900, 680, 1)
        loan_fraud_score = st.slider("Fraud Score", 0.0, 1.0, 0.08, 0.01)
        loan_dti_cap = st.slider("Debt-to-Income Cap", 0.10, 1.00, 0.45, 0.01)

    with c3:
        loan_property_value = st.number_input("Property Value (R)", min_value=0.0, value=0.0, step=1000.0)
        loan_deposit = st.number_input("Deposit (R)", min_value=0.0, value=0.0, step=1000.0)
        loan_secured = st.checkbox("Secured Loan", value=False)
        loan_days_past_due = st.number_input("Days Past Due", min_value=0, value=0, step=1)
        loan_sicr = st.checkbox("Significant Increase in Credit Risk (SICR)", value=False)
        loan_default = st.checkbox("Default Flag", value=False)
        loan_stress_rate = st.slider("Stress Rate Add-on (%)", 0.0, 10.0, 2.0, 0.1)

    if st.button("Run Loan Assessment", key="run_loan_assessment"):
        payload = {
            "customer_name": loan_customer_name,
            "product_type": loan_product_type,
            "requested_amount": loan_requested_amount,
            "annual_interest_rate": loan_interest,
            "term_months": int(loan_term_months),
            "net_monthly_income": loan_income,
            "monthly_expenses": loan_expenses,
            "existing_debt_payments": loan_existing_debt,
            "credit_score": loan_credit_score,
            "fraud_score": loan_fraud_score,
            "property_value": loan_property_value,
            "deposit": loan_deposit,
            "secured": loan_secured,
            "days_past_due": int(loan_days_past_due),
            "sicr_flag": loan_sicr,
            "default_flag": loan_default,
            "affordability_factor": loan_affordability,
            "debt_to_income_cap": loan_dti_cap,
            "stress_rate_addon": loan_stress_rate,
        }

        result = api_post("/api/loan/assess", payload)

        if result.get("error"):
            st.error(result["error"])
        else:
            save_status = result.get("save_status", {})
            if save_status.get("saved"):
                st.markdown(
                    '<div class="success-card">Loan application saved to Azure SQL successfully.</div>',
                    unsafe_allow_html=True,
                )
                clear_all_cache()
            else:
                st.warning(f"Loan calculated, but save failed: {save_status.get('message', 'Unknown error')}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Decision", result.get("final_decision", "N/A"))
            m2.metric("Approved Amount", fmt_currency(result.get("approved_amount", 0)))
            m3.metric("Monthly Payment", fmt_currency(result.get("monthly_payment", 0)))
            m4.metric("Risk Probability", fmt_percent(result.get("risk_probability", 0)))

            st.markdown("### LLM Explanation")
            st.markdown(
                f'<div class="info-card">{result.get("llm_explanation", "No explanation available.")}</div>',
                unsafe_allow_html=True,
            )

            explainability = result.get("explainability", [])
            explain_df = pd.DataFrame(explainability)

            if not explain_df.empty:
                st.markdown("### Explainable AI Dashboard")
                fig = px.bar(
                    explain_df.sort_values("abs_impact", ascending=True),
                    x="impact",
                    y="feature",
                    orientation="h",
                    text="impact",
                    title="Top Risk Drivers (SHAP-style view)",
                )
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 3: CREDIT APPLICATION
# =========================================================
with tabs[2]:
    st.subheader("Credit Application")

    c1, c2, c3 = st.columns(3)

    with c1:
        credit_customer_name = st.text_input("Customer Name", value="Jane Doe", key="credit_customer_name")
        credit_product_type = st.selectbox(
            "Product Type",
            ["credit_card", "store_card", "revolving_credit"],
            key="credit_product_type",
        )

    with c2:
        credit_income = st.number_input(
            "Net Monthly Income (R)",
            min_value=0.0,
            value=38000.0,
            step=500.0,
            key="credit_income",
        )
        credit_debt = st.number_input(
            "Existing Debt Payments (R)",
            min_value=0.0,
            value=2500.0,
            step=100.0,
            key="credit_debt",
        )
        credit_score = st.slider("Credit Score", 300, 900, 640, 1, key="credit_score")

    with c3:
        credit_fraud = st.slider("Fraud Score", 0.0, 1.0, 0.05, 0.01, key="credit_fraud")
        credit_dpd = st.number_input("Days Past Due", min_value=0, value=0, step=1, key="credit_dpd")

    if st.button("Run Credit Assessment", key="run_credit_assessment"):
        payload = {
            "customer_name": credit_customer_name,
            "product_type": credit_product_type,
            "net_monthly_income": credit_income,
            "existing_debt_payments": credit_debt,
            "credit_score": credit_score,
            "fraud_score": credit_fraud,
            "days_past_due": int(credit_dpd),
        }

        result = api_post("/api/credit/assess", payload)

        if result.get("error"):
            st.error(result["error"])
        else:
            save_status = result.get("save_status", {})
            if save_status.get("saved"):
                st.markdown(
                    '<div class="success-card">Credit application saved to Azure SQL successfully.</div>',
                    unsafe_allow_html=True,
                )
                clear_all_cache()
            else:
                st.warning(f"Credit calculated, but save failed: {save_status.get('message', 'Unknown error')}")

            m1, m2, m3 = st.columns(3)
            m1.metric("Decision", result.get("final_decision", "N/A"))
            m2.metric("Approved Limit", fmt_currency(result.get("approved_limit", 0)))
            m3.metric("Risk Probability", fmt_percent(result.get("risk_probability", 0)))

            st.markdown("### LLM Explanation")
            st.markdown(
                f'<div class="info-card">{result.get("llm_explanation", "No explanation available.")}</div>',
                unsafe_allow_html=True,
            )

            explainability = result.get("explainability", [])
            explain_df = pd.DataFrame(explainability)

            if not explain_df.empty:
                st.markdown("### Explainable AI Dashboard")
                fig = px.bar(
                    explain_df.sort_values("abs_impact", ascending=True),
                    x="impact",
                    y="feature",
                    orientation="h",
                    text="impact",
                    title="Top Risk Drivers (SHAP-style view)",
                )
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 4: FRAUD MONITORING
# =========================================================
with tabs[3]:
    st.subheader("Real-Time Fraud Monitoring")

    if not fraud_live_df.empty and "alert_level" in fraud_live_df.columns:
        critical = len(fraud_live_df[fraud_live_df["alert_level"] == "Critical"])
    else:
        critical = 0

    if critical > 0:
        st.markdown(
            f'<div class="alert-banner">🚨 Live Alert: {critical} critical fraud event(s) detected. Immediate review recommended.</div>',
            unsafe_allow_html=True,
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Events", len(fraud_live_df))
    col2.metric("Critical", int((fraud_live_df["alert_level"] == "Critical").sum()) if not fraud_live_df.empty and "alert_level" in fraud_live_df.columns else 0)
    col3.metric("High", int((fraud_live_df["alert_level"] == "High").sum()) if not fraud_live_df.empty and "alert_level" in fraud_live_df.columns else 0)
    col4.metric("Average Fraud Score", fmt_percent(fraud_live_df["fraud_score"].mean() if not fraud_live_df.empty and "fraud_score" in fraud_live_df.columns else 0))

    st.markdown("### Alert Level Distribution")
    if not fraud_live_df.empty and "alert_level" in fraud_live_df.columns:
        fig = px.histogram(fraud_live_df, x="alert_level")
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Fraud Score Trend")
    if not fraud_live_df.empty and "fraud_score" in fraud_live_df.columns:
        plot_df = fraud_live_df.copy()
        plot_df = plot_df.reset_index(drop=True)
        fig = px.line(plot_df, x=plot_df.index, y="fraud_score")
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Live Fraud Event Feed")
    if not fraud_live_df.empty:
        st.dataframe(fraud_live_df, use_container_width=True)
    else:
        st.info("No fraud events yet.")


# =========================================================
# TAB 5: RECENT APPLICATIONS
# =========================================================
with tabs[4]:
    st.subheader("Recent Applications")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Recent Loan Applications")
        if not loans_df.empty:
            st.dataframe(loans_df, use_container_width=True)
        else:
            st.info("No recent loan applications.")

    with col2:
        st.markdown("### Recent Credit Applications")
        if not credit_df.empty:
            st.dataframe(credit_df, use_container_width=True)
        else:
            st.info("No recent credit applications.")