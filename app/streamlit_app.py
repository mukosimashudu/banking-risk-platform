from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://banking-risk-app-mukosi.onrender.com"
).rstrip("/")

AUTO_REFRESH_SECONDS = int(os.getenv("AUTO_REFRESH_SECONDS", "30"))


st.set_page_config(
    page_title="Full Fintech Banking Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #050b18;
        color: white;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }
    .metric-box {
        background: #09101f;
        border-radius: 14px;
        padding: 14px 18px;
        border: 1px solid rgba(255,255,255,0.06);
        min-height: 110px;
    }
    .metric-label {
        font-size: 13px;
        color: #cbd5e1;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f8fafc;
    }
    .small-pill {
        display: inline-block;
        margin-top: 8px;
        background: rgba(34,197,94,0.18);
        color: #4ade80;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    .alert-banner {
        background: rgba(239,68,68,0.20);
        border: 1px solid rgba(239,68,68,0.25);
        color: #f87171;
        padding: 14px 16px;
        border-radius: 10px;
        margin-bottom: 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(endpoint: str, timeout: int = 60):
    try:
        r = requests.get(f"{API_BASE_URL}{endpoint}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(endpoint: str, payload: dict, timeout: int = 60):
    try:
        r = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def fmt_currency(v: Any) -> str:
    try:
        return f"R {float(v):,.2f}"
    except Exception:
        return "R 0.00"


def fmt_percent(v: Any) -> str:
    try:
        x = float(v)
        if x <= 1:
            x *= 100
        return f"{x:.2f}%"
    except Exception:
        return "0.00%"


def render_metric(label: str, value: str, pill: str = "") -> None:
    pill_html = f'<div class="small-pill">{pill}</div>' if pill else ""
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {pill_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > AUTO_REFRESH_SECONDS:
    st.session_state.last_refresh = time.time()
    st.rerun()


with st.sidebar:
    st.markdown("## Configuration")
    st.write(f"**API Base URL:** {API_BASE_URL}")
    st.write(f"**Auto refresh target:** {AUTO_REFRESH_SECONDS} sec")
    if st.button("🔄 Refresh now"):
        st.session_state.last_refresh = time.time()
        st.rerun()


st.markdown("# 🏢 Full Fintech Banking Platform")
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

summary = api_get("/api/portfolio/summary")
recent_loans = api_get("/api/portfolio/recent-loans")
recent_credit = api_get("/api/portfolio/recent-credit")
product_dist = api_get("/api/portfolio/product-distribution")
decision_dist = api_get("/api/portfolio/decision-distribution")
fraud_dist = api_get("/api/portfolio/fraud-distribution")
fraud_live = api_get("/api/fraud/live")


with tabs[0]:
    st.subheader("Executive Dashboard")

    if isinstance(summary, dict) and summary.get("critical_alerts", 0) > 0:
        st.markdown(
            f'<div class="alert-banner">🚨 Live Alert: {summary.get("critical_alerts", 0)} critical fraud event(s) detected. Immediate review recommended.</div>',
            unsafe_allow_html=True,
        )

    r1 = st.columns(4)
    with r1[0]:
        render_metric("Total Applications", f'{int(summary.get("total_applications", 0)):,}', "↑ recent")
    with r1[1]:
        render_metric("Approved Cases", f'{int(summary.get("approved_cases", 0)):,}', "↑ recent")
    with r1[2]:
        render_metric("Approval Rate", fmt_percent(summary.get("approval_rate", 0)), "live")
    with r1[3]:
        render_metric("Lifetime ECL", fmt_currency(summary.get("lifetime_ecl", 0)))

    r2 = st.columns(4)
    with r2[0]:
        render_metric("Loan Exposure", fmt_currency(summary.get("loan_exposure", 0)))
    with r2[1]:
        render_metric("Credit Limits", fmt_currency(summary.get("credit_limits", 0)))
    with r2[2]:
        render_metric("Average PD", fmt_percent(summary.get("average_pd", 0)))
    with r2[3]:
        render_metric("Average Fraud Score", fmt_percent(summary.get("average_fraud_score", 0)))

    r3 = st.columns(4)
    with r3[0]:
        render_metric("Critical Alerts", str(int(summary.get("critical_alerts", 0))))
    with r3[1]:
        render_metric("High Alerts", str(int(summary.get("high_alerts", 0))))
    with r3[2]:
        render_metric("Average SHAP Risk", fmt_percent(summary.get("average_shap_risk", 0)))
    with r3[3]:
        render_metric("Data Refresh", time.strftime("%H:%M:%S"))

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Product Distribution")
        pdf = pd.DataFrame(product_dist if isinstance(product_dist, list) else [])
        if not pdf.empty:
            fig = px.bar(pdf, x="product", y="count")
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product distribution yet.")

        st.markdown("### Decision Distribution")
        ddf = pd.DataFrame(decision_dist if isinstance(decision_dist, list) else [])
        if not ddf.empty:
            fig = px.pie(ddf, names="decision", values="count", hole=0.45)
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No decision distribution yet.")

    with c2:
        st.markdown("### Fraud Alert Distribution")
        fdf = pd.DataFrame(fraud_dist if isinstance(fraud_dist, list) else [])
        if not fdf.empty:
            fig = px.bar(fdf, x="alert_level", y="count")
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fraud distribution yet.")


with tabs[1]:
    st.subheader("Loan Application")

    c1, c2, c3 = st.columns(3)

    with c1:
        customer_name = st.text_input("Customer Name", value="John Doe", key="loan_customer_name")
        product_type = st.selectbox(
            "Product Type",
            ["personal_loan", "vehicle_finance", "home_loan", "business_loan"],
            key="loan_product_type",
        )
        requested_amount = st.number_input("Requested Amount (R)", min_value=0.0, value=150000.0, step=1000.0)
        annual_interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=15.5, step=0.1)
        term_months = st.number_input("Term (Months)", min_value=1, value=60, step=1)
        affordability_factor = st.slider("Affordability Factor", 0.10, 1.00, 0.70, 0.01)

    with c2:
        net_monthly_income = st.number_input("Net Monthly Income (R)", min_value=0.0, value=35000.0, step=500.0)
        monthly_expenses = st.number_input("Monthly Expenses (R)", min_value=0.0, value=12000.0, step=500.0)
        existing_debt_payments = st.number_input("Existing Debt Payments (R)", min_value=0.0, value=3500.0, step=100.0)
        credit_score = st.slider("Credit Score", 300, 900, 680, 1)
        fraud_score = st.slider("Fraud Score", 0.0, 1.0, 0.08, 0.01)
        debt_to_income_cap = st.slider("Debt-to-Income Cap", 0.10, 1.00, 0.45, 0.01)

    with c3:
        property_value = st.number_input("Property Value (R)", min_value=0.0, value=0.0, step=1000.0)
        deposit = st.number_input("Deposit (R)", min_value=0.0, value=0.0, step=1000.0)
        secured = st.checkbox("Secured Loan", value=False)
        days_past_due = st.number_input("Days Past Due", min_value=0, value=0, step=1)
        sicr_flag = st.checkbox("Significant Increase in Credit Risk (SICR)", value=False)
        default_flag = st.checkbox("Default Flag", value=False)
        stress_rate_addon = st.slider("Stress Rate Add-on (%)", 0.0, 10.0, 2.0, 0.1)

    if st.button("Run Loan Assessment", key="run_loan_assessment"):
        payload = {
            "customer_name": customer_name,
            "product_type": product_type,
            "requested_amount": requested_amount,
            "annual_interest_rate": annual_interest_rate,
            "term_months": int(term_months),
            "net_monthly_income": net_monthly_income,
            "monthly_expenses": monthly_expenses,
            "existing_debt_payments": existing_debt_payments,
            "credit_score": credit_score,
            "fraud_score": fraud_score,
            "property_value": property_value,
            "deposit": deposit,
            "secured": secured,
            "days_past_due": int(days_past_due),
            "sicr_flag": sicr_flag,
            "default_flag": default_flag,
            "affordability_factor": affordability_factor,
            "debt_to_income_cap": debt_to_income_cap,
            "stress_rate_addon": stress_rate_addon,
        }

        result = api_post("/api/loan/assess", payload)

        if result.get("error"):
            st.error(result["error"])
        else:
            st.success("Loan application saved to Azure SQL successfully.")
            a, b, c, d = st.columns(4)
            a.metric("Decision", result.get("final_decision", "N/A"))
            b.metric("Approved Amount", fmt_currency(result.get("approved_amount", 0)))
            c.metric("Monthly Payment", fmt_currency(result.get("monthly_payment", 0)))
            d.metric("Risk Probability", fmt_percent(result.get("risk_probability", 0)))

            st.markdown("### LLM Explanation")
            st.info(result.get("llm_explanation", "No explanation available."))

            shap_df = pd.DataFrame(result.get("top_shap_features", []))
            if not shap_df.empty:
                fig = px.bar(shap_df, x="shap_value", y="feature", orientation="h", text="shap_value")
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)


with tabs[2]:
    st.subheader("Credit Application")

    c1, c2, c3 = st.columns(3)

    with c1:
        c_customer_name = st.text_input("Customer Name", value="Jane Doe", key="credit_customer_name")
        c_product_type = st.selectbox(
            "Product Type",
            ["credit_card", "store_card", "revolving_credit"],
            key="credit_product_type",
        )

    with c2:
        c_income = st.number_input("Net Monthly Income (R)", min_value=0.0, value=28000.0, step=500.0, key="credit_income")
        c_debt = st.number_input("Existing Debt Payments (R)", min_value=0.0, value=2500.0, step=100.0, key="credit_debt")
        c_score = st.slider("Credit Score", 300, 900, 640, 1, key="credit_score_slider")

    with c3:
        c_fraud = st.slider("Fraud Score", 0.0, 1.0, 0.05, 0.01, key="credit_fraud")
        c_dpd = st.number_input("Days Past Due", min_value=0, value=0, step=1, key="credit_dpd")
        c_sicr = st.checkbox("SICR", value=False, key="credit_sicr")
        c_default = st.checkbox("Default Flag", value=False, key="credit_default")

    if st.button("Run Credit Assessment", key="run_credit_assessment"):
        payload = {
            "customer_name": c_customer_name,
            "product_type": c_product_type,
            "net_monthly_income": c_income,
            "existing_debt_payments": c_debt,
            "credit_score": c_score,
            "fraud_score": c_fraud,
            "days_past_due": int(c_dpd),
            "sicr_flag": c_sicr,
            "default_flag": c_default,
        }

        result = api_post("/api/credit/assess", payload)

        if result.get("error"):
            st.error(result["error"])
        else:
            st.success("Credit application saved to Azure SQL successfully.")
            a, b, c, d = st.columns(4)
            a.metric("Decision", result.get("final_decision", "N/A"))
            b.metric("Approved Limit", fmt_currency(result.get("approved_limit", 0)))
            c.metric("Risk Probability", fmt_percent(result.get("risk_probability", 0)))
            d.metric("IFRS 9 Stage", result.get("ifrs9_stage", "N/A"))

            st.markdown("### LLM Explanation")
            st.info(result.get("llm_explanation", "No explanation available."))

            shap_df = pd.DataFrame(result.get("top_shap_features", []))
            if not shap_df.empty:
                fig = px.bar(shap_df, x="shap_value", y="feature", orientation="h", text="shap_value")
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)


with tabs[3]:
    st.subheader("Real-Time Fraud Monitoring")

    if isinstance(summary, dict) and summary.get("critical_alerts", 0) > 0:
        st.markdown(
            f'<div class="alert-banner">🚨 Live Alert: {summary.get("critical_alerts", 0)} critical fraud event(s) detected. Immediate review recommended.</div>',
            unsafe_allow_html=True,
        )

    live_df = pd.DataFrame(fraud_live if isinstance(fraud_live, list) else [])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Events", len(live_df))
    k2.metric("Critical", int((live_df["alert_level"] == "Critical").sum()) if not live_df.empty else 0)
    k3.metric("High", int((live_df["alert_level"] == "High").sum()) if not live_df.empty else 0)
    k4.metric("Average Fraud Score", fmt_percent(live_df["fraud_score"].mean()) if not live_df.empty else 0)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Alert Level Distribution")
        if not live_df.empty:
            chart_df = live_df.groupby("alert_level").size().reset_index(name="count")
            fig = px.bar(chart_df, x="alert_level", y="count")
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Fraud Score Trend")
        if not live_df.empty:
            live_df["event_time"] = pd.to_datetime(live_df["event_time"], errors="coerce")
            trend_df = live_df.sort_values("event_time")
            fig = px.line(trend_df, x="event_time", y="fraud_score")
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Live Fraud Event Feed")
    if not live_df.empty:
        st.dataframe(live_df, use_container_width=True)
    else:
        st.info("No live fraud events.")


with tabs[4]:
    st.subheader("Recent Applications")

    left, right = st.columns(2)

    with left:
        st.markdown("### Recent Loan Applications")
        loans_df = pd.DataFrame(recent_loans if isinstance(recent_loans, list) else [])
        if not loans_df.empty:
            st.dataframe(loans_df, use_container_width=True)
        else:
            st.info("No recent loan applications.")

    with right:
        st.markdown("### Recent Credit Applications")
        credit_df = pd.DataFrame(recent_credit if isinstance(recent_credit, list) else [])
        if not credit_df.empty:
            st.dataframe(credit_df, use_container_width=True)
        else:
            st.info("No recent credit applications.")