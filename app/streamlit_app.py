from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_BASE_URL", "http://localhost:8001")

st.set_page_config(page_title="Full Fintech Banking Platform", page_icon="🏦", layout="wide")


def money(value: float) -> str:
    return f"R {value:,.2f}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: Dict[str, Any]):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def plot_horizontal_bar(df: pd.DataFrame, category_col: str, value_col: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df[category_col], df[value_col])
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


st.title("🏦 Full Fintech Banking Platform")
st.caption("Loans • Credit Cards • Vehicle Finance • Home Loans • IFRS 9 • Explainable AI • Fraud Monitoring")

with st.sidebar:
    st.header("Configuration")
    st.write(f"API Base URL: {API_URL}")

tabs = st.tabs(
    [
        "Executive Dashboard",
        "Loan Application",
        "Credit Application",
        "Fraud Monitoring",
        "Recent Applications",
    ]
)

with tabs[0]:
    st.subheader("Executive Dashboard")

    try:
        summary = api_get("/api/portfolio/summary")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Applications", f"{summary['total_applications']:,}")
        k2.metric("Approved Cases", f"{summary['total_approved_cases']:,}")
        k3.metric("Approval Rate", pct(summary["approval_rate"]))
        k4.metric("Lifetime ECL", money(summary["total_lifetime_ecl"]))

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Loan Exposure", money(summary["total_approved_amount"]))
        k6.metric("Credit Limits", money(summary["total_credit_limit"]))
        k7.metric("Average PD", pct(summary["average_pd_12m"]))
        k8.metric("Average Fraud Score", pct(summary["average_fraud_score"]))

        k9, k10 = st.columns(2)
        k9.metric("Critical Alerts", f"{summary['critical_alerts']:,}")
        k10.metric("High Alerts", f"{summary['high_alerts']:,}")

        left, right = st.columns(2)

        with left:
            st.markdown("### Product Distribution")
            product_df = pd.DataFrame(summary.get("product_distribution", []))
            if not product_df.empty:
                st.bar_chart(product_df.set_index("product")["count"])
            else:
                st.info("No product distribution yet.")

            st.markdown("### Decision Distribution")
            decision_df = pd.DataFrame(summary.get("decision_distribution", []))
            if not decision_df.empty:
                st.bar_chart(decision_df.set_index("decision")["count"])
            else:
                st.info("No decision distribution yet.")

        with right:
            st.markdown("### Fraud Alert Distribution")
            fraud_df = pd.DataFrame(summary.get("fraud_distribution", []))
            if not fraud_df.empty:
                st.bar_chart(fraud_df.set_index("alert_level")["count"])
            else:
                st.info("No fraud alerts yet.")

            st.markdown("### Average SHAP Risk")
            st.metric("Average SHAP Risk Probability", pct(summary["average_shap_risk_probability"]))

    except Exception as exc:
        st.error(f"Could not load executive dashboard: {exc}")

with tabs[1]:
    st.subheader("Loan Application")

    c1, c2, c3 = st.columns(3)

    with c1:
        customer_name = st.text_input("Customer Name", value="John Doe")
        product_type = st.selectbox("Product Type", ["personal_loan", "home_loan", "vehicle_loan", "credit_card"])
        requested_amount = st.number_input("Requested Amount (R)", min_value=1000.0, value=150000.0, step=1000.0)
        annual_interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=15.5, step=0.1)
        term_months = st.number_input("Term (Months)", min_value=1, value=60, step=1)

    with c2:
        net_monthly_income = st.number_input("Net Monthly Income (R)", min_value=0.0, value=35000.0, step=500.0)
        monthly_expenses = st.number_input("Monthly Expenses (R)", min_value=0.0, value=12000.0, step=500.0)
        existing_debt_payments = st.number_input("Existing Debt Payments (R)", min_value=0.0, value=3500.0, step=100.0)
        credit_score = st.slider("Credit Score", 300, 900, 680)
        fraud_score = st.slider("Fraud Score", 0.0, 1.0, 0.08, 0.01)

    with c3:
        property_value = st.number_input("Property Value (R)", min_value=0.0, value=0.0, step=10000.0)
        deposit = st.number_input("Deposit (R)", min_value=0.0, value=0.0, step=1000.0)
        secured = st.checkbox("Secured Loan", value=False)
        days_past_due = st.number_input("Days Past Due", min_value=0, value=0, step=1)
        sicr_flag = st.checkbox("Significant Increase in Credit Risk (SICR)", value=False)
        default_flag = st.checkbox("Default Flag", value=False)

    p1, p2, p3 = st.columns(3)
    with p1:
        affordability_factor = st.slider("Affordability Factor", 0.10, 1.00, 0.70, 0.01)
    with p2:
        debt_to_income_cap = st.slider("Debt-to-Income Cap", 0.10, 1.00, 0.45, 0.01)
    with p3:
        stress_rate_addon = st.slider("Stress Rate Add-on (%)", 0.0, 10.0, 2.0, 0.1)

    if st.button("Run Loan Assessment", type="primary"):
        payload = {
            "customer_name": customer_name,
            "product_type": product_type,
            "requested_amount": requested_amount,
            "annual_interest_rate": annual_interest_rate,
            "term_months": int(term_months),
            "net_monthly_income": net_monthly_income,
            "monthly_expenses": monthly_expenses,
            "existing_debt_payments": existing_debt_payments,
            "credit_score": int(credit_score),
            "fraud_score": fraud_score,
            "property_value": property_value if property_value > 0 else None,
            "deposit": deposit if deposit > 0 else None,
            "secured": secured,
            "days_past_due": int(days_past_due),
            "sicr_flag": sicr_flag,
            "default_flag": default_flag,
            "affordability_factor": affordability_factor,
            "debt_to_income_cap": debt_to_income_cap,
            "stress_rate_addon": stress_rate_addon,
        }

        try:
            result = api_post("/api/loan/assess", payload)

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Final Decision", result["final_decision"])
            r2.metric("Approved Amount", money(result["approved_amount"]))
            r3.metric("Monthly Payment", money(result["monthly_payment"]))
            r4.metric("Lifetime ECL", money(result["ecl_lifetime"]))

            st.info(result["decision_reason"])

            rr1, rr2, rr3, rr4 = st.columns(4)
            rr1.metric("Disposable Income", money(result["disposable_income"]))
            rr2.metric("DTI Ratio", pct(result["debt_to_income_ratio"]))
            rr3.metric("IFRS 9 Stage", result["ifrs9_stage"])
            rr4.metric("Fraud Alert", result["fraud_event"]["alert_level"])

            st.markdown("### Credit Narrative")
            st.write(result["llm_explanation"])

            st.markdown("### Explainable AI")
            shap_block = result.get("shap_explanation", {})
            if shap_block.get("available"):
                st.metric("Risk Probability", pct(shap_block["risk_probability"]))
                shap_df = pd.DataFrame(shap_block.get("top_features", []))
                if not shap_df.empty:
                    plot_horizontal_bar(shap_df.sort_values("abs_impact"), "feature", "shap_value", "Top Risk Drivers")
                    st.dataframe(shap_df, use_container_width=True)

            st.markdown("### Fraud Event")
            st.json(result["fraud_event"])

            schedule_df = pd.DataFrame(result["amortisation_schedule"])
            if not schedule_df.empty:
                st.markdown("### Amortisation Schedule")
                st.dataframe(schedule_df, use_container_width=True)
                st.line_chart(schedule_df.set_index("instalment_no")[["opening_balance", "closing_balance"]])

        except Exception as exc:
            st.error(f"Loan assessment failed: {exc}")

with tabs[2]:
    st.subheader("Credit Application")

    cc1, cc2 = st.columns(2)

    with cc1:
        cc_customer = st.text_input("Customer Name", value="Jane Credit")
        cc_product = st.selectbox("Credit Product", ["credit_card", "credit_line"])
        cc_income = st.number_input("Net Monthly Income (R)", min_value=0.0, value=30000.0, step=500.0)
        cc_debt = st.number_input("Existing Debt Payments (R)", min_value=0.0, value=2500.0, step=100.0)

    with cc2:
        cc_score = st.slider("Credit Score", 300, 900, 650, key="cc_score")

    if st.button("Run Credit Assessment", type="primary"):
        payload = {
            "customer_name": cc_customer,
            "product_type": cc_product,
            "net_monthly_income": cc_income,
            "existing_debt_payments": cc_debt,
            "credit_score": int(cc_score),
        }

        try:
            result = api_post("/api/credit/assess", payload)

            x1, x2, x3 = st.columns(3)
            x1.metric("Final Decision", result["final_decision"])
            x2.metric("Approved Limit", money(result["approved_limit"]))
            x3.metric("Risk Probability", pct(result["risk_probability"]))

            st.info(result["decision_reason"])

            st.markdown("### Credit Narrative")
            st.write(result["llm_explanation"])

        except Exception as exc:
            st.error(f"Credit assessment failed: {exc}")

with tabs[3]:
    st.subheader("Real-Time Fraud Monitoring")

    try:
        fraud_data = api_get("/api/fraud/recent")
        fraud_df = pd.DataFrame(fraud_data)

        if fraud_df.empty:
            st.info("No fraud events yet.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Events", f"{len(fraud_df):,}")
            a2.metric("Critical", f"{(fraud_df['alert_level'] == 'Critical').sum():,}")
            a3.metric("High", f"{(fraud_df['alert_level'] == 'High').sum():,}")

            st.markdown("### Alert Level Distribution")
            st.bar_chart(fraud_df["alert_level"].value_counts())

            st.markdown("### Fraud Score Trend")
            fraud_df["event_time"] = pd.to_datetime(fraud_df["event_time"])
            fraud_df = fraud_df.sort_values("event_time")
            st.line_chart(fraud_df.set_index("event_time")[["fraud_score"]])

            st.markdown("### Live Fraud Event Feed")
            st.dataframe(fraud_df, use_container_width=True)

            high_risk = fraud_df[fraud_df["alert_level"].isin(["Critical", "High"])]
            if not high_risk.empty:
                st.markdown("### Immediate Alerts")
                st.dataframe(high_risk, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not load fraud dashboard: {exc}")

with tabs[4]:
    st.subheader("Recent Applications")

    left, right = st.columns(2)

    with left:
        st.markdown("### Recent Loan Applications")
        try:
            recent_loans = api_get("/api/portfolio/recent")
            loans_df = pd.DataFrame(recent_loans)
            if loans_df.empty:
                st.info("No recent loan applications.")
            else:
                st.dataframe(loans_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load recent loan applications: {exc}")

    with right:
        st.markdown("### Recent Credit Applications")
        try:
            recent_credit = api_get("/api/credit/recent")
            credit_df = pd.DataFrame(recent_credit)
            if credit_df.empty:
                st.info("No recent credit applications.")
            else:
                st.dataframe(credit_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load recent credit applications: {exc}")