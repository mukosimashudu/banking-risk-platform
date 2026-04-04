from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
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
    layout="wide"
)


# ---------------------------
# Helpers
# ---------------------------
def money(value: float | int | None) -> str:
    if value is None:
        return "R 0.00"
    try:
        return f"R {float(value):,.2f}"
    except Exception:
        return "R 0.00"


def pct(value: float | int | None) -> str:
    if value is None:
        return "0.00%"
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def normalise_records(data: Any) -> list[dict]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "data", "results", "records"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    return []


def api_get(path: str):
    url = f"{API_BASE_URL}{path}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: Dict[str, Any]):
    url = f"{API_BASE_URL}{path}"
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def safe_api_get(path: str, default: Any):
    try:
        return api_get(path)
    except Exception:
        return default


def safe_api_post(path: str, payload: Dict[str, Any], default: Any):
    try:
        return api_post(path, payload)
    except Exception:
        return default


def plot_horizontal_bar(df: pd.DataFrame, category_col: str, value_col: str, title: str):
    plot_df = df.copy()
    if plot_df.empty or category_col not in plot_df.columns or value_col not in plot_df.columns:
        st.info("No explainability data available.")
        return

    plot_df = plot_df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(plot_df[category_col], plot_df[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel(category_col)
    st.pyplot(fig)
    plt.close(fig)


def render_alert_banner(alert_df: pd.DataFrame):
    if alert_df.empty or "alert_level" not in alert_df.columns:
        st.success("No critical fraud alerts at the moment.")
        return

    critical_count = (alert_df["alert_level"].astype(str).str.upper() == "CRITICAL").sum()
    high_count = (alert_df["alert_level"].astype(str).str.upper() == "HIGH").sum()

    if critical_count > 0:
        st.error(
            f"🚨 Live Alert: {critical_count} critical fraud event(s) detected. "
            f"Immediate review recommended."
        )
    elif high_count > 0:
        st.warning(
            f"⚠️ Live Alert: {high_count} high-risk fraud event(s) detected. "
            f"Please review urgently."
        )
    else:
        st.info("Fraud monitoring is active. No critical or high alerts right now.")


def coerce_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def recent_delta_text(current_value: float | int, base_value: float | int) -> str:
    try:
        current_value = float(current_value)
        base_value = float(base_value)
        if base_value == 0:
            return "n/a"
        delta = ((current_value - base_value) / base_value) * 100
        return f"{delta:+.1f}%"
    except Exception:
        return "n/a"


def build_summary_defaults() -> dict:
    return {
        "total_applications": 0,
        "total_approved_cases": 0,
        "approval_rate": 0.0,
        "total_lifetime_ecl": 0.0,
        "total_approved_amount": 0.0,
        "total_credit_limit": 0.0,
        "average_pd_12m": 0.0,
        "average_fraud_score": 0.0,
        "critical_alerts": 0,
        "high_alerts": 0,
        "average_shap_risk_probability": 0.0,
        "product_distribution": [],
        "decision_distribution": [],
        "fraud_distribution": [],
    }


# ---------------------------
# App header
# ---------------------------
st.title("🏦 Full Fintech Banking Platform")
st.caption(
    "Loans • Credit Cards • Vehicle Finance • Home Loans • IFRS 9 • Explainable AI • Fraud Monitoring"
)

with st.sidebar:
    st.header("Configuration")
    st.write(f"API Base URL: {API_BASE_URL}")
    st.write(f"Auto refresh target: {AUTO_REFRESH_SECONDS} sec")
    if st.button("🔄 Refresh now"):
        st.rerun()

tabs = st.tabs(
    [
        "Executive Dashboard",
        "Loan Application",
        "Credit Application",
        "Fraud Monitoring",
        "Recent Applications",
    ]
)

# ---------------------------
# Executive Dashboard
# ---------------------------
with tabs[0]:
    st.subheader("Executive Dashboard")

    try:
        summary = safe_api_get("/api/portfolio/summary", build_summary_defaults())
        fraud_data = normalise_records(safe_api_get("/api/fraud/recent", []))
        recent_loans = normalise_records(safe_api_get("/api/portfolio/recent", []))
        recent_credit = normalise_records(safe_api_get("/api/credit/recent", []))

        fraud_df = pd.DataFrame(fraud_data)
        loans_df = pd.DataFrame(recent_loans)
        credit_df = pd.DataFrame(recent_credit)

        if not fraud_df.empty:
            fraud_df = coerce_datetime(fraud_df, ["event_time", "created_at"])

        if not loans_df.empty:
            loans_df = coerce_datetime(loans_df, ["created_at"])

        if not credit_df.empty:
            credit_df = coerce_datetime(credit_df, ["created_at"])

        if not fraud_df.empty and "alert_level" in fraud_df.columns:
            render_alert_banner(fraud_df)
        else:
            st.info("Fraud monitoring feed is available but no alert rows were returned yet.")

        total_applications = safe_int(summary.get("total_applications", 0))
        total_approved_cases = safe_int(summary.get("total_approved_cases", 0))
        approval_rate = safe_float(summary.get("approval_rate", 0))
        total_lifetime_ecl = safe_float(summary.get("total_lifetime_ecl", 0))

        total_approved_amount = safe_float(summary.get("total_approved_amount", 0))
        total_credit_limit = safe_float(summary.get("total_credit_limit", 0))
        average_pd_12m = safe_float(summary.get("average_pd_12m", 0))
        average_fraud_score = safe_float(summary.get("average_fraud_score", 0))

        critical_alerts = safe_int(summary.get("critical_alerts", 0))
        high_alerts = safe_int(summary.get("high_alerts", 0))
        avg_shap = safe_float(summary.get("average_shap_risk_probability", 0))

        total_recent = len(loans_df) + len(credit_df)
        approved_recent_loans = 0
        approved_recent_credit = 0

        if not loans_df.empty and "final_decision" in loans_df.columns:
            approved_recent_loans = loans_df["final_decision"].astype(str).str.upper().isin(
                ["APPROVE", "APPROVED"]
            ).sum()

        if not credit_df.empty and "final_decision" in credit_df.columns:
            approved_recent_credit = credit_df["final_decision"].astype(str).str.upper().isin(
                ["APPROVE", "APPROVED"]
            ).sum()

        approved_recent_total = approved_recent_loans + approved_recent_credit
        recent_approval_rate = (approved_recent_total / total_recent) if total_recent > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Applications", f"{total_applications:,}", delta=f"{total_recent:,} recent")
        c2.metric("Approved Cases", f"{total_approved_cases:,}", delta=f"{approved_recent_total:,} recent")
        c3.metric(
            "Approval Rate",
            pct(approval_rate),
            delta=recent_delta_text(
                approval_rate,
                recent_approval_rate if recent_approval_rate > 0 else approval_rate
            )
        )
        c4.metric("Lifetime ECL", money(total_lifetime_ecl))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Loan Exposure", money(total_approved_amount))
        c6.metric("Credit Limits", money(total_credit_limit))
        c7.metric("Average PD", pct(average_pd_12m))
        c8.metric("Average Fraud Score", pct(average_fraud_score))

        c9, c10, c11, c12 = st.columns(4)
        c9.metric("Critical Alerts", f"{critical_alerts:,}")
        c10.metric("High Alerts", f"{high_alerts:,}")
        c11.metric("Average SHAP Risk", pct(avg_shap))
        c12.metric("Data Refresh", datetime.now().strftime("%H:%M:%S"))

        left, right = st.columns(2)

        with left:
            st.markdown("### Product Distribution")
            product_df = pd.DataFrame(summary.get("product_distribution", []))
            if not product_df.empty and {"product", "count"}.issubset(product_df.columns):
                st.bar_chart(product_df.set_index("product")["count"])
            else:
                st.info("No product distribution yet.")

            st.markdown("### Decision Distribution")
            decision_df = pd.DataFrame(summary.get("decision_distribution", []))
            if not decision_df.empty and {"decision", "count"}.issubset(decision_df.columns):
                st.bar_chart(decision_df.set_index("decision")["count"])
            else:
                st.info("No decision distribution yet.")

        with right:
            st.markdown("### Fraud Alert Distribution")
            fraud_dist_df = pd.DataFrame(summary.get("fraud_distribution", []))
            if not fraud_dist_df.empty and {"alert_level", "count"}.issubset(fraud_dist_df.columns):
                st.bar_chart(fraud_dist_df.set_index("alert_level")["count"])
            elif not fraud_df.empty and "alert_level" in fraud_df.columns:
                st.bar_chart(fraud_df["alert_level"].value_counts())
            else:
                st.info("No fraud alerts yet.")

            st.markdown("### Recent Approval Split")
            if total_recent > 0:
                pie_like = pd.DataFrame(
                    {
                        "category": ["Approved", "Not Approved"],
                        "count": [approved_recent_total, max(total_recent - approved_recent_total, 0)]
                    }
                )
                st.bar_chart(pie_like.set_index("category")["count"])
            else:
                st.info("No recent applications yet.")

        st.markdown("### Interview-Ready Commentary")
        commentary = []
        commentary.append(
            f"- Total book under management combines approved loan exposure ({money(total_approved_amount)}) "
            f"and granted credit limits ({money(total_credit_limit)})."
        )
        commentary.append(
            f"- Current portfolio approval rate is {pct(approval_rate)}, while average 12-month PD is {pct(average_pd_12m)}."
        )
        commentary.append(
            f"- Fraud environment remains under watch with {critical_alerts} critical and {high_alerts} high alerts."
        )
        commentary.append(
            f"- Average explainability risk output is {pct(avg_shap)}, useful for credit committee and audit discussions."
        )
        st.markdown("\n".join(commentary))

    except Exception as exc:
        st.error(f"Could not load executive dashboard: {exc}")


# ---------------------------
# Loan Application
# ---------------------------
with tabs[1]:
    st.subheader("Loan Application")

    c1, c2, c3 = st.columns(3)

    with c1:
        customer_name = st.text_input("Customer Name", value="John Doe")
        product_type = st.selectbox(
            "Product Type",
            ["personal_loan", "home_loan", "vehicle_loan", "credit_card"]
        )
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
            r1.metric("Final Decision", result.get("final_decision", "N/A"))
            r2.metric("Approved Amount", money(result.get("approved_amount")))
            r3.metric("Monthly Payment", money(result.get("monthly_payment")))
            r4.metric("Lifetime ECL", money(result.get("ecl_lifetime")))

            st.info(result.get("decision_reason", "No decision reason returned."))

            rr1, rr2, rr3, rr4 = st.columns(4)
            rr1.metric("Disposable Income", money(result.get("disposable_income")))
            rr2.metric("DTI Ratio", pct(result.get("debt_to_income_ratio")))
            rr3.metric("IFRS 9 Stage", str(result.get("ifrs9_stage", "N/A")))
            fraud_event = result.get("fraud_event", {})
            rr4.metric("Fraud Alert", str(fraud_event.get("alert_level", "N/A")))

            st.markdown("### Credit Narrative")
            st.write(result.get("llm_explanation", "No narrative returned."))

            st.markdown("### Explainable AI")
            shap_block = result.get("shap_explanation", {})
            if shap_block.get("available"):
                st.metric("Risk Probability", pct(shap_block.get("risk_probability")))
                shap_df = pd.DataFrame(shap_block.get("top_features", []))
                if not shap_df.empty:
                    if "abs_impact" not in shap_df.columns and "shap_value" in shap_df.columns:
                        shap_df["abs_impact"] = shap_df["shap_value"].abs()
                    plot_horizontal_bar(
                        shap_df,
                        "feature",
                        "shap_value",
                        "Top Risk Drivers"
                    )
                    st.dataframe(shap_df, use_container_width=True)
            else:
                st.info("Explainability output not available for this decision.")

            st.markdown("### Fraud Event")
            st.json(fraud_event)

            schedule_df = pd.DataFrame(result.get("amortisation_schedule", []))
            if not schedule_df.empty:
                st.markdown("### Amortisation Schedule")
                st.dataframe(schedule_df, use_container_width=True)

                chart_cols = [c for c in ["opening_balance", "closing_balance"] if c in schedule_df.columns]
                if "instalment_no" in schedule_df.columns and chart_cols:
                    st.line_chart(schedule_df.set_index("instalment_no")[chart_cols])

        except Exception as exc:
            st.error(f"Loan assessment failed: {exc}")


# ---------------------------
# Credit Application
# ---------------------------
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

    if st.button("Run Credit Assessment", type="primary", key="credit_assess"):
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
            x1.metric("Final Decision", result.get("final_decision", "N/A"))
            x2.metric("Approved Limit", money(result.get("approved_limit")))
            x3.metric("Risk Probability", pct(result.get("risk_probability")))

            st.info(result.get("decision_reason", "No decision reason returned."))

            st.markdown("### Credit Narrative")
            st.write(result.get("llm_explanation", "No narrative returned."))

        except Exception as exc:
            st.error(f"Credit assessment failed: {exc}")


# ---------------------------
# Fraud Monitoring
# ---------------------------
with tabs[3]:
    st.subheader("Real-Time Fraud Monitoring")

    try:
        fraud_data = normalise_records(safe_api_get("/api/fraud/recent", []))
        fraud_df = pd.DataFrame(fraud_data)

        if fraud_df.empty:
            st.info("No fraud events yet.")
        else:
            fraud_df = coerce_datetime(fraud_df, ["event_time", "created_at"])

            if "alert_level" in fraud_df.columns:
                render_alert_banner(fraud_df)

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Events", f"{len(fraud_df):,}")
            a2.metric(
                "Critical",
                f"{(fraud_df['alert_level'].astype(str).str.upper() == 'CRITICAL').sum():,}"
                if "alert_level" in fraud_df.columns else "0"
            )
            a3.metric(
                "High",
                f"{(fraud_df['alert_level'].astype(str).str.upper() == 'HIGH').sum():,}"
                if "alert_level" in fraud_df.columns else "0"
            )
            a4.metric(
                "Average Fraud Score",
                pct(fraud_df["fraud_score"].astype(float).mean())
                if "fraud_score" in fraud_df.columns else 0
            )

            left, right = st.columns(2)

            with left:
                st.markdown("### Alert Level Distribution")
                if "alert_level" in fraud_df.columns:
                    st.bar_chart(fraud_df["alert_level"].value_counts())
                else:
                    st.info("No alert levels available.")

            with right:
                st.markdown("### Fraud Score Trend")
                time_col = "event_time" if "event_time" in fraud_df.columns else "created_at"
                if time_col in fraud_df.columns and "fraud_score" in fraud_df.columns:
                    trend_df = fraud_df[[time_col, "fraud_score"]].dropna().sort_values(time_col)
                    if not trend_df.empty:
                        st.line_chart(trend_df.set_index(time_col)[["fraud_score"]])
                    else:
                        st.info("No trend data available.")
                else:
                    st.info("No fraud trend data available.")

            st.markdown("### Live Fraud Event Feed")
            st.dataframe(fraud_df, use_container_width=True)

            if "alert_level" in fraud_df.columns:
                high_risk = fraud_df[
                    fraud_df["alert_level"].astype(str).str.upper().isin(["CRITICAL", "HIGH"])
                ]
                if not high_risk.empty:
                    st.markdown("### Immediate Alerts")
                    st.dataframe(high_risk, use_container_width=True)

    except Exception as exc:
        st.error(f"Could not load fraud dashboard: {exc}")


# ---------------------------
# Recent Applications
# ---------------------------
with tabs[4]:
    st.subheader("Recent Applications")

    left, right = st.columns(2)

    with left:
        st.markdown("### Recent Loan Applications")
        try:
            recent_loans = normalise_records(safe_api_get("/api/portfolio/recent", []))
            loans_df = pd.DataFrame(recent_loans)
            if loans_df.empty:
                st.info("No recent loan applications.")
            else:
                loans_df = coerce_datetime(loans_df, ["created_at"])
                st.dataframe(loans_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load recent loan applications: {exc}")

    with right:
        st.markdown("### Recent Credit Applications")
        try:
            recent_credit = normalise_records(safe_api_get("/api/credit/recent", []))
            credit_df = pd.DataFrame(recent_credit)
            if credit_df.empty:
                st.info("No recent credit applications.")
            else:
                credit_df = coerce_datetime(credit_df, ["created_at"])
                st.dataframe(credit_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load recent credit applications: {exc}")