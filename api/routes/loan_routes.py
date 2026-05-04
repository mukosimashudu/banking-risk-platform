from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.config.db import engine
from src.llm.llm_assistant import generate_explanation


router = APIRouter(prefix="/api/loan", tags=["Loan Assessment"])


# =========================================================
# REQUEST MODEL
# =========================================================
class LoanRequest(BaseModel):
    customer_name: str = Field(..., min_length=1)
    product_type: str = "personal_loan"
    requested_amount: float
    annual_interest_rate: float = 15.5
    term_months: int = 60
    net_monthly_income: float
    monthly_expenses: float = 0.0
    existing_debt_payments: float = 0.0
    credit_score: int
    fraud_score: float = 0.05
    property_value: float = 0.0
    deposit: float = 0.0
    secured: bool = False
    days_past_due: int = 0
    sicr_flag: bool = False
    default_flag: bool = False
    affordability_factor: float = 0.70
    debt_to_income_cap: float = 0.45
    stress_rate_addon: float = 2.0


# =========================================================
# HELPERS
# =========================================================
def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def make_reference(prefix: str) -> str:
    return f"{prefix}-{str(uuid.uuid4())[:8].upper()}"


# =========================================================
# FRAUD + IFRS 9 LOGIC
# =========================================================
def normalize_fraud_score(score: float) -> float:
    score = to_float(score)
    return max(0.0, min(1.0, score))


def fraud_alert_from_score(score: float) -> str:
    score = normalize_fraud_score(score)

    if score >= 0.90:
        return "Critical"
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def fraud_review_required(score: float) -> bool:
    return normalize_fraud_score(score) >= 0.75


def fraud_decline_required(score: float) -> bool:
    return normalize_fraud_score(score) >= 0.90


def fraud_risk_component(score: float) -> float:
    score = normalize_fraud_score(score)

    if score < 0.45:
        return round(score * 0.30, 4)
    if score < 0.75:
        return round(0.135 + ((score - 0.45) / 0.30) * 0.20, 4)
    if score < 0.90:
        return round(0.335 + ((score - 0.75) / 0.15) * 0.30, 4)
    return round(0.635 + ((score - 0.90) / 0.10) * 0.20, 4)


def ifrs9_stage(days_past_due: int, sicr_flag: bool, default_flag: bool) -> str:
    if default_flag or days_past_due >= 90:
        return "Stage 3"
    if sicr_flag or days_past_due >= 30:
        return "Stage 2"
    return "Stage 1"


# =========================================================
# EXPLAINABILITY
# =========================================================
def build_explainability_features(
    *,
    credit_score: int,
    debt_to_income_ratio: float,
    income: float,
    fraud_score: float,
    amount: float,
) -> List[Dict[str, float]]:
    features = [
        {"feature": "credit_score", "impact": round((650 - credit_score) / 120, 3)},
        {"feature": "debt_to_income_ratio", "impact": round(debt_to_income_ratio, 3)},
        {"feature": "net_monthly_income", "impact": round(-(income / 100000), 3)},
        {"feature": "fraud_score", "impact": round(fraud_score, 3)},
        {"feature": "requested_amount", "impact": round(amount / 500000, 3)},
    ]

    for item in features:
        item["abs_impact"] = abs(item["impact"])

    features.sort(key=lambda x: x["abs_impact"], reverse=True)
    return features


def safe_llm_explanation(
    *,
    decision_type: str,
    credit_score: int,
    income: float,
    debt: float,
    decision: str,
    risk: float,
    dti: float,
    fraud_score: float,
    stage: str,
    top_features: List[Dict[str, float]],
) -> str:
    fallback = (
        f"{decision_type.title()} decision: {decision}. "
        f"Credit score is {credit_score}, debt-to-income ratio is {dti:.2f}, "
        f"risk probability is {risk:.2%}, fraud score is {fraud_score:.2f}, "
        f"and IFRS 9 stage is {stage}. "
        f"Main drivers: {', '.join([x['feature'] for x in top_features[:3]])}."
    )

    try:
        explanation = generate_explanation(
            {
                "credit_score": credit_score,
                "income": income,
                "debt": debt,
                "decision": decision,
                "risk": risk,
                "decision_type": decision_type,
            }
        )

        if not explanation:
            return fallback

        if "insufficient_quota" in str(explanation).lower():
            return fallback

        return str(explanation)

    except Exception:
        return fallback


# =========================================================
# DATABASE SAVE
# =========================================================
def ensure_data_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "application_reference": None,
        "customer_name": None,
        "decision_type": None,
        "product_type": None,
        "requested_amount": 0.0,
        "approved_amount": 0.0,
        "approved_limit": 0.0,
        "monthly_payment": 0.0,
        "net_monthly_income": 0.0,
        "monthly_expenses": 0.0,
        "existing_debt_payments": 0.0,
        "credit_score": 0,
        "debt_to_income_ratio": 0.0,
        "fraud_score": 0.0,
        "risk_probability": 0.0,
        "probability_default": 0.0,
        "ifrs9_stage": "Stage 1",
        "ecl_lifetime": 0.0,
        "final_decision": None,
        "llm_explanation": "",
        "created_at": None,
        "alert_level": "Low",
    }

    for key, value in defaults.items():
        data.setdefault(key, value)

    return data


def get_existing_columns(schema_name: str, table_name: str) -> List[str]:
    if engine is None:
        return []

    sql = text(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = :schema_name
          AND TABLE_NAME = :table_name
        ORDER BY ORDINAL_POSITION
        """
    )

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                sql,
                {
                    "schema_name": schema_name,
                    "table_name": table_name,
                },
            ).fetchall()

        return [row[0] for row in rows]

    except Exception:
        return []


def dynamic_insert(schema_name: str, table_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if engine is None:
        return {
            "saved": False,
            "message": "Database engine is not configured.",
        }

    columns = get_existing_columns(schema_name, table_name)

    if not columns:
        return {
            "saved": False,
            "message": f"Table {schema_name}.{table_name} was not found.",
        }

    insertable_columns = [col for col in columns if col in data]

    if "created_at" in columns:
        insertable_columns = [col for col in insertable_columns if col != "created_at"]

    if not insertable_columns:
        return {
            "saved": False,
            "message": f"No matching columns found for {schema_name}.{table_name}.",
        }

    column_sql = ", ".join(insertable_columns)
    value_sql = ", ".join([f":{col}" for col in insertable_columns])

    if "created_at" in columns:
        column_sql = f"{column_sql}, created_at"
        value_sql = f"{value_sql}, GETDATE()"

    sql = text(
        f"""
        INSERT INTO {schema_name}.{table_name} ({column_sql})
        VALUES ({value_sql})
        """
    )

    payload = {col: data.get(col) for col in insertable_columns}

    try:
        with engine.begin() as conn:
            conn.execute(sql, payload)

        return {
            "saved": True,
            "message": f"Saved to {schema_name}.{table_name}.",
        }

    except Exception as exc:
        return {
            "saved": False,
            "message": str(exc),
        }


def save_application(result: Dict[str, Any]) -> Dict[str, Any]:
    data = ensure_data_keys(result.copy())

    log_status = dynamic_insert("ml", "prediction_log", data)
    fact_status = dynamic_insert("analytics", "fact_applications", data)

    overall_saved = log_status.get("saved") or fact_status.get("saved")

    return {
        "saved": bool(overall_saved),
        "ml_prediction_log": log_status,
        "analytics_fact_applications": fact_status,
    }


# =========================================================
# LOAN SCORING ENGINE
# =========================================================
def assess_loan(req: LoanRequest) -> Dict[str, Any]:
    requested_amount = to_float(req.requested_amount)
    income = to_float(req.net_monthly_income)
    expenses = to_float(req.monthly_expenses)
    debt = to_float(req.existing_debt_payments)
    annual_interest_rate = to_float(req.annual_interest_rate)
    term_months = max(1, to_int(req.term_months, 60))
    credit_score = to_int(req.credit_score)
    fraud_score = normalize_fraud_score(req.fraud_score)
    affordability_factor = to_float(req.affordability_factor, 0.70)
    debt_to_income_cap = to_float(req.debt_to_income_cap, 0.45)
    stress_rate_addon = to_float(req.stress_rate_addon, 2.0)
    days_past_due = to_int(req.days_past_due)
    sicr_flag = to_bool(req.sicr_flag)
    default_flag = to_bool(req.default_flag)

    disposable_income = max(0.0, income - expenses - debt)
    dti = debt / income if income > 0 else 1.0

    credit_component = max(0.0, min(1.0, (700 - credit_score) / 500))
    dti_component = min(1.0, dti)
    amount_component = min(1.0, requested_amount / max(income * 12, 1))
    fraud_component = fraud_risk_component(fraud_score)

    risk_probability = round(
        min(
            0.95,
            max(
                0.02,
                0.50 * credit_component
                + 0.25 * dti_component
                + 0.15 * amount_component
                + 0.10 * fraud_component,
            ),
        ),
        4,
    )

    probability_default = risk_probability

    monthly_rate = ((annual_interest_rate + stress_rate_addon) / 100) / 12

    if monthly_rate > 0:
        monthly_payment = requested_amount * (
            (monthly_rate * (1 + monthly_rate) ** term_months)
            / (((1 + monthly_rate) ** term_months) - 1)
        )
    else:
        monthly_payment = requested_amount / term_months

    max_affordable_payment = disposable_income * affordability_factor
    stage = ifrs9_stage(days_past_due, sicr_flag, default_flag)

    if fraud_decline_required(fraud_score):
        final_decision = "DECLINED"
    elif default_flag:
        final_decision = "DECLINED"
    elif dti > debt_to_income_cap:
        final_decision = "DECLINED"
    elif monthly_payment > max_affordable_payment:
        final_decision = "DECLINED"
    elif risk_probability >= 0.60:
        final_decision = "DECLINED"
    elif fraud_review_required(fraud_score):
        final_decision = "REVIEW"
    elif risk_probability >= 0.40:
        final_decision = "REVIEW"
    else:
        final_decision = "APPROVED"

    approved_amount = requested_amount if final_decision == "APPROVED" else 0.0
    ecl_lifetime = round(probability_default * max(requested_amount, 1) * 0.12, 2)

    explainability = build_explainability_features(
        credit_score=credit_score,
        debt_to_income_ratio=dti,
        income=income,
        fraud_score=fraud_score,
        amount=requested_amount,
    )

    llm_explanation = safe_llm_explanation(
        decision_type="loan",
        credit_score=credit_score,
        income=income,
        debt=debt,
        decision=final_decision,
        risk=risk_probability,
        dti=dti,
        fraud_score=fraud_score,
        stage=stage,
        top_features=explainability,
    )

    return {
        "application_reference": make_reference("LOAN"),
        "customer_name": req.customer_name,
        "decision_type": "loan",
        "product_type": req.product_type,
        "requested_amount": round(requested_amount, 2),
        "approved_amount": round(approved_amount, 2),
        "approved_limit": 0.0,
        "monthly_payment": round(monthly_payment, 2),
        "net_monthly_income": round(income, 2),
        "monthly_expenses": round(expenses, 2),
        "existing_debt_payments": round(debt, 2),
        "credit_score": credit_score,
        "debt_to_income_ratio": round(dti, 4),
        "fraud_score": round(fraud_score, 4),
        "risk_probability": risk_probability,
        "probability_default": probability_default,
        "ifrs9_stage": stage,
        "ecl_lifetime": ecl_lifetime,
        "final_decision": final_decision,
        "llm_explanation": llm_explanation,
        "explainability": explainability,
        "alert_level": fraud_alert_from_score(fraud_score),
        "created_at": datetime.utcnow().isoformat(),
    }


# =========================================================
# ROUTE
# =========================================================
@router.post("/assess")
def loan_assess(req: LoanRequest):
    result = assess_loan(req)
    result["save_status"] = save_application(result)
    return result