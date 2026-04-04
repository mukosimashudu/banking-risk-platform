from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap


FEATURES: List[str] = [
    "net_monthly_income",
    "monthly_expenses",
    "existing_debt_payments",
    "requested_amount",
    "annual_interest_rate",
    "term_months",
    "credit_score",
    "fraud_score",
    "property_value",
    "deposit",
    "days_past_due",
    "sicr_flag",
    "default_flag",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _prepare_row(application: Dict[str, Any]) -> Dict[str, float]:
    return {
        "net_monthly_income": _safe_float(application.get("net_monthly_income")),
        "monthly_expenses": _safe_float(application.get("monthly_expenses")),
        "existing_debt_payments": _safe_float(application.get("existing_debt_payments")),
        "requested_amount": _safe_float(application.get("requested_amount")),
        "annual_interest_rate": _safe_float(application.get("annual_interest_rate")),
        "term_months": _safe_float(application.get("term_months")),
        "credit_score": _safe_float(application.get("credit_score")),
        "fraud_score": _safe_float(application.get("fraud_score")),
        "property_value": _safe_float(application.get("property_value")),
        "deposit": _safe_float(application.get("deposit")),
        "days_past_due": _safe_float(application.get("days_past_due")),
        "sicr_flag": 1.0 if application.get("sicr_flag", False) else 0.0,
        "default_flag": 1.0 if application.get("default_flag", False) else 0.0,
    }


def _load_credit_model():
    model_path = os.path.join("models", "credit_model_best.pkl")
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def _background_from_row(row: Dict[str, float]) -> pd.DataFrame:
    rows = []
    factors = [0.80, 0.95, 1.00, 1.10, 1.20]
    for f in factors:
        r = row.copy()
        r["net_monthly_income"] = row["net_monthly_income"] * f
        r["monthly_expenses"] = row["monthly_expenses"] * min(1.20, max(0.85, f))
        r["existing_debt_payments"] = row["existing_debt_payments"] * min(1.20, max(0.85, f))
        r["requested_amount"] = row["requested_amount"] * min(1.30, max(0.75, f))
        r["credit_score"] = max(300, min(900, row["credit_score"] + (f - 1.0) * 120))
        r["fraud_score"] = max(0.0, min(1.0, row["fraud_score"] + (1.0 - f) * 0.10))
        rows.append(r)
    return pd.DataFrame(rows, columns=FEATURES)


def _heuristic_explanation(row: Dict[str, float]) -> Dict[str, Any]:
    income = max(1.0, row["net_monthly_income"])
    debt = row["existing_debt_payments"]
    expenses = row["monthly_expenses"]
    requested = row["requested_amount"]
    credit_score = row["credit_score"]
    fraud_score = row["fraud_score"]
    dpd = row["days_past_due"]
    sicr = row["sicr_flag"]
    default = row["default_flag"]

    dti = debt / income
    expense_ratio = expenses / income
    requested_income_ratio = requested / max(1.0, income * 12)

    contributions = [
        {"feature": "existing_debt_payments", "feature_value": debt, "shap_value": round(dti * 1.8, 6)},
        {"feature": "monthly_expenses", "feature_value": expenses, "shap_value": round(expense_ratio * 1.2, 6)},
        {"feature": "requested_amount", "feature_value": requested, "shap_value": round(requested_income_ratio * 1.0, 6)},
        {"feature": "credit_score", "feature_value": credit_score, "shap_value": round(-((credit_score - 300) / 600) * 1.5, 6)},
        {"feature": "fraud_score", "feature_value": fraud_score, "shap_value": round(fraud_score * 2.5, 6)},
        {"feature": "days_past_due", "feature_value": dpd, "shap_value": round((dpd / 90.0) * 1.2, 6)},
        {"feature": "sicr_flag", "feature_value": sicr, "shap_value": round(sicr * 1.0, 6)},
        {"feature": "default_flag", "feature_value": default, "shap_value": round(default * 2.0, 6)},
    ]

    top = []
    for item in contributions:
        top.append(
            {
                "feature": item["feature"],
                "feature_value": float(item["feature_value"]),
                "shap_value": float(item["shap_value"]),
                "impact_direction": "increases risk" if item["shap_value"] > 0 else "reduces risk",
                "abs_impact": abs(float(item["shap_value"])),
            }
        )

    top = sorted(top, key=lambda x: x["abs_impact"], reverse=True)[:8]
    risk_probability = max(0.0, min(1.0, 0.50 + sum(i["shap_value"] for i in top[:4]) / 10.0))

    return {
        "available": True,
        "risk_probability": round(risk_probability, 6),
        "base_value": 0.50,
        "top_features": top,
        "message": "Fallback explainability used.",
    }


def explain_application(application: Dict[str, Any], max_features: int = 8) -> Dict[str, Any]:
    row = _prepare_row(application)
    model = _load_credit_model()

    if model is None:
        return _heuristic_explanation(row)

    try:
        sample_df = pd.DataFrame([row], columns=FEATURES)
        background_df = _background_from_row(row)

        def predict_fn(X):
            df = pd.DataFrame(X, columns=FEATURES)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df)
                if probs.ndim == 2 and probs.shape[1] > 1:
                    return probs[:, 1]
                return probs.reshape(-1)
            preds = model.predict(df)
            return np.array(preds).reshape(-1)

        explainer = shap.Explainer(predict_fn, background_df)
        explanation = explainer(sample_df)

        values = np.array(explanation.values)
        if values.ndim == 2:
            shap_row = values[0]
        elif values.ndim == 3:
            shap_row = values[0, :, 0]
        else:
            shap_row = values.reshape(-1)

        top = []
        for feature_name, shap_value in zip(FEATURES, shap_row):
            top.append(
                {
                    "feature": feature_name,
                    "feature_value": float(sample_df.iloc[0][feature_name]),
                    "shap_value": float(shap_value),
                    "impact_direction": "increases risk" if shap_value > 0 else "reduces risk",
                    "abs_impact": abs(float(shap_value)),
                }
            )

        top = sorted(top, key=lambda x: x["abs_impact"], reverse=True)[:max_features]
        risk_probability = float(predict_fn(sample_df.to_numpy())[0])

        base_value = explanation.base_values
        if isinstance(base_value, np.ndarray):
            base_value = float(np.array(base_value).reshape(-1)[0])
        else:
            base_value = float(base_value)

        return {
            "available": True,
            "risk_probability": round(risk_probability, 6),
            "base_value": round(base_value, 6),
            "top_features": top,
            "message": "SHAP explanation generated successfully.",
        }

    except Exception:
        return _heuristic_explanation(row)