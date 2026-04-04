from __future__ import annotations

from typing import Optional, Tuple


def make_final_decision(
    product_type: str,
    requested_amount: float,
    approved_amount: float,
    affordability_pass: bool,
    fraud_score: float,
    credit_score: int,
    ifrs9_stage: str,
    debt_to_income_ratio: float,
    ltv: Optional[float],
) -> Tuple[str, str]:
    reasons = []

    if fraud_score >= 0.85:
        return "Decline", "Very high fraud risk"

    if approved_amount <= 0:
        return "Decline", "No affordable amount available"

    if not affordability_pass:
        reasons.append("stress affordability failed")

    if debt_to_income_ratio > 0.50:
        reasons.append("high debt-to-income ratio")

    if credit_score < 560:
        reasons.append("weak credit score")

    if ifrs9_stage == "Stage 3":
        reasons.append("credit impaired exposure")

    if product_type == "home_loan" and ltv is not None and ltv > 0.95:
        reasons.append("high loan-to-value ratio")

    if fraud_score >= 0.60:
        reasons.append("elevated fraud risk")

    if ifrs9_stage == "Stage 2" and approved_amount == requested_amount:
        return "Refer", "Significant increase in credit risk; manual review required"

    if approved_amount < requested_amount:
        if reasons:
            return "Approve with Reduced Amount", "; ".join(reasons)
        return "Approve with Reduced Amount", "Requested amount reduced to affordable level"

    if reasons:
        return "Refer", "; ".join(reasons)

    return "Approve", "Application passes affordability, credit, and IFRS 9 checks"


def make_credit_card_decision(
    income: float,
    debt: float,
    credit_score: int,
    risk_probability: float,
) -> tuple[str, float, str]:
    dti = debt / income if income > 0 else 1.0

    if risk_probability >= 0.65 or credit_score < 540:
        return "Reject", 0.0, "High credit risk"

    if dti > 0.50:
        return "Reject", 0.0, "Debt burden too high"

    if risk_probability >= 0.45 or credit_score < 620:
        return "Approve with Limit", income * 1.5, "Moderate risk profile"

    if risk_probability >= 0.25 or credit_score < 680:
        return "Approve", income * 2.0, "Acceptable risk profile"

    return "Approve", income * 3.0, "Strong profile"