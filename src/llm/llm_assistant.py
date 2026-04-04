from __future__ import annotations

import os
from typing import Any, Dict

from openai import OpenAI


def _money(value: Any) -> str:
    try:
        return f"R {float(value):,.2f}"
    except Exception:
        return str(value)


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return str(value)


def _fallback_explanation(data: Dict[str, Any]) -> str:
    decision = data.get("decision", "Refer")
    reason = data.get("reason", "manual review required")
    product_type = data.get("product_type", "credit product")
    requested = _money(data.get("requested", 0))
    approved = _money(data.get("approved", 0))
    payment = _money(data.get("monthly_payment", 0))
    income = _money(data.get("income", 0))
    expenses = _money(data.get("expenses", 0))
    debt = _money(data.get("debt", 0))
    pd_12m = _pct(data.get("pd", 0))
    lgd = _pct(data.get("lgd", 0))
    ecl = _money(data.get("ecl", 0))
    stage = data.get("stage", "Stage 1")
    fraud_score = data.get("fraud_score", 0)

    return (
        f"The application for {product_type} was assessed as **{decision}**. "
        f"The applicant requested {requested} and the system approved {approved}. "
        f"Affordability was evaluated against income of {income}, expenses of {expenses}, "
        f"and existing debt commitments of {debt}. The estimated monthly repayment is {payment}. "
        f"From a credit risk perspective, the account was classified as {stage} under IFRS 9, "
        f"with PD of {pd_12m}, LGD of {lgd}, and expected credit loss of {ecl}. "
        f"The fraud score was {fraud_score:.2f}. Main decision reason: {reason}.\n\n"
        f"- **Affordability:** Based on income, expenses, and existing obligations.\n"
        f"- **Risk:** Credit risk and IFRS 9 impairment were included.\n"
        f"- **Outcome:** Final decision is {decision} with approved value of {approved}."
    )


def generate_explanation(data: Dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_explanation(data)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
You are a senior banking credit analyst.

Write a professional credit decision explanation for this case.

Inputs:
- Product type: {data.get("product_type")}
- Income: {data.get("income")}
- Expenses: {data.get("expenses")}
- Debt: {data.get("debt")}
- Credit score: {data.get("credit_score")}
- Fraud score: {data.get("fraud_score")}
- Requested amount: {data.get("requested")}
- Approved amount: {data.get("approved")}
- Monthly payment: {data.get("monthly_payment")}
- IFRS 9 stage: {data.get("stage")}
- PD 12M: {data.get("pd")}
- LGD: {data.get("lgd")}
- ECL: {data.get("ecl")}
- Final decision: {data.get("decision")}
- Reason: {data.get("reason")}

Instructions:
- Keep it short and professional.
- Use one paragraph and three bullet points.
- Mention affordability, credit risk, IFRS 9, and fraud risk.
- Do not say you are an AI.
"""

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = response.choices[0].message.content
        return text.strip() if text else _fallback_explanation(data)
    except Exception:
        return _fallback_explanation(data)