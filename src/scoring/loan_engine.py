from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LoanComputationResult:
    requested_amount: float
    recommended_amount: float
    approved_amount: float
    monthly_rate: float
    monthly_payment: float
    term_months: int
    total_repayment: float
    total_interest: float
    affordability_pass: bool
    debt_to_income_ratio: float
    expense_to_income_ratio: float
    disposable_income: float
    max_affordable_payment: float
    stressed_monthly_payment: float
    ltv: float | None
    approval_band: str
    principal_share_first_payment: float
    interest_share_first_payment: float


def safe_float(value: float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def monthly_payment(principal: float, annual_interest_rate: float, term_months: int) -> float:
    principal = max(0.0, safe_float(principal))
    annual_interest_rate = max(0.0, safe_float(annual_interest_rate))
    term_months = max(1, int(term_months))

    monthly_rate = annual_interest_rate / 12.0 / 100.0

    if monthly_rate == 0:
        return principal / term_months

    numerator = principal * monthly_rate * (1 + monthly_rate) ** term_months
    denominator = (1 + monthly_rate) ** term_months - 1
    return numerator / denominator


def loan_amount_from_payment(payment: float, annual_interest_rate: float, term_months: int) -> float:
    payment = max(0.0, safe_float(payment))
    annual_interest_rate = max(0.0, safe_float(annual_interest_rate))
    term_months = max(1, int(term_months))

    monthly_rate = annual_interest_rate / 12.0 / 100.0

    if monthly_rate == 0:
        return payment * term_months

    factor = ((1 + monthly_rate) ** term_months - 1) / (monthly_rate * (1 + monthly_rate) ** term_months)
    return payment * factor


def build_amortisation_schedule(
    principal: float,
    annual_interest_rate: float,
    term_months: int,
) -> List[Dict]:
    principal = max(0.0, safe_float(principal))
    term_months = max(1, int(term_months))
    payment = monthly_payment(principal, annual_interest_rate, term_months)
    monthly_rate = max(0.0, safe_float(annual_interest_rate)) / 12.0 / 100.0

    balance = principal
    schedule: List[Dict] = []

    for instalment_no in range(1, term_months + 1):
        interest_component = balance * monthly_rate
        principal_component = payment - interest_component

        if instalment_no == term_months:
            principal_component = balance
            payment_actual = principal_component + interest_component
        else:
            payment_actual = payment

        closing_balance = max(0.0, balance - principal_component)

        schedule.append(
            {
                "instalment_no": instalment_no,
                "opening_balance": round(balance, 2),
                "instalment": round(payment_actual, 2),
                "principal_component": round(principal_component, 2),
                "interest_component": round(interest_component, 2),
                "closing_balance": round(closing_balance, 2),
            }
        )

        balance = closing_balance

    return schedule


def affordability_engine(
    net_monthly_income: float,
    monthly_expenses: float,
    existing_debt_payments: float,
    requested_amount: float,
    annual_interest_rate: float,
    term_months: int,
    affordability_factor: float = 0.70,
    debt_to_income_cap: float = 0.45,
    stress_rate_addon: float = 2.0,
    property_value: float | None = None,
    deposit: float | None = None,
) -> LoanComputationResult:
    net_monthly_income = max(0.0, safe_float(net_monthly_income))
    monthly_expenses = max(0.0, safe_float(monthly_expenses))
    existing_debt_payments = max(0.0, safe_float(existing_debt_payments))
    requested_amount = max(0.0, safe_float(requested_amount))
    annual_interest_rate = max(0.0, safe_float(annual_interest_rate))
    term_months = max(1, int(term_months))
    property_value = None if property_value is None else max(0.0, safe_float(property_value))
    deposit = None if deposit is None else max(0.0, safe_float(deposit))

    disposable_income = max(0.0, net_monthly_income - monthly_expenses - existing_debt_payments)
    income_based_limit = net_monthly_income * debt_to_income_cap
    disposable_based_limit = disposable_income * affordability_factor
    max_affordable_payment = max(0.0, min(income_based_limit, disposable_based_limit))

    recommended_amount = loan_amount_from_payment(
        payment=max_affordable_payment,
        annual_interest_rate=annual_interest_rate,
        term_months=term_months,
    )

    approved_amount = min(requested_amount, recommended_amount)
    approved_monthly_payment = monthly_payment(approved_amount, annual_interest_rate, term_months)
    stressed_monthly_payment = monthly_payment(
        approved_amount,
        annual_interest_rate + stress_rate_addon,
        term_months,
    )

    debt_to_income_ratio = (
        (existing_debt_payments + approved_monthly_payment) / net_monthly_income
        if net_monthly_income > 0
        else 1.0
    )
    expense_to_income_ratio = (monthly_expenses / net_monthly_income) if net_monthly_income > 0 else 1.0

    ltv = None
    if property_value and property_value > 0:
        financed_amount = max(0.0, property_value - (deposit or 0.0))
        if approved_amount > 0:
            financed_amount = approved_amount
        ltv = financed_amount / property_value

    affordability_pass = approved_amount > 0 and stressed_monthly_payment <= max_affordable_payment

    if approved_amount <= 0:
        approval_band = "Decline"
    elif approved_amount < requested_amount:
        approval_band = "Approve with Reduced Amount"
    else:
        approval_band = "Approve"

    schedule = build_amortisation_schedule(approved_amount, annual_interest_rate, term_months)
    first_row = schedule[0] if schedule else {
        "principal_component": 0.0,
        "interest_component": 0.0,
    }

    total_repayment = sum(row["instalment"] for row in schedule)
    total_interest = sum(row["interest_component"] for row in schedule)

    return LoanComputationResult(
        requested_amount=round(requested_amount, 2),
        recommended_amount=round(recommended_amount, 2),
        approved_amount=round(approved_amount, 2),
        monthly_rate=round(annual_interest_rate / 12.0 / 100.0, 8),
        monthly_payment=round(approved_monthly_payment, 2),
        term_months=term_months,
        total_repayment=round(total_repayment, 2),
        total_interest=round(total_interest, 2),
        affordability_pass=affordability_pass,
        debt_to_income_ratio=round(debt_to_income_ratio, 4),
        expense_to_income_ratio=round(expense_to_income_ratio, 4),
        disposable_income=round(disposable_income, 2),
        max_affordable_payment=round(max_affordable_payment, 2),
        stressed_monthly_payment=round(stressed_monthly_payment, 2),
        ltv=round(ltv, 4) if ltv is not None else None,
        approval_band=approval_band,
        principal_share_first_payment=round(first_row["principal_component"], 2),
        interest_share_first_payment=round(first_row["interest_component"], 2),
    )