from pydantic import BaseModel, Field


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