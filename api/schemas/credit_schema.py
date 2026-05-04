from pydantic import BaseModel, Field


class CreditRequest(BaseModel):
    customer_name: str = Field(..., min_length=1)
    product_type: str = "credit_card"
    net_monthly_income: float
    existing_debt_payments: float = 0.0
    credit_score: int
    fraud_score: float = 0.05
    days_past_due: int = 0
    sicr_flag: bool = False
    default_flag: bool = False