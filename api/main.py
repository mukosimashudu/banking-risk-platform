# ==========================================
# FULL FINTECH BANKING PLATFORM API
# ==========================================

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import os
from datetime import datetime

# ==========================================
# DATABASE CONFIG
# ==========================================

DB_SERVER = os.getenv("DB_SERVER")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")

connection_string = (
    f"mssql+pyodbc://{DB_USERNAME}:{DB_PASSWORD}@{DB_SERVER}:1433/{DB_DATABASE}"
    f"?driver={DB_DRIVER.replace(' ', '+')}&Encrypt=yes&TrustServerCertificate=no"
)

engine = create_engine(connection_string, pool_pre_ping=True)

# ==========================================
# OPENAI LLM (SAFE VERSION)
# ==========================================

def generate_explanation(data: dict) -> str:
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            return "AI explanation unavailable (no API key configured)."

        client = OpenAI(api_key=api_key)

        prompt = f"""
        Explain this credit decision in simple business terms:

        Credit Score: {data.get("credit_score")}
        Income: {data.get("income")}
        Debt: {data.get("debt")}
        Decision: {data.get("decision")}
        Risk: {data.get("risk")}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    except Exception:
        return "Customer decision based on affordability, credit score and risk thresholds."

# ==========================================
# FASTAPI INIT
# ==========================================

app = FastAPI(title="Fintech Banking Platform API")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
def home():
    return {"message": "Full Fintech Banking Platform API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ==========================================
# REQUEST MODELS
# ==========================================

class CreditRequest(BaseModel):
    customer_name: str
    product_type: str
    net_monthly_income: float
    existing_debt_payments: float
    credit_score: int


class LoanRequest(BaseModel):
    customer_name: str
    requested_amount: float
    credit_score: int


# ==========================================
# CREDIT ASSESSMENT
# ==========================================

@app.post("/api/credit/assess")
def credit_assess(req: CreditRequest):

    income = req.net_monthly_income
    debt = req.existing_debt_payments
    score = req.credit_score

    # Risk calculation
    risk = max(0.05, min(0.95, (700 - score) / 1000 + debt / (income + 1)))

    decision = "APPROVED" if risk < 0.4 else "DECLINED"
    approved_limit = round(income * 5, 2)

    explanation = generate_explanation({
        "credit_score": score,
        "income": income,
        "debt": debt,
        "decision": decision,
        "risk": risk
    })

    # SAVE TO DATABASE
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO ml.prediction_log
                (customer_name, product_type, requested_amount, fraud_flag, created_at)
                VALUES (:name, :product, :amount, :fraud, :created_at)
            """), {
                "name": req.customer_name,
                "product": req.product_type,
                "amount": approved_limit,
                "fraud": 0,
                "created_at": datetime.utcnow()
            })
    except Exception as e:
        print("DB INSERT ERROR:", e)

    return {
        "decision": decision,
        "approved_limit": approved_limit,
        "risk_probability": risk,
        "explanation": explanation
    }


# ==========================================
# LOAN ASSESSMENT
# ==========================================

@app.post("/api/loan/assess")
def loan_assess(req: LoanRequest):

    risk = max(0.05, min(0.95, (700 - req.credit_score) / 1000))
    decision = "APPROVED" if risk < 0.5 else "DECLINED"

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO ml.prediction_log
                (customer_name, product_type, requested_amount, fraud_flag, created_at)
                VALUES (:name, 'loan', :amount, 0, :created_at)
            """), {
                "name": req.customer_name,
                "amount": req.requested_amount,
                "created_at": datetime.utcnow()
            })
    except Exception as e:
        print("DB INSERT ERROR:", e)

    return {
        "decision": decision,
        "risk_probability": risk
    }


# ==========================================
# EXECUTIVE DASHBOARD SUMMARY
# ==========================================

@app.get("/api/portfolio/summary")
def portfolio_summary():

    try:
        with engine.connect() as conn:

            total = conn.execute(text("SELECT COUNT(*) FROM ml.prediction_log")).scalar()

            approved = conn.execute(text("""
                SELECT COUNT(*) FROM ml.prediction_log
                WHERE requested_amount > 0
            """)).scalar()

            avg_amount = conn.execute(text("""
                SELECT AVG(CAST(requested_amount AS FLOAT))
                FROM ml.prediction_log
            """)).scalar() or 0

            fraud_avg = conn.execute(text("""
                SELECT AVG(CAST(fraud_flag AS FLOAT))
                FROM ml.prediction_log
            """)).scalar() or 0

        return {
            "total_applications": total,
            "approved_cases": approved,
            "approval_rate": (approved / total) if total else 0,
            "loan_exposure": float(avg_amount),
            "avg_fraud_score": float(fraud_avg)
        }

    except Exception as e:
        return {"error": str(e)}


# ==========================================
# RECENT APPLICATIONS
# ==========================================

@app.get("/api/portfolio/recent")
def recent_applications():

    try:
        with engine.connect() as conn:

            rows = conn.execute(text("""
                SELECT TOP 20
                    customer_name,
                    product_type,
                    requested_amount,
                    fraud_flag,
                    created_at
                FROM ml.prediction_log
                ORDER BY created_at DESC
            """)).fetchall()

        return [
            {
                "customer_name": r[0],
                "product_type": r[1],
                "requested_amount": float(r[2]),
                "fraud_flag": int(r[3]),
                "created_at": str(r[4])
            }
            for r in rows
        ]

    except Exception as e:
        return {"error": str(e)}