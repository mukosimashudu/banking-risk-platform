from fastapi import FastAPI

from api.routes.health_routes import router as health_router
from api.routes.loan_routes import router as loan_router
from api.routes.credit_routes import router as credit_router
from api.routes.portfolio_routes import router as portfolio_router
from api.routes.fraud_routes import router as fraud_router
from api.routes.chat_routes import router as chat_router
from api.routes.agent_routes import router as agent_router


app = FastAPI(
    title="Banking Risk Intelligence Platform API",
    description="FastAPI backend for loan assessment, credit assessment, fraud monitoring, portfolio analytics, SQL chatbot, and agentic AI investigation.",
    version="2.0.0",
)


app.include_router(health_router)
app.include_router(loan_router)
app.include_router(credit_router)
app.include_router(portfolio_router)
app.include_router(fraud_router)
app.include_router(chat_router)
app.include_router(agent_router)


@app.get("/")
def home():
    return {
        "message": "Banking Risk Intelligence Platform API is running",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }