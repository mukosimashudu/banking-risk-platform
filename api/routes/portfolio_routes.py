from fastapi import APIRouter
from sqlalchemy import text

from src.config.db import engine


router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


@router.get("/summary")
def portfolio_summary():
    if engine is None:
        return {}

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM analytics.v_portfolio_summary")
            ).mappings().first()

        return dict(row) if row else {}

    except Exception as exc:
        return {"error": str(exc)}


@router.get("/recent-loans")
def recent_loans():
    if engine is None:
        return []

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM analytics.v_recent_loans")
            ).mappings().all()

        return [dict(r) for r in rows]

    except Exception as exc:
        return {"error": str(exc)}


@router.get("/recent-credit")
def recent_credit():
    if engine is None:
        return []

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM analytics.v_recent_credit")
            ).mappings().all()

        return [dict(r) for r in rows]

    except Exception as exc:
        return {"error": str(exc)}


@router.get("/product-distribution")
def product_distribution():
    if engine is None:
        return []

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM analytics.v_product_distribution")
            ).mappings().all()

        return [dict(r) for r in rows]

    except Exception as exc:
        return {"error": str(exc)}


@router.get("/decision-distribution")
def decision_distribution():
    if engine is None:
        return []

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM analytics.v_decision_distribution")
            ).mappings().all()

        return [dict(r) for r in rows]

    except Exception as exc:
        return {"error": str(exc)}


@router.get("/fraud-distribution")
def fraud_distribution():
    if engine is None:
        return []

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM analytics.v_fraud_distribution")
            ).mappings().all()

        return [dict(r) for r in rows]

    except Exception as exc:
        return {"error": str(exc)}