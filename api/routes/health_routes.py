from fastapi import APIRouter
from sqlalchemy import text

from src.config.db import engine


router = APIRouter()


@router.get("/health")
def health():
    if engine is None:
        return {
            "status": "error",
            "database": "engine_not_initialized",
        }

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return {
            "status": "ok",
            "database": "connected",
        }

    except Exception as exc:
        return {
            "status": "error",
            "database": str(exc),
        }